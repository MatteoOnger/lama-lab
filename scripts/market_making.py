import argparse
import copy
import traceback
import yaml

import matplotlib.pyplot as plt
import torch

import lama_lab.analysis as analysis
import lama_lab.plotting as plotting
from lama_lab.agents import BaseAgent
from lama_lab.diagnostics import Exp3Diagnostics, build_log_checkpoints
from lama_lab.generators import BaseGenerator
from lama_lab.envs import MarketMakingEnvironment
from lama_lab.utils import ResultsManager, RingBuffer, build_from_config, setup_logger

# Numerical stability parameters
EPS = 1e-03
TOL = 1e-03

# Disable interactive plotting mode to optimize memory usage
plt.ioff()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def run_pipeline(config: dict, results_dir: str = "./results"):
    manager = ResultsManager(results_dir)
    exp_name = config.get("experiment_name", None)

    with manager.new_experiment(name=exp_name) as exp:

        logger = setup_logger(
            log_path=exp.file("execution.log"),
            capture_loggers=["lama_lab"],
        )

        logger.info("Experiment initialized. Starting pipeline.")
        logger.info(f"Device: {torch.get_default_device()}")

        try:
            # -------------------------------------------------------------------------
            # Configuration Setup
            # -------------------------------------------------------------------------
            n_makers = config["env"]["n_makers"]
            n_rounds = config["env"]["n_rounds"]
            n_episodes = config["env"]["n_episodes"]
            window = config["history_window"]
            seed = config.get("seed", None)

            logger.info(f"Final Configuration:\n{yaml.dump(config).removesuffix('\n')}")

            if seed is not None:
                torch.manual_seed(seed)
                logger.info(f"Random seed set to {seed}.")

            # -------------------------------------------------------------------------
            # Environment & Agents Initialization
            # -------------------------------------------------------------------------
            logger.info("Building environment, generators, and agents dynamically.")

            # Generator
            generator: BaseGenerator = build_from_config(config["generator"])

            # Environment
            env_kwargs = copy.deepcopy(config["env"])
            env = MarketMakingEnvironment(generator_v=generator, **env_kwargs)

            # Market Makers
            agent_base_cfg = config["agent"]
            makers: list[BaseAgent] = []

            for i in range(n_makers):
                agent_cfg = copy.deepcopy(agent_base_cfg)
                agent_cfg["name"] = f"Maker {i}"
                agent_cfg["n_episodes"] = n_episodes

                agent = build_from_config(agent_cfg)
                makers.append(agent)

            # Generate samples to study distribution
            samples = generator.generate(config["n_samples"])
            fixed_points = analysis.distributions.get_all_unique_fixed_points(
                samples=samples, eps=EPS, tol=TOL
            ).cpu()
            nash_points = analysis.nash.get_nash_market_making(
                samples=samples, fixed_points=fixed_points, tol=TOL
            ).cpu()

            colors = []
            for fp in fixed_points:
                is_nash = torch.any(
                    torch.all(torch.isclose(fp, nash_points), dim=1)
                ).item()
                colors.append("darkred" if is_nash else "tab:blue")

            # -------------------------------------------------------------------------
            # Diagnostics
            # -------------------------------------------------------------------------
            diagnostics_cfg = config.get("diagnostics", None)
            diagnostics = None

            if diagnostics_cfg is not None:
                if n_makers != 2:
                    raise ValueError(
                        f"Diagnostics are implemented for two makers only, "
                        f"got {n_makers}."
                    )
                if any(maker.get_policy() is None for maker in makers):
                    raise ValueError(
                        "Diagnostics require agents exposing a policy over a "
                        "finite action space, such as AgentExp3."
                    )

                # The joint distribution costs n_arms^2 values per episode, so
                # it is tracked on a subset of the independent episodes
                n_diag_episodes = min(
                    diagnostics_cfg.get("n_diag_episodes", 64), n_episodes
                )

                checkpoints = diagnostics_cfg.get("checkpoints", None)
                if checkpoints is None:
                    checkpoints = build_log_checkpoints(n_rounds)
                checkpoints = sorted(
                    {int(c) for c in checkpoints if 0 < int(c) <= n_rounds}
                )

                payoff = analysis.get_expected_payoff_matrix(
                    makers[0].action_space, samples, epsilon=env.epsilon
                )

                # Equilibria of the finite game the makers actually play, which
                # are unrelated to the fixed points of the continuous one
                grid_nash = analysis.get_pure_nash(payoff)

                diagnostics = Exp3Diagnostics(
                    payoff, n_episodes=n_diag_episodes, pure_nash=grid_nash
                )

                diagnostics_history = RingBuffer(
                    len(checkpoints),
                    shape=(len(Exp3Diagnostics.METRIC_NAMES), n_diag_episodes),
                    device="cpu",
                )

                logger.info(
                    f"Diagnostics enabled on {n_diag_episodes} episodes, "
                    f"{payoff.shape[0]} arms, {len(checkpoints)} checkpoints, "
                    f"{grid_nash.shape[0]} pure Nash profiles on the grid."
                )

            # -------------------------------------------------------------------------
            # Buffers
            # -------------------------------------------------------------------------
            action_history = {
                "mean": RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu"),
                "min": RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu"),
                "max": RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu"),
                "std": RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu"),
                "first": RingBuffer(
                    window, shape=(n_episodes, n_makers, 2), device="cpu"
                ),
                "last": RingBuffer(
                    window, shape=(n_episodes, n_makers, 2), device="cpu"
                ),
            }

            reward_history = {
                "mean": RingBuffer(n_rounds, shape=(n_makers,), device="cpu"),
                "min": RingBuffer(n_rounds, shape=(n_makers,), device="cpu"),
                "max": RingBuffer(n_rounds, shape=(n_makers,), device="cpu"),
                "std": RingBuffer(n_rounds, shape=(n_makers,), device="cpu"),
                "first": RingBuffer(window, shape=(n_episodes, n_makers), device="cpu"),
                "last": RingBuffer(window, shape=(n_episodes, n_makers), device="cpu"),
            }

            # -------------------------------------------------------------------------
            # Simulation Loop
            # -------------------------------------------------------------------------
            logger.info(f"Starting simulation loop for {n_rounds} rounds.")

            for round_idx in range(n_rounds):
                # The mixed strategies must be read before the actions are drawn
                if diagnostics is not None:
                    policies = torch.stack(
                        [maker.get_policy()[:n_diag_episodes] for maker in makers]
                    )

                last_actions = torch.stack([maker.act() for maker in makers], dim=1)
                rewards = env.step(last_actions)

                for j, maker in enumerate(makers):
                    maker.update(rewards[:, j])

                if diagnostics is not None:
                    diagnostics.update(
                        policies,
                        torch.stack(
                            [
                                maker.get_last_arms()[:n_diag_episodes]
                                for maker in makers
                            ]
                        ),
                    )

                    if (round_idx + 1) in checkpoints:
                        snapshot = diagnostics.snapshot()
                        diagnostics_history.append(
                            torch.stack(
                                [
                                    snapshot[name]
                                    for name in Exp3Diagnostics.METRIC_NAMES
                                ]
                            )
                        )

                action_history["mean"].append(last_actions.mean(dim=0))
                action_history["min"].append(last_actions.amin(dim=0))
                action_history["max"].append(last_actions.amax(dim=0))
                action_history["std"].append(last_actions.std(dim=0))

                reward_history["mean"].append(rewards.mean(dim=0))
                reward_history["min"].append(rewards.amin(dim=0))
                reward_history["max"].append(rewards.amax(dim=0))
                reward_history["std"].append(rewards.std(dim=0))

                if round_idx < window:
                    action_history["first"].append(last_actions)
                    reward_history["first"].append(rewards)

                if round_idx >= n_rounds - window:
                    action_history["last"].append(last_actions)
                    reward_history["last"].append(rewards)

                if (round_idx + 1) % max(1, (n_rounds // 10)) == 0:
                    logger.info(
                        f"Progress: {round_idx + 1}/{n_rounds} rounds completed."
                    )

            logger.info("Simulation completed successfully.")

            # -------------------------------------------------------------------------
            # Data Extraction
            # -------------------------------------------------------------------------
            logger.info("Extracting data from buffers into tensors.")

            actions_data = {k: v.get_all() for k, v in action_history.items()}
            rewards_data = {k: v.get_all() for k, v in reward_history.items()}

            # -------------------------------------------------------------------------
            # Analysis & Metrics Computation
            # -------------------------------------------------------------------------
            logger.info("Computing analysis metrics.")

            expected_spread = fixed_points[:, 2] - fixed_points[:, 0]

            # shape: (history_window * n_episodes, n_makers, 2)
            first_actions_reshaped = actions_data["first"].reshape(-1, n_makers, 2)
            last_actions_reshaped = actions_data["last"].reshape(-1, n_makers, 2)

            first_actions_dispersion = analysis.actions.compute_action_dispersion(
                first_actions_reshaped,
                reduce_action_dim=False,
            )
            last_actions_dispersion = analysis.actions.compute_action_dispersion(
                last_actions_reshaped,
                reduce_action_dim=False,
            )

            # Per-agent metrics
            # Actions: shape (n_makers, 2)
            agent_mean_actions = actions_data["mean"].mean(dim=0)
            agent_std_actions = actions_data["std"].mean(dim=0)
            agent_first_actions = actions_data["first"].mean(dim=(0, 1))
            agent_last_actions = actions_data["last"].mean(dim=(0, 1))

            # Spreads (Ask - Bid): shape (n_makers,)
            agent_spread_overall = agent_mean_actions[:, 1] - agent_mean_actions[:, 0]
            agent_spread_first = agent_first_actions[:, 1] - agent_first_actions[:, 0]
            agent_spread_last = agent_last_actions[:, 1] - agent_last_actions[:, 0]

            # Rewards: shape (n_makers,)
            agent_mean_rewards = rewards_data["mean"].mean(dim=0)
            agent_std_rewards = rewards_data["std"].mean(dim=0)
            agent_first_rewards = rewards_data["first"].mean(dim=(0, 1))
            agent_last_rewards = rewards_data["last"].mean(dim=(0, 1))

            metrics = {
                "continuous_fixed_points": {
                    "vals": fixed_points[:, [0, 2]].tolist(),
                    "expected_spread": expected_spread.tolist(),
                },
                "continuous_nash_points": {
                    "vals": nash_points[:, [0, 2]].tolist(),
                },
                "global": {
                    "actions": {
                        "mean": actions_data["mean"].mean(dim=(0, 1)).tolist(),
                        "std": actions_data["std"].mean(dim=(0, 1)).tolist(),
                        "first_window_mean": actions_data["first"]
                        .mean(dim=(0, 1, 2))
                        .tolist(),
                        "last_window_mean": actions_data["last"]
                        .mean(dim=(0, 1, 2))
                        .tolist(),
                        "first_window_dispersion": first_actions_dispersion.mean().item(),
                        "last_window_dispersion": last_actions_dispersion.mean().item(),
                    },
                    "spread": {
                        "overall": agent_spread_overall.mean().item(),
                        "first_window": agent_spread_first.mean().item(),
                        "last_window": agent_spread_last.mean().item(),
                    },
                    "rewards": {
                        "mean": rewards_data["mean"].mean().item(),
                        "std": rewards_data["std"].mean().item(),
                        "first_window_mean": rewards_data["first"].mean().item(),
                        "last_window_mean": rewards_data["last"].mean().item(),
                    },
                },
                "per_agent": {
                    "actions": {
                        "mean": agent_mean_actions.tolist(),
                        "std": agent_std_actions.tolist(),
                        "first_window_mean": agent_first_actions.tolist(),
                        "last_window_mean": agent_last_actions.tolist(),
                    },
                    "spread": {
                        "overall": agent_spread_overall.tolist(),
                        "first_window": agent_spread_first.tolist(),
                        "last_window": agent_spread_last.tolist(),
                    },
                    "rewards": {
                        "mean": agent_mean_rewards.tolist(),
                        "std": agent_std_rewards.tolist(),
                        "first_window_mean": agent_first_rewards.tolist(),
                        "last_window_mean": agent_last_rewards.tolist(),
                    },
                    "internal_states": [maker.get_internal_state() for maker in makers],
                },
            }

            if diagnostics is not None:
                # shape: (n_checkpoints, n_metrics, n_diag_episodes)
                diagnostics_table = diagnostics_history.get_all()

                grid_nash_quotes = makers[0].action_space[grid_nash.cpu()]
                metrics["finite_grid_pure_nash"] = {
                    "count": grid_nash.shape[0],
                    "indices": grid_nash.tolist(),
                    "quotes": grid_nash_quotes.tolist(),
                    "payoffs": [
                        [payoff[i, j].item(), payoff[j, i].item()]
                        for i, j in grid_nash.tolist()
                    ],
                }

                # Every episode is an independent replica, hence a seed: report
                # the spread across them rather than a single trajectory
                quantiles = torch.nanquantile(
                    diagnostics_table,
                    torch.tensor([0.25, 0.5, 0.75], device=diagnostics_table.device),
                    dim=-1,
                )

                metrics["diagnostics"] = {
                    "checkpoints": checkpoints,
                    "n_episodes": n_diag_episodes,
                    "payoff_range": diagnostics.payoff_range,
                } | {
                    label: {
                        name: quantiles[q, :, i].tolist()
                        for i, name in enumerate(Exp3Diagnostics.METRIC_NAMES)
                    }
                    for q, label in enumerate(("p25", "median", "p75"))
                }

                reported = (
                    "max_avg_regret",
                    "independence_tv",
                    "max_exploitability",
                    "max_last_exploitability",
                    "support_1",
                    "nash_mass",
                    "policy_drift_1",
                )
                columns = [Exp3Diagnostics.METRIC_NAMES.index(n) for n in reported]

                table = [
                    "Diagnostics (median over episodes):",
                    "       T  " + "  ".join(f"{name:>23}" for name in reported),
                ]
                for row, checkpoint in enumerate(checkpoints):
                    values = "  ".join(
                        f"{quantiles[1, row, col].item():>23.6f}" for col in columns
                    )
                    table.append(f"{checkpoint:>8}  {values}")
                logger.info("\n".join(table))

            # -------------------------------------------------------------------------
            # Visualization
            # -------------------------------------------------------------------------
            logger.info("Generating figures (headless mode).")

            fig_distribution = plotting.plot_1d_histogram(
                samples.cpu(),
                reference_values=fixed_points,
                reference_colors=colors,
                title="Distribution of V",
                xlabel="V",
            )
            fig_actions_history = plotting.plot_history(
                2,
                actions_data["mean"],
                actions_data["min"],
                actions_data["max"],
                actions_data["std"],
                reference_values=fixed_points[:, [0, 2]],
                agent_names=[maker.name for maker in makers],
                feature_names=["Bid", "Ask"],
                feature_colors=["tab:blue", "tab:orange"],
                ylabel="Price",
                title_prefix="Action History",
            )
            fig_rewards_history = plotting.plot_history(
                1,
                rewards_data["mean"],
                rewards_data["min"],
                rewards_data["max"],
                rewards_data["std"],
                reference_values=torch.tensor([0], device="cpu"),
                agent_names=[maker.name for maker in makers],
                feature_names=["Reward"],
                feature_colors=["tab:green"],
                ylabel="Reward",
                title_prefix="Reward History",
            )
            fig_first_actions_histo2d = plotting.plot_2d_histogram(
                first_actions_reshaped,
                reference_values=fixed_points[:, [0, 2]],
                reference_colors=colors,
                subplot_titles=[maker.name for maker in makers],
                feature_names=("Bid Price", "Ask Price"),
                hist_range=[0, 1],
                title="Bid/Ask Actions at the Beginning of Training",
            )
            fig_last_actions_histo2d = plotting.plot_2d_histogram(
                last_actions_reshaped,
                reference_values=fixed_points[:, [0, 2]],
                reference_colors=colors,
                subplot_titles=[maker.name for maker in makers],
                feature_names=("Bid Price", "Ask Price"),
                hist_range=[0, 1],
                title="Bid/Ask Actions at the End of Training",
            )
            fig_first_dispersion_histo2d = plotting.plot_2d_histogram(
                first_actions_dispersion,
                feature_names=("Bid Dispersion", "Ask Dispersion"),
                title="Market Makers Actions Dispersion at the Beginning of Training",
            )
            fig_last_dispersion_histo2d = plotting.plot_2d_histogram(
                last_actions_dispersion,
                feature_names=("Bid Dispersion", "Ask Dispersion"),
                title="Market Makers Actions Dispersion at the End of Training",
            )

            # -------------------------------------------------------------------------
            # Persistence
            # -------------------------------------------------------------------------
            logger.info("Saving all experiment artifacts to disk.")

            artifacts = {
                "config": config,
                "metrics": metrics,
                "trained_agents": makers,
                "actions": actions_data,
                "rewards": rewards_data,
                "01_distribution": fig_distribution,
                "02_actions_scatter": fig_actions_history,
                "03_rewards_scatter": fig_rewards_history,
                "04_first_actions_histo2d": fig_first_actions_histo2d,
                "05_last_actions_histo2d": fig_last_actions_histo2d,
                "06_first_dispersion_histo2d": fig_first_dispersion_histo2d,
                "07_last_dispersion_histo2d": fig_last_dispersion_histo2d,
            }

            if diagnostics is not None:
                distributions = diagnostics.get_distributions()
                artifacts["diagnostics"] = {
                    "table": diagnostics_table,
                    "checkpoints": torch.tensor(checkpoints, device="cpu"),
                    "payoff": diagnostics.payoff.cpu(),
                } | {name: value.cpu() for name, value in distributions.items()}

            exp.save_all(artifacts)

            plt.close("all")
            logger.info(f"Experiment successfully saved to: {exp.path}.")

        except Exception as e:
            logger.error("An error occurred during execution.")
            logger.error(traceback.format_exc())
            plt.close("all")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lama Lab - Market Making Simulation Pipeline"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory where experiment results will be saved.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/market_making/sanity_check.yml",
        help="Path to a YAML configuration file.",
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"Config file '{args.config}' is empty.")
    except Exception as e:
        print(f"Error reading config file {args.config}: {e}.")
        exit(1)

    run_pipeline(config=config, results_dir=args.results_dir)
