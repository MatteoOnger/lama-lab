import argparse
import copy
import traceback
import yaml

import matplotlib.pyplot as plt
import torch

import lama_lab.agents as agents
import lama_lab.analysis as analysis
import lama_lab.generators as generators
import lama_lab.plotting as plotting
import lama_lab.projectors as projectors
from lama_lab.envs import MarketMakingEnvironment
from lama_lab.utils import ResultsManager, RingBuffer, deep_update, setup_logger

# Numerical stability parameters
EPS = 1e-03
TOL = 1e-03

# Disable interactive plotting mode to optimize memory usage
plt.ioff()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def run_pipeline(results_dir: str = "./results", config_override: dict = None):
    manager = ResultsManager(results_dir)
    exp_name = config_override.get("experiment_name", None) if config_override else None

    with manager.new_experiment(name=exp_name) as exp:

        logger = setup_logger(
            log_path=exp.file("execution.log"),
            capture_loggers=["lama_lab"],
        )

        logger.info("Experiment initialized. Starting pipeline.")

        try:
            # -------------------------------------------------------------------------
            # Configuration
            # -------------------------------------------------------------------------
            default_config = {
                "env": {
                    "n_makers": 2,
                    "n_episodes": 100,
                    "n_rounds": 100_000,
                    "epsilon": 0.001,
                },
                "generator": {
                    "type": "GaussianMixtureGenerator",
                    "weights": [0.0, 1.0, 0.0],
                    "means": [0.15, 0.5, 0.85],
                    "stds": [0.03, 0.10, 0.03],
                    "low": 0.0,
                    "high": 1.0,
                },
                "projector": {
                    "type": "MarketMakingProjector",
                    "low": 0.0,
                    "high": 1.0,
                    "epsilon": 0.001,
                },
                "agent": {
                    "type": "AgentPZOMD",
                    "init_x": [0.0, 1.0],
                    "delta_0": 0.5,
                    "eta_0": 0.1,
                    "decay_delta": 0.25,
                    "decay_eta": 0.75,
                    "min_delta": 0.001,
                    "min_eta": 0.001,
                    "max_grad_norm": 5.0,
                },
                "n_samples": 1000000,
                "history_window": 10,
            }

            config = default_config
            if config_override:
                config = deep_update(default_config, config_override)

            window = config["history_window"]
            n_makers = config["env"]["n_makers"]
            n_rounds = config["env"]["n_rounds"]
            n_episodes = config["env"]["n_episodes"]

            logger.info(f"Final Configuration:\n{yaml.dump(config).removesuffix('\n')}")

            # -------------------------------------------------------------------------
            # Environment & Agents Initialization
            # -------------------------------------------------------------------------
            logger.info("Setting up environment, generators, projectors and agents.")

            # Generator
            generator_config: dict = copy.deepcopy(config["generator"])
            gen_type = generator_config.pop("type")

            if hasattr(generators, gen_type):
                GeneratorClass = getattr(generators, gen_type)
                generator: generators.BaseGenerator = GeneratorClass(**generator_config)
            else:
                raise ValueError(
                    f"Generator class '{gen_type}' not found in lama_lab.generators."
                )

            # Environment
            env_config: dict = copy.deepcopy(config["env"])
            env = MarketMakingEnvironment(generator_v=generator, **env_config)

            # Projector (if any)
            projector_config: dict = copy.deepcopy(config.get("projector"))
            projector = None

            if projector_config:
                proj_type = projector_config.pop("type")

                if hasattr(projectors, proj_type):
                    ProjClass = getattr(projectors, proj_type)
                    projector = ProjClass(**projector_config)
                else:
                    raise ValueError(
                        f"Projector class '{proj_type}' not found in lama_lab.projectors."
                    )

            # Market makers
            agent_config: dict = copy.deepcopy(config["agent"])
            agent_type = agent_config.pop("type")

            if hasattr(agents, agent_type):
                AgentClass = getattr(agents, agent_type)
            else:
                raise ValueError(
                    f"Agent class '{agent_type}' not found in lama_lab.agents."
                )

            makers: list[agents.BaseAgent] = []
            for i in range(n_makers):
                agent_kwargs = {
                    "n_episodes": n_episodes,
                    "name": f"Maker {i}",
                    **agent_config,
                }

                if projector is not None:
                    agent_kwargs["project_fn"] = projector

                makers.append(AgentClass(**agent_kwargs))

            # Generate samples to study distribution
            samples = generator.generate(config["n_samples"])
            fixed_points = analysis.distributions.get_all_unique_fixed_points(
                samples=samples, eps=EPS, tol=TOL
            ).cpu()
            nash_points = analysis.nash.get_nash_market_making(
                samples=samples, fixed_points=fixed_points, tol=TOL
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
                last_actions = torch.stack([maker.act() for maker in makers], dim=1)
                rewards = env.step(last_actions)

                for j, maker in enumerate(makers):
                    maker.update(rewards[:, j])

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
                "fixed_points": {
                    "vals": fixed_points[:, [0, 2]].tolist(),
                    "expected_spread": expected_spread.tolist(),
                },
                "nash_points": {
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

            # -------------------------------------------------------------------------
            # Visualization
            # -------------------------------------------------------------------------
            logger.info("Generating figures (headless mode).")

            fig_distribution = plotting.distributions.plot_distribution(
                samples.cpu(), fixed_points
            )
            fig_actions_scatter = plotting.actions.plot_market_makers_actions_scatter(
                actions_data["mean"],
                actions_data["min"],
                actions_data["max"],
                actions_data["std"],
                reference_prices=fixed_points[:, [0, 2]],
            )
            fig_rewards_scatter = plotting.rewards.plot_rewards_scatter(
                rewards_data["mean"],
                rewards_data["min"],
                rewards_data["max"],
                rewards_data["std"],
            )
            fig_first_actions_histo2d = (
                plotting.actions.plot_market_makers_actions_histo2d(
                    first_actions_reshaped,
                    title="Bid/Ask Actions at the Beginning of Training",
                    reference_prices=fixed_points[:, [0, 2]],
                )
            )
            fig_last_actions_histo2d = (
                plotting.actions.plot_market_makers_actions_histo2d(
                    last_actions_reshaped,
                    title="Bid/Ask Actions at the End of Training",
                    reference_prices=fixed_points[:, [0, 2]],
                )
            )
            fig_first_dispersion_histo2d = plotting.actions.plot_market_makers_actions_dispersion_histo2d(
                first_actions_dispersion,
                hist_range=None,
                title="Market Makers Actions Dispersion at the Beginning of Training",
            )
            fig_last_dispersion_histo2d = (
                plotting.actions.plot_market_makers_actions_dispersion_histo2d(
                    last_actions_dispersion,
                    hist_range=None,
                    title="Market Makers Actions Dispersion at the End of Training",
                )
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
                "02_actions_scatter": fig_actions_scatter,
                "03_rewards_scatter": fig_rewards_scatter,
                "04_first_actions_histo2d": fig_first_actions_histo2d,
                "05_last_actions_histo2d": fig_last_actions_histo2d,
                "06_first_dispersion_histo2d": fig_first_dispersion_histo2d,
                "07_last_dispersion_histo2d": fig_last_dispersion_histo2d,
            }
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
        default=None,
        help="Path to a YAML or JSON configuration file with overrides.",
    )
    args = parser.parse_args()

    overrides = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                overrides = yaml.safe_load(f)
                if overrides is None:
                    overrides = {}
        except Exception as e:
            print(f"Error reading config file {args.config}: {e}.")
            exit(1)

    run_pipeline(results_dir=args.results_dir, config_override=overrides)
