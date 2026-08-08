import argparse
import copy
import traceback
import yaml

import matplotlib.pyplot as plt
import torch

import lama_lab.agents as agents
import lama_lab.analysis as analysis
import lama_lab.plotting as plotting
from lama_lab.envs import MarketMakingEnvironment
from lama_lab.generators import GaussianMixtureGenerator
from lama_lab.projectors import MarketMakingProjector
from lama_lab.utils import ResultsManager, RingBuffer, deep_update, setup_logger

# Disable interactive plotting mode to optimize memory usage
plt.ioff()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def run_pipeline(results_dir="./results", config_override=None):
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
                "n_makers": 2,
                "n_episodes": 100,
                "n_rounds": 10_000,
                "epsilon": 0.001,
                "generator": {
                    "weights": [0.0, 1.0, 0.0],
                    "means": [0.15, 0.5, 0.85],
                    "stds": [0.03, 0.10, 0.03],
                    "clamp_min": 0.0,
                    "clamp_max": 1.0,
                },
                "agent": {
                    "type": "AgentPZOMD",
                    "init_x": [0.25, 0.75],
                    "eta_0": 0.05,
                    "delta_0": 1.0,
                    "min_eta": 0.001,
                    "min_delta": 0.001,
                },
                "n_samples": 2**15,
                "history_window": 10,
            }

            config = default_config
            if config_override:
                config = deep_update(default_config, config_override)

            logger.info(
                f"Final Configuration:\n{yaml.dump(config).removesuffix('\n')}."
            )
            logger.info("Setting up environment, generators, and agents.")

            # -------------------------------------------------------------------------
            # Environment & Agents Initialization
            # -------------------------------------------------------------------------
            generator = GaussianMixtureGenerator(
                weights=torch.tensor(config["generator"]["weights"]),
                means=torch.tensor(config["generator"]["means"]),
                stds=torch.tensor(config["generator"]["stds"]),
                clamp_min=config["generator"]["clamp_min"],
                clamp_max=config["generator"]["clamp_max"],
            )

            env = MarketMakingEnvironment(
                n_makers=config["n_makers"],
                n_episodes=config["n_episodes"],
                n_rounds=config["n_rounds"],
                generator_v=generator,
                epsilon=config["epsilon"],
            )

            projector = MarketMakingProjector(
                low=0.0,
                high=1.0,
                epsilon=config["epsilon"],
            )

            agent_config = copy.deepcopy(config["agent"])
            agent_type = agent_config.pop("type")

            if hasattr(agents, agent_type):
                AgentClass = getattr(agents, agent_type)
            else:
                raise ValueError(
                    f"Agent class '{agent_type}' not found in lama_lab.agents."
                )

            makers: list[agents.BaseAgent] = [
                AgentClass(
                    n_episodes=config["n_episodes"],
                    project_fn=projector,
                    name=f"Maker {i}",
                    **agent_config,
                )
                for i in range(config["n_makers"])
            ]

            samples = generator.generate(config["n_samples"])
            fixed_points = analysis.distributions.get_all_unique_fixed_points(
                initial_x_values=[i / 10 for i in range(1, 10)],
                samples=samples,
            )

            # -------------------------------------------------------------------------
            # Buffers
            # -------------------------------------------------------------------------
            n_makers = config["n_makers"]
            n_rounds = config["n_rounds"]
            n_episodes = config["n_episodes"]
            window = config["history_window"]

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
            # Analysis & Metrics Computation
            # -------------------------------------------------------------------------
            logger.info("Computing analysis metrics.")

            expected_spread = fixed_points[:, 2] - fixed_points[:, 0]

            # shape: (n_rounds, n_makers, 2)
            action_means_all = action_history["mean"].get_all()
            action_stds_all = action_history["std"].get_all()

            # shape: (n_rounds, n_makers)
            reward_means_all = reward_history["mean"].get_all()
            reward_stds_all = reward_history["std"].get_all()

            # shape: (history_window, n_episodes, n_makers, 2)
            first_actions = action_history["first"].get_all()
            last_actions = action_history["last"].get_all()

            # shape: (history_window, n_episodes, n_makers)
            first_rewards = reward_history["first"].get_all()
            last_rewards = reward_history["last"].get_all()

            first_actions_reshaped = first_actions.reshape(-1, n_makers, 2)
            first_actions_dispersion = analysis.actions.compute_action_dispersion(
                first_actions_reshaped,
                reduce_action_dim=False,
            )

            last_actions_reshaped = last_actions.reshape(-1, n_makers, 2)
            last_actions_dispersion = analysis.actions.compute_action_dispersion(
                last_actions_reshaped,
                reduce_action_dim=False,
            )

            # Aggregate calculations per agent, preserving the n_makers dimension.
            # shape: (n_makers, 2)
            agent_mean_actions = action_means_all.mean(dim=0)
            agent_std_actions = action_stds_all.mean(dim=0)

            # shape: (n_makers,)
            agent_spreads = agent_mean_actions[:, 1] - agent_mean_actions[:, 0]

            # shape: (n_makers,)
            agent_mean_rewards = reward_means_all.mean(dim=0)
            agent_std_rewards = reward_stds_all.mean(dim=0)

            # shape: (n_makers, 2)
            agent_first_actions = first_actions.mean(dim=(0, 1))
            agent_last_actions = last_actions.mean(dim=(0, 1))

            # shape: (n_makers,)
            agent_first_rewards = first_rewards.mean(dim=(0, 1))
            agent_last_rewards = last_rewards.mean(dim=(0, 1))

            metrics = {
                "fixed_points": {
                    "vals": fixed_points[:, [0, 2]].tolist(),
                    "spread": expected_spread.tolist(),
                },
                "global": {
                    "actions": {
                        "mean": action_means_all.mean(dim=(0, 1)).tolist(),
                        "std": action_stds_all.mean(dim=(0, 1)).tolist(),
                        "first_window_mean": first_actions.mean(dim=(0, 1, 2)).tolist(),
                        "last_window_mean": last_actions.mean(dim=(0, 1, 2)).tolist(),
                        "fist_window_dispersion": first_actions_dispersion.mean().item(),
                        "last_window_dispersion": last_actions_dispersion.mean().item(),
                    },
                    "rewards": {
                        "mean": reward_means_all.mean().item(),
                        "std": reward_stds_all.mean().item(),
                        "first_window_mean": first_rewards.mean().item(),
                        "last_window_mean": last_rewards.mean().item(),
                    },
                },
                "per_agent": {
                    "actions": {
                        "mean": agent_mean_actions.tolist(),
                        "std": agent_std_actions.tolist(),
                        "spread": agent_spreads.tolist(),
                        "first_window_mean": agent_first_actions.tolist(),
                        "last_window_mean": agent_last_actions.tolist(),
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
                action_history["mean"].get_all(),
                action_history["min"].get_all(),
                action_history["max"].get_all(),
                action_history["std"].get_all(),
                reference_prices=fixed_points[:, [0, 2]],
            )
            fig_rewards_scatter = plotting.rewards.plot_rewards_scatter(
                reward_history["mean"].get_all(),
                reward_history["min"].get_all(),
                reward_history["max"].get_all(),
                reward_history["std"].get_all(),
            )
            fig_actions_histo2d = plotting.actions.plot_market_makers_actions_histo2d(
                last_actions_reshaped,
                title="Bid/Ask Actions at the End of Training",
                reference_prices=fixed_points[:, [0, 2]],
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
                "01_distribution": fig_distribution,
                "02_actions_scatter": fig_actions_scatter,
                "03_rewards_scatter": fig_rewards_scatter,
                "04_last_actions_histo2d": fig_actions_histo2d,
                "05_first_dispersion_histo2d": fig_first_dispersion_histo2d,
                "06_last_dispersion_histo2d": fig_last_dispersion_histo2d,
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
            print(f"Error reading config file {args.config}: {e}")
            exit(1)

    run_pipeline(results_dir=args.results_dir, config_override=overrides)
