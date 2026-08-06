import logging
import traceback
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

import lama_lab.analysis as analysis
import lama_lab.plotting as plotting
from lama_lab.agents import AgentPZOMD
from lama_lab.envs import MarketMakingEnvironment
from lama_lab.generators import GaussianMixtureGenerator
from lama_lab.projectors import MarketMakingProjector
from lama_lab.utils import ResultsManager
from lama_lab.utils.buffers import RingBuffer

# Disable interactive plotting mode to optimize memory usage
plt.ioff()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def setup_logger(
    log_path: Path,
    level: int | str = logging.INFO,
    capture_loggers: list[str] | None = None,
) -> logging.Logger:
    """Sets up experiment logging to both console and file.

    Parameters
    ----------
    log_path : Path
        Path to the destination log file.
    level : int or str, default=logging.INFO
        Logging threshold level (e.g., logging.INFO, logging.DEBUG, "INFO").
    capture_loggers : list of str, optional
        Names of additional loggers to capture (e.g., ["lama_lab"]).

    Returns
    -------
    logging.Logger
        Main logger instance for the script.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Shared Handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Define all logger names to attach handlers to
    targets = ["ExperimentLogger"]
    if capture_loggers:
        targets.extend(capture_loggers)

    for logger_name in targets:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logging.getLogger("ExperimentLogger")


def run_pipeline():
    manager = ResultsManager("./results")

    with manager.new_experiment(name="market_making_pzomd") as exp:

        # Capture main script logs and all internal logs from lama_lab package
        logger = setup_logger(
            log_path=exp.file("execution.log"),
            level=logging.INFO,
            capture_loggers=["lama_lab"],
        )

        logger.info("Experiment initialized. Starting pipeline.")

        try:
            # -------------------------------------------------------------------------
            # Configuration
            # -------------------------------------------------------------------------
            config = {
                "n_makers": 2,
                "n_episodes": 2,
                "n_rounds": 10_000,
                "n_samples": 10_000,
                "epsilon": 0.001,
                "device": device,
                "generator": {
                    "weights": [0.0, 1.0, 0.0],
                    "means": [0.15, 0.5, 0.85],
                    "stds": [0.03, 0.10, 0.03],
                    "clamp_min": 0.0,
                    "clamp_max": 1.0,
                },
                "agent": {
                    "init_x": [0.25, 0.75],
                    "eta_0": 0.05,
                    "delta_0": 1.0,
                    "min_eta": 0.001,
                    "min_delta": 0.001,
                },
            }

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

            makers = [
                AgentPZOMD(
                    n_episodes=config["n_episodes"],
                    init_x=config["agent"]["init_x"],
                    project_fn=projector,
                    eta_0=config["agent"]["eta_0"],
                    delta_0=config["agent"]["delta_0"],
                    min_eta=config["agent"]["min_eta"],
                    min_delta=config["agent"]["min_delta"],
                    name=f"Maker {i}",
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

            mean_action_history = RingBuffer(
                n_rounds, shape=(n_makers, 2), device="cpu"
            )
            min_action_history = RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu")
            max_action_history = RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu")
            std_action_history = RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu")

            last_action_history = RingBuffer(
                10,
                shape=(config["n_episodes"], n_makers, 2),
                device="cpu",
            )

            mean_reward_history = RingBuffer(n_rounds, shape=(n_makers,), device="cpu")
            min_reward_history = RingBuffer(n_rounds, shape=(n_makers,), device="cpu")
            max_reward_history = RingBuffer(n_rounds, shape=(n_makers,), device="cpu")
            std_reward_history = RingBuffer(n_rounds, shape=(n_makers,), device="cpu")

            last_reward_history = RingBuffer(
                10, shape=(config["n_episodes"], n_makers), device="cpu"
            )

            # -------------------------------------------------------------------------
            # Simulation Loop
            # -------------------------------------------------------------------------
            logger.info(f"Starting simulation loop for {n_rounds} rounds.")

            for round_idx in range(n_rounds):
                last_actions = torch.stack([maker.act() for maker in makers], dim=1)
                rewards = env.step(last_actions)

                for j, maker in enumerate(makers):
                    maker.update(rewards[:, j])

                mean_action_history.append(last_actions.mean(dim=0))
                min_action_history.append(last_actions.amin(dim=0))
                max_action_history.append(last_actions.amax(dim=0))
                std_action_history.append(last_actions.std(dim=0))

                last_action_history.append(last_actions)

                mean_reward_history.append(rewards.mean(dim=0))
                min_reward_history.append(rewards.amin(dim=0))
                max_reward_history.append(rewards.amax(dim=0))
                std_reward_history.append(rewards.std(dim=0))

                last_reward_history.append(rewards)

                # Progress logging
                if (round_idx + 1) % (n_rounds // 10) == 0:
                    logger.info(
                        f"Progress: {round_idx + 1}/{n_rounds} rounds completed."
                    )

            logger.info("Simulation completed successfully.")

            # -------------------------------------------------------------------------
            # Analysis & Metrics
            # -------------------------------------------------------------------------
            logger.info("Computing analysis metrics.")

            final_actions = last_action_history.get_all().reshape(-1, n_makers, 2)
            final_rewards = last_reward_history.get_all().reshape(-1, n_makers)

            dispersion = analysis.actions.compute_action_dispersion(
                final_actions,
                reduce_action_dim=False,
            )

            metrics = {
                "mean_action": mean_action_history.get_all().mean(0).tolist(),
                "final_mean_action": final_actions.mean(0).tolist(),
                "mean_reward": mean_reward_history.get_all().mean(0).tolist(),
                "final_mean_reward": final_rewards.mean(0).tolist(),
                "dispersion": dispersion.mean().tolist(),
                "std_action": std_action_history.get_all().mean(0).tolist(),
                "std_reward": std_reward_history.get_all().mean(0).tolist(),
            }

            # -------------------------------------------------------------------------
            # Visualization
            # -------------------------------------------------------------------------
            logger.info("Generating figures (headless mode).")

            fig_distribution = plotting.distributions.plot_distribution(
                samples, fixed_points
            )
            fig_actions_scatter = plotting.actions.plot_market_makers_actions_scatter(
                mean_action_history.get_all(),
                min_action_history.get_all(),
                max_action_history.get_all(),
                std_action_history.get_all(),
                reference_prices=fixed_points[:, [0, 2]],
            )
            fig_rewards_scatter = plotting.rewards.plot_rewards_scatter(
                mean_reward_history.get_all(),
                min_reward_history.get_all(),
                max_reward_history.get_all(),
                std_reward_history.get_all(),
            )
            fig_actions_histo2d = plotting.actions.plot_market_makers_actions_histo2d(
                final_actions,
                title="Bid/Ask Actions at the End of Training",
                reference_prices=fixed_points[:, [0, 2]],
            )
            fig_dispersion_histo2d = (
                plotting.actions.plot_market_makers_actions_dispersion_histo2d(
                    dispersion,
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
                "fixed_points": fixed_points,
                "samples": samples,
                "final_actions": final_actions,
                "dispersion": dispersion,
                "mean_action_history": mean_action_history.get_all(),
                "mean_reward_history": mean_reward_history.get_all(),
                "trained_agents": makers,
                "fig_distribution": fig_distribution,
                "fig_actions_scatter": fig_actions_scatter,
                "fig_rewards_scatter": fig_rewards_scatter,
                "fig_actions_histo2d": fig_actions_histo2d,
                "fig_dispersion_histo2d": fig_dispersion_histo2d,
            }
            exp.save_all(artifacts)

            plt.close("all")
            logger.info(f"Experiment successfully saved to: {exp.path}")

        except Exception as e:
            logger.error("An error occurred during execution.")
            logger.error(traceback.format_exc())
            plt.close("all")
            raise


if __name__ == "__main__":
    run_pipeline()
