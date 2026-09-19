"""Sweep the learning rate of AgentBlumMansour's Exp3 experts, and compare the
learned symmetric equilibrium against the continuous Nash price envelope.

Blum-Mansour is a no-internal-regret (swap-regret) reduction over per-arm
Exp3 experts, which is the guarantee Corollary 4.10 needs; plain external-regret
Exp3 is not swept here for that reason. Blum-Mansour has no single top-level
learning rate: the swept "eta" is the learning rate of its underlying Exp3
experts.

For a fixed asset-value distribution and quote grid, learns the symmetric
equilibrium of the two-maker market-making game for several such learning
rates, then plots the resulting bid/ask/spread against the theoretical Nash
equilibrium of the continuous game (lama_lab.analysis.get_all_unique_fixed_points
/ get_nash_market_making). The learning rate that is analytically optimal for
a single Exp3 expert at the chosen horizon, ``sqrt(2 * ln(K) / (T * K))`` (see
the Notes of AgentExp3), is always included and marked on the plot.

Each round costs O(n_arms^3) per episode (Blum-Mansour solves one stationary
distribution per episode), so keep the grid coarse and n_episodes modest; the
defaults mirror configs/market_making/blum_1fp.yml.

Usage
-----
    python scripts/eta_sweep.py
    python scripts/eta_sweep.py --etas 1e-4 3e-4 1e-3 3e-3 1e-2 --rounds 50000
"""

import argparse
import math
import traceback

import matplotlib.pyplot as plt
import torch

import lama_lab.analysis as analysis
from lama_lab.generators import GaussianMixtureGenerator
from lama_lab.utils import ResultsManager, run_symmetric_duel, setup_logger

# Disable interactive plotting mode to optimize memory usage
plt.ioff()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def build_continuous_reference(
    generator: GaussianMixtureGenerator,
    n_samples: int,
    eps_tol: float,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed points and Nash equilibria of the continuous game for this distribution."""
    samples = generator.generate(n_samples)
    fixed_points = analysis.get_all_unique_fixed_points(samples, eps=eps_tol, tol=tol).cpu()
    nash_points = analysis.get_nash_market_making(samples, fixed_points, tol=tol).cpu()
    return fixed_points, nash_points


def build_agent_cfg(arms: torch.Tensor, reward_range: tuple, eta: float) -> dict:
    """Hydra-style config for a Blum-Mansour learner whose experts use this eta."""
    return {
        "_target_": "lama_lab.agents.AgentBlumMansour",
        "action_space": arms,
        "expert_cfg": {
            "_partial_": True,
            "_target_": "lama_lab.agents.AgentExp3",
            "reward_range": reward_range,
            "eta": eta,
            "gamma": 0.0,
        },
    }


def plot_eta_sweep(
    records: list[dict],
    nash_bid: float,
    nash_ask: float,
    nash_spread: float,
    optimal_eta: float,
) -> plt.Figure:
    fig, (ax_quotes, ax_spread) = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")

    etas = [r["eta"] for r in records]
    ax_quotes.errorbar(
        etas,
        [r["bid_mean"] for r in records],
        yerr=[r["bid_std"] for r in records],
        fmt="o-",
        color="tab:blue",
        markersize=4,
        capsize=3,
        label="Learned bid",
    )
    ax_quotes.errorbar(
        etas,
        [r["ask_mean"] for r in records],
        yerr=[r["ask_std"] for r in records],
        fmt="s--",
        color="tab:orange",
        markersize=4,
        capsize=3,
        label="Learned ask",
    )
    ax_spread.errorbar(
        etas,
        [r["spread_mean"] for r in records],
        yerr=[r["spread_std"] for r in records],
        fmt="o-",
        color="tab:blue",
        markersize=4,
        capsize=3,
        label="Learned spread",
    )

    ax_quotes.axhline(nash_bid, color="black", linestyle=":", label="Nash bid")
    ax_quotes.axhline(nash_ask, color="black", linestyle="-.", label="Nash ask")
    ax_spread.axhline(nash_spread, color="black", linestyle=":", label="Nash spread")

    ax_quotes.set_ylabel("Quote")
    ax_quotes.set_title("Learned bid/ask vs. learning rate")
    ax_spread.set_ylabel("Spread")
    ax_spread.set_title("Learned spread vs. learning rate")

    for ax in (ax_quotes, ax_spread):
        ax.set_xscale("log")
        ax.axvline(optimal_eta, color="gray", linestyle="--", alpha=0.7)
        ax.annotate(
            "optimal η",
            xy=(optimal_eta, 0.97),
            xycoords=("data", "axes fraction"),
            xytext=(3, 0),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="gray",
        )
        ax.set_xlabel("Learning rate η")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    return fig


def main(args: argparse.Namespace) -> None:
    manager = ResultsManager(args.results_dir)

    with manager.new_experiment(name=args.experiment_name) as exp:
        logger = setup_logger(exp.file("execution.log"), capture_loggers=["lama_lab"])
        logger.info(f"Device: {torch.get_default_device()}")

        try:
            torch.manual_seed(args.seed)

            generator = GaussianMixtureGenerator(
                weights=[1.0], means=[args.mean], stds=[args.std], low=0.0, high=1.0
            )
            arms = analysis.build_quote_grid(0.0, 1.0, args.delta, epsilon=args.epsilon)
            n_arms = arms.shape[0]
            logger.info(f"{n_arms} arms on a delta={args.delta} grid.")

            fixed_points, nash_points = build_continuous_reference(
                generator, args.n_samples, args.eps_tol, args.tol
            )
            if nash_points.numel() == 0:
                raise RuntimeError(
                    "No continuous Nash equilibrium found for this distribution."
                )
            if nash_points.shape[0] > 1:
                logger.warning(
                    f"{nash_points.shape[0]} continuous Nash equilibria found; "
                    "averaging their bid/ask for the reference line."
                )

            nash_bid = nash_points[:, 0].mean().item()
            nash_ask = nash_points[:, 2].mean().item()
            nash_spread = nash_ask - nash_bid
            logger.info(
                f"Continuous Nash: bid={nash_bid:.4f} ask={nash_ask:.4f} "
                f"spread={nash_spread:.4f}"
            )

            optimal_eta = math.sqrt(2.0 * math.log(n_arms) / (args.rounds * n_arms))
            etas = sorted(set(args.etas) | {optimal_eta})
            logger.info(f"Optimal eta at T={args.rounds}: {optimal_eta:.6f}")
            logger.info(f"Sweeping etas: {[f'{e:.6g}' for e in etas]}")

            reward_range = tuple(args.reward_range)
            records = []

            for eta in etas:
                logger.info(f"eta={eta:.6g}")
                summary = run_symmetric_duel(
                    generator=generator,
                    n_episodes=args.episodes,
                    n_rounds=args.rounds,
                    agent_cfg=build_agent_cfg(arms, reward_range, eta),
                    window=args.window,
                    epsilon=args.epsilon,
                    seed=args.seed,
                )
                records.append({"eta": eta} | {k: v.item() for k, v in summary.items()})

            fig = plot_eta_sweep(records, nash_bid, nash_ask, nash_spread, optimal_eta)

            exp.save_all(
                {
                    "config": vars(args),
                    "records": records,
                    "optimal_eta": optimal_eta,
                    "01_eta_sweep": fig,
                    "nash_reference": {
                        "bid": nash_bid,
                        "ask": nash_ask,
                        "spread": nash_spread,
                        "continuous_fixed_points": fixed_points.tolist(),
                        "continuous_nash_points": nash_points.tolist(),
                    },
                }
            )
            plt.close("all")
            logger.info(f"Saved to {exp.path}.")

        except Exception:
            logger.error(traceback.format_exc())
            plt.close("all")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lama Lab - Learning rate sweep for the market-making game"
    )
    parser.add_argument("--experiment_name", type=str, default="eta_sweep")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--mean", type=float, default=0.5, help="Mean of the Gaussian value distribution.")
    parser.add_argument("--std", type=float, default=0.1, help="Std of the Gaussian value distribution.")
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--delta", type=float, default=0.2, help="Quote grid tick size.")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Environment price tolerance.")
    parser.add_argument("--eps_tol", type=float, default=1e-3, help="Fixed-point separation tolerance.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Fixed-point/Nash validation tolerance.")

    parser.add_argument("--rounds", type=int, default=20_000)
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Blum-Mansour solves one stationary distribution per episode per "
        "round (O(n_arms^3)), so keep this modest; blum_1fp.yml uses 100.",
    )
    parser.add_argument("--window", type=int, default=1_000, help="Final rounds averaged for the learned quotes.")
    parser.add_argument(
        "--etas",
        type=float,
        nargs="*",
        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        help="Learning rates to sweep, besides the analytically optimal one.",
    )
    parser.add_argument(
        "--reward_range",
        type=float,
        nargs=2,
        default=[-0.8, 0.5],
        help="Bounds used by each Exp3 expert to normalize rewards into losses.",
    )

    main(parser.parse_args())
