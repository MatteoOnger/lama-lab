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

Every learning rate is an independent run, so on a CPU-only machine they are
run in separate processes (``--jobs``, default: one per logical core). Torch's
own intra-op threading does not help this workload (the per-round tensors are
tiny; measured on an 8-core i7, 8 threads in one process was slower than 1)
and each worker pins itself to a single thread to avoid oversubscription.

Usage
-----
    python scripts/eta_sweep.py
    python scripts/eta_sweep.py --etas 1e-4 3e-4 1e-3 3e-3 1e-2 --rounds 50000
    python scripts/eta_sweep.py --jobs 1  # disable multiprocessing
"""

import argparse
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

import lama_lab.analysis as analysis
from lama_lab.generators import GaussianMixtureGenerator
from lama_lab.utils import ExperimentManager, ResultsManager, run_symmetric_duel, setup_logger

# Disable interactive plotting mode to optimize memory usage
plt.ioff()

# Matches the manuscript's Libertine/newtxmath fonts. Figures are meant to be
# placed two to a column (roughly a quarter-page each once printed), hence the
# large base font size relative to the small figsize used below.
mpl.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "text.latex.preamble": r"""
            \usepackage{libertine}
            \usepackage[libertine]{newtxmath}
        """,
        "axes.linewidth": 0.4,
        "lines.linewidth": 1.5,
        "lines.markersize": 3.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "grid.linewidth": 0.2,
        "axes.grid": True,
        "grid.linestyle": "--",
        "savefig.pad_inches": 0.02,
    }
)

# MATLAB/pgfplots default color order, close to but distinct from matplotlib's
# own tab: palette.
COLOR_BID = "#0072BD"
COLOR_ASK = "#D95319"
COLOR_SPREAD = "#7E2F8E"

# Roughly half an ACM two-column's ~3.33in column width, since two of these
# are meant to sit side by side within one column. No legend is drawn (there
# is no room for one at this size and font); color code in the caption as
# blue = bid, orange = ask, purple = spread, dotted = continuous Nash
# reference, star = the analytically optimal eta.
FIGSIZE = (3, 3)

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


def run_one_eta(task: dict) -> dict:
    """Worker entry point: learn the equilibrium for one eta.

    Runs in its own process (see ``main``), so it only takes plain,
    picklable arguments and rebuilds everything it needs from them, rather
    than receiving a generator or an action-space tensor built by the caller.
    """
    torch.set_num_threads(1)

    generator = GaussianMixtureGenerator(
        weights=[1.0], means=[task["mean"]], stds=[task["std"]], low=0.0, high=1.0
    )
    arms = analysis.build_quote_grid(0.0, 1.0, task["delta"], epsilon=task["epsilon"])

    summary = run_symmetric_duel(
        generator=generator,
        n_episodes=task["n_episodes"],
        n_rounds=task["n_rounds"],
        agent_cfg=build_agent_cfg(arms, task["reward_range"], task["eta"]),
        window=task["window"],
        epsilon=task["epsilon"],
        seed=task["seed"],
    )
    return {"eta": task["eta"]} | {k: v.item() for k, v in summary.items()}


def _style_eta_axis(ax: plt.Axes, optimal_eta: float) -> None:
    """Log-scale eta axis shared by both panels, with eta* marked by a star tick."""
    ax.set_xscale("log")
    ax.axvline(optimal_eta, color="0.5", linestyle="--")
    ax.set_xticks([optimal_eta], labels=[r"$\eta^\star$"], minor=True)
    ax.tick_params(axis="x", which="minor", colors="0.5")
    ax.set_xlabel(r"$\eta$")


def plot_eta_sweep(
    records: list[dict],
    nash_bid: float,
    nash_ask: float,
    nash_spread: float,
    optimal_eta: float,
) -> tuple[plt.Figure, plt.Figure]:
    """Two standalone, quarter-page-sized figures meant to be placed side by side."""
    etas = [r["eta"] for r in records]

    fig_quotes, ax_quotes = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax_quotes.errorbar(
        etas,
        [r["bid_mean"] for r in records],
        yerr=[r["bid_std"] for r in records],
        fmt="o",
        color=COLOR_BID,
    )
    ax_quotes.errorbar(
        etas,
        [r["ask_mean"] for r in records],
        yerr=[r["ask_std"] for r in records],
        fmt="o",
        color=COLOR_ASK,
    )
    ax_quotes.axhline(nash_bid, color=COLOR_BID, linestyle=":")
    ax_quotes.axhline(nash_ask, color=COLOR_ASK, linestyle=":")
    ax_quotes.set_ylabel("Quote")
    _style_eta_axis(ax_quotes, optimal_eta)

    fig_spread, ax_spread = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax_spread.errorbar(
        etas,
        [r["spread_mean"] for r in records],
        yerr=[r["spread_std"] for r in records],
        fmt="o",
        color=COLOR_SPREAD,
    )
    ax_spread.axhline(nash_spread, color=COLOR_SPREAD, linestyle=":")
    ax_spread.set_ylim(bottom=0)
    ax_spread.set_ylabel("Spread")
    _style_eta_axis(ax_spread, optimal_eta)

    return fig_quotes, fig_spread


def replot(exp_dir: str) -> None:
    """Regenerate the figures from a previous run's saved records, no relearning.

    Reads back exactly what ``main`` wrote out (``records.json``,
    ``optimal_eta.json``, ``nash_reference.json``) and re-runs only
    ``plot_eta_sweep``, so plot-only tweaks don't require rerunning the sweep.
    """
    exp = ExperimentManager(exp_dir)
    records = exp.load_json("records")
    optimal_eta = exp.load_json("optimal_eta")
    nash = exp.load_json("nash_reference")

    fig_quotes, fig_spread = plot_eta_sweep(
        records, nash["bid"], nash["ask"], nash["spread"], optimal_eta
    )
    exp.save_figure("01a_eta_sweep_quotes", fig_quotes, fmt="pdf")
    exp.save_figure("01b_eta_sweep_spread", fig_spread, fmt="pdf")
    exp.save_figure("01a_eta_sweep_quotes", fig_quotes)
    exp.save_figure("01b_eta_sweep_spread", fig_spread)
    plt.close("all")
    print(f"Replotted from {exp.path}.")


def main(args: argparse.Namespace) -> None:
    if args.replot:
        replot(args.replot)
        return

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
            tasks = [
                {
                    "eta": eta,
                    "mean": args.mean,
                    "std": args.std,
                    "delta": args.delta,
                    "epsilon": args.epsilon,
                    "reward_range": reward_range,
                    "n_episodes": args.episodes,
                    "n_rounds": args.rounds,
                    "window": args.window,
                    "seed": args.seed,
                }
                for eta in etas
            ]

            n_jobs = max(1, min(args.jobs, len(tasks)))
            logger.info(f"Running {len(tasks)} etas across {n_jobs} process(es).")
            if n_jobs == 1:
                records = [run_one_eta(task) for task in tasks]
            else:
                with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    records = list(pool.map(run_one_eta, tasks))

            fig_quotes, fig_spread = plot_eta_sweep(
                records, nash_bid, nash_ask, nash_spread, optimal_eta
            )
            # Vector PDFs for the manuscript; save_all below also drops PNG
            # previews of the same figures for quick viewing.
            exp.save_figure("01a_eta_sweep_quotes", fig_quotes, fmt="pdf")
            exp.save_figure("01b_eta_sweep_spread", fig_spread, fmt="pdf")

            exp.save_all(
                {
                    "config": vars(args),
                    "records": records,
                    "optimal_eta": optimal_eta,
                    "01a_eta_sweep_quotes": fig_quotes,
                    "01b_eta_sweep_spread": fig_spread,
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
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        metavar="EXP_DIR",
        help="Path to a previous experiment directory (e.g. "
        "results/<timestamp>_eta_sweep); if given, every other flag is "
        "ignored and the figures are regenerated from its saved records.json "
        "instead of rerunning the sweep.",
    )

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
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of etas to learn in parallel worker processes. 1 disables "
        "multiprocessing.",
    )

    main(parser.parse_args())
