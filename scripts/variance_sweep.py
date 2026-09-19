"""Sweep the standard deviation of the asset value and compare the learned
equilibrium spread against the theoretical Nash spread.

Empirically validates Proposition 3.10 (the equilibrium bid-ask spread scales
linearly with the standard deviation of the asset value) by, for each
standard deviation, computing the exact continuous Nash spread on a sample of
the distribution (lama_lab.analysis.get_all_unique_fixed_points /
get_nash_market_making) and learning the symmetric equilibrium with two
AgentBlumMansour makers on the discretized grid, each run at its own
horizon-optimal expert learning rate. Blum-Mansour (rather than plain Exp3) is
used because it is the no-internal-regret algorithm Corollary 4.10 requires.

Each round costs O(n_arms^3) per episode (Blum-Mansour solves one stationary
distribution per episode), so the default grid is coarser than a plain-Exp3
sweep could afford; keep `--delta` and `--episodes` in mind if you widen it.

The asset value is a Gaussian clamped to [0, 1], which departs from the
unbounded location-scale family Proposition 3.10 assumes. Keep `--stds` small
relative to the distance from the mean to the bounds (the default range keeps
the clamped probability mass negligible) so the comparison stays meaningful;
the fraction of samples clamped at each std is logged and saved.

Every standard deviation is an independent run, so on a CPU-only machine they
are run in separate processes (``--jobs``, default: one per logical core).
Torch's own intra-op threading does not help this workload (the per-round
tensors are tiny; measured on an 8-core i7, 8 threads in one process was
slower than 1) and each worker pins itself to a single thread to avoid
oversubscription.

Usage
-----
    python scripts/variance_sweep.py
    python scripts/variance_sweep.py --stds 0.02 0.05 0.08 0.11 0.14 --rounds 50000
    python scripts/variance_sweep.py --jobs 1  # disable multiprocessing
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

# Matches the manuscript's Libertine/newtxmath fonts. This figure is meant to
# sit next to a companion plot within one column (roughly a quarter-page each
# once printed), hence the large base font size relative to the small figsize
# used below.
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
# own tab: palette. Purple matches eta_sweep.py's spread color.
COLOR_SPREAD = "#7E2F8E"

# Roughly half an ACM two-column's ~3.33in column width, matching
# eta_sweep.py's panels, since this is meant to pair with a companion figure.
# No legend is drawn (there is no room for one at this size and font); color
# code in the caption as gray dashed = theoretical Nash spread, purple = learned.
FIGSIZE = (3, 2.5)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def fit_linear_slope(records: list[dict]) -> float:
    """Least-squares slope of the line through the origin, spread = slope * std.

    A line through the origin, rather than a general affine fit, matches
    Proposition 3.10's S*_sigma = sigma * S*_1 exactly.
    """
    stds = torch.tensor([r["std"] for r in records])
    spreads = torch.tensor([r["learned_spread_mean"] for r in records])
    return (stds @ spreads / (stds @ stds)).item()


def plot_variance_sweep(records: list[dict], slope: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")

    stds = [r["std"] for r in records]
    ax.plot(
        stds,
        [r["theoretical_spread"] for r in records],
        "o--",
        color="0.3",
    )
    ax.errorbar(
        stds,
        [r["learned_spread_mean"] for r in records],
        yerr=[r["learned_spread_std"] for r in records],
        fmt="o",
        color=COLOR_SPREAD,
    )

    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("Spread")
    return fig


def run_one_std(task: dict) -> dict | None:
    """Worker entry point: learn the equilibrium for one standard deviation.

    Runs in its own process (see ``main``), so it only takes plain, picklable
    arguments and rebuilds everything it needs from them. Returns ``None`` if
    this distribution has no continuous Nash equilibrium to compare against.
    """
    torch.set_num_threads(1)
    std, seed = task["std"], task["seed"]

    generator = GaussianMixtureGenerator(
        weights=[1.0], means=[task["mean"]], stds=[std], low=0.0, high=1.0
    )
    torch.manual_seed(seed)
    samples = generator.generate(task["n_samples"])
    clamped_mass = ((samples <= 0.0) | (samples >= 1.0)).float().mean().item()

    fixed_points = analysis.get_all_unique_fixed_points(
        samples, eps=task["eps_tol"], tol=task["tol"]
    )
    nash_points = analysis.get_nash_market_making(samples, fixed_points, tol=task["tol"])
    if nash_points.numel() == 0:
        return None

    theoretical_spread = (nash_points[:, 2] - nash_points[:, 0]).mean().item()

    arms = analysis.build_quote_grid(0.0, 1.0, task["delta"], epsilon=task["epsilon"])
    n_arms = arms.shape[0]
    eta = math.sqrt(2.0 * math.log(n_arms) / (task["n_rounds"] * n_arms))
    agent_cfg = {
        "_target_": "lama_lab.agents.AgentBlumMansour",
        "action_space": arms,
        "expert_cfg": {
            "_partial_": True,
            "_target_": "lama_lab.agents.AgentExp3",
            "reward_range": task["reward_range"],
            "eta": eta,
            "gamma": 0.0,
        },
    }
    summary = run_symmetric_duel(
        generator=generator,
        n_episodes=task["n_episodes"],
        n_rounds=task["n_rounds"],
        agent_cfg=agent_cfg,
        window=task["window"],
        epsilon=task["epsilon"],
        seed=seed,
    )

    return {
        "std": std,
        "theoretical_spread": theoretical_spread,
        "learned_spread_mean": summary["spread_mean"].item(),
        "learned_spread_std": summary["spread_std"].item(),
        "clamped_mass": clamped_mass,
        "eta": eta,
    }


def replot(exp_dir: str) -> None:
    """Regenerate the figure from a previous run's saved records, no relearning.

    Reads back exactly what ``main`` wrote out (``records.json``,
    ``fitted_slope.json``) and re-runs only ``plot_variance_sweep``, so
    plot-only tweaks don't require rerunning the sweep.
    """
    exp = ExperimentManager(exp_dir)
    records = exp.load_json("records")
    slope = exp.load_json("fitted_slope")

    fig = plot_variance_sweep(records, slope)
    exp.save_figure("01_variance_sweep", fig, fmt="pdf")
    exp.save_figure("01_variance_sweep", fig)
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
            arms = analysis.build_quote_grid(0.0, 1.0, args.delta, epsilon=args.epsilon)
            n_arms = arms.shape[0]
            logger.info(f"{n_arms} arms on a delta={args.delta} grid.")

            reward_range = tuple(args.reward_range)
            tasks = [
                {
                    "std": std,
                    "mean": args.mean,
                    "n_samples": args.n_samples,
                    "eps_tol": args.eps_tol,
                    "tol": args.tol,
                    "delta": args.delta,
                    "epsilon": args.epsilon,
                    "reward_range": reward_range,
                    "n_episodes": args.episodes,
                    "n_rounds": args.rounds,
                    "window": args.window,
                    "seed": args.seed + i,
                }
                for i, std in enumerate(args.stds)
            ]

            n_jobs = max(1, min(args.jobs, len(tasks)))
            logger.info(f"Running {len(tasks)} stds across {n_jobs} process(es).")
            if n_jobs == 1:
                results = [run_one_std(task) for task in tasks]
            else:
                with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    results = list(pool.map(run_one_std, tasks))

            records = []
            for std, record in zip(args.stds, results):
                if record is None:
                    logger.warning(f"std={std}: no continuous Nash equilibrium found, skipping.")
                    continue
                logger.info(
                    f"std={std:.4f}  theoretical_spread={record['theoretical_spread']:.4f}  "
                    f"learned_spread={record['learned_spread_mean']:.4f}  "
                    f"clamped_mass={record['clamped_mass']:.2e}  eta={record['eta']:.6f}"
                )
                records.append(record)

            if len(records) < 2:
                raise RuntimeError(
                    "Need at least two valid standard deviations to fit a slope."
                )

            slope = fit_linear_slope(records)
            logger.info(f"Fitted slope (learned spread / std): {slope:.4f}")

            fig = plot_variance_sweep(records, slope)
            # Vector PDF for the manuscript; save_all below also drops a PNG
            # preview of the same figure for quick viewing.
            exp.save_figure("01_variance_sweep", fig, fmt="pdf")

            exp.save_all(
                {
                    "config": vars(args),
                    "records": records,
                    "fitted_slope": slope,
                    "01_variance_sweep": fig,
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
        description="Lama Lab - Asset value variance sweep for the market-making game"
    )
    parser.add_argument("--experiment_name", type=str, default="variance_sweep")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        metavar="EXP_DIR",
        help="Path to a previous experiment directory (e.g. "
        "results/<timestamp>_variance_sweep); if given, every other flag is "
        "ignored and the figure is regenerated from its saved records.json "
        "instead of rerunning the sweep.",
    )

    parser.add_argument("--mean", type=float, default=0.5, help="Mean of the Gaussian value distribution.")
    parser.add_argument(
        "--stds",
        type=float,
        nargs="*",
        default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
        help="Standard deviations to sweep.",
    )
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Quote grid tick size. Blum-Mansour's per-episode cost is "
        "O(n_arms^3), so a finer grid than this gets expensive fast; fall "
        "back to 0.2 (matching blum_1fp.yml) if this is too slow.",
    )
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Environment price tolerance.")
    parser.add_argument("--eps_tol", type=float, default=1e-3, help="Fixed-point separation tolerance.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Fixed-point/Nash validation tolerance.")

    parser.add_argument("--rounds", type=int, default=20_000)
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Kept modest because of Blum-Mansour's O(n_arms^3) per-episode cost.",
    )
    parser.add_argument("--window", type=int, default=1_000, help="Final rounds averaged for the learned quotes.")
    parser.add_argument(
        "--reward_range",
        type=float,
        nargs=2,
        default=[-1.0, 0.5],
        help="Bounds used by each Exp3 expert to normalize rewards into losses.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of standard deviations to learn in parallel worker "
        "processes. 1 disables multiprocessing.",
    )

    main(parser.parse_args())
