"""Sweep the quote-grid tick size Delta and show the learned equilibrium
approaching the continuous Nash price envelope as Delta -> 0 (Theorem 4.8).

For a fixed asset-value distribution, learns the symmetric equilibrium of
the two-maker market-making game with AgentBlumMansour on grids of several
tick sizes, then plots the learned bid and the learned ask, each against its
own continuous Nash reference (lama_lab.analysis.get_all_unique_fixed_points
/ get_nash_market_making), as Delta shrinks. This shows the two components of
the manuscript's envelope directly, rather than the single compounded
distance D_env = |bid - nash_bid| + |ask - nash_ask| (Theorem 4.8's target
quantity, still logged per delta for reference).

Raw per-round compute (O(n_arms^2) to O(n_arms^3) for Blum-Mansour's
stationary-distribution solve) is not actually the binding constraint here.
The manuscript's Corollary 4.10 gets its guarantee from Blum-Mansour's
classical external-to-swap-regret reduction over n_arms Exp3 experts, whose
swap regret is bounded by roughly n_arms times a single expert's own
O(sqrt(T * n_arms * log n_arms)) external regret, i.e.
O(n_arms^1.5 * sqrt(T log n_arms)). For that bound to be a small fraction of
T -- for the guarantee to be non-vacuous, let alone empirically visible --
needs T = Omega(n_arms^3). Empirically this checked out: n_arms=15 visibly
leaves uniform within a few thousand rounds and n_arms=28 partially
converges within 50,000, but n_arms=55/78/120 all plateau at the same
under-converged point within that same budget regardless of learning rate --
the round budget, not the grid, is what has to grow with n_arms. eta is
therefore a plain fixed learning rate (`--eta`, applied to every expert)
rather than the horizon/arm-count formula this script used before -- that
formula is the right one for a single Exp3 expert's own worst-case external
regret, not for what each expert inside this swap-regret construction needs
to move within a feasible T.

Uses AgentBlumMansourExp3 rather than AgentBlumMansour +
AgentExp3-expert_cfg: same algorithm, same math, bit-identical output on the
same seed, but every expert's policy and update is one vectorized tensor op
instead of a Python loop over n_arms expert objects -- 4-7x faster on CPU in
testing, and structured to batch far better on a GPU too, which matters a
lot once T needs to reach into the millions for n_arms > 30 or so.
`--jobs` (worker processes) is a CPU-parallelism device; on a single GPU
runtime pass `--jobs 1` and let one process use the whole card instead of
several processes fighting over its one CUDA context.

Usage
-----
    python scripts/delta_sweep.py
    python scripts/delta_sweep.py --deltas 0.5 0.2 --eta 0.1
    python scripts/delta_sweep.py --jobs 1  # single GPU runtime
"""

import argparse
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
# own tab: palette. Matches eta_sweep.py's bid/ask colors.
COLOR_BID = "#0072BD"
COLOR_ASK = "#D95319"

# Roughly half an ACM two-column's ~3.33in column width, matching the other
# two sweep scripts' panels. No legend is drawn (there is no room for one at
# this size and font); color code in the caption as blue = bid, orange = ask,
# dotted = the corresponding continuous Nash reference.
FIGSIZE = (3, 2.5)

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


def run_one_delta(task: dict) -> dict:
    """Worker entry point: learn the equilibrium for one tick size.

    Runs in its own process (see ``main``), so it only takes plain,
    picklable arguments and rebuilds everything it needs from them.
    """
    torch.set_num_threads(1)

    generator = GaussianMixtureGenerator(
        weights=[1.0], means=[task["mean"]], stds=[task["std"]], low=0.0, high=1.0
    )
    arms = analysis.build_quote_grid(0.0, 1.0, task["delta"], epsilon=task["epsilon"])
    n_arms = arms.shape[0]
    agent_cfg = {
        "_target_": "lama_lab.agents.AgentBlumMansourExp3",
        "action_space": arms,
        "reward_range": task["reward_range"],
        "eta": task["eta"],
    }
    summary = run_symmetric_duel(
        generator=generator,
        n_episodes=task["n_episodes"],
        n_rounds=task["n_rounds"],
        agent_cfg=agent_cfg,
        window=task["window"],
        epsilon=task["epsilon"],
        seed=task["seed"],
    )

    return {
        "delta": task["delta"],
        "n_arms": n_arms,
        "n_episodes": task["n_episodes"],
        "eta": task["eta"],
        "bid_mean": summary["bid_mean"].item(),
        "bid_std": summary["bid_std"].item(),
        "ask_mean": summary["ask_mean"].item(),
        "ask_std": summary["ask_std"].item(),
    }


def plot_delta_sweep(records: list[dict], nash_bid: float, nash_ask: float) -> plt.Figure:
    """Learned bid and ask vs. tick size, each against its own Nash reference.

    One quarter-page-sized figure showing both components of the envelope
    directly, rather than the single compounded
    D_env = |bid - nash_bid| + |ask - nash_ask|.
    """
    records = sorted(records, key=lambda r: r["delta"])
    deltas = [r["delta"] for r in records]

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.errorbar(
        deltas,
        [r["bid_mean"] for r in records],
        yerr=[r["bid_std"] for r in records],
        fmt="o-",
        color=COLOR_BID,
    )
    ax.errorbar(
        deltas,
        [r["ask_mean"] for r in records],
        yerr=[r["ask_std"] for r in records],
        fmt="o-",
        color=COLOR_ASK,
    )
    ax.axhline(nash_bid, color=COLOR_BID, linestyle=":")
    ax.axhline(nash_ask, color=COLOR_ASK, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel("Quote")

    return fig


def replot(exp_dir: str) -> None:
    """Regenerate the figure from a previous run's saved records, no relearning.

    Reads back exactly what ``main`` wrote out (``records.json``,
    ``nash_reference.json``) and re-runs only ``plot_delta_sweep``, so
    plot-only tweaks don't require rerunning the sweep.
    """
    exp = ExperimentManager(exp_dir)
    records = exp.load_json("records")
    nash = exp.load_json("nash_reference")

    fig = plot_delta_sweep(records, nash["bid"], nash["ask"])
    exp.save_figure("01_delta_sweep", fig, fmt="pdf")
    exp.save_figure("01_delta_sweep", fig)
    plt.close("all")
    print(f"Replotted from {exp.path}.")


def main(args: argparse.Namespace) -> None:
    if args.replot:
        replot(args.replot)
        return

    if len(args.deltas) != len(args.episodes):
        raise ValueError(
            f"--deltas and --episodes must have the same length, got "
            f"{len(args.deltas)} and {len(args.episodes)}."
        )

    manager = ResultsManager(args.results_dir)

    with manager.new_experiment(name=args.experiment_name) as exp:
        logger = setup_logger(exp.file("execution.log"), capture_loggers=["lama_lab"])
        logger.info(f"Device: {torch.get_default_device()}")

        try:
            generator = GaussianMixtureGenerator(
                weights=[1.0], means=[args.mean], stds=[args.std], low=0.0, high=1.0
            )
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
                    "averaging their bid/ask for the reference."
                )

            nash_bid = nash_points[:, 0].mean().item()
            nash_ask = nash_points[:, 2].mean().item()
            logger.info(f"Continuous Nash: bid={nash_bid:.4f} ask={nash_ask:.4f}")

            reward_range = tuple(args.reward_range)
            tasks = [
                {
                    "delta": delta,
                    "mean": args.mean,
                    "std": args.std,
                    "epsilon": args.epsilon,
                    "reward_range": reward_range,
                    "eta": args.eta,
                    "n_episodes": episodes,
                    "n_rounds": args.rounds,
                    "window": args.window,
                    "seed": args.seed,
                }
                for delta, episodes in zip(args.deltas, args.episodes)
            ]
            for task in tasks:
                n_arms = analysis.build_quote_grid(
                    0.0, 1.0, task["delta"], epsilon=task["epsilon"]
                ).shape[0]
                logger.info(
                    f"delta={task['delta']}: {n_arms} arms, "
                    f"{task['n_episodes']} episodes."
                )

            n_jobs = max(1, min(args.jobs, len(tasks)))
            logger.info(f"Running {len(tasks)} deltas across {n_jobs} process(es).")
            if n_jobs == 1:
                records = [run_one_delta(task) for task in tasks]
            else:
                with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    records = list(pool.map(run_one_delta, tasks))

            for r in records:
                logger.info(
                    f"delta={r['delta']}  n_arms={r['n_arms']}  "
                    f"bid={r['bid_mean']:.4f}  ask={r['ask_mean']:.4f}  "
                    f"D_env={abs(r['bid_mean'] - nash_bid) + abs(r['ask_mean'] - nash_ask):.4f}"
                )

            fig = plot_delta_sweep(records, nash_bid, nash_ask)
            # Vector PDF for the manuscript; save_all below also drops a PNG
            # preview of the same figure for quick viewing.
            exp.save_figure("01_delta_sweep", fig, fmt="pdf")

            exp.save_all(
                {
                    "config": vars(args),
                    "records": records,
                    "01_delta_sweep": fig,
                    "nash_reference": {
                        "bid": nash_bid,
                        "ask": nash_ask,
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
        description="Lama Lab - Tick size sweep for the market-making game"
    )
    parser.add_argument("--experiment_name", type=str, default="delta_sweep")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        metavar="EXP_DIR",
        help="Path to a previous experiment directory (e.g. "
        "results/<timestamp>_delta_sweep); if given, every other flag is "
        "ignored and the figure is regenerated from its saved records.json "
        "instead of rerunning the sweep.",
    )

    parser.add_argument("--mean", type=float, default=0.5, help="Mean of the Gaussian value distribution.")
    parser.add_argument("--std", type=float, default=0.1, help="Std of the Gaussian value distribution.")
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Environment price tolerance.")
    parser.add_argument("--eps_tol", type=float, default=1e-3, help="Fixed-point separation tolerance.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Fixed-point/Nash validation tolerance.")

    parser.add_argument("--rounds", type=int, default=50_000, help="Same for every tick size.")
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="*",
        default=[0.5, 1.0 / 3.0, 0.25, 0.2],
        help="Tick sizes to sweep, from coarsest to finest -- n_arms = 3, 6, "
        "10, 15 by default. Keep n_arms^3 well under --rounds (see module "
        "docstring); n_arms=210 (delta=0.05) needs tens of millions of "
        "rounds to have a chance of leaving its uniform start, which is why "
        "this script no longer defaults to it.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=[100, 100, 100, 100],
        help="Episode count paired 1:1 with --deltas (same length required). "
        "Uniform by default since n_arms^3 is small across this whole "
        "range, unlike the finer grids this script used to default to.",
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
        "--eta",
        type=float,
        default=0.05,
        help="Fixed learning rate for every Exp3 expert, at every tick size. "
        "Empirically (not the horizon/arm-count formula this script used "
        "before, see module docstring), this moves n_arms=15 visibly away "
        "from uniform within a few thousand rounds.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of deltas to learn in parallel worker processes. 1 "
        "disables multiprocessing. Since per-delta cost varies enormously, "
        "wall-clock is bounded by the single slowest (finest) delta, not the "
        "sum, as long as jobs >= len(deltas).",
    )

    main(parser.parse_args())
