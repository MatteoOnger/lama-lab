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

Usage
-----
    python scripts/variance_sweep.py
    python scripts/variance_sweep.py --stds 0.02 0.05 0.08 0.11 0.14 --rounds 50000
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


def fit_linear_slope(records: list[dict]) -> float:
    """Least-squares slope of the line through the origin, spread = slope * std.

    A line through the origin, rather than a general affine fit, matches
    Proposition 3.10's S*_sigma = sigma * S*_1 exactly.
    """
    stds = torch.tensor([r["std"] for r in records])
    spreads = torch.tensor([r["learned_spread_mean"] for r in records])
    return (stds @ spreads / (stds @ stds)).item()


def plot_variance_sweep(records: list[dict], slope: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")

    stds = [r["std"] for r in records]
    ax.plot(
        stds,
        [r["theoretical_spread"] for r in records],
        "o-",
        color="black",
        label="Theoretical Nash spread",
    )
    ax.errorbar(
        stds,
        [r["learned_spread_mean"] for r in records],
        yerr=[r["learned_spread_std"] for r in records],
        fmt="s",
        color="tab:blue",
        capsize=3,
        label="Learned spread (Blum-Mansour)",
    )

    fit_x = torch.linspace(0.0, max(stds), 100)
    ax.plot(
        fit_x,
        slope * fit_x,
        "--",
        color="tab:blue",
        alpha=0.6,
        label=f"Linear fit (slope={slope:.3f})",
    )

    ax.set_xlabel("Standard deviation σ of V")
    ax.set_ylabel("Equilibrium bid-ask spread")
    ax.set_title("Equilibrium spread scales linearly with σ (Prop. 3.10)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    return fig


def main(args: argparse.Namespace) -> None:
    manager = ResultsManager(args.results_dir)

    with manager.new_experiment(name=args.experiment_name) as exp:
        logger = setup_logger(exp.file("execution.log"), capture_loggers=["lama_lab"])
        logger.info(f"Device: {torch.get_default_device()}")

        try:
            arms = analysis.build_quote_grid(0.0, 1.0, args.delta, epsilon=args.epsilon)
            n_arms = arms.shape[0]
            logger.info(f"{n_arms} arms on a delta={args.delta} grid.")

            reward_range = tuple(args.reward_range)
            records = []

            for i, std in enumerate(args.stds):
                run_seed = args.seed + i
                torch.manual_seed(run_seed)

                generator = GaussianMixtureGenerator(
                    weights=[1.0], means=[args.mean], stds=[std], low=0.0, high=1.0
                )
                samples = generator.generate(args.n_samples)
                clamped_mass = ((samples <= 0.0) | (samples >= 1.0)).float().mean().item()

                fixed_points = analysis.get_all_unique_fixed_points(
                    samples, eps=args.eps_tol, tol=args.tol
                )
                nash_points = analysis.get_nash_market_making(
                    samples, fixed_points, tol=args.tol
                )
                if nash_points.numel() == 0:
                    logger.warning(f"std={std}: no continuous Nash equilibrium found, skipping.")
                    continue
                if nash_points.shape[0] > 1:
                    logger.warning(
                        f"std={std}: {nash_points.shape[0]} continuous Nash equilibria "
                        "found; averaging their spread for the reference value."
                    )

                theoretical_spread = (nash_points[:, 2] - nash_points[:, 0]).mean().item()

                eta = math.sqrt(2.0 * math.log(n_arms) / (args.rounds * n_arms))
                agent_cfg = {
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
                summary = run_symmetric_duel(
                    generator=generator,
                    n_episodes=args.episodes,
                    n_rounds=args.rounds,
                    agent_cfg=agent_cfg,
                    window=args.window,
                    epsilon=args.epsilon,
                    seed=run_seed,
                )

                logger.info(
                    f"std={std:.4f}  theoretical_spread={theoretical_spread:.4f}  "
                    f"learned_spread={summary['spread_mean'].item():.4f}  "
                    f"clamped_mass={clamped_mass:.2e}  eta={eta:.6f}"
                )
                records.append(
                    {
                        "std": std,
                        "theoretical_spread": theoretical_spread,
                        "learned_spread_mean": summary["spread_mean"].item(),
                        "learned_spread_std": summary["spread_std"].item(),
                        "clamped_mass": clamped_mass,
                        "eta": eta,
                    }
                )

            if len(records) < 2:
                raise RuntimeError(
                    "Need at least two valid standard deviations to fit a slope."
                )

            slope = fit_linear_slope(records)
            logger.info(f"Fitted slope (learned spread / std): {slope:.4f}")

            fig = plot_variance_sweep(records, slope)

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

    main(parser.parse_args())
