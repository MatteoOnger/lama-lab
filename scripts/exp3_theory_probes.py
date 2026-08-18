"""Theory probes for Exp3 equilibrium selection on the finite market-making grid.

Investigates why Exp3 converges to the payoff-dominant pure Nash equilibrium, by
splitting the question into a structural part, which depends only on the payoff
matrix, and a dynamic part, which needs learning runs.

Structural probes (cheap, no learning):

    1  iterated elimination of strictly dominated actions, pure and mixed
    2  strictness margin of every pure Nash equilibrium
    3  payoff ranking against a uniform opponent
    4  payoffs along the path from uniform play to the payoff-dominant action
    5  best-response region of the payoff-dominant action
    6  probes 2 to 5 repeated on the rationalizable subgame
    9  best-response basin of every equilibrium, by sampling the simplex
    13 the same structure across several tick sizes

Dynamic probes (require learning runs):

    7  does an Exp3 trajectory enter the payoff-dominant best-response region
    8  cumulative payoff advantage of the payoff-dominant action
    10 symmetry of the two trajectories
    11 deterministic multiplicative weights, i.e. the same dynamics without
       bandit noise
    12 dependence on the initial policy

Usage
-----
    python scripts/exp3_theory_probes.py --delta 0.2
    python scripts/exp3_theory_probes.py --delta 0.2 --skip-dynamic
    python scripts/exp3_theory_probes.py --delta 0.2 --grid-sweep 0.25 0.2 0.125 0.1

Writes ``probes.json`` and ``report.md`` into a timestamped results directory.
"""

import argparse
import json
import math
import traceback

import torch

from lama_lab.agents import AgentExp3
from lama_lab.analysis import (
    build_quote_grid,
    get_expected_payoff_matrix,
    get_exploitability,
    get_pure_nash,
    get_rationalizable_set,
)
from lama_lab.diagnostics import build_log_checkpoints
from lama_lab.envs import MarketMakingEnvironment
from lama_lab.generators import GaussianMixtureGenerator
from lama_lab.utils import ResultsManager, setup_logger

EPS = 1e-03
TOL = 1e-09


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def get_l1_distance_to_halfspace(
    direction: torch.Tensor,
    point: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Shortest move inside the simplex that makes a linear form non-positive.

    Solves ``min ||q - point||_1`` subject to ``q`` in the simplex and
    ``direction @ q <= 0``. Moving mass from one coordinate to another costs 2
    per unit and buys a reduction equal to the difference of their `direction`
    entries, so sending mass from the largest entries to the single smallest one
    is optimal.

    Parameters
    ----------
    direction : torch.Tensor
        One-dimensional tensor defining the half-space.
    point : torch.Tensor
        Distribution to move away from, of the same shape.

    Returns
    -------
    distance : float
        The L1 distance, ``inf`` when the half-space does not meet the simplex.
    boundary : torch.Tensor
        The closest distribution satisfying the constraint.
    """
    excess = (direction @ point).item()
    if excess <= 0.0:
        return 0.0, point.clone()

    sink = int(direction.argmin())
    if direction[sink].item() > 0.0:
        # Every distribution keeps the form positive, the action is dominant
        return float("inf"), point.clone()

    boundary = point.clone()
    moved = 0.0

    for source in torch.argsort(direction, descending=True).tolist():
        if source == sink or excess <= 1e-15:
            break

        rate = (direction[source] - direction[sink]).item()
        if rate <= 0.0:
            break

        take = min(boundary[source].item(), excess / rate)
        boundary[source] -= take
        boundary[sink] += take
        excess -= take * rate
        moved += take

    return 2.0 * moved, boundary


# ---------------------------------------------------------------------------
# Structural probes
# ---------------------------------------------------------------------------
def probe_iterated_dominance(
    payoff: torch.Tensor,
    tol: float = 1e-06,
    max_mixed_arms: int = 400,
) -> dict:
    """Probe 1: iterated elimination of strictly dominated actions.

    Thin wrapper over :func:`~lama_lab.analysis.get_rationalizable_set`, which
    the pipeline also uses, so the elimination layers reported here are the same
    ones the diagnostics track.
    """
    layers, rounds = get_rationalizable_set(
        payoff, tol=tol, max_mixed_arms=max_mixed_arms
    )
    survivors = layers[-1]

    return {
        "rounds": rounds,
        "layers": layers[:-1],
        "rationalizable_indices": survivors,
        "n_survivors": len(survivors),
        "mixed_test_applied": payoff.shape[0] <= max_mixed_arms,
    }


def probe_nash_strictness(payoff: torch.Tensor, nash: torch.Tensor) -> list[dict]:
    """Probe 2: strictness margin of every symmetric pure Nash equilibrium.

    Parameters
    ----------
    payoff : torch.Tensor
        Payoff matrix of the row maker.
    nash : torch.Tensor
        Pure Nash profiles, of shape ``(n_profiles, 2)``.

    Returns
    -------
    equilibria : list of dict
        One entry per symmetric equilibrium, sorted by payoff.
    """
    equilibria = []

    for action in sorted({i for i, j in nash.tolist() if i == j}):
        column = payoff[:, action].clone()
        own = column[action].item()
        column[action] = -float("inf")

        runner_up = int(column.argmax())
        equilibria.append(
            {
                "action": action,
                "payoff": own,
                "best_deviation_action": runner_up,
                "best_deviation_payoff": column[runner_up].item(),
                "strictness_margin": own - column[runner_up].item(),
            }
        )

    return sorted(equilibria, key=lambda e: -e["payoff"])


def probe_uniform_ranking(payoff: torch.Tensor, dagger: int) -> dict:
    """Probe 3: expected payoff of every action against a uniform opponent."""
    uniform = torch.full((payoff.shape[0],), 1.0 / payoff.shape[0], dtype=payoff.dtype)
    value = payoff @ uniform
    order = torch.argsort(value, descending=True).tolist()

    competitor = value.clone()
    competitor[dagger] = -float("inf")

    return {
        "ranking": [
            {"action": a, "value": value[a].item(), "rank": r}
            for r, a in enumerate(order)
        ],
        "best_action": order[0],
        "payoff_dominant_is_best": order[0] == dagger,
        "gap_to_runner_up": (value[dagger] - competitor.max()).item(),
    }


def probe_uniform_path(payoff: torch.Tensor, dagger: int, steps: int = 1001) -> dict:
    """Probe 4: payoff gaps along the segment from uniform play to the equilibrium.

    Expected payoff is linear in the opponent distribution, so each gap is affine
    in the mixing weight and the endpoints decide the sign. The full curve is
    still evaluated, as a check on that reasoning.
    """
    n_arms = payoff.shape[0]
    uniform = torch.full((n_arms,), 1.0 / n_arms, dtype=payoff.dtype)
    pure = torch.zeros(n_arms, dtype=payoff.dtype)
    pure[dagger] = 1.0

    lam = torch.linspace(0.0, 1.0, steps, dtype=payoff.dtype).unsqueeze(-1)
    opponents = (1.0 - lam) * uniform + lam * pure  # (steps, n_arms)

    value = opponents @ payoff.T  # (steps, n_arms)
    gaps = value[:, dagger : dagger + 1] - value
    gaps[:, dagger] = float("inf")

    minimum, argmin = gaps.min(dim=-1)
    worst = int(minimum.argmin())

    competitors = [a for a in range(n_arms) if a != dagger]
    return {
        "min_gap_over_path": minimum[worst].item(),
        "lambda_of_min": lam[worst].item(),
        "worst_competitor": int(argmin[worst]),
        "always_strict": bool((minimum > 0).all()),
        "best_response_is_dagger_throughout": bool(
            (value.argmax(dim=-1) == dagger).all()
        ),
        "per_competitor": [
            {
                "action": a,
                "gap_at_uniform": gaps[0, a].item(),
                "gap_at_equilibrium": gaps[-1, a].item(),
                "min_gap": gaps[:, a].min().item(),
                "lambda_of_min": lam[int(gaps[:, a].argmin())].item(),
            }
            for a in competitors
        ],
    }


def probe_best_response_region(payoff: torch.Tensor, dagger: int) -> dict:
    """Probe 5: is uniform play inside the payoff-dominant best-response region,
    and how far is its boundary."""
    n_arms = payoff.shape[0]
    uniform = torch.full((n_arms,), 1.0 / n_arms, dtype=payoff.dtype)

    nearest = {"action": None, "distance": float("inf"), "boundary": None}
    gaps = []

    for action in range(n_arms):
        if action == dagger:
            continue

        # The payoff-dominant action stops being a best response once this
        # form turns non-positive, so that is the boundary to reach
        direction = payoff[dagger] - payoff[action]
        gaps.append((direction @ uniform).item())

        distance, boundary = get_l1_distance_to_halfspace(direction, uniform)
        if distance < nearest["distance"]:
            nearest = {
                "action": action,
                "distance": distance,
                "boundary": boundary.tolist(),
            }

    # A single-action game has no competitor, so the region is the whole simplex
    smallest = min(gaps) if gaps else float("inf")

    return {
        "min_gap_at_uniform": smallest,
        "uniform_inside": smallest > 0.0,
        "nearest_competitor": nearest["action"],
        "l1_distance_to_boundary": nearest["distance"],
        "boundary_distribution": nearest["boundary"],
    }


def probe_basins(
    payoff: torch.Tensor,
    equilibria: list[dict],
    n_samples: int = 100_000,
    chunk: int = 10_000,
) -> list[dict]:
    """Probe 9: fraction of the simplex on which each equilibrium is a best response."""
    n_arms = payoff.shape[0]
    actions = [e["action"] for e in equilibria]

    unique = torch.zeros(len(actions), dtype=torch.float64)
    weak = torch.zeros(len(actions), dtype=torch.float64)

    dirichlet = torch.distributions.Dirichlet(
        torch.ones(n_arms, dtype=payoff.dtype, device=payoff.device)
    )

    drawn = 0
    while drawn < n_samples:
        size = min(chunk, n_samples - drawn)
        opponents = dirichlet.sample((size,))
        value = opponents @ payoff.T

        best = value.amax(dim=-1, keepdim=True)
        for k, action in enumerate(actions):
            own = value[:, action : action + 1]
            weak[k] += (own >= best - 1e-12).sum().item()

            # Unique when every other action is strictly worse
            beaten = (value < own - 1e-12).sum(dim=-1)
            unique[k] += (beaten == n_arms - 1).sum().item()
        drawn += size

    uniform = torch.full((n_arms,), 1.0 / n_arms, dtype=payoff.dtype)
    results = []

    for k, action in enumerate(actions):
        # How far uniform play sits from the region where this action wins
        distance = 0.0
        for other in range(n_arms):
            if other == action:
                continue
            direction = payoff[action] - payoff[other]
            if (direction @ uniform).item() > 0.0:
                continue
            # Uniform is outside, so move until this competitor is beaten
            step, _ = get_l1_distance_to_halfspace(-direction, uniform)
            distance = max(distance, step)

        results.append(
            {
                "action": action,
                "unique_best_response_fraction": (unique[k] / n_samples).item(),
                "weak_best_response_fraction": (weak[k] / n_samples).item(),
                "l1_distance_from_uniform": distance,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Dynamic probes
# ---------------------------------------------------------------------------
def _best_response_gap(payoff: torch.Tensor, opponent: torch.Tensor, dagger: int):
    """Advantage of the payoff-dominant action over its best competitor.

    Parameters
    ----------
    payoff : torch.Tensor
        Payoff matrix of the row maker.
    opponent : torch.Tensor
        Opponent policies, of shape ``(n_episodes, n_arms)``.
    dagger : int
        Index of the payoff-dominant action.

    Returns
    -------
    gap : torch.Tensor
        Shape ``(n_episodes,)``, positive when the payoff-dominant action is the
        unique best response.
    best : torch.Tensor
        Index of the best response, of shape ``(n_episodes,)``.
    """
    value = opponent @ payoff.T
    own = value[:, dagger].clone()

    competitors = value.clone()
    competitors[:, dagger] = -float("inf")

    return own - competitors.amax(dim=-1), value.argmax(dim=-1)


def probe_trajectory(
    payoff: torch.Tensor,
    arms: torch.Tensor,
    dagger: int,
    generator,
    n_rounds: int,
    n_episodes: int,
    eta: float,
    reward_range: tuple[float, float],
    logger,
) -> dict:
    """Probes 7, 8 and 10: best-response gap, cumulative advantage, symmetry.

    Runs the real learner against the real environment, and records per round
    the gap of the payoff-dominant action against the opponent's current policy,
    the cumulative advantage over every competitor, and the distance between the
    two players' policies.
    """
    n_arms = payoff.shape[0]
    env = MarketMakingEnvironment(2, n_episodes, n_rounds, generator, epsilon=EPS)
    makers = [
        AgentExp3(n_episodes, arms, reward_range=reward_range, eta=eta)
        for _ in range(2)
    ]

    advantage = torch.zeros((2, n_episodes, n_arms), dtype=payoff.dtype)
    gap_history = []
    symmetry_history = []
    dagger_history = []

    first_entry = [None, None]
    permanent_entry = [None, None]
    positive_rounds = [0, 0]

    for step in range(n_rounds):
        policies = torch.stack([m.get_policy().to(payoff.dtype) for m in makers])

        for player in range(2):
            opponent = policies[1 - player]
            gap, _ = _best_response_gap(payoff, opponent, dagger)

            # Advantage of the payoff-dominant action over each competitor
            value = opponent @ payoff.T
            advantage[player] += value[:, dagger : dagger + 1] - value

            median = gap.median().item()
            if player == 0:
                gap_history.append(median)

            if median > 0.0:
                positive_rounds[player] += 1
                if first_entry[player] is None:
                    first_entry[player] = step + 1
                if permanent_entry[player] is None:
                    permanent_entry[player] = step + 1
            else:
                permanent_entry[player] = None

        symmetry_history.append(
            (policies[0] - policies[1]).abs().sum(-1).median().item()
        )
        dagger_history.append(policies[:, :, dagger].median(dim=-1).values.tolist())

        actions = torch.stack([m.act() for m in makers], dim=1)
        rewards = env.step(actions)
        for j, maker in enumerate(makers):
            maker.update(rewards[:, j])

        if (step + 1) % max(1, n_rounds // 5) == 0:
            logger.info(f"  trajectory probe: {step + 1}/{n_rounds}")

    final = torch.stack([m.get_policy().to(payoff.dtype) for m in makers])
    gap, best = _best_response_gap(payoff, final[1], dagger)

    tail = [
        g
        for g, s in zip(gap_history, range(n_rounds))
        if permanent_entry[0] is not None and s + 1 >= permanent_entry[0]
    ]

    return {
        "first_entry_round": first_entry,
        "permanent_entry_round": permanent_entry,
        "fraction_rounds_positive": [p / n_rounds for p in positive_rounds],
        "min_gap_after_permanent_entry": min(tail) if tail else None,
        "final_gap": gap.median().item(),
        "final_best_response": int(best.median()),
        "final_prob_on_dagger": final[:, :, dagger].median(dim=-1).values.tolist(),
        "gap_history": gap_history,
        "symmetry_history": symmetry_history,
        "dagger_prob_history": dagger_history,
        "cumulative_advantage": {
            "median": advantage[0].median(dim=0).values.tolist(),
            "all_positive": bool(
                (advantage[0].median(dim=0).values > 0).sum() >= n_arms - 1
            ),
        },
    }


def probe_deterministic_mw(
    payoff: torch.Tensor,
    dagger: int,
    n_rounds: int,
    etas: list[float],
    reward_range: tuple[float, float],
    initials: torch.Tensor | None = None,
    labels: list[str] | None = None,
) -> list[dict]:
    """Probes 11 and 12: exponential weights without bandit noise.

    Each player updates against the exact expected payoff of every action
    against the opponent's current mixed strategy, rather than against a sampled
    reward. Under full information gains and losses give the same trajectory,
    because the offset is common to every action and cancels in the
    normalization; the loss form is used here to match the learner.

    Every configuration runs as one batch entry, so a sweep over learning rates
    or initial policies costs the same as a single run.

    Parameters
    ----------
    payoff : torch.Tensor
        Payoff matrix of the row maker.
    dagger : int
        Index of the payoff-dominant action.
    n_rounds : int
        Rounds to simulate.
    etas : list of float
        One learning rate per configuration.
    reward_range : tuple of float
        Bounds used to convert payoffs to losses, matching the learner.
    initials : torch.Tensor, optional
        Initial policies, of shape ``(n_configs, 2, n_arms)``. Defaults to
        uniform.
    labels : list of str, optional
        Names for the configurations, used in the returned records.

    Returns
    -------
    results : list of dict
        One record per configuration.
    """
    n_arms = payoff.shape[0]
    low, high = reward_range

    eta = torch.tensor(etas, dtype=payoff.dtype).reshape(-1, 1, 1)
    n_configs = eta.shape[0]

    if initials is None:
        weights = torch.zeros((n_configs, 2, n_arms), dtype=payoff.dtype)
    else:
        weights = torch.log(initials.clamp_min(1e-300)) / eta

    entered = [None] * n_configs
    checkpoints = build_log_checkpoints(n_rounds)
    history = {"round": [], "prob_on_dagger": [], "gap": []}

    for step in range(n_rounds):
        policies = torch.softmax(eta * weights, dim=-1)

        # Each player faces the other, so the opponent axis is the flipped one
        opponents = policies.flip(1)
        value = opponents @ payoff.T

        own = value[:, :, dagger]
        competitors = value.clone()
        competitors[:, :, dagger] = -float("inf")
        gap = own - competitors.amax(dim=-1)

        for config in range(n_configs):
            if gap[config].min().item() > 0.0:
                if entered[config] is None:
                    entered[config] = step + 1
            else:
                entered[config] = None

        if (step + 1) in checkpoints:
            history["round"].append(step + 1)
            history["prob_on_dagger"].append(policies[:, 0, dagger].tolist())
            history["gap"].append(gap[:, 0].tolist())

        weights = weights - eta * (high - value) / (high - low)

    policies = torch.softmax(eta * weights, dim=-1)
    e1, e2 = get_exploitability(payoff, policies[:, 0], policies[:, 1])
    selected = policies[:, 0].argmax(dim=-1)

    return [
        {
            "label": labels[c] if labels else f"eta={etas[c]:g}",
            "eta": etas[c],
            "selected_action": int(selected[c]),
            "selects_dagger": int(selected[c]) == dagger,
            "prob_on_dagger": policies[c, :, dagger].tolist(),
            "max_prob": policies[c].amax(dim=-1).tolist(),
            "exploitability": max(e1[c].item(), e2[c].item()),
            "permanent_entry_round": entered[c],
            "history": {
                "round": history["round"],
                "prob_on_dagger": [p[c] for p in history["prob_on_dagger"]],
                "gap": [g[c] for g in history["gap"]],
            },
        }
        for c in range(n_configs)
    ]


def probe_initialisation(
    payoff: torch.Tensor,
    dagger: int,
    equilibria: list[dict],
    n_rounds: int,
    eta: float,
    reward_range: tuple[float, float],
    seed: int = 0,
) -> list[dict]:
    """Probe 12: does the selected equilibrium depend on the initial policy."""
    n_arms = payoff.shape[0]
    torch.manual_seed(seed)

    starts: dict[str, torch.Tensor] = {
        "uniform": torch.full((n_arms,), 1.0 / n_arms, dtype=payoff.dtype)
    }

    perturbed = torch.full((n_arms,), 1.0 / n_arms, dtype=payoff.dtype)
    perturbed = perturbed + 0.01 * torch.rand(n_arms, dtype=payoff.dtype) / n_arms
    starts["perturbed_uniform"] = perturbed / perturbed.sum()

    for equilibrium in equilibria:
        action = equilibrium["action"]
        biased = torch.full((n_arms,), 0.1 / n_arms, dtype=payoff.dtype)
        biased[action] += 0.9
        starts[f"near_nash_{action}"] = biased / biased.sum()

    widest = int((payoff.diagonal()).argmax())
    biased = torch.full((n_arms,), 0.1 / n_arms, dtype=payoff.dtype)
    biased[widest] += 0.9
    starts[f"near_collusive_{widest}"] = biased / biased.sum()

    for alpha in (0.1, 1.0, 10.0):
        sample = torch.distributions.Dirichlet(
            torch.full((n_arms,), alpha, dtype=payoff.dtype)
        ).sample()
        starts[f"dirichlet_{alpha}"] = sample

    labels = list(starts)
    initials = torch.stack([starts[name].repeat(2, 1) for name in labels])

    results = probe_deterministic_mw(
        payoff,
        dagger,
        n_rounds,
        [eta] * len(labels),
        reward_range,
        initials=initials,
        labels=labels,
    )
    for record, name in zip(results, labels):
        record["initial_prob_on_dagger"] = starts[name][dagger].item()

    return results


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_game(delta: float, samples: torch.Tensor) -> dict:
    """Build the finite game for one tick size."""
    arms = build_quote_grid(0.0, 1.0, delta, epsilon=EPS)
    payoff = get_expected_payoff_matrix(arms, samples, epsilon=EPS)
    nash = get_pure_nash(payoff)

    equilibria = probe_nash_strictness(payoff, nash)
    dagger = equilibria[0]["action"] if equilibria else None

    return {
        "delta": delta,
        "arms": arms,
        "payoff": payoff,
        "nash": nash,
        "equilibria": equilibria,
        "dagger": dagger,
    }


def run_structural(game: dict, logger, max_mixed_arms: int, basin_samples: int) -> dict:
    """Probes 1 to 6 and 9 on one game."""
    payoff, dagger = game["payoff"], game["dagger"]
    arms = game["arms"]

    logger.info("Probe 1: iterated strict dominance.")
    dominance = probe_iterated_dominance(payoff, max_mixed_arms=max_mixed_arms)
    survivors = dominance["rationalizable_indices"]
    dominance["rationalizable_quotes"] = arms[survivors].tolist()

    nash_actions = sorted({e["action"] for e in game["equilibria"]})
    dominance["pure_nash_actions"] = nash_actions
    dominance["nash_actions_surviving"] = [
        a for a in nash_actions if a in set(survivors)
    ]
    dominance["equals_nash_set"] = set(survivors) == set(nash_actions)

    logger.info("Probes 3 to 5: payoff geometry around the payoff-dominant action.")
    result = {
        "delta": game["delta"],
        "n_arms": int(payoff.shape[0]),
        "payoff_dominant_action": dagger,
        "payoff_dominant_quote": arms[dagger].tolist(),
        "equilibria": [
            {**e, "quote": arms[e["action"]].tolist()} for e in game["equilibria"]
        ],
        "iterated_dominance": dominance,
        "uniform_ranking": probe_uniform_ranking(payoff, dagger),
        "uniform_path": probe_uniform_path(payoff, dagger),
        "best_response_region": probe_best_response_region(payoff, dagger),
    }

    logger.info("Probe 9: best-response basins.")
    result["basins"] = probe_basins(payoff, game["equilibria"], basin_samples)

    # Probe 6: the same geometry once dominated actions are gone
    if len(survivors) < payoff.shape[0] and dagger in set(survivors):
        index = torch.tensor(survivors, device=payoff.device)
        reduced = payoff[index][:, index]
        local = survivors.index(dagger)

        result["after_elimination"] = {
            "n_arms": len(survivors),
            "uniform_ranking": probe_uniform_ranking(reduced, local),
            "uniform_path": probe_uniform_path(reduced, local),
            "best_response_region": probe_best_response_region(reduced, local),
        }

    return result


def main(args) -> None:
    manager = ResultsManager(args.results_dir)

    with manager.new_experiment(name="exp3_theory_probes") as exp:
        logger = setup_logger(
            log_path=exp.file("execution.log"), capture_loggers=["lama_lab"]
        )

        try:
            generator = GaussianMixtureGenerator([1.0], [0.5], [0.1], 0.0, 1.0)
            torch.manual_seed(args.seed)
            samples = generator.generate(args.n_samples)

            game = build_game(args.delta, samples)
            logger.info(
                f"delta={args.delta}  arms={game['payoff'].shape[0]}  "
                f"pure Nash={game['nash'].shape[0]}  "
                f"payoff-dominant action={game['dagger']}"
            )

            report = {
                "config": vars(args),
                "main": run_structural(
                    game, logger, args.max_mixed_arms, args.basin_samples
                ),
            }

            if not args.skip_dynamic:
                reward_range = (-1.0 + args.delta, 0.5)
                n_arms = game["payoff"].shape[0]
                eta = args.eta or 2.5 * math.sqrt(
                    2 * math.log(n_arms) / (args.rounds * n_arms)
                )
                logger.info(f"Probes 7, 8, 10: Exp3 trajectory with eta={eta:.6f}.")
                report["trajectory"] = probe_trajectory(
                    game["payoff"],
                    game["arms"],
                    game["dagger"],
                    generator,
                    args.rounds,
                    args.episodes,
                    eta,
                    reward_range,
                    logger,
                )

                logger.info("Probe 11: deterministic multiplicative weights.")
                report["deterministic_mw"] = probe_deterministic_mw(
                    game["payoff"],
                    game["dagger"],
                    args.mw_rounds,
                    [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
                    reward_range,
                )

                logger.info("Probe 12: initialisation dependence.")
                report["initialisation"] = probe_initialisation(
                    game["payoff"],
                    game["dagger"],
                    game["equilibria"],
                    args.mw_rounds,
                    2e-3,
                    reward_range,
                )

            if args.grid_sweep:
                logger.info("Probe 13: structural probes across tick sizes.")
                sweep = []
                for delta in args.grid_sweep:
                    other = build_game(delta, samples)
                    logger.info(f"  delta={delta}  arms={other['payoff'].shape[0]}")
                    sweep.append(
                        run_structural(
                            other, logger, args.max_mixed_arms, args.basin_samples
                        )
                    )
                report["grid_sweep"] = sweep

            exp.save_all(
                {
                    "probes": report,
                    "payoff": game["payoff"].cpu(),
                    "arms": game["arms"].cpu(),
                    "report": render_report(report),
                }
            )
            logger.info(f"Saved to {exp.path}.")
            print(render_report(report))

        except Exception:
            logger.error(traceback.format_exc())
            raise


def render_report(report: dict) -> str:
    """Render the minimal result bundle as markdown."""
    main_result = report["main"]
    lines = [
        "# Exp3 equilibrium selection probes",
        "",
        f"Delta {main_result['delta']}, {main_result['n_arms']} arms. "
        f"Payoff-dominant action {main_result['payoff_dominant_action']} "
        f"= {main_result['payoff_dominant_quote']}.",
        "",
        "## Pure Nash equilibria",
        "",
        "| action | quote | payoff | strictness margin |",
        "|---|---|---|---|",
    ]
    for e in main_result["equilibria"]:
        lines.append(
            f"| {e['action']} | {e['quote']} | {e['payoff']:.5f} | "
            f"{e['strictness_margin']:+.5f} |"
        )

    dominance = main_result["iterated_dominance"]
    lines += [
        "",
        "## Iterated strict dominance",
        "",
        f"- elimination rounds: {len(dominance['rounds'])}",
        f"- survivors: {dominance['n_survivors']} of {main_result['n_arms']}",
        f"- rationalizable set equals the Nash action set: "
        f"{dominance['equals_nash_set']}",
        f"- mixed dominance tested: {dominance['mixed_test_applied']}",
        "",
        "## Against a uniform opponent",
        "",
        f"- best action: {main_result['uniform_ranking']['best_action']}",
        f"- payoff-dominant action is the best response: "
        f"{main_result['uniform_ranking']['payoff_dominant_is_best']}",
        f"- gap to runner-up: {main_result['uniform_ranking']['gap_to_runner_up']:+.5f}",
        "",
        "## Path from uniform play to the equilibrium",
        "",
        f"- minimum gap over the path: "
        f"{main_result['uniform_path']['min_gap_over_path']:+.5f}",
        f"- at lambda {main_result['uniform_path']['lambda_of_min']:.3f}, "
        f"competitor {main_result['uniform_path']['worst_competitor']}",
        f"- strict throughout: {main_result['uniform_path']['always_strict']}",
        "",
        "## Best-response region",
        "",
        f"- uniform inside: {main_result['best_response_region']['uniform_inside']}",
        f"- minimum gap at uniform: "
        f"{main_result['best_response_region']['min_gap_at_uniform']:+.5f}",
        f"- L1 distance to the nearest boundary: "
        f"{main_result['best_response_region']['l1_distance_to_boundary']:.5f} "
        f"(competitor {main_result['best_response_region']['nearest_competitor']})",
    ]

    if "trajectory" in report:
        t = report["trajectory"]
        lines += [
            "",
            "## Exp3 trajectory",
            "",
            f"- first round with a positive gap: {t['first_entry_round']}",
            f"- permanent entry: {t['permanent_entry_round']}",
            f"- minimum gap after permanent entry: "
            f"{t['min_gap_after_permanent_entry']}",
            f"- final gap: {t['final_gap']:+.5f}",
            f"- final probability on the payoff-dominant action: "
            f"{[round(p, 4) for p in t['final_prob_on_dagger']]}",
            f"- cumulative advantage positive for every competitor: "
            f"{t['cumulative_advantage']['all_positive']}",
        ]

    if "deterministic_mw" in report:
        lines += [
            "",
            "## Deterministic multiplicative weights",
            "",
            "| eta | selected | selects payoff-dominant | max prob | exploitability |",
            "|---|---|---|---|---|",
        ]
        for r in report["deterministic_mw"]:
            lines.append(
                f"| {r['eta']:g} | {r['selected_action']} | "
                f"{r['selects_dagger']} | {max(r['max_prob']):.4f} | "
                f"{r['exploitability']:.6f} |"
            )

    if "initialisation" in report:
        lines += [
            "",
            "## Initialisation dependence",
            "",
            "| start | selected | selects payoff-dominant | max prob | exploitability |",
            "|---|---|---|---|---|",
        ]
        for r in report["initialisation"]:
            lines.append(
                f"| {r['label']} | {r['selected_action']} | "
                f"{r['selects_dagger']} | {max(r['max_prob']):.4f} | "
                f"{r['exploitability']:.6f} |"
            )

    if "grid_sweep" in report:
        lines += [
            "",
            "## Across tick sizes",
            "",
            "| delta | arms | CR survivors | pure Nash | dominant | gap vs uniform | "
            "boundary distance |",
            "|---|---|---|---|---|---|---|",
        ]
        for g in report["grid_sweep"]:
            lines.append(
                f"| {g['delta']} | {g['n_arms']} | "
                f"{g['iterated_dominance']['n_survivors']} | "
                f"{len(g['equilibria'])} | {g['payoff_dominant_action']} | "
                f"{g['uniform_ranking']['gap_to_runner_up']:+.5f} | "
                f"{g['best_response_region']['l1_distance_to_boundary']:.5f} |"
            )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lama Lab - Exp3 equilibrium selection probes"
    )
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--rounds", type=int, default=100_000)
    parser.add_argument("--mw_rounds", type=int, default=500_000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--basin_samples", type=int, default=100_000)
    parser.add_argument("--max_mixed_arms", type=int, default=400)
    parser.add_argument("--skip-dynamic", action="store_true", dest="skip_dynamic")
    parser.add_argument("--grid-sweep", type=float, nargs="*", dest="grid_sweep")
    main(parser.parse_args())
