"""Metrics of the learning process of the players of a finite game.

A :class:`LearningTracker` follows the policies of the players round after round
and, at a few checkpoints, evaluates the requested metrics and stores them. The metrics
are the entries of ``METRICS``. A metric of shape ``(n_episodes,)`` describes the
episodes, and one of shape ``(n_episodes, n_players)`` describes each player in
each episode.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import torch

from ...utils import RingBuffer
from ..games.finite_game import action_values, exploitability


class Metric(NamedTuple):
    """A metric, with the running statistics and the references it needs.

    Parameters
    ----------
    fn : Callable
        Function ``fn(payoffs, stats, references)`` returning the value of the
        metric, with the episodes along its first axis.
    stats : tuple of str
        Names of the running statistics that ``fn`` reads from ``stats``.
    references : tuple of str
        Names of the entries that ``fn`` reads from ``references``.
    """

    fn: Callable[[torch.Tensor, dict[str, Any], dict[str, Any]], torch.Tensor]
    stats: tuple[str, ...] = ()
    references: tuple[str, ...] = ()


def _exploitability(
    payoffs: torch.Tensor, stats: dict[str, Any], references: dict[str, Any]
) -> torch.Tensor:
    """Best-response gain of each player against the current policies."""
    return exploitability(payoffs, stats["last_policies"])


def _avg_exploitability(
    payoffs: torch.Tensor, stats: dict[str, Any], references: dict[str, Any]
) -> torch.Tensor:
    """Best-response gain of each player against the time-averaged policies."""
    average = [total / stats["t"] for total in stats["policy_sum"]]
    return exploitability(payoffs, average)


def _avg_regret(
    payoffs: torch.Tensor, stats: dict[str, Any], references: dict[str, Any]
) -> torch.Tensor:
    """Average regret of each player against its best fixed action."""
    best = torch.stack([total.amax(dim=-1) for total in stats["value_sum"]], dim=-1)
    return (best - stats["payoff_sum"]) / stats["t"]


def _entropy(
    payoffs: torch.Tensor, stats: dict[str, Any], references: dict[str, Any]
) -> torch.Tensor:
    """Entropy of the current policy of each player."""
    entropies = [torch.special.entr(p).sum(dim=-1) for p in stats["last_policies"]]
    return torch.stack(entropies, dim=-1)


def _nash_mass(
    payoffs: torch.Tensor, stats: dict[str, Any], references: dict[str, Any]
) -> torch.Tensor:
    """Probability that the current policies play a pure Nash equilibrium."""
    profiles = references["pure_nash"]

    # Probability of each equilibrium profile, as the product over the players
    mass = torch.ones(profiles.shape[0], dtype=torch.float64)
    mass = mass.expand(stats["last_policies"][0].shape[0], -1)
    for player, policy in enumerate(stats["last_policies"]):
        mass = mass * policy[:, profiles[:, player]]
    return mass.sum(dim=-1)


def _layer_mass(
    payoffs: torch.Tensor, stats: dict[str, Any], references: dict[str, Any]
) -> torch.Tensor:
    """Probability that each player plays an action of each elimination layer."""
    rounds = references["rationalizable"]
    n_layers = max(int(r.max()) for r in rounds) + 1

    # Eliminated layers in order of removal, then the rationalizable actions
    layers = [*range(n_layers), -1]
    masses = [
        torch.stack([policy[:, r == layer].sum(dim=-1) for layer in layers], dim=-1)
        for policy, r in zip(stats["last_policies"], rounds)
    ]
    return torch.stack(masses, dim=1)


METRICS: dict[str, Metric] = {
    "exploitability": Metric(_exploitability),
    "avg_exploitability": Metric(_avg_exploitability, stats=("policy_sum",)),
    "avg_regret": Metric(_avg_regret, stats=("value_sum", "payoff_sum")),
    "entropy": Metric(_entropy),
    "nash_mass": Metric(_nash_mass, references=("pure_nash",)),
    "layer_mass": Metric(_layer_mass, references=("rationalizable",)),
}


class LearningTracker:
    """Track metrics of the learning process of the players of a finite game.

    The policies are passed round after round. At each checkpoint, the requested
    metrics are evaluated and stored, and :meth:`history` returns them.

    Parameters
    ----------
    payoffs : torch.Tensor
        Tensor of shape ``(n_players, n_1, ..., n_N)``, as returned by
        :func:`~lama_lab.analysis.games.finite_game.build_payoffs`.
    n_episodes : int
        Number of independent episodes tracked.
    n_rounds : int
        Number of rounds of the simulation.
    metrics : sequence of str
        Names of the metrics to track, among the keys of ``METRICS``.
    references : dict of str to Any or None, optional
        Reference data used by some metrics. ``"pure_nash"`` holds the equilibria
        returned by :func:`~lama_lab.analysis.games.finite_game.pure_nash`, and
        ``"rationalizable"`` the elimination rounds returned by
        :func:`~lama_lab.analysis.games.finite_game.rationalizable_set`.
    checkpoints : sequence of int or None, optional
        One-based rounds at which the metrics are evaluated. If ``None``, they are
        log-spaced, as returned by :meth:`log_spaced_checkpoints`.

    Attributes
    ----------
    checkpoints : list of int
        Sorted rounds at which the metrics are evaluated, within ``[1, n_rounds]``.
    t : int
        Number of rounds seen so far.

    Raises
    ------
    ValueError
        If a metric is unknown, or if a metric needs a reference that is missing.
    """

    def __init__(
        self,
        payoffs: torch.Tensor,
        n_episodes: int,
        n_rounds: int,
        metrics: Sequence[str],
        references: dict[str, Any] | None = None,
        checkpoints: Sequence[int] | None = None,
    ) -> None:
        unknown = [name for name in metrics if name not in METRICS]
        if unknown:
            raise ValueError(
                f"Unknown metrics {unknown}. Choose among {list(METRICS)}."
            )

        references = {} if references is None else references
        missing = {r for name in metrics for r in METRICS[name].references}
        missing -= set(references)
        if missing:
            raise ValueError(f"Missing references {sorted(missing)} for the metrics.")

        if checkpoints is None:
            checkpoints = self.log_spaced_checkpoints(n_rounds)

        self.payoffs = payoffs.to(torch.float64)
        self.n_episodes = n_episodes
        self.metrics = list(metrics)
        self.references = references
        self.checkpoints = sorted({int(c) for c in checkpoints if 0 < c <= n_rounds})
        self.t = 0

        # Only the running statistics that the metrics need are accumulated
        self._stats: dict[str, Any] = {}
        needed = {stat for name in metrics for stat in METRICS[name].stats}
        sizes = payoffs.shape[1:]
        for stat in ("policy_sum", "value_sum"):
            if stat in needed:
                self._stats[stat] = [
                    torch.zeros(n_episodes, n, dtype=torch.float64) for n in sizes
                ]
        if "payoff_sum" in needed:
            self._stats["payoff_sum"] = torch.zeros(
                n_episodes, len(sizes), dtype=torch.float64
            )
        self._history: dict[str, RingBuffer] = {}
        return

    @staticmethod
    def log_spaced_checkpoints(
        n_rounds: int, start: int = 100, per_decade: int = 2
    ) -> list[int]:
        """Build logarithmically spaced rounds at which to evaluate the metrics.

        Parameters
        ----------
        n_rounds : int
            Total number of rounds, which is always included in the result.
        start : int, optional
            First checkpoint. Ignored if larger than ``n_rounds``.
        per_decade : int, optional
            Number of checkpoints per decade, so that the default of two gives
            100, 300, 1000, 3000 and so on.

        Returns
        -------
        checkpoints : list of int
            Strictly increasing one-based rounds.

        Notes
        -----
        Checkpoints are rounded to one significant digit, which turns the
        geometric sequence into round numbers that read well in a table.
        """
        checkpoints = {n_rounds}

        step = 0
        while True:
            value = start * 10 ** (step / per_decade)
            magnitude = 10 ** math.floor(math.log10(value))
            value = round(value / magnitude) * magnitude

            if value > n_rounds:
                break
            checkpoints.add(value)
            step += 1
        return sorted(checkpoints)

    def update(self, policies: Sequence[torch.Tensor]) -> None:
        """Track one round of play, and evaluate the metrics if it is a checkpoint.

        Parameters
        ----------
        policies : sequence of torch.Tensor
            Mixed strategy of each player, with shape ``(n_episodes, n_i)``, read
            before the actions of the round were sampled.

        Raises
        ------
        ValueError
            If the shapes of the policies are not the expected ones.
        """
        expected = [(self.n_episodes, n) for n in self.payoffs.shape[1:]]
        shapes = [tuple(policy.shape) for policy in policies]
        if shapes != expected:
            raise ValueError(f"Expected policies of shapes {expected}. Got {shapes}.")

        policies = [p.to(dtype=torch.float64, copy=True) for p in policies]
        for total, policy in zip(self._stats.get("policy_sum", ()), policies):
            total += policy

        if "value_sum" in self._stats or "payoff_sum" in self._stats:
            values = action_values(self.payoffs, policies)
            for total, value in zip(self._stats.get("value_sum", ()), values):
                total += value
            if "payoff_sum" in self._stats:
                self._stats["payoff_sum"] += torch.stack(
                    [(p * v).sum(dim=-1) for p, v in zip(policies, values)], dim=-1
                )

        self.t += 1
        self._stats["t"] = self.t
        self._stats["last_policies"] = policies

        if self.t in self.checkpoints:
            self._evaluate()
        return

    def history(self) -> dict[str, torch.Tensor]:
        """Return the stored values of every metric, on the CPU.

        Returns
        -------
        history : dict of str to torch.Tensor
            For each metric, a tensor of shape ``(n_checkpoints, *metric_shape)``
            whose rows follow ``checkpoints``. It is empty before the first
            checkpoint.
        """
        return {name: buffer.get_all() for name, buffer in self._history.items()}

    def _evaluate(self) -> None:
        """Evaluate the metrics and store their values."""
        for name in self.metrics:
            value = METRICS[name].fn(self.payoffs, self._stats, self.references)

            # The buffer takes its shape from the first value of the metric
            if name not in self._history:
                shape = tuple(value.shape)
                self._history[name] = RingBuffer(len(self.checkpoints), shape, "cpu")
            self._history[name].append(value)
        return
