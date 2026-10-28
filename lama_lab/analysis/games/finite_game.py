"""Generic tools for finite games.

A finite game is stored as a ``payoffs`` tensor of shape
``(n_players, n_1, ..., n_N)``, where ``n_i`` is the number of actions of player
``i`` and ``payoffs[i][a_1, ..., a_N]`` is the payoff of player ``i`` when every
player ``j`` plays its action ``a_j``.
"""

import itertools
import math
import string
from collections.abc import Callable, Sequence

import numpy as np
import torch
from scipy.optimize import linprog

from ...spaces import DiscreteSpace


def build_payoffs(
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    spaces: Sequence[DiscreteSpace],
    batch_size: int = 100_000,
) -> torch.Tensor:
    """Evaluate the payoffs of a game on every joint action profile.

    The game is described by a function returning the payoff of every player for
    a batch of profiles, such as
    :meth:`~lama_lab.envs.MarketMakingEnvironment.expected_payoff`, and by the
    discrete action space of each player.

    Parameters
    ----------
    payoff_fn : Callable[[torch.Tensor], torch.Tensor]
        Function taking the profiles, with shape ``(B, n_players, *action_shape)``,
        and returning the payoff of each player, with shape ``(B, n_players)``.
    spaces : sequence of DiscreteSpace
        Action space of each player. All of them must have the same action shape.
    batch_size : int, optional
        Number of profiles passed to ``payoff_fn`` at a time, which bounds the
        memory used by the evaluation.

    Returns
    -------
    payoffs : torch.Tensor
        Tensor of shape ``(n_players, n_1, ..., n_N)``, with ``n_i = len(spaces[i])``.

    Notes
    -----
    The result has ``n_players * n_1 * ... * n_N`` entries, so its size grows
    exponentially with the number of players.
    """
    sizes = [len(space) for space in spaces]
    elements = [space[torch.arange(size)] for space, size in zip(spaces, sizes)]

    n_profiles = math.prod(sizes)
    chunks: list[torch.Tensor] = []
    for start in range(0, n_profiles, batch_size):
        # Split each flat profile index into one action index per player
        flat = torch.arange(start, min(start + batch_size, n_profiles))
        indices = torch.unravel_index(flat, sizes)

        profiles = torch.stack([e[i] for e, i in zip(elements, indices)], dim=1)
        chunks.append(payoff_fn(profiles))

    return torch.cat(chunks).T.reshape(len(spaces), *sizes)


def pure_nash(payoffs: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
    """Enumerate the pure Nash equilibria of a finite game.

    A profile is a pure Nash equilibrium when no player gains by unilaterally
    switching to another action.

    Parameters
    ----------
    payoffs : torch.Tensor
        Tensor of shape ``(n_players, n_1, ..., n_N)``, as returned by
        :func:`build_payoffs`.
    tol : float, optional
        Slack allowed on each best-response condition, so that profiles within
        ``tol`` of optimal are retained.

    Returns
    -------
    profiles : torch.Tensor
        Tensor of shape ``(n_profiles, n_players)`` with the action index of each
        player at every equilibrium, in lexicographic order.
    """
    is_equilibrium = torch.ones(payoffs.shape[1:], dtype=torch.bool)

    for player in range(payoffs.shape[0]):
        # The actions of this player run along axis 'player' of its own payoffs
        best = payoffs[player].amax(dim=player, keepdim=True)
        is_equilibrium &= payoffs[player] >= best - tol

    return torch.nonzero(is_equilibrium)


def action_values(
    payoffs: torch.Tensor, policies: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    """Compute the expected payoff of every pure action of every player.

    The expectation is taken over the actions of the other players, who follow
    their mixed strategies independently of each other.

    Parameters
    ----------
    payoffs : torch.Tensor
        Tensor of shape ``(n_players, n_1, ..., n_N)``, as returned by
        :func:`build_payoffs`.
    policies : sequence of torch.Tensor
        Mixed strategy of each player, with shape ``(..., n_i)``. The leading
        dimensions form a batch shared by all the players.

    Returns
    -------
    values : list of torch.Tensor
        One tensor of shape ``(..., n_i)`` per player, in the dtype of ``payoffs``.
        Its entry ``a`` is the expected payoff of player ``i`` when it plays
        action ``a``.
    """
    n_players = payoffs.shape[0]
    letters = string.ascii_lowercase[:n_players]

    values = []
    for player in range(n_players):
        others = [j for j in range(n_players) if j != player]

        # Contract the payoffs of this player with the policies of the others
        inputs = [letters] + [f"...{letters[j]}" for j in others]
        equation = ",".join(inputs) + f"->...{letters[player]}"
        operands = [payoffs[player]] + [policies[j].to(payoffs.dtype) for j in others]

        values.append(torch.einsum(equation, *operands))
    return values


def exploitability(
    payoffs: torch.Tensor, policies: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Compute how much each player gains by best responding to the others.

    The exploitability of a player is the increase in its expected payoff when
    it switches to its best pure action, while the other players keep their mixed
    strategies. All the exploitabilities are zero if and only if the policies form
    a mixed Nash equilibrium.

    Parameters
    ----------
    payoffs : torch.Tensor
        Tensor of shape ``(n_players, n_1, ..., n_N)``, as returned by
        :func:`build_payoffs`.
    policies : sequence of torch.Tensor
        Mixed strategy of each player, with shape ``(..., n_i)``. The leading
        dimensions form a batch shared by all the players.

    Returns
    -------
    gains : torch.Tensor
        Tensor of shape ``(..., n_players)`` with the best-response gain of each
        player.
    """
    values = action_values(payoffs, policies)

    gains = [
        value.amax(dim=-1) - (policy.to(value.dtype) * value).sum(dim=-1)
        for value, policy in zip(values, policies)
    ]
    return torch.stack(gains, dim=-1)


def _lp_value(matrix: torch.Tensor) -> float:
    """Value of the zero-sum matrix game in which the row player maximizes.

    The row player mixes its rows to maximize its worst case over the columns,
    and the value is that worst case. It is found with a linear program.
    """
    n_rows, n_cols = matrix.shape
    payoff = matrix.double().cpu().numpy()

    # Variables are (sigma, v): maximize v subject to sigma @ payoff >= v per column
    cost = np.zeros(n_rows + 1)
    cost[-1] = -1.0
    upper = np.hstack([-payoff.T, np.ones((n_cols, 1))])
    equality = np.hstack([np.ones((1, n_rows)), np.zeros((1, 1))])
    bounds = [(0.0, None)] * n_rows + [(None, None)]

    solution = linprog(
        cost,
        A_ub=upper,
        b_ub=np.zeros(n_cols),
        A_eq=equality,
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Game value LP failed: {solution.message}")
    return float(solution.x[-1])


def _is_dominated(margin: torch.Tensor, tol: float) -> bool:
    """Whether some strategy beats a candidate action by more than ``tol`` everywhere.

    ``margin[k, c]`` is how much better the action ``k`` is than the candidate
    against the column ``c``.
    """
    # A column where no action beats the candidate rules out any mixture
    if margin.amax(dim=0).amin() <= tol:
        return False

    # A single action that beats the candidate against every column
    if margin.amin(dim=1).amax() > tol:
        return True

    # Otherwise, the candidate can only be dominated by a mixture of actions
    return _lp_value(margin) > tol


def dominated_actions(
    payoffs: torch.Tensor,
    alive: Sequence[torch.Tensor] | None = None,
    tol: float = 1e-6,
) -> list[torch.Tensor]:
    """Find the strictly dominated actions among the actions in play.

    An action is strictly dominated when a pure or mixed strategy of the same
    player earns more than it, by a margin larger than ``tol``, against every
    profile of the other players. Only the actions in play are considered, both as
    dominators and in the profiles of the other players.

    Parameters
    ----------
    payoffs : torch.Tensor
        Tensor of shape ``(n_players, n_1, ..., n_N)``, as returned by
        :func:`build_payoffs`.
    alive : sequence of torch.Tensor or None, optional
        Boolean mask of shape ``(n_i,)`` for each player, marking the actions in
        play. If ``None``, all the actions are in play.
    tol : float, optional
        Margin by which a strategy must beat an action to dominate it.

    Returns
    -------
    dominated : list of torch.Tensor
        Boolean mask of shape ``(n_i,)`` for each player, marking the strictly
        dominated actions. An action that is not in play is never marked.
    """
    if alive is None:
        alive = [torch.ones(n, dtype=torch.bool) for n in payoffs.shape[1:]]
    in_play = [torch.where(mask)[0] for mask in alive]

    dominated = []
    for player in range(payoffs.shape[0]):
        # One row per action of this player and one column per profile of the others,
        # keeping only the actions in play
        restricted = payoffs[player]
        for axis, index in enumerate(in_play):
            restricted = restricted.index_select(axis, index)
        matrix = restricted.movedim(player, 0).reshape(len(in_play[player]), -1)

        mask = torch.zeros(payoffs.shape[player + 1], dtype=torch.bool)
        for row, action in enumerate(in_play[player]):
            mask[action] = _is_dominated(matrix - matrix[row], tol)
        dominated.append(mask)
    return dominated


def rationalizable_set(payoffs: torch.Tensor, tol: float = 1e-6) -> list[torch.Tensor]:
    """Find the rationalizable actions of every player by iterated elimination.

    The strictly dominated actions are removed, for all the players at the same
    time, and the test is repeated on what remains until nothing else is removed.
    Domination by mixed strategies is included. For two players the survivors are
    the rationalizable actions. For more players they are the actions that survive
    iterated strict dominance, which is rationalizability with correlated beliefs.

    Parameters
    ----------
    payoffs : torch.Tensor
        Tensor of shape ``(n_players, n_1, ..., n_N)``, as returned by
        :func:`build_payoffs`.
    tol : float, optional
        Margin by which a strategy must beat an action to dominate it.

    Returns
    -------
    rounds : list of torch.Tensor
        One long tensor of shape ``(n_i,)`` per player, with the round at which
        each action was eliminated, starting from 0, and ``-1`` for the
        rationalizable actions.

    Notes
    -----
    Like the payoff tensor, the cost grows with the number of profiles of the
    other players.
    """
    alive = [torch.ones(n, dtype=torch.bool) for n in payoffs.shape[1:]]
    rounds = [torch.full((n,), -1, dtype=torch.long) for n in payoffs.shape[1:]]

    for round_index in itertools.count():
        dominated = dominated_actions(payoffs, alive, tol)
        if not any(mask.any() for mask in dominated):
            break

        for player, mask in enumerate(dominated):
            rounds[player][mask] = round_index
            alive[player] &= ~mask
    return rounds
