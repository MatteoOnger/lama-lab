from typing import Any

import torch

from .exp3 import AgentExp3


class AgentExp3MeanBased(AgentExp3):
    r"""Exp3 with the schedule required by mean-based learning theory.

    Differs from :class:`~lama_lab.agents.AgentExp3` in two ways, and in nothing
    else: the learning rate and the exploration weight follow a fixed schedule
    instead of being constants, and the weights accumulate the normalized
    *reward* rather than the negated loss.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to simulate in parallel.
    action_space : torch.Tensor or list of list of float
        A 2D tensor of shape ``(n_arms, action_dim)`` containing the discrete
        pool of possible actions.
    reward_range : tuple of float
        Analytical lower and upper bound of the reward, used to normalize it to
        [0, 1]. Bounds observed empirically must not be used here, since the
        guarantee assumes the range holds for every realization.
    eta_exponent : float, optional
        Decay exponent of the learning rate.
    exploration_exponent : float, optional
        Decay exponent of the exploration weight. Must be strictly between zero
        and `eta_exponent`, which is what keeps the importance weight bounded.
    name : str, optional
        Human-readable identifier for the agent.

    Notes
    -----
    The schedule carries no tuning multiplier, only exponents:

    .. math:: \eta_t = t^{-\alpha}, \qquad \epsilon_t = t^{-\beta},
              \qquad 0 < \beta < \alpha.

    At the first round both equal one, so play starts uniform. Both decrease to
    zero, while the exploration weight decreases more slowly, which is what
    keeps the gain-based estimator usable: mixing in uniform exploration floors
    every probability at :math:`\epsilon_t / K`, so the importance weight is
    bounded by :math:`K / \epsilon_t` and

    .. math:: \eta_t \frac{K}{\epsilon_t} = K t^{\beta - \alpha} \to 0,

    which is why `exploration_exponent` must stay below `eta_exponent`.

    Any admissible pair satisfies the theory, but they are not equally quick.
    Measured over 100k rounds on the 15-arm grid, the two theorem targets pull
    in opposite directions: small exponents eliminate faster, large exponents
    concentrate on the payoff-dominant action faster. The defaults are the best
    compromise found, reaching the lowest exploitability of the pairs tried,
    against :math:`(1/3, 1/4)` as originally proposed.

    The reward form is deliberate here, and is the reason the exploration term
    is not optional. Accumulating :math:`\hat\sigma_t(a) = \sum_s \hat x_s(a)`
    and sampling from :math:`\exp(\eta_t \hat\sigma_t)` is the canonical
    mean-based algorithm the theorem is stated for, whereas the parent class
    accumulates losses, which is no-regret without exploration but is not the
    object of that theorem.

    That bound is loose at practical horizons for any admissible pair, since
    the exponent gap is small; the schedule matches the theory rather than
    being tight over a given number of rounds.
    """

    def __init__(
        self,
        n_episodes: int,
        action_space: torch.Tensor | list[list[float]],
        reward_range: tuple[float, float],
        eta_exponent: float = 1.0 / 2.0,
        exploration_exponent: float = 2.0 / 5.0,
        name: str = "AgentExp3MeanBased",
    ):
        if not 0.0 < exploration_exponent < eta_exponent:
            raise ValueError(
                f"Need 0 < exploration_exponent < eta_exponent, so that the "
                f"importance weight stays bounded. Got {exploration_exponent} "
                f"and {eta_exponent}."
            )

        # The stored constants are placeholders: both getters below ignore them
        super().__init__(
            n_episodes=n_episodes,
            action_space=action_space,
            reward_range=reward_range,
            eta=1.0,
            gamma=1.0,
            name=name,
        )

        self.eta_exponent = eta_exponent
        self.exploration_exponent = exploration_exponent
        return

    def estimate_signal(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalized reward of the current round, in ``[0, 1]``."""
        return (reward - self.reward_low) / (self.reward_high - self.reward_low)

    def get_learning_rate(self) -> float:
        return float(self._t) ** -self.eta_exponent

    def get_exploration(self) -> float:
        return float(self._t) ** -self.exploration_exponent

    def get_internal_state(self) -> dict[str, Any]:
        return super().get_internal_state() | {
            "eta_exponent": self.eta_exponent,
            "exploration_exponent": self.exploration_exponent,
        }
