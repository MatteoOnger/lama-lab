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
    signal : {"reward", "loss"}, optional
        Quantity accumulated in the weights. The default is the mean-based
        algorithm; ``"loss"`` keeps the schedule but takes the estimator of the
        fixed-rate learner, which isolates one change from the other.
    name : str, optional
        Human-readable identifier for the agent.

    Notes
    -----
    The schedule carries no tuning multiplier:

    .. math:: \eta_t = t^{-1/3}, \qquad \epsilon_t = t^{-1/4}.

    At the first round both equal one, so play starts uniform. Both decrease to
    zero, while the exploration weight decreases more slowly, which is what
    keeps the gain-based estimator usable: mixing in uniform exploration floors
    every probability at :math:`\epsilon_t / K`, so the importance weight is
    bounded by :math:`K / \epsilon_t` and

    .. math:: \eta_t \frac{K}{\epsilon_t} = K t^{-1/12} \to 0.

    The reward form is deliberate here, and is the reason the exploration term
    is not optional. Accumulating :math:`\hat\sigma_t(a) = \sum_s \hat x_s(a)`
    and sampling from :math:`\exp(\eta_t \hat\sigma_t)` is the canonical
    mean-based algorithm the theorem is stated for, whereas the parent class
    accumulates losses, which is no-regret without exploration but is not the
    object of that theorem.

    Note that :math:`K t^{-1/12}` decays very slowly, so the bound above is
    loose at practical horizons; the schedule is chosen to match the theory
    rather than to be tight over a given number of rounds.
    """

    def __init__(
        self,
        n_episodes: int,
        action_space: torch.Tensor | list[list[float]],
        reward_range: tuple[float, float],
        signal: str = "reward",
        name: str = "AgentExp3MeanBased",
    ):
        if signal not in ("reward", "loss"):
            raise ValueError(f"signal must be 'reward' or 'loss'. Got {signal!r}.")

        # The stored constants are placeholders: both getters below ignore them
        super().__init__(
            n_episodes=n_episodes,
            action_space=action_space,
            reward_range=reward_range,
            eta=1.0,
            gamma=1.0,
            name=name,
        )

        self.signal = signal
        return

    def estimate_signal(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalized reward of the current round, in ``[0, 1]``.

        Set `signal` to ``"loss"`` to accumulate the negated loss instead, which
        is what the fixed-rate learner does. That combination is not the
        mean-based algorithm, but it isolates the schedule from the estimator
        when comparing against the fixed-rate control, which otherwise differs
        in both at once.
        """
        if self.signal == "loss":
            return super().estimate_signal(reward)
        return (reward - self.reward_low) / (self.reward_high - self.reward_low)

    def get_internal_state(self) -> dict[str, Any]:
        return super().get_internal_state() | {"signal": self.signal}

    def get_learning_rate(self) -> float:
        return float(self._t) ** (-1.0 / 3.0)

    def get_exploration(self) -> float:
        return float(self._t) ** (-1.0 / 4.0)
