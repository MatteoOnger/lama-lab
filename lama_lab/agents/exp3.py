import logging
from typing import Any

import torch

from .agent import Agent
from ..spaces import DiscreteSpace

logger = logging.getLogger(__name__)


class Exp3(Agent):
    """Exponential-weight algorithm for Exploration and Exploitation (Exp3).

    This agent treats the optimization problem as an adversarial multi-armed
    bandit [1]_. Given a discrete set of possible actions (arms), it maintains a
    probability distribution over them. Rewards are converted to losses in
    :math:`[0, 1]` and the distribution is updated with an importance-sampling
    estimator of those losses. The learning rate and the exploration weight are
    constant.

    Parameters
    ----------
    n_episodes : int
        Number of parallel episodes to simulate.
    action_space : DiscreteSpace
        Discrete space containing the pool of valid discrete actions.
    reward_range : tuple of float
        Lower and upper bound of the reward, used to normalize it. Exp3 needs a
        bounded signal and the bound cannot be inferred dynamically, so it must
        be provided. Rewards outside this range are clamped.
    eta : float, optional
        Learning rate (or inverse temperature) controlling how aggressively the
        probability distribution updates based on estimated losses. Must be
        strictly positive.
    gamma : float, optional
        Exploration parameter in :math:`[0, 1]`. Mixes the exponential weights with a
        uniform distribution.
    name : str, optional
        Human-readable name of the agent.

    Attributes
    ----------
    reward_low : float
        Lower bound of ``reward_range``.
    reward_high : float
        Upper bound of ``reward_range``.
    weights : torch.Tensor
        Cumulative importance-weighted negated loss estimates of each arm, with
        shape ``(n_episodes, n_arms)``.
    warned_out_of_range : bool
        Flag indicating whether a warning for out-of-range rewards was issued.
    last_action_info : dict of str to Any
        Metadata about the last selected action (inherited from :class:`Agent`).
        Besides ``"action"`` and ``"pending_update"``, :meth:`act` stores
        ``"action_index"``, the indices of the selected arms with shape
        ``(n_episodes,)``, and ``"action_probability"``, their sampling
        probabilities with shape ``(n_episodes,)``.

    Raises
    ------
    ValueError
        If ``reward_range`` is not an increasing pair of numbers.
        If ``action_space`` is not an instance of :class:`~lama_lab.spaces.DiscreteSpace`.
        If ``eta`` is not strictly positive.
        If ``gamma`` is not in :math:`[0, 1]`.

    References
    ----------
    .. [1] P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire, "The
       nonstochastic multiarmed bandit problem," SIAM Journal on Computing,
       vol. 32, no. 1, pp. 48-77, 2002.
    """

    def __init__(
        self,
        n_episodes: int,
        action_space: DiscreteSpace,
        reward_range: tuple[float, float],
        eta: float = 0.1,
        gamma: float = 0.0,
        name: str = "Exp3",
    ) -> None:
        reward_low, reward_high = float(reward_range[0]), float(reward_range[1])

        if reward_high <= reward_low:
            raise ValueError(
                f"'reward_range' must be an increasing pair. Got {reward_range}."
            )
        if not isinstance(action_space, DiscreteSpace):
            raise ValueError("'Exp3' requires a discrete action space.")
        if eta <= 0.0:
            raise ValueError(f"'eta' must be strictly positive. Got {eta}.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"'gamma' must be in [0, 1]. Got {gamma}.")

        super().__init__(n_episodes=n_episodes, action_space=action_space, name=name)

        self.reward_low = reward_low
        self.reward_high = reward_high
        self.eta = eta
        self.gamma = gamma

        self.weights = torch.zeros((self.n_episodes, len(self.action_space)))
        self.warned_out_of_range = False
        return

    def _act(self) -> torch.Tensor:
        batch_index = torch.arange(self.n_episodes)

        # Compute the current sampling probabilities for all arms
        prob = self.get_policy()

        # Sample one arm per episode
        action_index = torch.multinomial(prob, num_samples=1).squeeze(-1)
        action = self.action_space[action_index]

        # Store the selected arm indices and their probabilities for the update step
        self.last_action_info["action_index"] = action_index
        self.last_action_info["action_probability"] = prob[batch_index, action_index]
        return action

    def _update(self, reward: torch.Tensor) -> None:
        batch_index = torch.arange(self.n_episodes)

        # Clamping an out-of-range reward silently invalidates the regret guarantee,
        # so warn about it once
        if not self.warned_out_of_range:
            out_of_range = (reward < self.reward_low) | (reward > self.reward_high)
            if out_of_range.any():
                logger.warning(
                    f"{self.name} received a reward outside 'reward_range' "
                    f"[{self.reward_low:g}, {self.reward_high:g}]. Values are "
                    "clamped, but the regret guarantee assumes the range is correct."
                )
                self.warned_out_of_range = True

        reward = reward.clamp(self.reward_low, self.reward_high)
        action_index: torch.Tensor = self.last_action_info["action_index"]
        action_probability: torch.Tensor = self.last_action_info["action_probability"]

        # Negated loss in [-1, 0], importance-weighted, for the played arm only
        signal = -(self.reward_high - reward) / (self.reward_high - self.reward_low)
        signal = signal / action_probability
        self.weights[batch_index, action_index] += signal.to(self.weights.dtype)
        return

    def get_internal_state(self) -> dict[str, Any]:
        return {
            "eta": float(self.eta),
            "gamma": float(self.gamma),
            "reward_range": [self.reward_low, self.reward_high],
        }

    def get_policy(self) -> torch.Tensor:
        w = self.eta * self.weights

        # Softmax of the scaled weights, shifted by their max to avoid overflow in exp
        w_max = torch.amax(w, dim=1, keepdim=True)
        exp_w = torch.exp(w - w_max)
        probs = exp_w / torch.sum(exp_w, dim=1, keepdim=True)

        # Mix with the uniform distribution to guarantee exploration
        if self.gamma > 0.0:
            probs = (1.0 - self.gamma) * probs + (self.gamma / len(self.action_space))
        return probs
