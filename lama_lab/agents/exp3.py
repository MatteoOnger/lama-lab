import logging
from typing import Any

import torch

from .base import BaseAgent

logger = logging.getLogger(__name__)


class AgentExp3(BaseAgent):
    r"""Exponential-weight algorithm for Exploration and Exploitation (Exp3).

    This agent treats the optimization problem as an adversarial multi-armed
    bandit. Given a discrete set of possible actions (arms), it maintains a
    probability distribution over them. Rewards are converted to losses in
    [0, 1] and the distribution is updated with an importance-sampling estimator
    of those losses.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to simulate in parallel.
    action_space : torch.Tensor or list of list of float
        A 2D tensor of shape ``(n_arms, action_dim)`` containing the discrete
        pool of possible actions.
    reward_range : tuple of float
        Lower and upper bound of the reward, used to normalize it. Exp3 needs a
        bounded signal and the bound cannot be inferred from the rewards seen so
        far, so it must be stated. Rewards outside the range are clamped.
    eta : float, optional
        Learning rate (or inverse temperature) controlling how aggressively the
        probability distribution updates based on estimated losses. Higher
        values lead to faster exploitation.
    gamma : float, optional
        Exploration parameter in [0, 1]. Mixes the exponential weights with a
        uniform distribution. Not needed by the loss-based update rule, which is
        already no-regret at 0.0.
    name : str, optional
        Human-readable identifier for the agent.

    Attributes
    ----------
    n_arms : int
        Number of discrete actions available in the action space.
    weights : torch.Tensor
        Current unnormalized log-weights for each arm and episode, of shape
        ``(n_episodes, n_arms)``. They hold the negated cumulative loss
        estimates, hence are non-positive and non-increasing.

    Raises
    ------
    ValueError
        If `action_space` is not a 2D matrix/tensor, or if `reward_range` is not
        an increasing pair.

    Notes
    -----
    This implementation supports batched execution. All episodes share the same
    hyperparameters but maintain independent weight vectors, allowing :meth:`_act`
    and :meth:`_update` to operate on tensors using vectorized PyTorch operations.

    The update rule accumulates the importance-sampled *loss* estimator
    :math:`\hat\ell_{t,a} = \ell_t \mathbf 1\{a_t = a\} / p_{t,a}` on the chosen
    arm, so that :math:`w_{t,a} = -\hat L_{t,a}` and the sampling distribution is
    :math:`p_t \propto \exp(-\eta \hat L_t)`. With losses in [0, 1] this is
    no-regret without any explicit exploration, with

    .. math:: \mathbb E[R_T] \le \frac{\ln K}{\eta} + \frac{\eta T K}{2},

    minimized at :math:`\eta = \sqrt{2 \ln K / (TK)}`. The regret is expressed in
    normalized units and scales back to reward units by the width of
    `reward_range`.

    Losses rather than gains are essential here, and the two are *not*
    interchangeable under bandit feedback. Writing :math:`\ell = 1 - r`, only the
    played arm is updated, so
    :math:`\hat L_{t,a} = \sum_s \mathbf 1\{a_s = a\}/p_{s,a} - \hat R_{t,a}`,
    whose first term depends on the arm and therefore does not cancel in the
    normalization the way it does under full information. The loss analysis
    relies on :math:`e^{-x} \le 1 - x + x^2/2` for every :math:`x \ge 0`, while
    the gain version would need :math:`e^{x} \le 1 + x + x^2`, which holds only
    for :math:`\eta / p \le 1` and thus fails as probabilities shrink. That is
    why gain-based Exp3 requires an explicit exploration term and this one does
    not.

    `eta` is constant: it is optimal for a known horizon, which is the case in
    these experiments.
    """

    def __init__(
        self,
        n_episodes: int,
        action_space: torch.Tensor | list[list[float]],
        reward_range: tuple[float, float],
        eta: float = 0.1,
        gamma: float = 0.0,
        name: str = "AgentExp3",
    ):
        action_space = torch.as_tensor(action_space)

        if action_space.ndim != 2:
            raise ValueError(
                f"action_space must be a 2D matrix of shape (n_arms, action_dim). "
                f"Got shape {action_space.shape}."
            )

        reward_low, reward_high = float(reward_range[0]), float(reward_range[1])
        if reward_high <= reward_low:
            raise ValueError(
                f"reward_range must be an increasing pair. Got {reward_range}."
            )

        n_arms, action_dim = action_space.shape
        super().__init__(n_episodes=n_episodes, action_dim=action_dim, name=name)

        self.n_arms = n_arms
        self.action_space = action_space
        self.reward_low = reward_low
        self.reward_high = reward_high
        self.eta = eta
        self.gamma = gamma

        self.weights = torch.zeros((self.n_episodes, self.n_arms))

        self._t = 1
        self._last_arms = None
        self._last_probs = None
        self._warned_out_of_range = False
        return

    def _act(self) -> torch.Tensor:
        probs = self.compute_probs()

        # Sample one arm for each parallel episode
        arm_idxs = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Store the sampled arms and their specific probabilities for the update step
        batch_idxs = torch.arange(self.n_episodes, device=probs.device)
        self._last_arms = arm_idxs
        self._last_probs = probs[batch_idxs, arm_idxs]

        # Map discrete arm indices to their corresponding continuous vectors
        x_chosen = self.action_space[arm_idxs]
        return x_chosen

    def _update(self, reward: torch.Tensor) -> None:
        batch_idxs = torch.arange(self.n_episodes, device=reward.device)

        # A reward outside the declared range would leave the loss outside
        # [0, 1] and silently invalidate the regret guarantee
        if not self._warned_out_of_range:
            if (reward < self.reward_low).any() or (reward > self.reward_high).any():
                logger.warning(
                    "%s received a reward outside reward_range [%g, %g]. Values are "
                    "clamped, but the regret guarantee assumes the range is correct.",
                    self.name,
                    self.reward_low,
                    self.reward_high,
                )
                self._warned_out_of_range = True

        reward = reward.clamp(self.reward_low, self.reward_high)

        # Importance sampling estimator, applied only to the chosen arm
        self.weights[batch_idxs, self._last_arms] += (
            self.estimate_signal(reward) / self._last_probs
        )

        self._t += 1
        return

    def estimate_signal(self, reward: torch.Tensor) -> torch.Tensor:
        """Quantity accumulated in the weights for the arm that was played.

        Returns the negated normalized loss, so that the weights hold minus the
        cumulative loss estimate and the sampling distribution is
        ``exp(-eta * L_hat)``.

        Parameters
        ----------
        reward : torch.Tensor
            Rewards of the current round, already clamped to `reward_range`.

        Returns
        -------
        signal : torch.Tensor
            Tensor of the same shape as `reward`, in ``[-1, 0]``.
        """
        return -(self.reward_high - reward) / (self.reward_high - self.reward_low)

    def get_learning_rate(self) -> float:
        """Learning rate of the current round, constant unless overridden."""
        return self.eta

    def get_exploration(self) -> float:
        """Uniform exploration weight of the round, constant unless overridden."""
        return self.gamma

    def compute_probs(self) -> torch.Tensor:
        """Compute the current sampling probabilities for all arms.

        Returns
        -------
        probs : torch.Tensor
            Tensor of shape ``(n_episodes, n_arms)`` containing the probabilities.
        """
        gamma = self.get_exploration()
        w = self.get_learning_rate() * self.weights

        # Subtract max for numerical stability (prevents overflow in exp)
        w_max = torch.amax(w, dim=1, keepdim=True)
        exp_w = torch.exp(w - w_max)

        # Base exponential weights probabilities
        probs = exp_w / torch.sum(exp_w, dim=1, keepdim=True)

        # Optional: Mix with uniform distribution for guaranteed exploration
        if gamma > 0.0:
            probs = (1.0 - gamma) * probs + (gamma / self.n_arms)
        return probs

    def get_internal_state(self) -> dict[str, Any]:
        return {
            "t": int(self._t),
            "eta": float(self.get_learning_rate()),
            "gamma": float(self.get_exploration()),
            "reward_range": [self.reward_low, self.reward_high],
        }

    def get_policy(self) -> torch.Tensor:
        return self.compute_probs()

    def get_last_arms(self) -> torch.Tensor | None:
        return self._last_arms
