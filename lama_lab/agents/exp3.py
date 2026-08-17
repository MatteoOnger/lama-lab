from typing import Any

import torch

from .base import BaseAgent


class AgentExp3(BaseAgent):
    """Exponential-weight algorithm for Exploration and Exploitation (Exp3).

    This agent treats the optimization problem as an adversarial multi-armed
    bandit. Given a discrete set of possible actions (arms), it maintains a
    probability distribution over them. The distribution is updated using an
    importance-sampling estimator of the observed rewards, ensuring robust
    exploration and exploitation even in non-stationary environments.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to simulate in parallel.
    action_space : torch.Tensor or list of list of float
        A 2D tensor of shape ``(n_arms, action_dim)`` containing the discrete
        pool of possible actions.
    eta : float, optional
        Learning rate (or inverse temperature) controlling how aggressively
        the probability distribution updates based on estimated rewards.
        Higher values lead to faster exploitation.
    gamma : float, optional
        Exploration parameter in [0, 1]. Mixes the exponential weights with a
        uniform distribution. If 0.0, the algorithm behaves purely as Hedge.
    name : str, optional
        Human-readable identifier for the agent.

    Attributes
    ----------
    n_arms : int
        Number of discrete actions available in the action space.
    weights : torch.Tensor
        Current unnormalized log-weights for each arm and episode, of shape
        ``(n_episodes, n_arms)``.

    Raises
    ------
    ValueError
        If `action_space` is not a 2D matrix/tensor.

    Notes
    -----
    This implementation supports batched execution. All episodes share the same
    hyperparameters but maintain independent weight vectors, allowing :meth:`_act`
    and :meth:`_update` to operate on tensors using vectorized PyTorch operations.
    The update rule uses the standard importance-sampled reward estimator:
    :math:`w_{t+1, a} = w_{t, a} + \frac{r_t}{p_{t, a}}`.
    """

    def __init__(
        self,
        n_episodes: int,
        action_space: torch.Tensor | list[list[float]],
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

        n_arms, action_dim = action_space.shape
        super().__init__(n_episodes=n_episodes, action_dim=action_dim, name=name)

        self.n_arms = n_arms
        self.action_space = action_space
        self.eta = eta
        self.gamma = gamma

        self.weights = torch.zeros((self.n_episodes, self.n_arms))

        self._t = 1
        self._last_arms = None
        self._last_probs = None
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

        # Importance sampling reward estimator: r_hat = r / p
        # Only the chosen arm gets its weight updated.
        estimated_reward = reward / self._last_probs
        self.weights[batch_idxs, self._last_arms] += estimated_reward

        self._t += 1
        return

    def compute_probs(self) -> torch.Tensor:
        """Compute the current sampling probabilities for all arms.

        Returns
        -------
        probs : torch.Tensor
            Tensor of shape ``(n_episodes, n_arms)`` containing the probabilities.
        """
        w = self.eta * self.weights

        # Subtract max for numerical stability (prevents overflow in exp)
        w_max = torch.amax(w, dim=1, keepdim=True)
        exp_w = torch.exp(w - w_max)

        # Base exponential weights probabilities
        probs = exp_w / torch.sum(exp_w, dim=1, keepdim=True)

        # Optional: Mix with uniform distribution for guaranteed exploration
        if self.gamma > 0.0:
            probs = (1.0 - self.gamma) * probs + (self.gamma / self.n_arms)
        return probs

    def get_internal_state(self) -> dict[str, Any]:
        return {
            "t": int(self._t),
            "eta": float(self.eta),
            "gamma": float(self.gamma),
        }

    def get_policy(self) -> torch.Tensor:
        return self.compute_probs()

    def get_last_arms(self) -> torch.Tensor | None:
        return self._last_arms
