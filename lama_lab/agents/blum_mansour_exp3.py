import logging
from typing import Any

import torch

from .base import BaseAgent

logger = logging.getLogger(__name__)


class AgentBlumMansourExp3(BaseAgent):
    r"""Vectorized Blum-Mansour meta-agent, specialized for constant-eta,
    gamma=0 Exp3 experts.

    Mathematically identical to
    :class:`~lama_lab.agents.AgentBlumMansour` configured with a
    constant-eta, ``gamma=0`` :class:`~lama_lab.agents.AgentExp3` as
    ``expert_cfg``: same Markov-matrix-of-expert-policies construction, same
    stationary-distribution solve, same per-expert importance-weighted
    update. The difference is purely computational: :class:`AgentBlumMansour`
    represents its ``n_arms`` experts as ``n_arms`` separate
    :class:`~lama_lab.agents.AgentExp3` objects and loops over them in
    Python every round (to read each one's policy, and to feed each one its
    importance-weighted update); this class instead stores every expert's
    weight vector as one row of a single ``(n_episodes, n_arms, n_arms)``
    tensor and computes every expert's policy, and every expert's update, as
    a single vectorized tensor operation. For n_arms in the hundreds, that
    Python-level per-expert loop is a meaningful fraction of the per-round
    cost on top of the O(n_arms^3) stationary-distribution solve, so this
    class is meant as a drop-in, faster alternative wherever the experts
    would all be identical constant-eta Exp3 instances anyway -- it is not a
    different algorithm, and is not an approximation of one.

    Parameters
    ----------
    n_episodes : int
        Number of parallel episodes.
    action_space : torch.Tensor or list of list of float
        Discrete pool of available actions of shape ``(n_arms, action_dim)``.
    reward_range : tuple of float
        Lower and upper bound of the reward, as in
        :class:`~lama_lab.agents.AgentExp3`.
    eta : float, optional
        Constant learning rate shared by every expert.
    name : str, optional
        Human-readable identifier for the agent.

    Attributes
    ----------
    weights : torch.Tensor
        Tensor of shape ``(n_episodes, n_arms, n_arms)``. ``weights[:, i, :]``
        is expert ``i``'s own negated cumulative-loss weight vector, exactly
        as :attr:`AgentExp3.weights` would hold it for a standalone expert
        ``i``.
    """

    def __init__(
        self,
        n_episodes: int,
        action_space: torch.Tensor | list[list[float]],
        reward_range: tuple[float, float],
        eta: float = 0.1,
        name: str = "AgentBlumMansourExp3",
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

        # weights[:, i, a]: expert i's own negated cumulative-loss weight for arm a
        self.weights = torch.zeros((n_episodes, n_arms, n_arms))
        self._current_stationary_dist = torch.ones((n_episodes, n_arms)) / n_arms

        self._t = 1
        self._warned_out_of_range = False
        return

    def get_policy(self) -> torch.Tensor:
        """Return the current stationary distribution over action arms."""
        return self._current_stationary_dist

    def get_internal_state(self) -> dict[str, Any]:
        return {
            "t": int(self._t),
            "eta": float(self.eta),
            "gamma": 0.0,
            "reward_range": [self.reward_low, self.reward_high],
        }

    def _all_expert_policies(self) -> torch.Tensor:
        """Every expert's softmax policy at once, shape (n_episodes, n_arms, n_arms).

        Identical to calling :meth:`AgentExp3.get_policy` on each of
        ``n_arms`` separate constant-eta, gamma=0 experts and stacking the
        results, but as one vectorized softmax instead of a Python loop.
        """
        w = self.eta * self.weights
        w_max = w.amax(dim=-1, keepdim=True)
        exp_w = torch.exp(w - w_max)
        return exp_w / exp_w.sum(dim=-1, keepdim=True)

    def _act(self) -> torch.Tensor:
        """Build Markov matrix, solve stationary distribution, and sample actions."""
        markov_matrix = self._all_expert_policies()
        pi_t = self._solve_stationary_distribution(markov_matrix)
        self._current_stationary_dist = pi_t

        sampled_indices = torch.multinomial(pi_t, num_samples=1).squeeze(1)
        batch_idxs = torch.arange(self.n_episodes, device=pi_t.device)
        sampled_probs = pi_t[batch_idxs, sampled_indices]

        self._last_action_index = sampled_indices
        self._last_action_probability = sampled_probs
        return self.action_space[sampled_indices]

    def _update(self, reward: torch.Tensor) -> None:
        """Feed every expert its importance-weighted update in one vectorized step."""
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

        # Negated normalized loss, matching AgentExp3.estimate_signal
        signal = -(self.reward_high - reward) / (self.reward_high - self.reward_low)

        A_t = self._last_action_index  # (n_episodes,)
        pi_A_t = self._last_action_probability  # (n_episodes,)
        pi = self._current_stationary_dist  # (n_episodes, n_arms), pi[:, i] = pi_i

        # Per-expert i, the update is estimate_signal(reward) / (pi_A_t / pi_i)
        importance = pi_A_t.unsqueeze(-1) / torch.clamp(pi, min=1e-12)  # (n_episodes, n_arms)
        delta = signal.unsqueeze(-1) / importance  # (n_episodes, n_arms)

        batch_idxs = torch.arange(self.n_episodes, device=reward.device)
        self.weights[batch_idxs, :, A_t] += delta

        self._t += 1
        return

    def _solve_stationary_distribution(self, P: torch.Tensor) -> torch.Tensor:
        """Solve pi (I - P) = 0 subject to sum(pi) = 1 via batched linear solve.

        Identical to :meth:`AgentBlumMansour._solve_stationary_distribution`.
        """
        batch_size, n, _ = P.shape
        device = P.device

        I = torch.eye(n, device=device).expand(batch_size, n, n)
        A = (I - P).transpose(1, 2)
        A[:, -1, :] = 1.0

        b = torch.zeros(batch_size, n, 1, device=device)
        b[:, -1, 0] = 1.0

        try:
            pi = torch.linalg.solve(A, b).squeeze(-1)
        except RuntimeError:
            pi = torch.ones(batch_size, n, device=device) / n

        pi = torch.clamp(pi, min=0.0)
        return pi / pi.sum(dim=-1, keepdim=True)
