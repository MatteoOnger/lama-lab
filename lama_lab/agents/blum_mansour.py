import copy
from typing import Any

import torch

from lama_lab.agents.base import BaseAgent
from lama_lab.utils import build_from_config


class AgentBlumMansour(BaseAgent):
    """Blum-Mansour meta-agent reducing external regret to internal (swap) regret.

    Instantiates N independent experts (one for each action arm). At each step,
    it forms a Markov transition matrix from the experts' distributions,
    solves for its stationary distribution, samples actions, and updates each
    expert with appropriately scaled importance-sampling estimators.

    Parameters
    ----------
    expert_cfg : dict
        Hydra-style configuration dictionary for the underlying base expert.
    n_episodes : int
        Number of parallel episodes.
    action_space : torch.Tensor or list of list of float
        Discrete pool of available actions of shape ``(n_arms, action_dim)``.
    name : str, optional
        Human-readable identifier for the agent.
    """

    def __init__(
        self,
        expert_cfg: dict,
        n_episodes: int,
        action_space: torch.Tensor | list[list[float]],
        name: str = "AgentBlumMansour",
    ):
        if not isinstance(action_space, torch.Tensor):
            action_space = torch.tensor(action_space, dtype=torch.float32)

        self.action_space = action_space
        self.n_arms = action_space.shape[0]
        action_dim = action_space.shape[1]

        super().__init__(n_episodes=n_episodes, action_dim=action_dim, name=name)

        # Instantiate one expert per action arm
        self.experts: list[BaseAgent] = []
        for i in range(self.n_arms):
            cfg = copy.deepcopy(expert_cfg)
            cfg["n_episodes"] = n_episodes
            cfg["action_space"] = action_space
            cfg["name"] = f"{name}_Expert_{i}"

            expert = build_from_config(cfg)
            self.experts.append(expert)

        self._current_stationary_dist: torch.Tensor | None = (
            torch.ones((self.n_episodes, self.n_arms)) / self.n_arms
        )
        return

    def get_policy(self) -> torch.Tensor | None:
        """Return the current stationary distribution over action arms."""
        return self._current_stationary_dist

    def get_internal_state(self) -> dict[str, Any]:
        """Collect internal state metadata from all underlying experts."""
        return {
            "eta": None,
            "gamma": None,
        }

    def _act(self) -> torch.Tensor:
        """Build Markov matrix, solve stationary distribution, and sample actions."""
        probs_list = []
        for i, expert in enumerate(self.experts):
            dist = expert.get_policy()
            if dist is None:
                raise ValueError(
                    f"Expert {i} ({expert.name}) returned None for get_distribution(). "
                    f"Blum-Mansour requires discrete experts with explicit distributions."
                )
            probs_list.append(dist)

        # Transition matrix shape: (n_episodes, n_arms, n_arms)
        markov_matrix = torch.stack(probs_list, dim=1)

        # Solve stationary distribution pi_t = pi_t * P_t
        pi_t = self._solve_stationary_distribution(markov_matrix)
        self._current_stationary_dist = pi_t

        # Sample global action arm indices for each episode
        sampled_indices = torch.multinomial(pi_t, num_samples=1).squeeze(1)

        # Retrieve global action probabilities pi_t(A_t)
        sampled_probs = pi_t.gather(dim=1, index=sampled_indices.unsqueeze(1)).squeeze(
            1
        )

        # Populate inherited state attributes
        self._last_action_index = sampled_indices
        self._last_action_probability = sampled_probs

        return self.action_space[sampled_indices]

    def _update(self, reward: torch.Tensor) -> None:
        """Distribute scaled reward feedback to experts according to Blum-Mansour rule."""
        if self._current_stationary_dist is None or self._last_action_index is None:
            raise RuntimeError("Cannot update before calling _act().")

        A_t = self._last_action_index  # shape: (n_episodes,)
        pi_A_t = self._last_action_probability  # shape: (n_episodes,)

        for i, expert in enumerate(self.experts):
            pi_i = self._current_stationary_dist[:, i]

            # Inject the played global action into the expert
            expert._last_action_index = A_t
            expert._last_action = self.action_space[A_t]

            # Set virtual action probability so Exp3 computes:
            # estimate_signal(reward) / (pi_A_t / pi_i) = estimate_signal(reward) * (pi_i / pi_A_t)
            # Clamp pi_i to avoid division by zero if an expert's weight drops to zero.
            expert._last_action_probability = pi_A_t / torch.clamp(pi_i, min=1e-12)

            # Pass unmodified environment reward to keep it strictly within reward_range
            expert.update(reward)

    def _solve_stationary_distribution(self, P: torch.Tensor) -> torch.Tensor:
        """Solve pi (I - P) = 0 subject to sum(pi) = 1 via batched linear solve."""
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
