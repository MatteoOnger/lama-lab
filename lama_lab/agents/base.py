from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseAgent(ABC):
    """Abstract base class for learning agents.

    This class defines the public API for all agents and handles common
    parameter validation. Subclasses must implement the specific logic
    in the protected methods :meth:`_act` and :meth:`_update`.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to simulate in parallel. Must be positive.
    action_dim : int
        Dimensionality of the action space. Must be positive.
    name : str
        Human-readable name of the agent.

    Raises
    ------
    ValueError
        If `n_episodes` or `action_dim` are not strictly positive.
    """

    def __init__(self, n_episodes: int, action_dim: int, name: str):
        if n_episodes <= 0:
            raise ValueError("n_episodes must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")

        self.n_episodes = n_episodes
        self.action_dim = action_dim
        self.name = name
        return

    def act(self) -> torch.Tensor:
        """Select action(s) for the current step and validate their shape.

        Returns
        -------
        action : torch.Tensor
            Tensor containing the action(s) selected by the agent.
            Guaranteed to have shape `(n_episodes, action_dim)`.

        Raises
        ------
        ValueError
            If the selected action tensor does not match the expected
            shape `(n_episodes, action_dim)`.
        """
        action = self._act()

        if action.shape != (self.n_episodes, self.action_dim):
            raise ValueError(
                f"Expected action shape ({self.n_episodes}, {self.action_dim}), "
                f"got {action.shape}."
            )
        return action

    def update(self, reward: torch.Tensor) -> None:
        """Update the agent's internal parameters using the received reward(s).

        Parameters
        ----------
        reward : torch.Tensor
            Reward(s) obtained by executing the selected action(s).

        Raises
        ------
        ValueError
            If the reward is not a 1D tensor or its length does not
            match `n_episodes`.
        """
        if reward.ndim != 1 or reward.shape[0] != self.n_episodes:
            raise ValueError(
                f"reward must be a 1D tensor of shape ({self.n_episodes},). "
                f"Got shape {reward.shape}."
            )

        self._update(reward)
        return

    def get_internal_state(self) -> dict[str, Any]:
        """Retrieve the agent's internal state for logging purposes.

        Returns
        -------
        state : dict
            A dictionary containing the agent's internal variables
            (e.g., learning rates, exploration parameters). Defaults
            to an empty dictionary if not overridden.
        """
        return {}

    def get_policy(self) -> torch.Tensor | None:
        """Retrieve the current distribution over a discrete action space.

        Returns
        -------
        policy : torch.Tensor or None
            Tensor of shape ``(n_episodes, n_arms)`` containing the probability
            assigned to each arm, or ``None`` for agents that do not maintain
            an explicit distribution over a finite set of actions.

        Notes
        -----
        Intended for diagnostics that need the mixed strategy of the agent
        *before* it is sampled from, and therefore before :meth:`act` is
        called.
        """
        return None

    def get_last_arms(self) -> torch.Tensor | None:
        """Retrieve the indices of the arms selected by the last call to :meth:`act`.

        Returns
        -------
        arms : torch.Tensor or None
            Tensor of shape ``(n_episodes,)`` containing the index of the
            selected arm within the action space, or ``None`` for agents that
            do not act on a finite set of actions or have not acted yet.
        """
        return None

    @abstractmethod
    def _act(self) -> torch.Tensor:
        """Internal action-selection routine.

        To be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _update(self, reward: torch.Tensor) -> None:
        """Internal update routine.

        To be implemented by subclasses.
        """
        raise NotImplementedError
