from abc import ABC, abstractmethod
from typing import Any

import torch

from ..spaces import Space


class Agent(ABC):
    """Abstract base class for learning agents.

    Defines the public agent API, validates common inputs, and delegates
    algorithm-specific behavior to :meth:`_act` and :meth:`_update`.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to simulate in parallel. Must be positive.
    action_space : Space
        Action space defining the environment's actions.
    name : str
        Human-readable name of the agent.

    Attributes
    ----------
    last_action_info : dict of str to Any
        Metadata about the last selected action, overwritten at every step.
        :meth:`act` stores the selected actions under ``"action"`` and sets
        ``"pending_update"`` to ``True``, which :meth:`update` resets to
        ``False``. Subclasses may store additional keys and should document
        them.

    Raises
    ------
    ValueError
        If ``n_episodes`` is not strictly positive.
    """

    def __init__(self, n_episodes: int, action_space: Space, name: str) -> None:
        if n_episodes <= 0:
            raise ValueError("The 'n_episodes' parameter must be positive.")

        self.n_episodes = n_episodes
        self.action_space = action_space
        self.name = name

        self.last_action_info: dict[str, Any] = {}
        return

    def act(self) -> torch.Tensor:
        """Select actions for the current step and validate their shape.

        Returns
        -------
        torch.Tensor
            Selected actions with shape ``(n_episodes, *action_space.shape)``.

        Raises
        ------
        ValueError
            If the selected action tensor does not have shape
            ``(n_episodes, *action_space.shape)``.
        """
        action = self._act()
        expected_shape = (self.n_episodes, *self.action_space.shape)

        if action.shape != expected_shape:
            raise ValueError(
                f"Expected action shape {expected_shape}, got {action.shape}."
            )

        self.last_action_info["action"] = action
        self.last_action_info["pending_update"] = True
        return action

    def update(self, reward: torch.Tensor) -> None:
        """Update the agent using rewards received for the selected actions.

        Parameters
        ----------
        reward : torch.Tensor
            Rewards obtained by executing the selected actions. Must have
            shape ``(n_episodes,)``.

        Raises
        ------
        RuntimeError
            If no action is awaiting an update, i.e. :meth:`act` has not been
            called since the last update.
        ValueError
            If ``reward`` does not have shape ``(n_episodes,)``.
        """
        if not self.last_action_info.get("pending_update", False):
            raise RuntimeError("No action is awaiting an update. Call 'act()' first.")

        if reward.shape != (self.n_episodes,):
            raise ValueError(
                f"The 'reward' tensor must have shape ({self.n_episodes},), got {reward.shape}."
            )

        self._update(reward)
        self.last_action_info["pending_update"] = False
        return

    @abstractmethod
    def _act(self) -> torch.Tensor:
        """Select actions for the current step in a subclass implementation."""
        pass

    @abstractmethod
    def _update(self, reward: torch.Tensor) -> None:
        """Update subclass state using rewards for the selected actions."""
        pass

    @abstractmethod
    def get_internal_state(self) -> dict[str, Any]:
        """Retrieve the agent's internal state for logging purposes.

        Returns
        -------
        dict of str to Any
            Dictionary containing the agent's internal variables.
        """
        pass

    @abstractmethod
    def get_policy(self) -> torch.Tensor | None:
        """Retrieve the current policy over a discrete action space.

        Returns
        -------
        torch.Tensor or None
            Current action distribution with shape ``(n_episodes, n_arms)``, or
            ``None`` if the agent does not maintain an explicit policy.
        """
        pass
