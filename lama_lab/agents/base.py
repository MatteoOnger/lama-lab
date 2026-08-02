from abc import ABC, abstractmethod

import torch


class BaseAgent(ABC):
    """Abstract base class for reinforcement-learning agents.

    Subclasses implement the action-selection and update routines used by the
    training loop.

    Parameters
    ----------
    name : str, optional
        Human-readable name of the agent.
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        return

    @abstractmethod
    def act(self) -> torch.Tensor:
        """Select an action for the current step.

        Returns
        -------
        out : torch.Tensor
            Tensor containing the action values produced by the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, rewards: torch.Tensor) -> None:
        """Update the agent's internal parameters using the received rewards.

        Parameters
        ----------
        rewards : torch.Tensor
            Reward values associated with the currently evaluated actions.
        """
        raise NotImplementedError
