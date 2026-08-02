from abc import ABC, abstractmethod

import torch


class BaseProjector(ABC):
    """Abstract base class for tensor projection operators.

    Subclasses define how input tensors are projected into a valid domain.
    """

    @abstractmethod
    def project(self, vec: torch.Tensor) -> torch.Tensor:
        """Project a tensor into the admissible domain.

        Parameters
        ----------
        vec : torch.Tensor
            Input tensor to project.

        Returns
        -------
        out : torch.Tensor
            Projected tensor.
        """
        raise NotImplementedError

    def __call__(self, vec: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`project`."""
        return self.project(vec)
