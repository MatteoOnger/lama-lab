from abc import ABC, abstractmethod

import torch


class BaseGenerator(ABC):
    """Abstract base class for data generators.

    Subclasses define how to generate samples from a specific distribution or data source.
    """

    @abstractmethod
    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate ``n_samples`` samples.


        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        tensor : torch.Tensor
            A tensor of shape ``(n_samples,)`` containing the generated values.
        """
        raise NotImplementedError
