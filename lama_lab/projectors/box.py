import torch

from .base import BaseProjector


class BoxProjector(BaseProjector):
    """Clamp values into a rectangular box domain.

    Parameters
    ----------
    low : float
        Lower bound of the admissible interval.
    high : float
        Upper bound of the admissible interval.
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        return

    def project(self, vec: torch.Tensor) -> torch.Tensor:
        """Clamp each entry of ``vec`` to the interval ``[low, high]``.

        Parameters
        ----------
        vec : torch.Tensor
            Input tensor.

        Returns
        -------
        out : torch.Tensor
            Clamped tensor.
        """
        return torch.clamp(vec, min=self.low, max=self.high)
