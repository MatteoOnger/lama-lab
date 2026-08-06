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

    Raises
    ------
    ValueError
        If ``low > high``.
    """

    def __init__(self, low: float, high: float):
        if low > high:
            raise ValueError("low must be less than or equal to high.")
        self.low = low
        self.high = high
        return

    def project(self, vec: torch.Tensor) -> torch.Tensor:
        """Clamp each entry of ``vec`` to the interval ``[low, high]``.

        Parameters
        ----------
        vec : torch.Tensor
            Input tensor to clamp.

        Returns
        -------
        out : torch.Tensor
            Clamped tensor with the same shape as ``vec``.
        """
        return torch.clamp(vec, min=self.low, max=self.high)
