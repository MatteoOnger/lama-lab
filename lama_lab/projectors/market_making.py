import torch

from .base import BaseProjector


class MarketMakingProjector(BaseProjector):
    """Project two-dimensional vectors into a valid bid/ask spread region.

    Parameters
    ----------
    low : float
        Lower bound for the bid and ask prices.
    high : float
        Upper bound for the bid and ask prices.
    epsilon : float, optional
        Minimum allowed spread between the bid and ask prices. It ensures that
        the projected pair remains separated by a positive gap and avoids
        degenerate or inverted spreads.
    """

    def __init__(self, low: float, high: float, epsilon: float = 0.001):
        self.low = low
        self.high = high
        self.epsilon = epsilon
        return

    def project(self, vec: torch.Tensor) -> torch.Tensor:
        """Project a batch of ``(bid, ask)`` pairs into a valid market-making domain.

        Parameters
        ----------
        vec : torch.Tensor
            Tensor of shape ``(n_samples, 2)`` containing bid/ask pairs.

        Returns
        -------
        out : torch.Tensor
            Projected tensor with the same shape as ``vec``.
        """
        if vec.ndim != 2 or vec.shape[1] != 2:
            raise ValueError("vec must have shape (n_samples, 2)")

        x = torch.clamp(vec, min=self.low, max=self.high)
        b, a = x[:, 0], x[:, 1]

        swapped_mask = torch.abs(a - b) < self.epsilon
        if swapped_mask.any():
            mid = (a[swapped_mask] + b[swapped_mask]) / 2.0
            a[swapped_mask] = mid + self.epsilon / 2.0
            b[swapped_mask] = mid - self.epsilon / 2.0

            below_low_mask = b < self.low
            if below_low_mask.any():
                b[below_low_mask] = self.low
                a[below_low_mask] = self.low + self.epsilon

            above_high_mask = a > self.high
            if above_high_mask.any():
                b[above_high_mask] = self.high - self.epsilon
                a[above_high_mask] = self.high
        return torch.stack((b, a), dim=1)
