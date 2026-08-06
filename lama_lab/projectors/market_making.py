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

    Raises
    ------
    ValueError
        If ``low >= high`` or if ``high - low < epsilon``.
    """

    def __init__(self, low: float, high: float, epsilon: float = 0.001):
        if low >= high:
            raise ValueError("low must be strictly less than high.")
        if high - low < epsilon:
            raise ValueError(
                "The interval (high - low) must be at least equal to epsilon."
            )

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

        Raises
        ------
        ValueError
            If ``vec`` does not have shape ``(n_samples, 2)``.
        """
        if vec.ndim != 2 or vec.shape[1] != 2:
            raise ValueError("vec must have shape (n_samples, 2).")

        # Clamp values to global bounds
        x = torch.clamp(vec, min=self.low, max=self.high)

        # Ensure bid <= ask by swapping values where bid > ask
        b = torch.minimum(x[:, 0], x[:, 1])
        a = torch.maximum(x[:, 0], x[:, 1])

        # Enforce minimum spread epsilon only when ask - bid < epsilon
        too_close_mask = (a - b) < self.epsilon
        if too_close_mask.any():
            mid = (a[too_close_mask] + b[too_close_mask]) / 2.0
            a[too_close_mask] = mid + self.epsilon / 2.0
            b[too_close_mask] = mid - self.epsilon / 2.0

            # Re-align to bounds if the expansion exceeded limits
            below_low_mask = b < self.low
            if below_low_mask.any():
                b[below_low_mask] = self.low
                a[below_low_mask] = self.low + self.epsilon

            above_high_mask = a > self.high
            if above_high_mask.any():
                b[above_high_mask] = self.high - self.epsilon
                a[above_high_mask] = self.high
        return torch.stack((b, a), dim=1)
