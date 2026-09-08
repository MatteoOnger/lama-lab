import torch

from .space import ContinuousSpace
from .space import DiscreteSpace


class ContinuousMMActionSpace(ContinuousSpace):
    """Continuous market-making action space for two-dimensional ``(bid, ask)`` price pairs.

    Projects arbitrary continuous vectors into a valid domain bounded by global limits
    and separated by a minimum spread.

    Parameters
    ----------
    low : float
        Lower bound for the bid and ask prices.
    high : float
        Upper bound for the bid and ask prices.
    epsilon : float, optional
        Minimum allowed spread between the bid and ask prices. Ensures that
        the pair remains separated by a positive gap and avoids degenerate
        or inverted spreads.
    dtype : torch.dtype, optional
        Data type of the tensor elements.

    Raises
    ------
    ValueError
        If ``low >= high`` or if ``high - low < epsilon``.
    """

    def __init__(
        self,
        low: float,
        high: float,
        epsilon: float = 0.001,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if low >= high:
            raise ValueError("'low' must be strictly less than 'high'.")
        if high - low < epsilon:
            raise ValueError(
                "The difference between 'high' and 'low' must be at least 'epsilon'."
            )

        super().__init__(shape=(2,), dtype=dtype, low=low, high=high)
        self.epsilon = epsilon
        return

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Check if a batch of ``(bid, ask)`` pairs belongs to the valid space.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of candidate values to validate. It must have shape
            ``(N, 2)`` containing bid/ask pairs.

        Returns
        -------
        mask : torch.Tensor
            Boolean mask tensor of shape ``(N,)`` indicating validity for
            each sample in the batch.
        """
        if x.ndim != 2 or x.shape[1] != 2:
            return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

        bids, asks = x[:, 0], x[:, 1]
        in_bounds = (bids >= self.low) & (asks <= self.high)
        valid_spread = (asks - bids) >= self.epsilon
        return in_bounds & valid_spread

    def project(self, raw_x: torch.Tensor) -> torch.Tensor:
        """Project a batch of unconstrained ``(bid, ask)`` pairs into a valid market-making domain.

        Parameters
        ----------
        raw_x : torch.Tensor
            Tensor of shape ``(N, 2)`` containing unconstrained bid/ask pairs.

        Returns
        -------
        valid_x : torch.Tensor
            Projected tensor with the same shape as ``raw_x``.

        Raises
        ------
        ValueError
            If ``raw_x`` does not have shape ``(N, 2)``.
        """
        if raw_x.ndim != 2 or raw_x.shape[1] != 2:
            raise ValueError("'raw_x' must have shape (N, 2).")

        # Clamp values to global bounds
        x = torch.clamp(raw_x, min=self.low, max=self.high)

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


class DiscreteMMActionSpace(DiscreteSpace):
    """Discrete market-making action space based on a uniform price grid.

    Generates all valid combinations of ``(bid, ask)`` price pairs on a uniform
    grid defined by ``epsilon``. Since the bid must be strictly less than the ask,
    the minimum spread is inherently equal to ``epsilon``.

    Parameters
    ----------
    low : float
        Lowest price allowed on the grid.
    high : float
        Highest price allowed on the grid.
    epsilon : float
        Price increment. The price range ``(high - low)`` must be an
        integer multiple of ``epsilon``.
    dtype : torch.dtype, optional
        Data type of the physical tensor elements.

    Raises
    ------
    ValueError
        If ``epsilon <= 0.0``, if ``high <= low``, or if ``(high - low)`` is
        not an integer multiple of ``epsilon``.
    """

    def __init__(
        self,
        low: float,
        high: float,
        epsilon: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"'epsilon' must be strictly positive. Got {epsilon}.")
        if high <= low:
            raise ValueError(
                f"'high' must be greater than 'low'. Got {low} and {high}."
            )

        n_ticks = round((high - low) / epsilon)
        if abs((high - low) - n_ticks * epsilon) > 1e-9:
            raise ValueError(
                f"The price range ({high - low}) must be an integer multiple of "
                f"'epsilon' ({epsilon})."
            )

        # Generate all valid discrete (bid, ask) pairs on the price grid.
        prices = torch.linspace(low, high, n_ticks + 1, dtype=torch.float64)
        quotes = torch.combinations(prices, r=2)
        actions = quotes.to(dtype=dtype)

        super().__init__(shape=(2,), dtype=dtype, n=actions.shape[0])
        self.actions = actions
        return

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Check if a batch of ``(bid, ask)`` pairs belongs to the discrete grid.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of candidate values to validate. It must have shape
            ``(N, 2)`` containing bid/ask pairs.

        Returns
        -------
        mask : torch.Tensor
            Boolean mask tensor of shape ``(N,)`` indicating whether each pair
            matches a valid grid action within floating-point tolerance.
        """
        if x.ndim != 2 or x.shape[1] != 2:
            return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

        # Vectorized Euclidean distance matrix to grid actions: shape (N, n_actions)
        distances = torch.cdist(x.to(self.dtype), self.actions)
        min_distances = torch.min(distances, dim=1).values
        return min_distances < 1e-9

    def project(self, raw_x: torch.Tensor) -> torch.Tensor:
        """Project continuous ``(bid, ask)`` pairs to the nearest discrete grid action.

        Parameters
        ----------
        raw_x : torch.Tensor
            Tensor of shape ``(N, 2)`` containing unconstrained coordinates.

        Returns
        -------
        valid_x : torch.Tensor
            Tensor of shape ``(N, 2)`` containing nearest valid discrete actions.

        Raises
        ------
        ValueError
            If ``raw_x`` does not have shape ``(N, 2)``.
        """
        if raw_x.ndim != 2 or raw_x.shape[1] != 2:
            raise ValueError("'raw_x' must have shape (N, 2).")

        indices = self.to_indices(raw_x)
        return self.from_indices(indices)

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map discrete action indices to their physical ``(bid, ask)`` pairs.

        Clips indices to the valid range ``[0, self.n - 1]`` before mapping.

        Parameters
        ----------
        indices : torch.Tensor
            1D tensor of shape ``(N,)`` containing discrete indices in range
            ``[0, self.n - 1]``.

        Returns
        -------
        values : torch.Tensor
            Tensor of shape ``(N, 2)`` containing physical bid/ask pairs.
        """
        clamped_indices = torch.clamp(indices, 0, self.n - 1)
        return self.actions[clamped_indices]

    def to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Map physical ``(bid, ask)`` coordinates to nearest discrete action indices.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of physical values with shape ``(N, 2)``.

        Returns
        -------
        indices : torch.Tensor
            1D tensor of shape ``(N,)`` containing mapped discrete indices.

        Raises
        ------
        ValueError
            If ``x`` does not have shape ``(N, 2)``.
        """
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("'x' must have shape (N, 2).")

        # Vectorized minimum distance search: shape (N, n_actions)
        distances = torch.cdist(x.to(self.dtype), self.actions)
        return torch.argmin(distances, dim=1)
