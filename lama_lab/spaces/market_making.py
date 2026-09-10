import torch

from .space import ContinuousSpace, DiscreteSpace


class ContinuousMMActionSpace(ContinuousSpace):
    """Continuous market-making action space for two-dimensional ``(bid, ask)`` price pairs.

    Parameters
    ----------
    low : float
        Lower bound for the bid and ask prices.
    high : float
        Upper bound for the bid and ask prices.
    tick_size : float, optional
        Minimum allowed spread between the bid and ask prices. Ensures that
        the pair remains separated by a positive gap and avoids degenerate
        or inverted spreads.
    dtype : torch.dtype, optional
        Data type of the tensor elements.

    Attributes
    ----------
    shape : tuple of int
        Shape of a single action, ``(2,)`` (inherited from :class:`Space`).
    ndim : int
        Number of dimensions of a single action (inherited from :class:`Space`).

    Raises
    ------
    ValueError
        If ``low >= high`` or if ``high - low < tick_size``.
    """

    def __init__(
        self,
        low: float,
        high: float,
        tick_size: float = 0.001,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if low >= high:
            raise ValueError("'low' must be strictly less than 'high'.")
        if high - low < tick_size:
            raise ValueError(
                "The difference between 'high' and 'low' must be at least 'tick_size'."
            )

        super().__init__(shape=(2,), dtype=dtype, low=low, high=high)
        self.tick_size = tick_size
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
        valid_spread = (asks - bids) >= self.tick_size
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

        # Keep both quotes within the admissible price interval
        x = torch.clamp(raw_x, min=self.low, max=self.high)

        # Order the quotes so the first coordinate is the bid and the second is the ask
        b = torch.minimum(x[:, 0], x[:, 1])
        a = torch.maximum(x[:, 0], x[:, 1])

        # Expand only pairs whose spread is too small, preserving wider spreads
        too_close_mask = (a - b) < self.tick_size
        if too_close_mask.any():
            mid = (a[too_close_mask] + b[too_close_mask]) / 2.0
            a[too_close_mask] = mid + self.tick_size / 2.0
            b[too_close_mask] = mid - self.tick_size / 2.0

            # Restore feasibility when the expansion crossed a global bound
            below_low_mask = b < self.low
            if below_low_mask.any():
                b[below_low_mask] = self.low
                a[below_low_mask] = self.low + self.tick_size

            above_high_mask = a > self.high
            if above_high_mask.any():
                b[above_high_mask] = self.high - self.tick_size
                a[above_high_mask] = self.high
        return torch.stack((b, a), dim=1)


class DiscreteMMActionSpace(DiscreteSpace):
    """Discrete market-making action space for two-dimensional ``(bid, ask)`` price pairs
    based on a uniform price grid.

    Parameters
    ----------
    low : float
        Lowest price allowed on the grid.
    high : float
        Highest price allowed on the grid.
    n_ticks : int
        Number of intervals (ticks) between 'low' and 'high'. This implicitly
        defines the 'tick_size' (step size) of the grid.
    dtype : torch.dtype, optional
        Data type of the physical tensor elements.

    Attributes
    ----------
    actions : torch.Tensor
        Tensor of shape ``(n, 2)`` containing all valid discrete
        ``(bid, ask)`` pairs on the grid.
    prices : torch.Tensor
        Tensor of shape ``(n_ticks + 1,)`` containing all valid price ticks.
    tick_size : float
        The price increment derived from 'low', 'high', and 'n_ticks'.
    shape : tuple of int
        Shape of a single action, ``(2,)`` (inherited from :class:`Space`).
    ndim : int
        Number of dimensions of a single action (inherited from :class:`Space`).
    num_elements : int
        Total number of valid discrete actions (inherited from :class:`DiscreteSpace`).

    Raises
    ------
    ValueError
        If ``high <= low`` or if ``n_ticks <= 0``.
    """

    def __init__(
        self,
        low: float,
        high: float,
        n_ticks: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if high <= low:
            raise ValueError(
                f"'high' must be greater than 'low'. Got {low} and {high}."
            )
        if n_ticks <= 0:
            raise ValueError(
                f"'n_ticks' must be a strictly positive integer. Got {n_ticks}."
            )

        tick_size = (high - low) / n_ticks

        # Generate the grid and all possible ordered (bid, ask) combinations
        prices = torch.linspace(low, high, n_ticks + 1, dtype=dtype)
        actions = torch.combinations(prices, r=2)

        super().__init__(shape=(2,), dtype=dtype, num_elements=actions.shape[0])
        self.low = low
        self.high = high
        self.tick_size = tick_size
        self.n_ticks = n_ticks
        self.prices = prices
        self.actions = actions

        # Tolerance to handle floating-point inaccuracies
        self._grid_tolerance = max(
            1e-9,
            4.0
            * torch.finfo(dtype).eps
            * max(1.0, abs(low), abs(high), abs(tick_size)),
        )
        return

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Check whether a batch of ``(bid, ask)`` pairs belongs to the grid.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of candidate values with shape ``(N, 2)``.

        Returns
        -------
        mask : torch.Tensor
            Boolean tensor of shape ``(N,)`` indicating whether each pair is a
            valid discrete action.
        """
        if x.ndim != 2 or x.shape[1] != 2:
            return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

        steps = torch.round((x - self.low) / self.tick_size)
        grid_values = self.low + steps * self.tick_size

        on_grid = torch.isclose(
            x, grid_values, rtol=0.0, atol=self._grid_tolerance
        ).all(dim=1)
        valid_order = x[:, 0] < x[:, 1]
        in_range = (x >= self.low).all(dim=1) & (x <= self.high).all(dim=1)
        return on_grid & valid_order & in_range

    def project(self, raw_x: torch.Tensor) -> torch.Tensor:
        """Project continuous ``(bid, ask)`` pairs to the nearest valid grid action.

        Each pair is first ordered so that ``bid <= ask``. The ordered prices
        are then independently snapped to the nearest price tick. If both prices
        snap to the same tick, the closest valid pair with two distinct ticks is
        selected.

        Parameters
        ----------
        raw_x : torch.Tensor
            Tensor of shape ``(N, 2)`` containing unconstrained bid/ask pairs.

        Returns
        -------
        valid_x : torch.Tensor
            Tensor of shape ``(N, 2)`` containing the nearest valid grid actions.

        Raises
        ------
        ValueError
            If ``raw_x`` does not have shape ``(N, 2)``.
        """
        if raw_x.ndim != 2 or raw_x.shape[1] != 2:
            raise ValueError("'raw_x' must have shape (N, 2).")

        # Ensure bid <= ask
        x = torch.sort(raw_x, dim=1).values

        # Map values to integer grid coordinates and clamp to valid range
        ticks = torch.round((x - self.low) / self.tick_size).long()
        ticks.clamp_(0, self.n_ticks)

        i, j = ticks[:, 0], ticks[:, 1]

        # Handle collisions where bid and ask fall on the same tick (i == j)
        invalid = i == j
        if invalid.any():
            k = i[invalid]
            x_inv = x[invalid]

            has_lower = k > 0
            has_upper = k < self.n_ticks

            dist_lower = torch.full_like(x_inv[:, 0], torch.inf)
            dist_upper = torch.full_like(x_inv[:, 0], torch.inf)

            if has_lower.any():
                bid_l = self.low + (k[has_lower] - 1).to(x.dtype) * self.tick_size
                ask_l = self.low + k[has_lower].to(x.dtype) * self.tick_size
                dist_lower[has_lower] = (x_inv[has_lower, 0] - bid_l) ** 2 + (
                    x_inv[has_lower, 1] - ask_l
                ) ** 2

            if has_upper.any():
                bid_u = self.low + k[has_upper].to(x.dtype) * self.tick_size
                ask_u = self.low + (k[has_upper] + 1).to(x.dtype) * self.tick_size
                dist_upper[has_upper] = (x_inv[has_upper, 0] - bid_u) ** 2 + (
                    x_inv[has_upper, 1] - ask_u
                ) ** 2

            # Choose the shift direction (up or down) that minimizes the distance
            use_upper = dist_upper < dist_lower
            i[invalid] = torch.where(use_upper, k, k - 1)
            j[invalid] = torch.where(use_upper, k + 1, k)

        indices = torch.stack((i, j), dim=1)
        return self.low + indices.to(x.dtype) * self.tick_size

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map discrete action indices to their physical ``(bid, ask)`` pairs.

        Parameters
        ----------
        indices : torch.Tensor
            1D tensor of shape ``(N,)`` containing discrete action indices.

        Returns
        -------
        values : torch.Tensor
            Tensor of shape ``(N, 2)`` containing physical bid/ask pairs.

        Raises
        ------
        ValueError
            If ``indices`` does not have shape ``(N,)`` or contains indices
            outside the valid range.
        """
        if indices.ndim != 1:
            raise ValueError("'indices' must have shape (N,).")

        if (indices < 0).any() or (indices >= self.num_elements).any():
            raise ValueError(
                f"All indices must be in the range [0, {self.num_elements - 1}]."
            )
        return self.actions[indices]

    def to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Map physical ``(bid, ask)`` coordinates to discrete action indices.

        Every input pair must exactly match a valid action on the price
        grid with ``bid < ask``. Otherwise, a ``ValueError`` is raised.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of physical values with shape ``(N, 2)``.

        Returns
        -------
        indices : torch.Tensor
            1D tensor of shape ``(N,)`` containing the corresponding discrete
            action indices.

        Raises
        ------
        ValueError
            If ``x`` does not have shape ``(N, 2)`` or if one or more entries do
            not correspond to a valid discrete action.
        """
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("'x' must have shape (N, 2).")

        steps = torch.round((x - self.low) / self.tick_size).long()
        i, j = steps[:, 0], steps[:, 1]

        valid_ticks = (i >= 0) & (j >= 0) & (i < j) & (j <= self.n_ticks)

        reconstructed = self.low + steps.to(x.dtype) * self.tick_size
        on_grid = torch.isclose(
            x, reconstructed, rtol=0.0, atol=self._grid_tolerance
        ).all(dim=1)

        valid = valid_ticks & on_grid

        if not valid.all():
            invalid_indices = torch.nonzero(~valid, as_tuple=False).squeeze(1)
            raise ValueError(
                "All entries of 'x' must match a valid discrete action on the "
                f"price grid. Invalid entries at batch indices {invalid_indices.tolist()}."
            )
        n = self.n_ticks + 1
        return i * (n - 1) - (i * (i - 1)) // 2 + j - i - 1
