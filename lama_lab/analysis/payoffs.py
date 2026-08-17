import torch


def build_ecdf(
    samples: torch.Tensor,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the empirical CDF machinery of a sample of latent values.

    The returned pair supports O(1) evaluation of :math:`P(V < c)` and
    :math:`E[V \\mathbf{1}\\{V < c\\}]` for any threshold, once the threshold has
    been located with :func:`torch.searchsorted`.

    Parameters
    ----------
    samples : torch.Tensor
        Tensor containing samples drawn from the target distribution. It is
        flattened before sorting.
    dtype : torch.dtype, optional
        Floating point type used for the sorted samples and the prefix sums.
        Defaults to ``torch.float64``: with a large number of samples the
        prefix sums grow well beyond the resolution of ``torch.float32``, and
        the upper tail is evaluated as a difference of two prefix sums, which
        is a catastrophic cancellation pattern.

    Returns
    -------
    samples_sorted : torch.Tensor
        One-dimensional tensor of shape ``(n_samples,)`` with the samples in
        ascending order.
    cum_sums : torch.Tensor
        One-dimensional tensor of shape ``(n_samples + 1,)`` containing the
        prefix sums of ``samples_sorted``, with a leading zero, so that
        ``cum_sums[i]`` is the sum of the ``i`` smallest samples.
    """
    samples_sorted = torch.sort(samples.flatten().to(dtype)).values

    cum_sums = torch.cat(
        [
            torch.zeros(1, dtype=dtype, device=samples_sorted.device),
            torch.cumsum(samples_sorted, dim=0),
        ]
    )
    return samples_sorted, cum_sums


def build_quote_grid(
    low: float,
    high: float,
    delta: float,
    min_spread: float | None = None,
    epsilon: float | None = None,
) -> torch.Tensor:
    """Build the finite set of ``(bid, ask)`` quotes on a uniform price grid.

    Every pair of grid prices with ``ask - bid >= min_spread`` is returned, in
    lexicographic order. The result is meant to be used as the ``action_space``
    of :class:`~lama_lab.agents.AgentExp3`, and is directly constructible from a
    configuration file through the ``_target_`` mechanism.

    Parameters
    ----------
    low : float
        Lowest price of the grid.
    high : float
        Highest price of the grid.
    delta : float
        Tick size. ``(high - low)`` must be an integer multiple of it.
    min_spread : float, optional
        Minimum admissible value of ``ask - bid``. Defaults to ``delta``, i.e.
        only degenerate zero-spread quotes are excluded.
    epsilon : float, optional
        Price tolerance of the environment. If given, the grid is validated
        against it, since a tick size comparable to the tolerance makes the
        tie-breaking rule of the environment ill-defined.

    Returns
    -------
    action_space : torch.Tensor
        Tensor of shape ``(n_arms, 2)`` containing the ``(bid, ask)`` quotes.

    Raises
    ------
    ValueError
        If ``delta`` is not strictly positive, if ``high`` is not greater than
        ``low``, if the price range is not an integer multiple of ``delta``, if
        no quote satisfies the minimum spread, or if ``delta`` is within a
        factor of two of ``epsilon``.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be strictly positive. Got {delta}.")
    if high <= low:
        raise ValueError(f"high must be greater than low. Got {low} and {high}.")

    min_spread = delta if min_spread is None else min_spread

    n_ticks = round((high - low) / delta)
    if abs((high - low) - n_ticks * delta) > 1e-9:
        raise ValueError(
            f"The price range ({high - low}) must be an integer multiple of "
            f"delta ({delta})."
        )

    # A tick of the order of the tolerance would let the environment treat two
    # distinct quotes as a tie, discontinuously halving the payoff of a maker.
    if epsilon is not None and delta < 2.0 * epsilon:
        raise ValueError(
            f"delta ({delta}) must be at least twice the environment tolerance "
            f"epsilon ({epsilon}), otherwise distinct quotes are seen as ties."
        )

    prices = torch.linspace(low, high, n_ticks + 1, dtype=torch.float64)

    # All (bid, ask) pairs with bid < ask, in lexicographic order
    quotes = torch.combinations(prices, r=2)
    quotes = quotes[quotes[:, 1] - quotes[:, 0] >= min_spread - 1e-9]

    if quotes.numel() == 0:
        raise ValueError(
            f"No quote on the grid satisfies min_spread ({min_spread}). "
            f"The widest available spread is {high - low}."
        )
    return quotes.to(dtype=torch.get_default_dtype())


def get_expected_payoff_matrix(
    action_space: torch.Tensor | list[list[float]],
    samples: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    r"""Compute the exact expected payoff matrix of the two-maker game.

    Entry ``(i, j)`` is the expected reward of the row maker when it quotes arm
    ``i`` and the column maker quotes arm ``j``, evaluated in closed form
    against the empirical distribution of ``samples``. The semantics replicate
    :meth:`~lama_lab.envs.MarketMakingEnvironment.step` exactly, including the
    tolerance band within which the trader is indifferent and picks a side at
    random, the tolerance used to detect price ties, and the equal split of the
    profit among the makers quoting the best price.

    Parameters
    ----------
    action_space : torch.Tensor or list of list of float
        Tensor of shape ``(n_arms, 2)`` containing the ``(bid, ask)`` arms.
    samples : torch.Tensor
        Tensor containing samples drawn from the latent value distribution.
    epsilon : float, optional
        Numerical tolerance of the environment. Must match the ``epsilon`` of
        the environment the matrix is compared against.

    Returns
    -------
    payoff : torch.Tensor
        Tensor of shape ``(n_arms, n_arms)``, in double precision. The payoff
        matrix of the column maker is its transpose.

    Raises
    ------
    ValueError
        If `action_space` is not of shape ``(n_arms, 2)``, or if `epsilon` is
        not strictly positive.

    Notes
    -----
    Writing $A = \min(a_i, a_j)$ for the best ask, $B = \max(b_i, b_j)$ for the
    best bid and $m = (A + B) / 2$, the trader executes against the ask when
    $V > m + \epsilon/2$, against the bid when $V < m - \epsilon/2$, and picks a
    side with a fair coin on the closed band in between. Hence

    .. math::

        u_i = s^{ask}_i \left(
            \mathbb{E}[(A - V) \mathbf{1}\{V > m + \epsilon/2\}]
            + \tfrac12 \mathbb{E}[(A - V) \mathbf{1}\{|V - m| \le \epsilon/2\}]
        \right) + s^{bid}_i \left(
            \mathbb{E}[(V - B) \mathbf{1}\{V < m - \epsilon/2\}]
            + \tfrac12 \mathbb{E}[(V - B) \mathbf{1}\{|V - m| \le \epsilon/2\}]
        \right),

    where the shares $s^{ask}_i$ and $s^{bid}_i$ are the reciprocal of the
    number of makers within `epsilon` of the best price on that side, or zero if
    the row maker is not among them.

    The matrix is exact with respect to the *empirical* distribution of
    `samples`, whose own error with respect to the underlying law is of order
    :math:`\sigma / \sqrt{n}` and is correlated across entries.

    This function is specific to two makers. With more makers the payoff depends
    on the whole profile through the best bid and the best ask, so the exact
    object is a tensor with one axis per maker rather than a matrix.
    """
    action_space = torch.as_tensor(action_space)

    if action_space.ndim != 2 or action_space.shape[1] != 2:
        raise ValueError(
            f"action_space must be a 2D matrix of shape (n_arms, 2). "
            f"Got shape {tuple(action_space.shape)}."
        )
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be strictly positive. Got {epsilon}.")

    dtype = torch.float64
    device = samples.device

    arms = action_space.to(device=device, dtype=dtype)
    bids, asks = arms[:, 0], arms[:, 1]
    n_arms = arms.shape[0]

    samples_sorted, cum_sums = build_ecdf(samples, dtype=dtype)
    n_samples = samples_sorted.numel()

    # Best quotes available to the trader, for every ordered pair of arms
    best_bid = torch.maximum(bids[:, None], bids[None, :])
    best_ask = torch.minimum(asks[:, None], asks[None, :])

    # Makers within epsilon of the best price, on each side separately
    row_on_ask = (asks[:, None] - best_ask).abs() < epsilon
    col_on_ask = (asks[None, :] - best_ask).abs() < epsilon
    row_on_bid = (bids[:, None] - best_bid).abs() < epsilon
    col_on_bid = (bids[None, :] - best_bid).abs() < epsilon

    n_on_ask = row_on_ask.to(dtype) + col_on_ask.to(dtype)
    n_on_bid = row_on_bid.to(dtype) + col_on_bid.to(dtype)

    # Every selected maker is paid from the *best* price rather than from its
    # own quote, so a maker quoting slightly worse than the best still collects
    # the full profit of the best price, shared with the other selected makers
    zero = torch.zeros((), dtype=dtype, device=device)
    share_ask = torch.where(row_on_ask, 1.0 / n_on_ask, zero)
    share_bid = torch.where(row_on_bid, 1.0 / n_on_bid, zero)

    # Indifference band of the trader, on which the environment flips a coin
    mid = (best_ask + best_bid) / 2.0
    c_lo = mid - epsilon / 2.0
    c_hi = mid + epsilon / 2.0

    # Both sides are strict in the environment, hence the asymmetric sidedness
    idx_lo = torch.searchsorted(samples_sorted, c_lo.reshape(-1).contiguous())
    idx_hi = torch.searchsorted(
        samples_sorted, c_hi.reshape(-1).contiguous(), right=True
    )
    idx_lo = idx_lo.reshape(n_arms, n_arms)
    idx_hi = idx_hi.reshape(n_arms, n_arms)

    prob_bid = idx_lo.to(dtype) / n_samples
    mass_bid = cum_sums[idx_lo] / n_samples

    prob_ask = (n_samples - idx_hi).to(dtype) / n_samples
    mass_ask = (cum_sums[-1] - cum_sums[idx_hi]) / n_samples

    # Taken as a difference of indices, so that the three masses sum to one
    prob_tie = (idx_hi - idx_lo).to(dtype) / n_samples
    mass_tie = (cum_sums[idx_hi] - cum_sums[idx_lo]) / n_samples

    profit_ask = (best_ask * prob_ask - mass_ask) + 0.5 * (
        best_ask * prob_tie - mass_tie
    )
    profit_bid = (mass_bid - best_bid * prob_bid) + 0.5 * (
        mass_tie - best_bid * prob_tie
    )

    payoff = share_ask * profit_ask + share_bid * profit_bid
    return payoff


def get_pure_nash(payoff: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
    """Enumerate the pure Nash equilibria of the finite game.

    The profile ``(i, j)`` is a pure Nash equilibrium when neither maker gains
    by unilaterally switching arm. This is computed directly on the finite
    action set, and is therefore unrelated to the fixed points of the
    continuous game returned by
    :func:`~lama_lab.analysis.get_all_unique_fixed_points`.

    Parameters
    ----------
    payoff : torch.Tensor
        Payoff matrix of the row maker, of shape ``(n_arms, n_arms)``, as
        returned by :func:`get_expected_payoff_matrix`. The payoff matrix of the
        column maker is its transpose.
    tol : float, optional
        Slack allowed on each best-response condition, so that profiles within
        `tol` of optimal are retained.

    Returns
    -------
    profiles : torch.Tensor
        Tensor of shape ``(n_profiles, 2)`` containing the arm indices of the
        row and column maker at each equilibrium, in lexicographic order.
    """
    best = payoff.amax(dim=0)

    # Row maker plays i against column j, column maker plays j against row i
    row_is_best = payoff >= best[None, :] - tol
    col_is_best = payoff.T >= best[:, None] - tol

    return torch.nonzero(row_is_best & col_is_best)


def get_exploitability(
    payoff: torch.Tensor,
    row_policy: torch.Tensor,
    col_policy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute how much each maker gains by best responding to the other.

    Given a product distribution over the joint action space, the exploitability
    of a maker is the increase in expected payoff it obtains by switching to its
    best pure response while the opponent keeps its own mixed strategy. Both
    values are zero if and only if the pair is a mixed Nash equilibrium.

    Parameters
    ----------
    payoff : torch.Tensor
        Payoff matrix of the row maker, of shape ``(n_arms, n_arms)``, as
        returned by :func:`get_expected_payoff_matrix`.
    row_policy : torch.Tensor
        Mixed strategy of the row maker, of shape ``(..., n_arms)``. Leading
        dimensions are treated as a batch.
    col_policy : torch.Tensor
        Mixed strategy of the column maker, of shape ``(..., n_arms)``.

    Returns
    -------
    row_exploitability : torch.Tensor
        Best-response gain of the row maker, of shape ``(...)``.
    col_exploitability : torch.Tensor
        Best-response gain of the column maker, of shape ``(...)``.
    """
    # payoff @ p gives the payoff of each pure action against the opponent
    row_best = (col_policy @ payoff.T).amax(dim=-1)
    col_best = (row_policy @ payoff.T).amax(dim=-1)

    row_value = torch.einsum("...i,ij,...j->...", row_policy, payoff, col_policy)
    col_value = torch.einsum("...j,ji,...i->...", col_policy, payoff, row_policy)

    return row_best - row_value, col_best - col_value
