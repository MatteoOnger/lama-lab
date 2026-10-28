"""Equilibria of the continuous market-making game.

The trader buys at the best ask or sells at the best bid, depending on whether
the latent value ``V`` lies above or below the midpoint of the best quotes. A
symmetric profile with bid ``b`` and ask ``a`` is a fixed point when the makers
earn nothing: ``b`` is the mean of ``V`` below the midpoint ``(a + b) / 2`` and
``a`` is its mean above it.

Both functions work on the closed-form functionals of the generator of ``V``,
within an interval ``[low, high]`` that contains its support.
"""

import torch

from ...generators import Generator


def _conditional_means(
    generator: Generator, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of the value below and above each ``x``, or ``x`` if there is none."""
    below, above = generator.cdf(x), generator.ccdf(x)
    tiny = torch.finfo(x.dtype).tiny

    lower = generator.lower_partial_mean(x) / below.clamp_min(tiny)
    upper = generator.upper_partial_mean(x) / above.clamp_min(tiny)
    return torch.where(below > 0, lower, x), torch.where(above > 0, upper, x)


def fixed_points(
    generator: Generator,
    low: float,
    high: float,
    grid_size: int = 5000,
    eps: float = 1e-3,
    tol: float = 1e-3,
) -> torch.Tensor:
    r"""Find the fixed points of the continuous market-making game.

    A fixed point is a midpoint $x^*$ at which

    .. math:: x^* = \frac{1}{2} \left( E[V \mid V \le x^*] + E[V \mid V > x^*] \right),

    and the corresponding quotes are the two conditional means. The residual of
    this equation is evaluated on a grid, and its zero crossings are refined by
    linear interpolation.

    Parameters
    ----------
    generator : Generator
        Distribution of the latent value ``V``.
    low : float
        Lower end of an interval containing the support of ``V``.
    high : float
        Upper end of an interval containing the support of ``V``.
    grid_size : int, optional
        Number of points of the grid over ``[low, high]``.
    eps : float, optional
        Minimum distance between two fixed points. Closer ones are merged.
    tol : float, optional
        Largest residual of the equation at a candidate, which discards the
        jumps of an empirical estimate that cross zero without a root.

    Returns
    -------
    points : torch.Tensor
        Tensor of shape ``(n_points, 3)``, in double precision, with the bid
        ``E[V | V <= x*]``, the midpoint ``x*`` and the ask ``E[V | V > x*]`` of
        each fixed point, sorted by midpoint. It has no rows if there is none.
    """
    x = torch.linspace(low, high, grid_size, dtype=torch.float64)
    lower, upper = _conditional_means(generator, x)
    residual = (lower + upper) / 2 - x

    # Zero crossings between consecutive grid points, with the root interpolated
    crossing = torch.nonzero(residual[1:].sign() != residual[:-1].sign()).squeeze(1)
    x0, x1 = x[crossing], x[crossing + 1]
    y0, y1 = residual[crossing], residual[crossing + 1]
    roots = x0 - y0 * (x1 - x0) / (y1 - y0)

    lower, upper = _conditional_means(generator, roots)
    valid = ((lower + upper) / 2 - roots).abs() <= tol
    points = torch.stack([lower, roots, upper], dim=1)[valid]

    # The roots are sorted, so it is enough to compare each with the last kept
    kept: list[int] = []
    for i, root in enumerate(points[:, 1].tolist()):
        if not kept or root - last >= eps:
            kept.append(i)
            last = root
    return points[torch.tensor(kept, dtype=torch.long)]


def nash_fixed_points(
    generator: Generator,
    points: torch.Tensor,
    low: float,
    high: float,
    grid_size: int = 5000,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Keep the fixed points that are Nash equilibria of the continuous game.

    A maker can only gain by quoting a better price than the market, since a
    worse one wins no trade. So a fixed point is tested against undercutting the
    ask, with any ask in ``[low, a*)``, and against outbidding the bid, with any
    bid in ``(b*, high]``, one side at a time.

    Parameters
    ----------
    generator : Generator
        Distribution of the latent value ``V``.
    points : torch.Tensor
        Candidates of shape ``(n_points, 3)``, as returned by
        :func:`fixed_points`.
    low : float
        Lower end of an interval containing the support of ``V``.
    high : float
        Upper end of an interval containing the support of ``V``.
    grid_size : int, optional
        Number of deviations tried on each side.
    tol : float, optional
        A deviation counts only if its expected profit exceeds ``tol``.

    Returns
    -------
    nash : torch.Tensor
        The rows of ``points`` that no deviation beats.

    Notes
    -----
    The deviating maker is assumed to win the whole trade on its side, so the
    test ignores ``epsilon`` and the sharing of ties. It judges the continuum of
    quotes, unlike :func:`~lama_lab.analysis.games.finite_game.pure_nash` on the
    discrete grid of quotes that the agents actually play, and the two sets can
    be far apart when the tick is coarse.
    """
    is_nash = []
    for bid, _, ask in points.tolist():
        asks = torch.linspace(low, ask, grid_size, dtype=torch.float64)[:-1]
        mid = (asks + bid) / 2
        gain_ask = asks * generator.ccdf(mid) - generator.upper_partial_mean(mid)

        bids = torch.linspace(bid, high, grid_size, dtype=torch.float64)[1:]
        mid = (ask + bids) / 2
        gain_bid = generator.lower_partial_mean(mid) - bids * generator.cdf(mid)

        is_nash.append(max(gain_ask.amax().item(), gain_bid.amax().item()) <= tol)
    return points[torch.tensor(is_nash, dtype=torch.bool)]
