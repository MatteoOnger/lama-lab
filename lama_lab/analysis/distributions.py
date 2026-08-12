import torch


def get_all_unique_fixed_points(
    samples: torch.Tensor,
    grid_size: int = 5000,
    eps: float = 1e-3,
    tol: float = 1e-3,
) -> torch.Tensor:
    r"""Find all unique fixed points (both stable and unstable) of a distribution.

    Given an empirical distribution represented by 1D ``samples``, a fixed point
    $x^*$ satisfies the equilibrium condition:

        $$x^* = \frac{1}{2} \left( \mu_{lower}(x^*) + \mu_{upper}(x^*) \right)$$

    where $\mu_{lower}(x^*)$ is the conditional mean of samples strictly less than
    $x^*$, and $\mu_{upper}(x^*)$ is the conditional mean of samples greater than
    or equal to $x^*$.

    This function evaluates $g(x) = \frac{1}{2}(\mu_{lower}(x) + \mu_{upper}(x)) - x$
    over a grid, identifies zero-crossings, and validates candidates against exact
    conditional means to filter interpolation artifacts.

    Parameters
    ----------
    samples : torch.Tensor
        One-dimensional tensor containing samples drawn from the target distribution.
    grid_size : int, optional
        Number of evaluation points in the search grid.
    eps : float, optional
        Minimum separation distance required to consider two fixed points distinct.
    tol : float, optional
        Maximum allowable deviation between a candidate root and its evaluated
        fixed-point condition for validation.

    Returns
    -------
    out : torch.Tensor
        Tensor of shape ``(n_fixed_points, 3)`` containing, for each validated
        unique fixed point, the lower mean $\mu_{lower}$, the fixed point $x^*$,
        and the upper mean $\mu_{upper}$. Rows are sorted in ascending order by $x^*$.

    Raises
    ------
    ValueError
        If no valid unique fixed points are found within the sample range.
    """
    # Sort samples and define grid boundaries over the sample range
    samples_sorted = torch.sort(samples.flatten()).values
    n_samples = len(samples_sorted)

    x_min, x_max = samples_sorted[0].item(), samples_sorted[-1].item()
    x_grid = torch.linspace(x_min, x_max, steps=grid_size, device=samples.device)

    # Precompute prefix sums for O(1) conditional mean evaluation
    cum_sums = torch.cat(
        [
            torch.zeros(1, dtype=samples.dtype, device=samples.device),
            torch.cumsum(samples_sorted, dim=0),
        ]
    )

    # Find sample split indices for every grid point using binary search
    idx = torch.searchsorted(samples_sorted, x_grid)

    # Compute vectorized conditional lower means: E[S | S < x]
    lower_counts = idx.to(samples.dtype)
    safe_lower_counts = lower_counts.clamp(min=1)  # Prevent division by zero
    lower_sums = cum_sums[idx]
    mbtx = torch.where(lower_counts > 0, lower_sums / safe_lower_counts, x_grid)

    # Compute vectorized conditional upper means: E[S | S >= x]
    upper_counts = (n_samples - idx).to(samples.dtype)
    safe_upper_counts = upper_counts.clamp(min=1)  # Prevent division by zero
    upper_sums = cum_sums[-1] - cum_sums[idx]
    matx = torch.where(upper_counts > 0, upper_sums / safe_upper_counts, x_grid)

    # Evaluate residual function g(x) = f(x) - x, where roots satisfy g(x*) = 0
    f_x = (mbtx + matx) / 2.0
    g_x = f_x - x_grid

    # Detect zero-crossings
    signs = torch.sign(g_x)
    sign_diffs = signs[1:] - signs[:-1]
    crossing_indices = torch.where(sign_diffs != 0)[0]

    if len(crossing_indices) == 0:
        raise ValueError("No unique fixed points found.")

    unique_points = []
    for i in crossing_indices:
        # Interpolate linearly between grid points to estimate candidate root
        x0, x1 = x_grid[i], x_grid[i + 1]
        y0, y1 = g_x[i], g_x[i + 1]

        if y1 == y0:
            root = x0
        else:
            root = x0 - y0 * (x1 - x0) / (y1 - y0)

        # Ensure uniqueness by keeping candidates separated by at least eps
        if not unique_points or all(
            torch.abs(root - u[1]) >= eps for u in unique_points
        ):
            # Compute exact conditional means at candidate root scalar
            root_scalar = root.item()
            lower = samples[samples < root_scalar]
            upper = samples[samples >= root_scalar]

            root_mbtx = lower.mean() if lower.numel() > 0 else root
            root_matx = upper.mean() if upper.numel() > 0 else root

            # Validate fixed-point condition to discard phantom step-function roots
            expected_root = (root_mbtx + root_matx) / 2.0
            if torch.abs(expected_root - root) > tol:
                continue

            unique_points.append(torch.stack([root_mbtx, root, root_matx]))

    if not unique_points:
        raise ValueError(
            "No valid unique fixed points found after validation filtering."
        )

    # Stack validated roots and return sorted by fixed point value
    result = torch.stack(unique_points)
    return result[result[:, 1].argsort()]
