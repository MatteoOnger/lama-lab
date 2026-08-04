import torch


def get_all_unique_fixed_points(
    initial_x_values: list[float],
    samples: torch.Tensor,
    max_iter: int = 1000,
    tol: float = 1e-6,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Find unique fixed points of the lower/upper mean iteration.

    Starting from multiple initial values, iteratively updates each value by
    replacing it with the midpoint between the mean of samples below it and
    the mean of samples above or equal to it.
    A value is considered a fixed point when this midpoint is equal to the current
    value within the convergence tolerance.

    Converged solutions are collected and filtered to keep only unique fixed points.

    Parameters
    ----------
    initial_x_values : list of float
        Initial values from which to start the iteration.
    samples : torch.Tensor
        One-dimensional tensor containing the samples used to compute the
        lower and upper means.
    max_iter : int, optional
        Maximum number of iterations for each initial value.
    tol : float, optional
        Convergence tolerance for the fixed-point update.
    epsilon : float, optional
        Tolerance used to identify distinct fixed points.

    Returns
    -------
    out : torch.Tensor
        Tensor of shape ``(n_fixed_points, 3)`` containing, for each unique
        fixed point, the lower mean, the fixed point value, and the upper mean.
        Rows are sorted by the fixed-point value.
    """
    unique_b = []
    unique_x = []
    unique_a = []

    for x in initial_x_values:
        for _ in range(max_iter):
            lower = samples[samples < x]
            upper = samples[samples >= x]

            mbtx = lower.mean()  # if lower.numel() else x
            matx = upper.mean()  # if upper.numel() else x

            new_x = (mbtx + matx) / 2

            if torch.abs(new_x - x) < tol:
                if not any(torch.abs(new_x - u) < epsilon for u in unique_x):
                    unique_b.append(mbtx.cpu())
                    unique_x.append(new_x.cpu())
                    unique_a.append(matx.cpu())
                break

            x = new_x

    if not unique_x:
        raise ValueError("no unique fixed points found")

    result = torch.stack(
        (
            torch.stack(unique_b),
            torch.stack(unique_x),
            torch.stack(unique_a),
        ),
        dim=1,
    )
    return result[result[:, 1].argsort()]
