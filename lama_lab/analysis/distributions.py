import torch


def get_all_unique_fixed_points(
    initial_x_values: list[float],
    samples: torch.Tensor,
    max_iter: int = 1000,
    eps: float = 1e-4,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Find unique fixed points of a distribution.

    In this context, given the underlying distribution represented by the
    provided ``samples``, we define a "fixed point" $x^*$ as a value that
    satisfies the following equilibrium condition:

    $$x^* = \\frac{1}{2} ( \\mu_{lower}(x^*) + \\mu_{upper}(x^*) )$$

    where $\\mu_{lower}(x^*)$ is the conditional mean of the distribution for
    values strictly less than $x^*$, and $\\mu_{upper}(x^*)$ is the conditional
    mean for values greater than or equal to $x^*$, both estimated from the samples.

    Starting from multiple initial values, the function iteratively updates each
    value using the formula above. Convergence is reached when the updated value
    differs from the current one by less than ``tol``. Converged solutions are
    collected and filtered to keep only those separated by at least ``eps``.

    Parameters
    ----------
    initial_x_values : list of float
        Initial values from which to start the iteration.
    samples : torch.Tensor
        One-dimensional tensor containing the samples drawn from the target
        distribution, used to compute the lower and upper conditional means.
    max_iter : int, optional
        Maximum number of iterations for each initial value.
    eps : float, optional
        Tolerance used to identify and separate distinct fixed points.
    tol : float, optional
        Convergence tolerance for the fixed-point update.

    Returns
    -------
    out : torch.Tensor
        Tensor of shape ``(n_fixed_points, 3)`` containing, for each unique
        fixed point, the lower mean, the fixed point value, and the upper mean.
        Rows are sorted in ascending order by the fixed-point value.

    Raises
    ------
    ValueError
        If no unique fixed points are found.
    """
    unique_b = []
    unique_x = []
    unique_a = []

    for x_init in initial_x_values:
        x = torch.as_tensor(x_init, dtype=samples.dtype, device=samples.device)

        for _ in range(max_iter):
            lower = samples[samples < x]
            upper = samples[samples >= x]

            mbtx = lower.mean() if lower.numel() > 0 else x
            matx = upper.mean() if upper.numel() > 0 else x

            new_x = (mbtx + matx) / 2.0

            if torch.abs(new_x - x) < tol:
                if not any(torch.abs(new_x - u) < eps for u in unique_x):
                    unique_b.append(mbtx.cpu())
                    unique_x.append(new_x.cpu())
                    unique_a.append(matx.cpu())
                break

            x = new_x

    if not unique_x:
        raise ValueError("No unique fixed points found.")

    result = torch.stack(
        (
            torch.stack(unique_b),
            torch.stack(unique_x),
            torch.stack(unique_a),
        ),
        dim=1,
    )
    return result[result[:, 1].argsort()]
