from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_distribution(
    samples: torch.Tensor,
    fixed_points: Optional[torch.Tensor] = None,
    ax: Optional[Union[Axes, Tuple[Axes, Axes]]] = None,
    bins: int = 1000,
    hist_range: Tuple[float, float] = (0, 1),
    density: bool = True,
    figsize: Tuple[float, float] = (8, 5),
) -> Figure:
    """Plot the distribution of samples.

    Optionally display fixed-point intervals beneath the histogram.

    Parameters
    ----------
    samples : torch.Tensor
        Samples from the distribution to visualize.
    fixed_points : torch.Tensor, optional
        Array-like of shape ``(N, 3)`` containing ``(b, m, a)`` tuples for each
        interval.
    ax : matplotlib.axes.Axes or tuple of matplotlib.axes.Axes, optional
        Single axes when ``fixed_points`` is ``None``. Otherwise, pass a tuple
        ``(ax_hist, ax_fp)``.
    bins : int
        Number of histogram bins.
    hist_range : tuple of float
        Histogram range.
    density : bool
        Whether to normalize the histogram.
    figsize : tuple of float
        Figure size, used only when ``ax`` is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    """
    if fixed_points is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Histogram
        ax.hist(
            samples,
            bins=bins,
            density=density,
            range=hist_range,
            color="tab:blue",
            alpha=0.75,
        )
        ax.set_title("Distribution of V")
        ax.set_xlabel("v")
        ax.set_ylabel("Density")
        return fig

    if ax is None:
        fig, (ax_hist, ax_fp) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={
                "height_ratios": [4, 1],
                "hspace": 0.05,
            },
        )
    else:
        ax_hist, ax_fp = ax
        fig = ax_hist.figure

    # Histogram
    ax_hist.hist(
        samples,
        bins=bins,
        density=density,
        range=hist_range,
        color="tab:blue",
        alpha=0.75,
    )
    ax_hist.set_title("Distribution of V")
    ax_hist.set_ylabel("Density")

    # Fixed-point intervals
    n = len(fixed_points)
    for i, (b, m, a) in enumerate(fixed_points):
        y = n - i

        # Interval [b, a]
        ax_fp.hlines(
            y,
            b,
            a,
            color="black",
            linewidth=2,
        )

        # Endpoints
        ax_fp.plot(
            [b, a],
            [y, y],
            "|",
            color="black",
            markersize=10,
            markeredgewidth=2,
        )

        # Midpoint
        ax_fp.scatter(
            m,
            y,
            marker="D",
            color="tab:blue",
            s=35,
            zorder=3,
        )

    ax_fp.set_ylim(0.5, n + 0.5)
    ax_fp.set_yticks([])
    ax_fp.set_xlabel("v")
    ax_fp.set_ylabel("FP")

    # Clean up
    ax_fp.spines["left"].set_visible(False)
    ax_fp.spines["right"].set_visible(False)
    ax_fp.spines["top"].set_visible(False)
    return fig
