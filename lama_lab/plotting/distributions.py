from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_distribution(
    samples: torch.Tensor,
    fixed_points: Optional[torch.Tensor] = None,
    bins: int = 1000,
    density: bool = True,
    hist_range: Tuple[float, float] = (0, 1),
    axes: Optional[list[Axes]] = None,
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
    bins : int, optional
        Number of histogram bins.
    density : bool, optional
        Whether to normalize the histogram.
    hist_range : tuple of float, optional
        Histogram range.
    axes : list of matplotlib.axes.Axes, optional
        Axes to draw into. Pass ``[ax]`` for a single histogram plot, or
        ``[ax_hist, ax_fp]`` when fixed-point intervals are shown. When
        provided, the plot is drawn into the supplied axes instead of creating a
        new figure.
    figsize : tuple of float, optional
        Figure size used when ``axes`` is not provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    """
    if axes is None:
        if fixed_points is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax_hist = ax
            ax_fp = None
        else:
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
        if fixed_points is None:
            if len(axes) != 1:
                raise ValueError("axes must contain one Axes object for a single-panel plot")
            ax_hist = axes[0]
            ax_fp = None
        else:
            if len(axes) != 2:
                raise ValueError("axes must contain two Axes objects for fixed-point plots")
            ax_hist, ax_fp = axes
        fig = ax_hist.figure

    if fixed_points is None:
        ax_hist.hist(
            samples,
            bins=bins,
            density=density,
            range=hist_range,
            color="tab:blue",
            alpha=0.75,
        )
        ax_hist.set_title("Distribution of V")
        ax_hist.set_xlabel("v")
        ax_hist.set_ylabel("Density")
        return fig

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

    n = len(fixed_points)
    for i, (b, m, a) in enumerate(fixed_points):
        y = n - i

        ax_fp.hlines(
            y,
            b,
            a,
            color="black",
            linewidth=2,
        )
        ax_fp.plot(
            [b, a],
            [y, y],
            "|",
            color="black",
            markersize=10,
            markeredgewidth=2,
        )
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

    ax_fp.spines["left"].set_visible(False)
    ax_fp.spines["right"].set_visible(False)
    ax_fp.spines["top"].set_visible(False)
    return fig
