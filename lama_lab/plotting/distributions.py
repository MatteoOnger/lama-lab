import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_distribution(
    samples: torch.Tensor,
    reference_intervals: torch.Tensor | None = None,
    hist_range: tuple[float, float] = (0.0, 1.0),
    density: bool = True,
    bins: int = 1000,
    title: str = "Distribution",
    xlabel: str = "Value",
    figsize: tuple[float, float] = (8, 5),
    axes: list[Axes] | None = None,
) -> Figure:
    """Plot the distribution of samples.

    Optionally display reference intervals beneath the histogram.

    Parameters
    ----------
    samples : torch.Tensor
        Samples from the distribution to visualize. Expected to be located
        on the CPU.
    reference_intervals : torch.Tensor, optional
        Tensor of shape ``(N, 3)`` containing ``(lower, marker, upper)``
        tuples for each interval to display below the distribution. Expected
        to be located on the CPU.
    hist_range : tuple of float, optional
        Histogram range for the x-axis.
    density : bool, optional
        Whether to normalize the histogram.
    bins : int, optional
        Number of histogram bins.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    figsize : tuple of float, optional
        Figure size used when ``axes`` is not provided.
    axes : list of matplotlib.axes.Axes, optional
        Axes to draw into. Pass ``[ax]`` for a single histogram plot, or
        ``[ax_hist, ax_ref]`` when reference intervals are shown. When
        provided, the plot is drawn into the supplied axes instead of creating a
        new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.

    Raises
    ------
    ValueError
        If ``axes`` is provided but does not contain the correct number of
        Axes objects (1 if ``reference_intervals`` is None, 2 otherwise).
    """
    if axes is None:
        if reference_intervals is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax_hist = ax
            ax_ref = None
        else:
            fig, (ax_hist, ax_ref) = plt.subplots(
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
        if reference_intervals is None:
            if len(axes) != 1:
                raise ValueError(
                    "axes must contain exactly one Axes object for a single-panel plot."
                )
            ax_hist = axes[0]
            ax_ref = None
        else:
            if len(axes) != 2:
                raise ValueError(
                    "axes must contain exactly two Axes objects for reference interval plots."
                )
            ax_hist, ax_ref = axes
        fig = ax_hist.figure

    # Single-panel plot (Histogram only)
    if reference_intervals is None:
        ax_hist.hist(
            samples,
            bins=bins,
            density=density,
            range=hist_range,
            color="tab:blue",
            alpha=0.75,
        )
        ax_hist.set_title(title)
        ax_hist.set_xlabel(xlabel)
        ax_hist.set_ylabel("Density")
        return fig

    # Two-panel plot (Histogram + Reference Intervals)
    ax_hist.hist(
        samples,
        bins=bins,
        density=density,
        range=hist_range,
        color="tab:blue",
        alpha=0.75,
    )
    ax_hist.set_title(title)
    ax_hist.set_ylabel("Density")

    n = len(reference_intervals)
    for i, (lower, marker, upper) in enumerate(reference_intervals):
        y = n - i

        # Draw the horizontal line for the interval
        ax_ref.hlines(
            y,
            lower,
            upper,
            color="black",
            linewidth=2,
        )
        # Draw the vertical pipes at the edges
        ax_ref.plot(
            [lower, upper],
            [y, y],
            "|",
            color="black",
            markersize=10,
            markeredgewidth=2,
        )
        # Draw the marker in the middle (or specific point)
        ax_ref.scatter(
            marker,
            y,
            marker="D",
            color="tab:blue",
            s=35,
            zorder=3,
        )

    ax_ref.set_ylim(0.5, n + 0.5)
    ax_ref.set_yticks([])
    ax_ref.set_xlabel(xlabel)
    ax_ref.set_ylabel("Ref")

    ax_ref.spines["left"].set_visible(False)
    ax_ref.spines["right"].set_visible(False)
    ax_ref.spines["top"].set_visible(False)
    return fig
