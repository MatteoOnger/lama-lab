import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_1d_histogram(
    data: torch.Tensor,
    reference_values: torch.Tensor | None = None,
    reference_colors: str | list[str] | None = None,
    hist_range: tuple[float, float] | None = (0.0, 1.0),
    density: bool = True,
    bins: int = 1000,
    hist_color: str = "tab:blue",
    alpha: float = 0.75,
    title: str = "1D Histogram",
    xlabel: str = "Value",
    figsize: tuple[float, float] = (8, 5),
    axes: list[Axes] | None = None,
) -> Figure:
    """Plot a 1D histogram of data with optional reference intervals.

    Draws a 1D histogram of the provided data. When reference values are
    provided, the figure includes a second panel with horizontal interval
    markers aligned beneath the histogram's x-axis.

    Parameters
    ----------
    data : torch.Tensor
        Data to visualize with shape ``(N,)`` or ``(N, 1)``. The tensor must
        be located on the CPU.
    reference_values : torch.Tensor, optional
        Tensor of shape ``(K, 3)`` containing ``(lower, marker, upper)`` tuples
        or ``(K, 2)`` containing ``(lower, upper)`` tuples. The tensor must be
        located on the CPU.
    reference_colors : str or list of str, optional
        Colors for the reference intervals. A single color string applies to all
        intervals; a list of ``K`` color strings colors them individually.
    hist_range : tuple of float, optional
        Histogram range for the x-axis as ``(min, max)``.
    density : bool, optional
        Whether to normalize the histogram to form a probability density.
    bins : int, optional
        Number of histogram bins.
    hist_color : str, optional
        Color of the histogram bars.
    alpha : float, optional
        Transparency of the histogram bars.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    figsize : tuple of float, optional
        Figure size used when ``axes`` is not provided.
    axes : list of Axes, optional
        Axes to draw into. Use ``[ax]`` for a single histogram or
        ``[ax_hist, ax_ref]`` when reference values are shown.

    Returns
    -------
    fig : Figure
        Figure containing the plotted distribution and optional intervals.

    Raises
    ------
    ValueError
        If ``axes`` does not match the required panel count (1 without intervals,
        2 with intervals), if ``reference_values`` does not have shape ``(K, 2)``
        or ``(K, 3)``, or if ``reference_colors`` length does not match ``K``.
    """
    # Reduce the optional singleton feature dimension for histogram input
    data_flat = data.squeeze()

    # Axes layout resolution
    if axes is None:
        if reference_values is None:
            fig, ax_hist = plt.subplots(figsize=figsize, layout="constrained")
            ax_ref = None
        else:
            fig, (ax_hist, ax_ref) = plt.subplots(
                2,
                1,
                figsize=figsize,
                sharex=True,
                layout="constrained",
                gridspec_kw={
                    "height_ratios": [4, 1],
                    "hspace": 0.05,
                },
            )
    else:
        if reference_values is None:
            if len(axes) != 1:
                raise ValueError(
                    "'axes' must contain exactly one Axes object when 'reference_values' is None."
                )
            ax_hist = axes[0]
            ax_ref = None
        else:
            if len(axes) != 2:
                raise ValueError(
                    "'axes' must contain exactly two Axes objects when 'reference_values' is provided."
                )
            ax_hist, ax_ref = axes
        fig = ax_hist.figure

    # Plot main histogram
    y_label = "Density" if density else "Count"

    ax_hist.hist(
        data_flat,
        bins=bins,
        density=density,
        range=hist_range,
        color=hist_color,
        alpha=alpha,
    )
    ax_hist.set_title(title)
    ax_hist.set_ylabel(y_label)
    ax_hist.grid(True, linestyle="--", alpha=0.7)

    # Single panel early return
    if reference_values is None:
        ax_hist.set_xlabel(xlabel)
        return fig

    # Validate & process reference intervals (Panel 2)
    if reference_values.ndim != 2 or reference_values.shape[-1] not in (2, 3):
        raise ValueError(
            f"'reference_values' must have shape (K, 2) or (K, 3). Got {reference_values.shape}."
        )

    k_refs = len(reference_values)

    # Resolve colors
    if reference_colors is None:
        colors = [hist_color] * k_refs
    elif isinstance(reference_colors, str):
        colors = [reference_colors] * k_refs
    elif len(reference_colors) == k_refs:
        colors = reference_colors
    else:
        raise ValueError(
            f"Length of 'reference_colors' ({len(reference_colors)}) must match "
            f"the number of reference values ({k_refs})."
        )

    for i, ref in enumerate(reference_values):
        y_pos = k_refs - i
        c = colors[i]

        if len(ref) == 3:
            lower, marker, upper = ref
        else:
            lower, upper = ref
            marker = None

        # Horizontal interval line
        ax_ref.hlines(
            y_pos,
            lower,
            upper,
            color="black",
            linewidth=2,
            alpha=0.8,
        )
        # Vertical end pipes
        ax_ref.plot(
            [lower, upper],
            [y_pos, y_pos],
            "|",
            color=c,
            markersize=10,
            markeredgewidth=2,
            alpha=0.8,
        )
        # Center/Marker diamond (plotted only if marker is present)
        if marker is not None:
            ax_ref.scatter(
                marker,
                y_pos,
                marker="D",
                color=c,
                s=35,
                zorder=3,
            )

    # Reference panel styling
    ax_ref.set_ylim(0.5, k_refs + 0.5)
    ax_ref.set_yticks([])
    ax_ref.set_xlabel(xlabel)
    ax_ref.set_ylabel("Ref")
    ax_ref.grid(True, linestyle="--", alpha=0.7, axis="x")

    # Clean borders for reference plot
    for spine in ["left", "right", "top"]:
        ax_ref.spines[spine].set_visible(False)
    return fig


def plot_2d_histogram(
    data: torch.Tensor,
    reference_values: torch.Tensor | None = None,
    reference_colors: str | list[str] = "tab:red",
    subplot_titles: list[str] | None = None,
    feature_names: tuple[str, str] = ("Dimension 0", "Dimension 1"),
    hist_range: (
        tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] | None
    ) = None,
    density: bool = True,
    bins: int = 100,
    cmap: str = "viridis",
    show_origin: bool = False,
    title: str = "2D Histogram",
    nrows: int = 1,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    axes: list[Axes] | None = None,
) -> Figure:
    """Plot a 2D histogram of continuous features for single or multiple groups.

    Accepts either multi-group data, such as actions per agent, or aggregated
    single-group data, such as dispersion.

    Parameters
    ----------
    data : torch.Tensor
        Tensor of shape ``(N, n_groups, 2)`` or ``(N, 2)``. A tensor with shape
        ``(N, 2)`` is treated as a single group. The tensor must be on the CPU.
    reference_values : torch.Tensor, optional
        Tensor of shape ``(K, 2)`` containing reference points to plot as markers.
    reference_colors : str or list of str, optional
        Colors for the reference markers. A single color string applies to all
        points; a list of ``K`` colors assigns them individually.
    subplot_titles : list of str, optional
        Titles for each subplot. When omitted for multiple groups, titles use the
        ``"Agent_i"`` naming pattern. The value is ignored for a single group.
    feature_names : tuple of (str, str), optional
        Names of the two dimensions used as axis labels.
    hist_range : tuple of float or tuple of tuples, optional
        Range of the histogram axes. Can be a single tuple ``(min, max)`` applied to
        both dimensions or ``((xmin, xmax), (ymin, ymax))``.
    density : bool, optional
        Whether to plot the histogram as a probability density.
    bins : int, optional
        Number of bins used for the histogram.
    cmap : str, optional
        Colormap used for the 2D histogram.
    show_origin : bool, optional
        Whether to draw dashed lines at ``x = 0`` and ``y = 0``.
    title : str, optional
        Base title of the plot.
    nrows : int, optional
        Number of subplot rows used when ``axes`` is not provided.
    ncols : int, optional
        Number of subplot columns used when ``axes`` is not provided.
    figsize : tuple of float, optional
        Figure size. When omitted, uses ``(18, 8)`` for multiple groups and
        ``(8, 8)`` for a single group.
    axes : list of Axes, optional
        Axes to draw into. Provide one axis per group.

    Returns
    -------
    fig : Figure
        Figure containing the plotted 2D histograms.

    Raises
    ------
    ValueError
        If the shape of ``data`` is invalid, or if list lengths don't match.
    """
    # Input normalization
    if data.ndim == 2 and data.shape[-1] == 2:
        data = data.unsqueeze(1)
    elif data.ndim != 3 or data.shape[-1] != 2:
        raise ValueError(
            f"'data' must have shape (N, 2) or (N, n_groups, 2). Got {data.shape}."
        )

    _, n_groups, _ = data.shape

    # Setup defaults for metadata
    if n_groups > 1:
        if subplot_titles is None:
            subplot_titles = [f"Agent_{i}" for i in range(n_groups)]
        elif len(subplot_titles) != n_groups:
            raise ValueError("Length of 'subplot_titles' must match number of groups.")
    else:
        subplot_titles = [title] if subplot_titles is None else subplot_titles

    if figsize is None:
        figsize = (8, 8) if n_groups == 1 else (18, 8)

    # Histogram range processing
    if hist_range is not None and isinstance(hist_range[0], (int, float)):
        hist_range_2d = [hist_range, hist_range]
    else:
        hist_range_2d = hist_range

    cbar_label = "Density" if density else "Count"

    # Axes setup
    if axes is None:
        if ncols is None:
            ncols = -(-n_groups // nrows)
        fig, created_axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
        )
        axes = created_axes.flatten()
    else:
        if len(axes) < n_groups:
            raise ValueError("Not enough 'axes' provided for the number of groups.")
        fig = axes[0].figure

    # Reference colors processing
    if reference_values is not None:
        k_refs = reference_values.shape[0]
        if isinstance(reference_colors, str):
            c_refs = [reference_colors] * k_refs
        else:
            if len(reference_colors) != k_refs:
                raise ValueError(
                    "Length of 'reference_colors' must match number of reference points."
                )
            c_refs = reference_colors

    # Plotting loop
    for i in range(n_groups):
        ax = axes[i]
        points = data[:, i, :]

        feat_0 = points[:, 0]
        feat_1 = points[:, 1]

        hist = ax.hist2d(
            feat_0,
            feat_1,
            bins=bins,
            cmap=cmap,
            range=hist_range_2d,
            density=density,
        )

        fig.colorbar(hist[3], ax=ax, label=cbar_label)

        if show_origin:
            ax.axhline(0, color="white", linestyle="--", alpha=0.5, zorder=5)
            ax.axvline(0, color="white", linestyle="--", alpha=0.5, zorder=5)

        if reference_values is not None:
            ax.scatter(
                reference_values[:, 0],
                reference_values[:, 1],
                marker="x",
                c=c_refs,
                s=100,
                linewidths=2,
                zorder=10,
                label="Reference Values",
            )
            ax.legend(loc="upper right")

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

        if n_groups > 1:
            ax.set_title(f"{title} - {subplot_titles[i]}")
        else:
            ax.set_title(subplot_titles[i])

        ax.grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout()
    return fig
