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

    Draws a 1D histogram of the provided data. If reference values are
    provided, a two-panel figure is created (or used) displaying horizontal
    interval markers directly aligned underneath the x-axis of the histogram.

    Parameters
    ----------
    data : torch.Tensor
        Data from the distribution to visualize. Expected shape ``(N,)`` or
        ``(N, 1)``. Must be located on the CPU.
    reference_values : torch.Tensor, optional
        Tensor of shape ``(K, 3)`` containing ``(lower, marker, upper)`` tuples,
        or ``(K, 2)`` containing ``(lower, upper)`` tuples for each interval to
        display below the distribution. Must be located on the CPU.
    reference_colors : str or list of str, optional
        Color(s) for the reference intervals. Can be a single color string applied
        to all intervals, or a list of color strings of length ``K`` to color each
        interval individually. Defaults to ``"tab:blue"``.
    hist_range : tuple of float, optional
        Histogram range for the x-axis ``(min, max)``. Default is ``(0.0, 1.0)``.
    density : bool, optional
        Whether to normalize the histogram to form a probability density. Default is True.
    bins : int, optional
        Number of histogram bins. Default is 1000.
    hist_color : str, optional
        Color of the histogram bars. Default is ``"tab:blue"``.
    alpha : float, optional
        Transparency of the histogram bars. Default is 0.75.
    title : str, optional
        Title of the plot. Default is ``"1D Histogram"``.
    xlabel : str, optional
        Label for the x-axis. Default is ``"Value"``.
    figsize : tuple of float, optional
        Figure size used when ``axes`` is not provided. Default is ``(8, 5)``.
    axes : list of matplotlib.axes.Axes, optional
        Axes to draw into. Pass ``[ax]`` for a single histogram plot, or
        ``[ax_hist, ax_ref]`` when reference values are shown.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plotted distribution and optional intervals.

    Raises
    ------
    ValueError
        If ``axes`` does not match the required panel count (1 without intervals,
        2 with intervals), if ``reference_values`` does not have shape ``(K, 2)``
        or ``(K, 3)``, or if ``reference_colors`` length does not match ``K``.
    """
    # Flattens (N, 1) to (N,) if necessary
    data_flat = data.squeeze()

    # Axes Layout Resolution
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
                    "axes must contain exactly one Axes object when reference_values is None."
                )
            ax_hist = axes[0]
            ax_ref = None
        else:
            if len(axes) != 2:
                raise ValueError(
                    "axes must contain exactly two Axes objects when reference_values is provided."
                )
            ax_hist, ax_ref = axes
        fig = ax_hist.figure

    # Plot Main Histogram
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

    # Single Panel Early Return
    if reference_values is None:
        ax_hist.set_xlabel(xlabel)
        return fig

    # Validate & Process Reference Intervals (Panel 2)
    if reference_values.ndim != 2 or reference_values.shape[-1] not in (2, 3):
        raise ValueError(
            f"reference_values must have shape (K, 2) or (K, 3), got {reference_values.shape}."
        )

    k_refs = len(reference_values)

    # Resolve Colors
    if reference_colors is None:
        colors = [hist_color] * k_refs
    elif isinstance(reference_colors, str):
        colors = [reference_colors] * k_refs
    elif len(reference_colors) == k_refs:
        colors = reference_colors
    else:
        raise ValueError(
            f"Length of reference_colors ({len(reference_colors)}) must match "
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

    This function handles both multi-agent data (e.g., actions per agent) and
    aggregated single-group data (e.g., dispersion).

    Parameters
    ----------
    data : torch.Tensor
        Tensor of shape ``(N, n_groups, 2)`` or ``(N, 2)``. If ``(N, 2)`` is provided,
        it is treated as a single group (``n_groups = 1``). Expected to be on the CPU.
    reference_values : torch.Tensor, optional
        Tensor of shape ``(K, 2)`` containing reference points to plot as markers.
    reference_colors : str or list of str, optional
        Color(s) for the reference markers. Can be a single color string applied to all,
        or a list of colors of length ``K`` to color each point differently.
    subplot_titles : list of str, optional
        Titles for each subplot. If None and ``n_groups > 1``, defaults to
        ``["Agent_0", "Agent_1", ...]``. If ``n_groups == 1``, this is ignored.
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
        If True, draws dashed lines at x=0 and y=0 (useful for dispersion plots).
    title : str, optional
        Base title of the plot.
    nrows : int, optional
        Number of subplot rows used when ``axes`` is not provided.
    ncols : int, optional
        Number of subplot columns used when ``axes`` is not provided.
    figsize : tuple of float, optional
        Figure size. If None, defaults to ``(18, 8)`` for multi-group data,
        and ``(8, 8)`` for single-group data.
    axes : list of matplotlib.axes.Axes, optional
        Axes to draw into. Provide one axis per group.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plotted 2D histograms.

    Raises
    ------
    ValueError
        If the shape of ``data`` is invalid, or if list lengths don't match.
    """
    # Input Normalization
    if data.ndim == 2 and data.shape[-1] == 2:
        data = data.unsqueeze(1)
    elif data.ndim != 3 or data.shape[-1] != 2:
        raise ValueError(
            f"data must have shape (N, 2) or (N, n_groups, 2). Got {data.shape}."
        )

    _, n_groups, _ = data.shape

    # Setup Defaults for Metadata
    if n_groups > 1:
        if subplot_titles is None:
            subplot_titles = [f"Agent_{i}" for i in range(n_groups)]
        elif len(subplot_titles) != n_groups:
            raise ValueError("Length of subplot_titles must match number of groups.")
    else:
        subplot_titles = [title] if subplot_titles is None else subplot_titles

    if figsize is None:
        figsize = (8, 8) if n_groups == 1 else (18, 8)

    # Histogram Range Processing
    if hist_range is not None and isinstance(hist_range[0], (int, float)):
        hist_range_2d = [hist_range, hist_range]
    else:
        hist_range_2d = hist_range

    cbar_label = "Density" if density else "Count"

    # Axes Setup
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
            raise ValueError("Not enough axes provided for the number of groups.")
        fig = axes[0].figure

    # Reference Colors Processing
    if reference_values is not None:
        k_refs = reference_values.shape[0]
        if isinstance(reference_colors, str):
            c_refs = [reference_colors] * k_refs
        else:
            if len(reference_colors) != k_refs:
                raise ValueError(
                    "Length of reference_colors must match number of reference points."
                )
            c_refs = reference_colors

    # Plotting Loop
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
