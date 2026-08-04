import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_market_makers_actions_histo2d(
    actions: torch.Tensor,
    maker_names: list[str] | None = None,
    hist_range: tuple[float, float] = (0.0, 1.0),
    density: bool = True,
    bins: int = 100,
    figsize: tuple[float, float] = (18, 8),
    axes: list[Axes] | None = None,
) -> Figure:
    """Plot a 2D histogram of bid/ask actions for multiple market makers.

    Parameters
    ----------
    actions : torch.Tensor
        Tensor of shape ``(N, n_makers, 2)`` containing bid/ask actions.
    maker_names : list of str, optional
        Names of the market makers.
    hist_range : tuple of float, optional
        Range of the histogram axes for both bid and ask prices.
    density : bool, optional
        Whether to plot the histogram as a probability density.
    bins : int, optional
        Number of bins used for the histogram.
    figsize : tuple of float, optional
        Figure size used when ``axes`` is not provided.
    axes : list of matplotlib.axes.Axes, optional
        Axes to draw into. Provide one axes per maker, e.g. ``[ax_0, ax_1]``.
        When provided, the plots will be drawn on these axes instead of
        creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plotted 2D histograms.
    """
    _, n_makers, _ = actions.shape

    if maker_names is None:
        maker_names = [f"Maker {i}" for i in range(n_makers)]
    elif len(maker_names) != n_makers:
        raise ValueError("length of maker_names must match number of makers")

    if axes is None:
        fig, created_axes = plt.subplots(
            1,
            n_makers,
            figsize=figsize,
            squeeze=False,
        )
        axes = created_axes.flatten()
    else:
        if len(axes) < n_makers:
            raise ValueError("not enough axes provided")
        fig = axes[0].figure

    for i in range(n_makers):
        ax = axes[i]
        points = actions[:, i, :]

        bid_prices = points[:, 0]
        ask_prices = points[:, 1]

        hist = ax.hist2d(
            bid_prices,
            ask_prices,
            bins=bins,
            cmap="viridis",
            range=[
                [hist_range[0], hist_range[1]],
                [hist_range[0], hist_range[1]],
            ],
            density=density,
        )

        fig.colorbar(hist[3], ax=ax, label="Density")

        ax.set_xlabel("Bid Price")
        ax.set_ylabel("Ask Price")
        ax.set_title(f"Bid/ask actions for {maker_names[i]}")
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_market_makers_actions_dispersion_histo2d(
    actions_dispersion: torch.Tensor,
    hist_range: tuple[float, float] = (-1.0, 1.0),
    density: bool = True,
    bins: int = 100,
    title: str = "Market makers actions dispersion",
    figsize: tuple[float, float] = (8, 8),
    ax: Axes | None = None,
) -> Figure:
    """Plot a 2D histogram of market makers actions dispersion.

    The input is expected to contain one dispersion value for each action
    dimension and sample. Typically, it is the output of
    ``compute_action_dispersion(..., reduce_features=False)``, although any
    tensor with the same shape can be provided.

    Each point represents the dispersion of the market makers' actions along
    the two action dimensions:
    - x-axis: bid action dispersion
    - y-axis: ask action dispersion

    Points concentrated around the origin indicate highly synchronized
    actions, while a wider spread indicates greater disagreement among
    market makers.

    Parameters
    ----------
    actions_dispersion : torch.Tensor
        Tensor of shape ``(N, 2)`` containing the bid and ask action
        dispersion for each sample.
    hist_range : tuple of float, optional
        Range of the histogram axes for both action dimensions.
    density : bool, optional
        Whether to plot the histogram as a probability density.
    bins : int, optional
        Number of bins used for the histogram.
    title : str, optional
        Title of the plot.
    figsize : tuple of float, optional
        Figure size used when ``ax`` is not provided.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. When provided, the plot is drawn on this axis
        instead of creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plotted 2D histogram.
    """
    if actions_dispersion.ndim != 2 or actions_dispersion.shape[-1] != 2:
        raise ValueError("actions_dispersion must have shape (N, 2)")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    points = actions_dispersion.detach().cpu()

    bid_dispersion = points[:, 0]
    ask_dispersion = points[:, 1]

    hist = ax.hist2d(
        bid_dispersion,
        ask_dispersion,
        bins=bins,
        cmap="viridis",
        range=[
            [hist_range[0], hist_range[1]],
            [hist_range[0], hist_range[1]],
        ],
        density=density,
    )

    fig.colorbar(hist[3], ax=ax, label="Density")

    ax.axhline(
        0,
        color="white",
        linestyle="--",
        alpha=0.5,
    )
    ax.axvline(
        0,
        color="white",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel("Bid action dispersion")
    ax.set_ylabel("Ask action dispersion")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout()
    return fig


def plot_market_makers_actions_scatter(
    mean_action_history: torch.Tensor,
    min_action_history: torch.Tensor | None = None,
    max_action_history: torch.Tensor | None = None,
    std_action_history: torch.Tensor | None = None,
    fixed_points: torch.Tensor | None = None,
    maker_names: list[str] | None = None,
    start_step: int = 0,
    nrows: int = 1,
    ncols: int | None = None,
    figsize: tuple[float, float] = (18, 6),
    axes: list[Axes] | None = None,
) -> Figure:
    """Plot action history (bid/ask prices) for multiple agents.

    Parameters
    ----------
    mean_action_history : torch.Tensor
        Tensor of shape ``(n_rounds, n_makers, 2)`` containing mean bid and ask
        prices for each maker.
    min_action_history : torch.Tensor, optional
        Tensor of shape ``(n_rounds, n_makers, 2)`` containing minimum bid and
        ask prices.
    max_action_history : torch.Tensor, optional
        Tensor of shape ``(n_rounds, n_makers, 2)`` containing maximum bid and
        ask prices.
    std_action_history : torch.Tensor, optional
        Tensor of shape ``(n_rounds, n_makers, 2)`` containing bid and ask
        standard deviations.
    fixed_points : torch.Tensor, optional
        Array-like of shape ``(N, 3)`` containing ``(b, m, a)`` tuples for each
        interval. The lower and upper bounds ``b`` and ``a`` are used as the
        fixed horizontal levels to draw.
    maker_names : list of str, optional
        Names of the market makers.
    start_step : int, optional
        First time step for the plotted history.
    nrows : int, optional
        Number of subplot rows used when ``axes`` is not provided.
    ncols : int, optional
        Number of subplot columns used when ``axes`` is not provided.
    figsize : tuple of float, optional
        Figure size used when ``axes`` is not provided.
    axes : list of matplotlib.axes.Axes, optional
        Axes to draw into. Provide one axes per maker, e.g. ``[ax_0, ax_1]``.
        When provided, the plots will be drawn on these axes instead of
        creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plotted action history.
    """
    n_rounds, n_makers, _ = mean_action_history.shape
    time = range(start_step, start_step + n_rounds)

    if maker_names is None:
        maker_names = [f"Maker {i}" for i in range(n_makers)]
    else:
        if len(maker_names) != n_makers:
            raise ValueError("length of maker_names must match number of makers")

    if axes is None:
        if ncols is None:
            ncols = -(-n_makers // nrows)
        fig, created_axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
        )
        axes = created_axes.flatten()
    else:
        fig = axes[0].figure

    if len(axes) < n_makers:
        raise ValueError("not enough axes provided for the number of makers")

    handles = []
    labels = []

    for i in range(n_makers):
        ax = axes[i]

        mean = mean_action_history[:, i]
        bid_mean = mean[:, 0]
        ask_mean = mean[:, 1]

        ax.plot(
            time,
            bid_mean,
            color="blue",
            label="Mean Bid Price",
        )
        ax.plot(
            time,
            ask_mean,
            color="orange",
            label="Mean Ask Price",
        )

        if std_action_history is not None:
            std = std_action_history[:, i]
            bid_std = std[:, 0]
            ask_std = std[:, 1]

            ax.fill_between(
                time,
                bid_mean - bid_std,
                bid_mean + bid_std,
                color="blue",
                alpha=0.1,
                label="Bid Std Dev",
            )
            ax.fill_between(
                time,
                ask_mean - ask_std,
                ask_mean + ask_std,
                color="orange",
                alpha=0.1,
                label="Ask Std Dev",
            )

        if min_action_history is not None:
            minimum = min_action_history[:, i]
            bid_min = minimum[:, 0]
            ask_min = minimum[:, 1]

            ax.plot(
                time,
                bid_min,
                color="lightblue",
                linestyle=":",
                label="Min Bid",
            )
            ax.plot(
                time,
                ask_min,
                color="peachpuff",
                linestyle=":",
                label="Min Ask",
            )

        if max_action_history is not None:
            maximum = max_action_history[:, i]
            bid_max = maximum[:, 0]
            ask_max = maximum[:, 1]

            ax.plot(
                time,
                bid_max,
                color="darkblue",
                linestyle=":",
                label="Max Bid",
            )
            ax.plot(
                time,
                ask_max,
                color="darkorange",
                linestyle=":",
                label="Max Ask",
            )

        if fixed_points is not None:
            fixed_bid_points = fixed_points[:, 0]
            fixed_ask_points = fixed_points[:, 2]

            ax.hlines(
                fixed_bid_points,
                xmin=start_step,
                xmax=start_step + n_rounds - 1,
                color="blue",
                alpha=0.5,
                linestyle="--",
                label="Fixed Bid",
            )
            ax.hlines(
                fixed_ask_points,
                xmin=start_step,
                xmax=start_step + n_rounds - 1,
                color="orange",
                alpha=0.5,
                linestyle="--",
                label="Fixed Ask",
            )

        ax.set_title(f"{maker_names[i]} Actions")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Price")

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=4 if fixed_points is None else 5,
    )
    return fig
