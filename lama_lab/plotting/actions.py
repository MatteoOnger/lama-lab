from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


def plot_market_makers_actions_history(
    mean_action_history: torch.Tensor,
    min_action_history: Optional[torch.Tensor] = None,
    max_action_history: Optional[torch.Tensor] = None,
    std_action_history: Optional[torch.Tensor] = None,
    fixed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    maker_names: Optional[Sequence[str]] = None,
    start_step: int = 0,
    nrows: int = 1,
    ncols: Optional[int] = None,
    figsize: Tuple[float, float] = (18, 6),
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
    fixed_points : tuple of torch.Tensor, optional
        Fixed price levels to plot as ``(fixed_b, fixed_a)``.
    maker_names : sequence of str, optional
        Names of the market makers.
    start_step : int, optional
        First time step for the plotted history.
    nrows : int, optional
        Number of subplot rows.
    ncols : int, optional
        Number of subplot columns.
    figsize : tuple of float, optional
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plotted action history.
    """
    n_rounds, n_makers, _ = mean_action_history.shape
    time = range(start_step, start_step + n_rounds)

    if maker_names is None:
        maker_names = [f"Maker {i}" for i in range(n_makers)]

    if ncols is None:
        ncols = -(-n_makers // nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
    )

    axes = axes.flatten()

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
            fixed_b, fixed_a = fixed_points
            ax.hlines(
                fixed_b,
                xmin=start_step,
                xmax=start_step + n_rounds - 1,
                color="blue",
                alpha=0.5,
                linestyle="--",
                label="Fixed Bid",
            )
            ax.hlines(
                fixed_a,
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

    for j in range(n_makers, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=4 if fixed_points is None else 5,
    )
    return fig
