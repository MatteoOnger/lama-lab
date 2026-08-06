import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_rewards_scatter(
    mean_reward_history: torch.Tensor,
    min_reward_history: torch.Tensor | None = None,
    max_reward_history: torch.Tensor | None = None,
    std_reward_history: torch.Tensor | None = None,
    maker_names: list[str] | None = None,
    start_step: int = 0,
    nrows: int = 1,
    ncols: int | None = None,
    figsize: tuple[float, float] = (18, 6),
    axes: list[Axes] | None = None,
) -> Figure:
    """Plot reward history for multiple agents.

    Parameters
    ----------
    mean_reward_history : torch.Tensor
        Tensor of shape ``(n_rounds, n_makers)`` containing mean rewards.
        Expected to be located on the CPU.
    min_reward_history : torch.Tensor, optional
        Tensor of shape ``(n_rounds, n_makers)`` containing minimum rewards.
        Expected to be located on the CPU.
    max_reward_history : torch.Tensor, optional
        Tensor of shape ``(n_rounds, n_makers)`` containing maximum rewards.
        Expected to be located on the CPU.
    std_reward_history : torch.Tensor, optional
        Tensor of shape ``(n_rounds, n_makers)`` containing reward standard
        deviations. Expected to be located on the CPU.
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
        Figure containing the plotted reward history.

    Raises
    ------
    ValueError
        If the length of ``maker_names`` does not match ``n_makers``, or if
        the provided list of ``axes`` contains fewer elements than ``n_makers``.
    """
    n_rounds, n_makers = mean_reward_history.shape
    time = range(start_step, start_step + n_rounds)

    if maker_names is None:
        maker_names = [f"Maker_{i}" for i in range(n_makers)]
    elif len(maker_names) != n_makers:
        raise ValueError("Length of maker_names must match number of makers.")

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
        raise ValueError("Not enough axes provided for the number of makers.")

    handles = []
    labels = []

    for i in range(n_makers):
        ax = axes[i]
        reward_mean = mean_reward_history[:, i]

        ax.plot(
            time,
            reward_mean,
            color="green",
            label="Mean Reward",
        )

        if std_reward_history is not None:
            reward_std = std_reward_history[:, i]

            ax.fill_between(
                time,
                reward_mean - reward_std,
                reward_mean + reward_std,
                color="lightgreen",
                alpha=0.2,
                label="Reward Std Dev",
            )

        if min_reward_history is not None:
            reward_min = min_reward_history[:, i]

            ax.plot(
                time,
                reward_min,
                color="darkgreen",
                linestyle=":",
                label="Min Reward",
            )

        if max_reward_history is not None:
            reward_max = max_reward_history[:, i]

            ax.plot(
                time,
                reward_max,
                color="limegreen",
                linestyle=":",
                label="Max Reward",
            )

        ax.set_title(f"Rewards - {maker_names[i]}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Reward")
        ax.grid(True, linestyle="--", alpha=0.7)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=4,
        )
    return fig
