from typing import Optional, Sequence, Tuple
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_rewards_history(
    mean_reward_history: torch.Tensor,
    min_reward_history: Optional[torch.Tensor] = None,
    max_reward_history: Optional[torch.Tensor] = None,
    std_reward_history: Optional[torch.Tensor] = None,
    maker_names: Optional[Sequence[str]] = None,
    start_step: int = 0,
    nrows: int = 1,
    ncols: Optional[int] = None,
    figsize: Tuple[float, float] = (18, 6),
) -> Figure:
    """Plot reward history for multiple agents.

    Parameters
    ----------
    mean_reward_history : torch.Tensor
        Tensor of shape ``(n_rounds, n_makers)`` containing mean rewards.
    min_reward_history : torch.Tensor, optional
        Tensor of shape ``(n_rounds, n_makers)`` containing minimum rewards.
    max_reward_history : torch.Tensor, optional
        Tensor of shape ``(n_rounds, n_makers)`` containing maximum rewards.
    std_reward_history : torch.Tensor, optional
        Tensor of shape ``(n_rounds, n_makers)`` containing reward standard
        deviations.
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
        Figure containing the plotted reward history.
    """

    n_rounds, n_makers = mean_reward_history.shape
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

        ax.set_title(f"{maker_names[i]} Rewards")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Reward")
        ax.grid(True, linestyle="--", alpha=0.7)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    # Hide unused axes
    for j in range(n_makers, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=4,
    )
    return fig
