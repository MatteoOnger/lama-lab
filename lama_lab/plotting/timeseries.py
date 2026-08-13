import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_history(
    feature_dim: int,
    mean_history: torch.Tensor,
    min_history: torch.Tensor | None = None,
    max_history: torch.Tensor | None = None,
    std_history: torch.Tensor | None = None,
    reference_values: torch.Tensor | None = None,
    agent_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    feature_colors: list[str] | None = None,
    start_step: int = 0,
    nrows: int = 1,
    ncols: int | None = None,
    ylabel: str = "Value",
    title_prefix: str = "History",
    figsize: tuple[float, float] = (18, 6),
    axes: list[Axes] | None = None,
) -> Figure:
    """Plot time series history for multiple agents and N-dimensional features.

    Parameters
    ----------
    feature_dim : int
        The explicit number of feature dimensions to plot (e.g., 1 for scalars like
        Rewards, 2 for 2D coordinates).
    mean_history : torch.Tensor
        Tensor of shape ``(n_rounds, n_agents, feature_dim)`` containing mean values.
        If ``feature_dim == 1``, shape ``(n_rounds, n_agents)`` is also accepted.
        Expected to be located on the CPU.
    min_history : torch.Tensor, optional
        Tensor containing minimum values. Shape must match ``mean_history``.
        Expected to be located on the CPU.
    max_history : torch.Tensor, optional
        Tensor containing maximum values. Shape must match ``mean_history``.
        Expected to be located on the CPU.
    std_history : torch.Tensor, optional
        Tensor containing standard deviations. Shape must match ``mean_history``.
        Expected to be located on the CPU.
    reference_values : torch.Tensor, optional
        Tensor of shape ``(K, feature_dim)`` containing reference values (e.g. Nash Equilibria)
        to draw as fixed horizontal lines. If ``feature_dim == 1``, ``(K,)`` is accepted.
        Expected to be located on the CPU.
    agent_names : list of str, optional
        Names of the agents. If None, defaults to ``["Agent_0", "Agent_1", ...]``.
    feature_names : list of str, optional
        Names of the dimensions. Defaults to ``["Metric"]`` for 1D, or
        ``["Dim 0", "Dim 1", ...]`` for multi-dimensional data.
    feature_colors : list of str, optional
        Colors for each dimension. If None, uses matplotlib's default color cycle.
    start_step : int, optional
        First time step index for the x-axis.
    nrows : int, optional
        Number of subplot rows used when ``axes`` is not provided.
    ncols : int, optional
        Number of subplot columns used when ``axes`` is not provided.
    ylabel : str, optional
        Label for the y-axis.
    title_prefix : str, optional
        Prefix for the subplot titles.
    figsize : tuple of float, optional
        Figure size used when ``axes`` is not provided.
    axes : list of matplotlib.axes.Axes, optional
        Axes to draw into. Provide one axis per agent.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plotted history.

    Raises
    ------
    ValueError
        If tensor shapes do not match the explicitly provided ``feature_dim``.
    """
    # Input Normalization & Validation
    if mean_history.ndim == 2 and feature_dim == 1:
        mean_history = mean_history.unsqueeze(-1)

    if mean_history.ndim != 3 or mean_history.shape[-1] != feature_dim:
        raise ValueError(
            f"mean_history shape {mean_history.shape} does not match "
            f"expected feature_dim={feature_dim}."
        )

    n_rounds, n_agents, _ = mean_history.shape
    time = range(start_step, start_step + n_rounds)

    def normalize_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is not None and t.ndim == 2 and feature_dim == 1:
            return t.unsqueeze(-1)
        return t

    min_history = normalize_tensor(min_history)
    max_history = normalize_tensor(max_history)
    std_history = normalize_tensor(std_history)

    if reference_values is not None and reference_values.ndim == 1 and feature_dim == 1:
        reference_values = reference_values.unsqueeze(-1)

    # Setup Defaults for Metadata
    if agent_names is None:
        agent_names = [f"Agent_{i}" for i in range(n_agents)]
    elif len(agent_names) != n_agents:
        raise ValueError("Length of agent_names must match number of agents.")

    if feature_names is None:
        if feature_dim == 1:
            feature_names = ["Metric"]
        else:
            feature_names = [f"Dim {d}" for d in range(feature_dim)]
    elif len(feature_names) != feature_dim:
        raise ValueError("Length of feature_names must match feature_dim.")

    if feature_colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        feature_colors = [prop_cycle[i % len(prop_cycle)] for i in range(feature_dim)]
    elif len(feature_colors) != feature_dim:
        raise ValueError("Length of feature_colors must match feature_dim.")

    # Axes Setup
    if axes is None:
        if ncols is None:
            ncols = -(-n_agents // nrows)
        fig, created_axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
        )
        axes = created_axes.flatten()
    else:
        fig = axes[0].figure

    if len(axes) < n_agents:
        raise ValueError("Not enough axes provided for the number of agents.")

    # Plotting Loop
    handles = []
    labels = []

    for i in range(n_agents):
        ax = axes[i]

        for d in range(feature_dim):
            c_dim = feature_colors[d]
            f_name = feature_names[d]

            mean_vals = mean_history[:, i, d]
            ax.plot(time, mean_vals, color=c_dim, linewidth=1.5, label=f"Mean {f_name}")

            if std_history is not None:
                std_vals = std_history[:, i, d]
                ax.fill_between(
                    time,
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    color=c_dim,
                    alpha=0.15,
                    label=f"{f_name} Std Dev",
                )

            if min_history is not None:
                min_vals = min_history[:, i, d]
                ax.plot(
                    time,
                    min_vals,
                    color=c_dim,
                    linestyle=":",
                    alpha=0.7,
                    label=f"Min {f_name}",
                )

            if max_history is not None:
                max_vals = max_history[:, i, d]
                ax.plot(
                    time,
                    max_vals,
                    color=c_dim,
                    linestyle="--",
                    alpha=0.7,
                    label=f"Max {f_name}",
                )

            if reference_values is not None:
                ref_vals = reference_values[:, d]
                ax.hlines(
                    ref_vals,
                    xmin=start_step,
                    xmax=start_step + n_rounds - 1,
                    color=c_dim,
                    alpha=0.5,
                    linestyle="-.",
                    label=f"Reference {f_name}",
                )

        ax.set_title(f"{title_prefix} - {agent_names[i]}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.7)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    # Deduplicate Legend and Finalize
    unique_labels = dict(zip(labels, handles))
    ncol_labels = (
        (mean_history is not None)
        + (min_history is not None)
        + (max_history is not None)
        + (std_history is not None)
        + (reference_values is not None)
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    if unique_labels:
        ncol = ncol_labels
        fig.legend(
            unique_labels.values(),
            unique_labels.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=ncol,
        )
    return fig
