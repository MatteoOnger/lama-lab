import torch


def action_dispersion(
    actions: torch.Tensor, agent_axis: int = -2, p: float = 2.0
) -> torch.Tensor:
    """Compute the dispersion of the agents' actions around their centroid.

    For each component of the actions, the dispersion is the mean over the agents
    of ``|a - centroid| ** p``, where the centroid is the mean action of the
    agents. For ``p=2`` it is the variance of the actions across the agents.

    Parameters
    ----------
    actions : torch.Tensor
        Tensor of actions with one axis indexing the agents, for example of shape
        ``(n_samples, n_agents, n_components)``.
    agent_axis : int, optional
        Axis of ``actions`` that indexes the agents.
    p : float, optional
        Exponent applied to the absolute deviations. Larger values give more
        weight to large deviations, while ``p=1`` is more robust.

    Returns
    -------
    dispersion : torch.Tensor
        Tensor with the shape of ``actions`` without the agent axis. Averaging it
        over the components gives a single value for each sample.
    """
    centroid = actions.mean(dim=agent_axis, keepdim=True)
    return (actions - centroid).abs().pow(p).mean(dim=agent_axis)
