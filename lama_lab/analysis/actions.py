import torch


def compute_action_dispersion(
    actions: torch.Tensor,
    action_axis: int = -1,
    agent_axis: int = -2,
    reduce_action_dim: bool = True,
    p: float = 2.0,
) -> torch.Tensor:
    """Compute the dispersion of agent actions around their centroid.

    The dispersion is computed as the average absolute deviation of agent
    actions from their centroid raised to a power ``p``. Larger values of
    ``p`` increase the sensitivity to large deviations, while ``p=1``
    provides a more robust measure.

    Parameters
    ----------
    actions : torch.Tensor
        Tensor containing agent actions. Must have at least 2 dimensions.
    action_axis : int, optional
        Axis corresponding to the action dimensions (components).
    agent_axis : int, optional
        Axis corresponding to agents.
    reduce_action_dim : bool, optional
        If ``True``, average the dispersion over all action dimensions and
        return a single value for each remaining index combination.
        If ``False``, preserve the action dimension axis and return one
        dispersion value per action component.
    p : float, optional
        Exponent applied to the absolute deviations from the centroid.

    Returns
    -------
    out : torch.Tensor
        Action dispersion around the centroid.
        If ``reduce_action_dim=True``, the output has the same shape as the
        input tensor with the agent and action dimension axes removed.
        If ``reduce_action_dim=False``, the output preserves the action
        dimension in its original relative position.
    """
    centroid = actions.mean(dim=agent_axis, keepdim=True)
    deviation = torch.abs(actions - centroid).pow(p)

    if reduce_action_dim:
        return deviation.mean(dim=(agent_axis, action_axis))
    return deviation.mean(dim=agent_axis)
