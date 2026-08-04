import torch


def compute_action_dispersion(
    x: torch.Tensor,
    agent_axis: int = -2,
    feature_axis: int = -1,
    reduce_features: bool = True,
    p: float = 2,
) -> torch.Tensor:
    """Compute the dispersion of agent actions around their centroid.

    The dispersion is computed as the average absolute deviation of agent
    actions from their centroid raised to a power ``p``. Larger values of
    ``p`` increase the sensitivity to large deviations, while ``p=1``
    provides a more robust measure.

    Parameters
    ----------
    x : torch.Tensor
        Tensor containing agent actions.
    agent_axis : int, optional
        Axis corresponding to agents.
    feature_axis : int, optional
        Axis corresponding to action features.
    reduce_features : bool, optional
        If ``True``, average the dispersion over all action features and
        return a single value for each remaining index combination.
        If ``False``, preserve the action feature dimension and return one
        dispersion value per action feature.
    p : float, optional
        Exponent applied to the absolute deviations from the centroid.

    Returns
    -------
    out : torch.Tensor
        Action dispersion around the centroid.
        If ``reduce_features=True``, the output has the same shape as the
        input tensor with the agent and feature dimensions removed.
        If ``reduce_features=False``, the output preserves the action feature
        dimension.
    """
    x = torch.movedim(x, (agent_axis, feature_axis), (-2, -1))
    centroid = x.mean(dim=-2, keepdim=True)

    deviation = torch.abs(x - centroid).pow(p)
    dim = (-2, -1) if reduce_features else (-2,)

    return deviation.mean(dim=dim)
