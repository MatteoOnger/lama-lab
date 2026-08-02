import torch

from .base import BaseGenerator


class GaussianMixtureGenerator(BaseGenerator):
    """Generate samples from a Gaussian mixture distribution.

    Parameters
    ----------
    weights : torch.Tensor
        Non-negative mixing weights for each component.
    means : torch.Tensor
        Mean vector for each Gaussian component.
    stds : torch.Tensor
        Standard deviation vector for each Gaussian component.
    clamp_min : float
        Lower bound used to clamp generated values.
    clamp_max : float
        Upper bound used to clamp generated values.
    device : torch.device, optional
        Device on which to store the parameters and perform computations.
        If not provided, defaults to the current default device.

    Notes
    -----
    The mixture parameters are moved to the specified device during
    initialization. All subsequent sampling operations are performed on the
    same device.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        clamp_min: float,
        clamp_max: float,
        device: torch.device = None,
    ):
        self.device = device if device is not None else torch.get_default_device()
        self.weights = weights.to(device=self.device)
        self.means = means.to(device=self.device)
        self.stds = stds.to(device=self.device)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        return

    def generate(self, n_samples: int) -> torch.Tensor:
        sampled_indices = torch.multinomial(
            self.weights, num_samples=n_samples, replacement=True
        )
        selected_means = self.means[sampled_indices]
        selected_stds = self.stds[sampled_indices]
        samples = torch.normal(selected_means, selected_stds)
        return torch.clamp(samples, min=self.clamp_min, max=self.clamp_max)
