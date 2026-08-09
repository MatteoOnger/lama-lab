import torch

from .base import BaseGenerator


class GaussianMixtureGenerator(BaseGenerator):
    """Generate samples from a Gaussian mixture distribution.

    Parameters
    ----------
    weights : list of float
        Non-negative mixing weights for each component.
    means : list of float
        Mean vector for each Gaussian component.
    stds : list of float
        Standard deviation vector for each Gaussian component.
    low : float
        Lower bound used to clamp generated values.
    high : float
        Upper bound used to clamp generated values.
    """

    def __init__(
        self,
        weights: list[float],
        means: list[float],
        stds: list[float],
        low: float,
        high: float,
    ):
        self.weights = torch.as_tensor(weights)
        self.means = torch.as_tensor(means)
        self.stds = torch.as_tensor(stds)
        self.low = low
        self.high = high
        return

    def generate(self, n_samples: int) -> torch.Tensor:
        sampled_indices = torch.multinomial(
            self.weights, num_samples=n_samples, replacement=True
        )
        selected_means = self.means[sampled_indices]
        selected_stds = self.stds[sampled_indices]
        samples = torch.normal(selected_means, selected_stds)
        return torch.clamp(samples, min=self.low, max=self.high)
