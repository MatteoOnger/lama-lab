import math

import torch

from .generator import Generator

# Smallest admissible acceptance rate of the rejection sampler
_MIN_MASS = 1e-6


def _pdf(z: torch.Tensor) -> torch.Tensor:
    """Density of the standard normal distribution."""
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_prob(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Probability that a standard normal variable lies in ``(lo, hi)``."""
    return torch.special.ndtr(hi) - torch.special.ndtr(lo)


class GaussianMixtureGenerator(Generator):
    """Gaussian mixture truncated to the interval ``[low, high]``.

    The generated variable follows the mixture conditioned on lying in
    ``[low, high]``. All four closed-form functionals of :class:`Generator` are
    available.

    Parameters
    ----------
    weights : list of float or torch.Tensor
        Non-negative mixing weights of the components. They are
        normalized to sum to one.
    means : list of float or torch.Tensor
        Mean of each component.
    stds : list of float or torch.Tensor
        Standard deviation of each component.
    low : float
        Lower end of the truncation interval.
    high : float
        Upper end of the truncation interval.

    Raises
    ------
    ValueError
        If ``weights``, ``means`` and ``stds`` are not one-dimensional with the
        same length.
        If ``stds`` is not strictly positive.
        If the untruncated mixture puts almost no mass in ``[low, high]``, which
        includes the case ``high <= low``.
    """

    def __init__(
        self,
        weights: list[float] | torch.Tensor,
        means: list[float] | torch.Tensor,
        stds: list[float] | torch.Tensor,
        low: float,
        high: float,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float64)
        means = torch.as_tensor(means, dtype=torch.float64)
        stds = torch.as_tensor(stds, dtype=torch.float64)

        if (
            weights.ndim != 1
            or means.shape != weights.shape
            or stds.shape != weights.shape
        ):
            raise ValueError(
                "'weights', 'means' and 'stds' must be 1D with the same length. "
                f"Got shapes {tuple(weights.shape)}, {tuple(means.shape)} "
                f"and {tuple(stds.shape)}."
            )
        if (stds <= 0.0).any():
            raise ValueError("'stds' must be strictly positive.")

        self.weights = weights / weights.sum()
        self.means = means
        self.stds = stds
        self.low = low
        self.high = high

        # Bounds of the interval, standardized against every component
        self._a = (low - self.means) / self.stds
        self._b = (high - self.means) / self.stds

        # Probability that the untruncated mixture lies in the interval, which is
        # the acceptance rate of the sampler and the normalization of the density
        self._mass = float((self.weights * _normal_prob(self._a, self._b)).sum())
        if self._mass < _MIN_MASS:
            raise ValueError(
                f"The mixture has too little mass in [{low}, {high}] "
                f"({self._mass:.2e}) for rejection sampling. Check that 'high' is "
                "greater than 'low' and that 'means' and 'stds' fit the interval."
            )
        return

    def generate(self, n_samples: int) -> torch.Tensor:
        accepted: list[torch.Tensor] = []
        n_accepted = 0

        while n_accepted < n_samples:
            # Oversample by the expected rejection rate, so one pass is usually enough
            n_draw = math.ceil(1.1 * (n_samples - n_accepted) / self._mass) + 1

            idxs = torch.multinomial(self.weights, num_samples=n_draw, replacement=True)
            samples = torch.normal(self.means[idxs], self.stds[idxs])

            samples = samples[(samples >= self.low) & (samples <= self.high)]
            accepted.append(samples)
            n_accepted += samples.numel()
        return torch.cat(accepted)[:n_samples].to(torch.get_default_dtype())

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        z = self._standardize(x)
        terms = self.weights * _normal_prob(self._a, z)
        return (terms.sum(dim=-1) / self._mass).to(x.dtype)

    def ccdf(self, x: torch.Tensor) -> torch.Tensor:
        z = self._standardize(x)
        terms = self.weights * _normal_prob(-self._b, -z)
        return (terms.sum(dim=-1) / self._mass).to(x.dtype)

    def lower_partial_mean(self, x: torch.Tensor) -> torch.Tensor:
        z = self._standardize(x)
        terms = self.weights * (
            self.means * _normal_prob(self._a, z)
            - self.stds * (_pdf(z) - _pdf(self._a))
        )
        return (terms.sum(dim=-1) / self._mass).to(x.dtype)

    def upper_partial_mean(self, x: torch.Tensor) -> torch.Tensor:
        z = self._standardize(x)
        terms = self.weights * (
            self.means * _normal_prob(-self._b, -z)
            + self.stds * (_pdf(z) - _pdf(self._b))
        )
        return (terms.sum(dim=-1) / self._mass).to(x.dtype)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize ``x`` against every component."""
        x = x.double().clamp(self.low, self.high)
        return (x.unsqueeze(-1) - self.means) / self.stds
