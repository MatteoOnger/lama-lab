import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class Generator(ABC):
    """Abstract base class for data generators.

    A generator samples from a distribution :meth:`generate`.
    It may also override the closed-form functionals of its distribution:
    :meth:`cdf`, :meth:`ccdf`, :meth:`lower_partial_mean` and :meth:`upper_partial_mean`.
    A generator that does not override them gets an empirical estimate from a fixed
    sample of ``n_fallback_samples`` draws of :meth:`generate`, and a warning
    is logged the first time.

    Parameters
    ----------
    event_shape : tuple of int, optional
        Shape of a single sample, so that :meth:`generate` returns tensors of
        shape ``(n_samples, *event_shape)``. The default ``()`` is a scalar.
    n_fallback_samples : int, optional
        Size of the sample behind the empirical estimate of the functionals.

    Notes
    -----
    The four functionals are defined only for scalar-valued generators, and
    raise ``NotImplementedError`` for any other.
    """

    def __init__(
        self,
        event_shape: tuple[int, ...] = (),
        n_fallback_samples: int = 1_000_000,
    ) -> None:
        self.event_shape = event_shape
        self.n_fallback_samples = n_fallback_samples
        self._empirical_law: tuple[torch.Tensor, torch.Tensor] | None = None
        return

    @abstractmethod
    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate ``n_samples`` samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        samples : torch.Tensor
            Tensor of shape ``(n_samples, *event_shape)``.
        """
        pass

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Cumulative distribution function, ``P(X <= x)``.

        Parameters
        ----------
        x : torch.Tensor
            Points at which the function is evaluated, with shape ``(n_points,)``.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as ``x``.
        """
        index, _, n = self._locate(x)
        return (index.double() / n).to(x.dtype)

    def ccdf(self, x: torch.Tensor) -> torch.Tensor:
        """Complementary cumulative distribution function, ``P(X > x)``.

        Parameters
        ----------
        x : torch.Tensor
            Points at which the function is evaluated, with shape ``(n_points,)``.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as ``x``.
        """
        index, _, n = self._locate(x)
        return ((n - index).double() / n).to(x.dtype)

    def lower_partial_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Lower partial mean, ``E[X * I{X <= x}]``.

        Parameters
        ----------
        x : torch.Tensor
            Points at which the function is evaluated, with shape ``(n_points,)``.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as ``x``.
        """
        index, prefix, n = self._locate(x)
        return (prefix[index] / n).to(x.dtype)

    def upper_partial_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Upper partial mean, ``E[X * I{X > x}]``.

        Parameters
        ----------
        x : torch.Tensor
            Points at which the function is evaluated, with shape ``(n_points,)``.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as ``x``.
        """
        index, prefix, n = self._locate(x)
        return ((prefix[-1] - prefix[index]) / n).to(x.dtype)

    def _locate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Locate ``x`` in the cached sample, with its prefix sums and size."""
        if self._empirical_law is None:
            if self.event_shape:
                raise NotImplementedError(
                    f"Generator '{type(self).__name__}' has no closed-form "
                    "functionals and its samples are not scalar."
                )
            samples = self.generate(self.n_fallback_samples)
            logger.warning(
                f"Generator '{type(self).__name__}' has no closed-form functionals. "
                f"Estimating them from {samples.numel()} samples."
            )
            ordered = torch.sort(samples.double()).values
            zero = torch.zeros(1, dtype=torch.float64)
            prefix = torch.cat([zero, torch.cumsum(ordered, dim=0)])
            self._empirical_law = (ordered, prefix)

        ordered, prefix = self._empirical_law
        index = torch.searchsorted(ordered, x.double().contiguous(), right=True)
        return index, prefix, ordered.numel()
