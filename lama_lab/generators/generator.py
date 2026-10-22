from abc import ABC, abstractmethod

import torch


class Generator(ABC):
    """Abstract base class for data generators.

    A generator samples from a distribution or a data source through
    :meth:`generate`. It may also expose closed-form functionals of its
    distribution through :meth:`cdf`, :meth:`ccdf`, :meth:`lower_partial_mean`
    and :meth:`upper_partial_mean`. These four are optional: a generator without
    a closed form does not override them, and calling them raises
    ``NotImplementedError``.

    Notes
    -----
    The four functionals are defined only for scalar-valued generators.
    """

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
        raise self._no_closed_form("cdf")

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
        raise self._no_closed_form("ccdf")

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
        raise self._no_closed_form("lower_partial_mean")

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
        raise self._no_closed_form("upper_partial_mean")

    def _no_closed_form(self, name: str) -> NotImplementedError:
        """Build the error raised by an optional functional that is not overridden."""
        return NotImplementedError(
            f"Generator '{type(self).__name__}' does not provide a closed-form "
            f"'{name}()'."
        )
