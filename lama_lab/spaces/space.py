from abc import ABC, abstractmethod
import torch


class Space(ABC):
    """Abstract root class for all mathematical spaces.

    Parameters
    ----------
    shape : tuple of int
        The shape of a single element belonging to this space.
    dtype : torch.dtype
        The data type of the elements in this space.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        self.shape = shape
        self.dtype = dtype
        return

    @property
    def ndim(self) -> int:
        """Number of dimensions of a single element in the space.

        Returns
        -------
        ndim : int
            The length of the ``shape`` tuple.
        """
        return len(self.shape)

    @abstractmethod
    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Check if a batch of elements mathematically belongs to the space.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of candidate values to validate. It must have shape
            ``(N, *self.shape)``, where ``N`` is the batch size.

        Returns
        -------
        mask : torch.Tensor
            Boolean mask tensor of shape ``(N,)`` indicating validity for
            each sample in the batch.
        """
        pass

    @abstractmethod
    def project(self, raw_x: torch.Tensor) -> torch.Tensor:
        """Geometrically project unconstrained elements into the valid space.

        Parameters
        ----------
        raw_x : torch.Tensor
            Unconstrained tensor, typically produced by an agent or a raw
            generation process. It must have shape ``(N, *self.shape)``.

        Returns
        -------
        valid_x : torch.Tensor
            Tensor projected into the valid domain, guaranteed to have
            shape ``(N, *self.shape)``.
        """
        pass


class ContinuousSpace(Space):
    """Abstract base class for continuous spaces defined by global bounds.

    Parameters
    ----------
    shape : tuple of int
        The shape of a single element belonging to this space.
    dtype : torch.dtype
        The data type of the elements in this space.
    low : float
        Lower bound for the space elements.
    high : float
        Upper bound for the space elements.
    """

    def __init__(
        self, shape: tuple[int, ...], dtype: torch.dtype, low: float, high: float
    ) -> None:
        super().__init__(shape, dtype)
        self.low = low
        self.high = high
        return


class DiscreteSpace(Space):
    """Abstract base class for discrete spaces with a finite set of elements.

    Parameters
    ----------
    shape : tuple of int
        The shape of a single element belonging to this space.
    dtype : torch.dtype
        The data type of the elements in this space.
    n : int
        The total number of valid discrete elements in the space.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, n: int) -> None:
        super().__init__(shape, dtype)
        self.n = n
        return

    @abstractmethod
    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map discrete indices to their tensor representation.

        Parameters
        ----------
        indices : torch.Tensor
            1D tensor of shape ``(N,)`` containing discrete indices. Each index
            must be in the range ``[0, self.n - 1]``.

        Returns
        -------
        values : torch.Tensor
            Tensor values corresponding to the indices, with shape ``(N, *self.shape)``.
        """
        pass

    @abstractmethod
    def to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Map tensor values back to their corresponding discrete indices.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of elements with shape ``(N, *self.shape)``.

        Returns
        -------
        indices : torch.Tensor
            1D tensor of shape ``(N,)`` containing the mapped discrete indices.
        """
        pass

    def __getitem__(self, indices: torch.Tensor) -> torch.Tensor:
        """Wrapper around :meth:`DiscreteSpace.from_indices`."""
        return self.from_indices(indices)
