import torch


class RingBuffer:
    """A simple circular buffer for storing tensors in order.

    Parameters
    ----------
    capacity : int
        Maximum number of elements stored by the buffer.
    shape : tuple of int, optional
        Shape of each stored element, by default ``()``.
    device : torch.device, optional
        Device on which the underlying tensor is allocated.

    Attributes
    ----------
    size : int
        Current number of elements stored in the buffer.

    Raises
    ------
    ValueError
        If ``capacity`` is less than or equal to 0.
    """

    def __init__(
        self,
        capacity: int,
        shape: tuple[int, ...] = (),
        device: torch.device | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0.")

        self.capacity = capacity
        self.shape = tuple(shape)
        self.device = device if device is not None else torch.get_default_device()

        self._buffer = torch.zeros((capacity, *self.shape), device=self.device)
        self._idx = 0
        self.size = 0
        return

    def append(self, x: torch.Tensor) -> None:
        """Append a new element to the buffer.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to store in the buffer.

        Raises
        ------
        ValueError
            If the shape of ``x`` does not match the buffer's element shape.
        """
        if x.shape != self.shape:
            raise ValueError(
                f"Tensor shape {x.shape} does not match buffer element shape {self.shape}."
            )

        x = x.to(device=self.device)
        self._buffer[self._idx] = x
        self._idx = (self._idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_all(self, device: torch.device | None = None) -> torch.Tensor:
        """Return all stored elements in chronological order.

        Parameters
        ----------
        device : torch.device, optional
            Device on which to return the buffered data. If ``None``, the data
            is returned on the device where the buffer is stored.

        Returns
        -------
        tensor : torch.Tensor
            Tensor containing the buffered elements.
        """
        if self.size < self.capacity:
            ordered = self._buffer[: self.size]
        else:
            ordered = torch.cat(
                (self._buffer[self._idx :], self._buffer[: self._idx]), dim=0
            )

        if device is None:
            return ordered
        return ordered.to(device=device)

    def get_last(self, n: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """Return the most recent ``n`` stored elements.

        Parameters
        ----------
        n : int, optional
            Number of recent elements to retrieve.
        device : torch.device, optional
            Device on which to return the buffered data. If ``None``, the data
            is returned on the device where the buffer is stored.

        Returns
        -------
        tensor : torch.Tensor
            Tensor containing the requested recent elements.

        Raises
        ------
        IndexError
            If the buffer is empty.
        ValueError
            If ``n`` is less than 1.
        """
        if self.size == 0:
            raise IndexError("Empty buffer.")
        if n < 1:
            raise ValueError("n must be > 0")

        ordered = self.get_all()
        ordered = ordered[-n:]

        if device is None:
            return ordered
        return ordered.to(device=device)

    def __len__(self) -> int:
        """Return the number of currently stored elements."""
        return self.size
