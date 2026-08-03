import torch


class RingBuffer:
    """A simple circular buffer for storing tensors in order.

    Parameters
    ----------
    capacity : int
        Maximum number of elements stored by the buffer.
    shape : tuple of int, optional
        Shape of each stored element, by default ().
    device : torch.device, optional
        Device on which the underlying tensor is allocated.
    """

    def __init__(
        self, capacity: int, shape: tuple[int, ...] = (), device: torch.device | None = None
    ) -> None:
        self.capacity = capacity
        self.shape = tuple(shape)
        self.device = device if device is not None else torch.get_default_device()

        self.buffer = torch.zeros((capacity, *self.shape), device=self.device)
        self.size = 0
        self.idx = 0
        return

    def append(self, x: torch.Tensor) -> None:
        """Append a new element to the buffer.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to store in the buffer.
        """
        x = x.to(device=self.device)
        self.buffer[self.idx] = x
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return

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
            ordered = self.buffer[: self.size]
        else:
            ordered = torch.cat(
                (self.buffer[self.idx :], self.buffer[: self.idx]), dim=0
            )

        if device is None:
            return ordered
        return ordered.to(device=device)

    def get_last(self, n: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """Return the most recent ``n`` stored elements.

        Parameters
        ----------
        n : int, optional
            Number of recent elements to retrieve, by default 1.
        device : torch.device, optional
            Device on which to return the buffered data. If ``None``, the data
            is returned on the device where the buffer is stored.

        Returns
        -------
        tensor : torch.Tensor
            Tensor containing the requested recent elements.
        """
        if self.size == 0:
            raise IndexError("empty buffer")
        if n < 1:
            raise ValueError("n must be > 0")

        if n == 1:
            ordered = self.buffer[(self.idx - 1) % self.capacity]
        else:
            ordered = self.get_all()
            ordered = ordered if n >= self.size else ordered[-n:]

        if device is None:
            return ordered
        return ordered.to(device=device)

    def __len__(self) -> int:
        """Return the number of currently stored elements."""
        return self.size
