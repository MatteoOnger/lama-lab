from typing import Callable

import torch

from .base import BaseAgent


class AgentPZOMD(BaseAgent):
    """Perturbation-based zeroth-order optimization agent.

    The agent estimates the gradient by sampling a random perturbation direction,
    evaluating the perturbed action, and updating the current estimate based on
    the observed reward. Learning rates and perturbation radii follow predefined
    decay schedules.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to simulate in parallel.
    init_x : torch.Tensor or list of float
        Initial action used to initialize the optimization process in each episode.
    project_fn : Callable[[torch.Tensor], torch.Tensor], optional
        Projection function applied after perturbation and update steps.
    eta_0 : float, optional
        Initial learning rate.
    delta_0 : float, optional
        Initial perturbation radius.
    decay_eta : float, optional
        Exponent for the learning rate decay schedule (t^-decay_eta).
    decay_delta : float, optional
        Exponent for the perturbation radius decay schedule (t^-decay_delta).
    min_delta : float, optional
        Minimum allowed perturbation radius.
    min_eta : float, optional
        Minimum allowed learning rate.
    max_grad_norm : float, optional
        Maximum norm used to clip the estimated gradient.
    name : str, optional
        Human-readable identifier for the agent.

    Attributes
    ----------
    x : torch.Tensor
        Current action estimates, one for each parallel episode.
    eta : float
        Current learning rate.
    delta : float
        Current perturbation radius.

    Raises
    ------
    ValueError
        If `init_x` is not a 1D vector.

    Notes
    -----
    This implementation supports batched execution by maintaining an independent
    optimizer state for each episode. All episodes share the same hyperparameters
    but evolve independently, allowing :meth:`_act` and :meth:`_update` to operate
    on tensors of shape ``(n_episodes, ...)`` using vectorized PyTorch operations.
    """

    def __init__(
        self,
        n_episodes: int,
        init_x: torch.Tensor | list[float],
        project_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        eta_0: float = 0.05,
        delta_0: float = 0.01,
        decay_eta: float = 0.75,
        decay_delta: float = 0.25,
        min_delta: float = 0.005,
        min_eta: float = 0.001,
        max_grad_norm: float = 5.0,
        name: str = "AgentPZOMD",
    ):
        init_x = torch.as_tensor(init_x)
        action_dim = init_x.numel()

        if init_x.ndim != 1:
            raise ValueError("init_x must be a 1D action vector.")

        super().__init__(n_episodes=n_episodes, action_dim=action_dim, name=name)

        self.project_fn = project_fn if project_fn is not None else (lambda x: x)
        self.eta_0 = eta_0
        self.delta_0 = delta_0
        self.decay_eta = decay_eta
        self.decay_delta = decay_delta
        self.min_delta = min_delta
        self.min_eta = min_eta
        self.max_grad_norm = max_grad_norm

        self.x = self.project_fn(init_x.repeat(self.n_episodes, 1))

        self._t = 1
        self.eta = eta_0
        self.delta = delta_0
        self._u = None
        return

    def _act(self) -> torch.Tensor:
        self.delta = max(self.delta_0 * (self._t**-self.decay_delta), self.min_delta)
        self.eta = max(self.eta_0 * (self._t**-self.decay_eta), self.min_eta)

        z = torch.randn(
            self.n_episodes,
            self.action_dim,
            device=self.x.device,
            dtype=self.x.dtype,
        )
        self._u = z / torch.clamp(
            torch.linalg.norm(z, dim=1, keepdim=True),
            min=1e-8,
        )
        x_perturbed = self.project_fn(self.x + self.delta * self._u)
        return x_perturbed

    def _update(self, reward: torch.Tensor) -> None:
        g_hat = (self.action_dim / self.delta) * reward[:, None] * self._u
        grad_norm = torch.linalg.norm(g_hat, dim=1, keepdim=True)
        clip_scale = torch.clamp(
            self.max_grad_norm / (grad_norm + 1e-8),
            max=1.0,
        )
        g_hat = g_hat * clip_scale

        self.x = self.project_fn(self.x + self.eta * g_hat)
        self._t += 1
        return
