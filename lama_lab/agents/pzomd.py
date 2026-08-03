from typing import Callable, Sequence, Optional

import torch

from .base import BaseAgent


class AgentPZOMD(BaseAgent):
    """Perturbation-based zeroth-order optimization agent.

    The agent estimates the gradient by sampling a random perturbation direction,
    evaluating the perturbed action, and updating the current iterate using the
    observed reward. Learning rates and perturbation radii follow predefined
    decay schedules.

    Parameters
    ----------
    n_episodes : int
        Number of parallel games to simulate at once.
    init_x : torch.Tensor or sequence
        Initial point used to initialize the candidate actions.
    project_fn : Callable[[torch.Tensor], torch.Tensor], optional
        Projection function applied after perturbation and update steps.
    eta_0 : float, optional
        Initial learning rate.
    delta_0 : float, optional
        Initial perturbation scale.
    min_delta : float, optional
        Minimum allowed perturbation scale.
    min_eta : float, optional
        Minimum allowed learning rate.
    max_grad_norm : float, optional
        Maximum norm used to clip the estimated gradient.
    name : str, optional
        Human-readable name of the agent.

    Notes
    -----
    This implementation supports batched execution by maintaining one optimizer
    state per episode. All episodes share the same hyperparameters and evolve
    independently, allowing :meth:`act` and :meth:`update` to operate on
    tensors of shape ``(n_episodes, ...)`` using vectorized PyTorch operations.
    """

    def __init__(
        self,
        n_episodes: int,
        init_x: torch.Tensor | Sequence,
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        eta_0: float = 0.05,
        delta_0: float = 0.01,
        min_delta: float = 0.005,
        min_eta: float = 0.001,
        max_grad_norm: float = 5.0,
        name: str = "AgentPZOMD",
    ):
        super().__init__(name=name)
        self.n_parallel_games = n_episodes
        self.project_fn = project_fn if project_fn is not None else (lambda x: x)
        self.eta_0 = eta_0
        self.delta_0 = delta_0
        self.min_delta = min_delta
        self.min_eta = min_eta
        self.max_grad_norm = max_grad_norm

        init_x_tensor = torch.as_tensor(init_x)
        self.x = self.project_fn(init_x_tensor.repeat(self.n_parallel_games, 1))
        self.action_dim = self.x.shape[1]

        self.t = 1
        self.eta = eta_0
        self.delta = delta_0
        self.u = torch.zeros_like(self.x)
        return

    def act(self) -> torch.Tensor:
        self.delta = max(self.delta_0 * (self.t**-0.5), self.min_delta)
        self.eta = max(self.eta_0 * (self.t**-0.75), self.min_eta)

        z = torch.randn(self.n_parallel_games, self.action_dim)
        norm_z = torch.linalg.norm(z, dim=1, keepdim=True)

        norm_z = torch.clamp(norm_z, min=1e-8)
        self.u = z / norm_z

        # TODO (Future Refinement):
        # Currently, act() projects the perturbed action x_perturbed directly. Near domain
        # boundaries, this projection truncates the sampled direction u, introducing bias
        # into the zeroth-order gradient estimate g_hat.
        #
        # To guarantee strict theoretical unbiasedness (Flaxman et al., 2005 - FKM):
        # - Constrain the nominal state self.x to a delta-shrunk domain (A_delta) during update().
        # - Maintain a safety buffer >= delta from all boundaries so that (x + delta * u)
        #   is natively feasible in act(), eliminating the need to project perturbed actions.
        x_perturbed = self.project_fn(self.x + self.delta * self.u)
        return x_perturbed

    def update(self, rewards: torch.Tensor) -> None:
        g_hat = (self.action_dim / self.delta) * rewards[:, None] * self.u
        norm_g = torch.linalg.norm(g_hat, dim=1, keepdim=True)

        too_big_mask = (norm_g > self.max_grad_norm).squeeze(1)
        if too_big_mask.any():
            g_hat[too_big_mask] = (
                g_hat[too_big_mask] / norm_g[too_big_mask]
            ) * self.max_grad_norm

        self.x = self.project_fn(self.x + self.eta * g_hat)
        self.t += 1
        return
