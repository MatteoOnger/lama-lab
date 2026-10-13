from typing import Any

import torch

from .agent import Agent
from ..spaces import ContinuousSpace


class PZOMD(Agent):
    """Perturbation-based Zeroth-Order Online Mirror Descent (PZOMD) agent.

    Each episode keeps its own action estimate ``x`` and improves it using only
    reward feedback, without access to gradients. At every round the agent plays
    a randomly perturbed action, observes the reward, and moves ``x`` by a
    projected gradient-ascent step (Euclidean mirror map) along a one-point
    gradient estimate [1]_.

    At round :math:`t`, a direction :math:`u_t` is drawn uniformly from the unit
    sphere :math:`\\mathbb{S}^{d-1}`, the action :math:`x_t + \\delta_t u_t` is
    played (projected onto the action space), and the received reward
    :math:`R_t` gives the gradient estimate

    .. math::

       \\hat{g}_t = \\frac{d}{\\delta_t} R_t u_t,

    where :math:`d` is the number of coordinates of an action
    (``action_space.numel``). The estimate is clipped to norm ``max_grad_norm``,
    and :math:`x_{t+1}` is the projection onto the action space of
    :math:`x_t + \\eta_t \\hat{g}_t`.

    Parameters
    ----------
    n_episodes : int
        Number of independent episodes simulated in parallel. Each episode has
        its own estimate and evolves independently.
    action_space : ContinuousSpace
        Continuous space in which actions are defined. It is also used to
        project perturbed and updated actions back onto the feasible set.
    init_x : torch.Tensor or float
        Initial action estimate, shared by all episodes. It must have shape
        ``action_space.shape``; a scalar is accepted only if actions have a
        single coordinate. It is cast to ``action_space.dtype`` and projected
        onto the action space.
    init_delta : float, optional
        Perturbation radius at the first round. At round ``t`` the radius is
        ``max(init_delta * t**-decay_delta, min_delta)``. It should not exceed
        the radius of the largest ball contained in the action space.
    init_eta : float, optional
        Learning rate at the first round. At round ``t`` the rate is
        ``max(init_eta * t**-decay_eta, min_eta)``.
    decay_delta : float, optional
        Decay exponent of the perturbation radius.
    decay_eta : float, optional
        Decay exponent of the learning rate.
    min_delta : float, optional
        Lower bound on the perturbation radius.
    min_eta : float, optional
        Lower bound on the learning rate.
    max_grad_norm : float, optional
        Maximum norm of the gradient estimate. Larger estimates are rescaled to
        this norm.
    name : str, optional
        Human-readable name of the agent.

    Attributes
    ----------
    x : torch.Tensor
        Current action estimates with shape ``(n_episodes, *action_space.shape)``,
        one row per episode. Each episode owns its row, since episodes evolve
        independently.
    eta : float
        Learning rate of the current round, updated at every :meth:`act` call.
    delta : float
        Perturbation radius of the current round, updated at every :meth:`act`
        call.
    last_action_info : dict of str to Any
        Metadata about the last selected action, containing only ``"action"``
        (inherited from :class:`Agent`).

    Raises
    ------
    ValueError
        If ``init_x`` does not match ``action_space.shape`` (a scalar matches
        only single-coordinate actions).
        If ``action_space`` is not an instance of :class:`~lama_lab.spaces.ContinuousSpace`.

    Notes
    -----
    Estimates are stored with their real shape and flattened to
    ``(n_episodes, numel)`` only for the vectorized per-episode math. They are
    reshaped back before being projected or returned.

    The gradient estimator is unbiased only if the perturbed point
    ``x + delta * u`` stays inside the action space. The agent plays the
    projected point instead, so the estimate is biased in rounds where ``x`` is
    within ``delta`` of the boundary. The bias is small when ``init_x`` is well
    inside the space, ``delta`` is small compared to the distance of ``x`` from
    the boundary, and the optimum is not on the boundary; it can persist if the
    iterates converge to the boundary.

    The gradient clipping controlled by ``max_grad_norm`` introduces a further,
    deliberate bias in exchange for lower variance. Finally, the guarantees of
    mirror descent assume a Euclidean projection, which depends on how
    ``action_space.project`` is implemented.

    References
    ----------
    .. [1] A. D. Flaxman, A. T. Kalai, and H. B. McMahan, "Online convex
       optimization in the bandit setting: gradient descent without a
       gradient," in Proceedings of the 16th Annual ACM-SIAM Symposium on
       Discrete Algorithms (SODA), 2005. arXiv:cs/0408007.
    """

    def __init__(
        self,
        n_episodes: int,
        action_space: ContinuousSpace,
        init_x: torch.Tensor | float,
        init_delta: float = 0.5,
        init_eta: float = 0.1,
        decay_delta: float = 0.25,
        decay_eta: float = 0.75,
        min_delta: float = 0.001,
        min_eta: float = 0.001,
        max_grad_norm: float = 5.0,
        name: str = "PZOMD",
    ) -> None:
        if not isinstance(action_space, ContinuousSpace):
            raise ValueError("'PZOMD' requires a continuous action space.")

        init_x = torch.as_tensor(init_x, dtype=action_space.dtype)

        # A scalar is a valid initial point only for single-coordinate actions
        if init_x.ndim == 0 and action_space.numel == 1:
            init_x = init_x.reshape(action_space.shape)

        if init_x.shape != action_space.shape:
            raise ValueError(
                "The shape of 'init_x' must match 'action_space.shape'; a scalar is "
                "accepted only if actions have a single coordinate. "
                f"Got {init_x.shape} and {action_space.shape}."
            )

        super().__init__(
            n_episodes=n_episodes,
            action_space=action_space,
            name=name,
        )

        self.init_delta = init_delta
        self.init_eta = init_eta
        self.decay_delta = decay_delta
        self.decay_eta = decay_eta
        self.min_delta = min_delta
        self.min_eta = min_eta
        self.max_grad_norm = max_grad_norm

        # All episodes start from the same point, so a zero-copy view suffices here
        init_x_batched = init_x.expand(self.n_episodes, *self.action_space.shape)

        # Store estimates with the real action shape, one row per episode. Clone so
        # that each episode owns its row, whatever 'project' returns, since episodes
        # evolve independently
        self.x = self.action_space.project(init_x_batched).clone()

        self.delta = init_delta
        self.eta = init_eta
        self._u: torch.Tensor | None = None
        self._t: int = 1
        return

    def _act(self) -> torch.Tensor:
        self.delta = float(
            max(self.init_delta * (self._t**-self.decay_delta), self.min_delta)
        )
        self.eta = float(max(self.init_eta * (self._t**-self.decay_eta), self.min_eta))

        # Flat view of the estimates for the vectorized per-episode math
        x_flat = self.x.reshape(self.n_episodes, -1)

        z = torch.randn(
            self.n_episodes,
            self.action_space.numel,
            device=x_flat.device,
            dtype=x_flat.dtype,
        )
        self._u = z / torch.clamp(
            torch.linalg.norm(z, dim=1, keepdim=True),
            min=1e-8,
        )

        x_perturbed = x_flat + self.delta * self._u
        return self.action_space.project(
            x_perturbed.reshape(-1, *self.action_space.shape)
        )

    def _update(self, reward: torch.Tensor) -> None:
        if self._u is None:
            raise RuntimeError("Cannot update agent state before calling 'act()'.")

        # Estimate the gradient from one perturbed reward observation
        g_hat = (self.action_space.numel / self.delta) * reward[:, None] * self._u

        # Limit the update magnitude when large rewards produce unstable estimates
        grad_norm = torch.linalg.norm(g_hat, dim=1, keepdim=True)
        clip_scale = torch.clamp(
            self.max_grad_norm / (grad_norm + 1e-8),
            max=1.0,
        )
        g_hat = g_hat * clip_scale

        # Apply gradient ascent on the flat view while keeping estimates inside the
        # action space
        x_flat = self.x.reshape(self.n_episodes, -1)
        self.x = self.action_space.project(
            (x_flat + self.eta * g_hat).reshape(-1, *self.action_space.shape)
        )

        self._u = None
        self._t += 1
        return

    def get_internal_state(self) -> dict[str, Any]:
        return {
            "t": int(self._t),
            "delta": float(self.delta),
            "eta": float(self.eta),
        }

    def get_policy(self) -> torch.Tensor | None:
        return None
