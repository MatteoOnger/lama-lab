import copy
import logging
from typing import Any

import torch

from .agent import Agent
from ..spaces import DiscreteSpace
from ..utils import build_from_config

logger = logging.getLogger(__name__)


class BlumMansour(Agent):
    """Blum-Mansour meta-agent reducing external regret to internal (swap) regret.

    The agent wraps a single batched expert that simulates one virtual expert per
    action in each episode. At every round, the policies of the virtual experts of
    an episode form a Markov matrix, and the agent samples its action from the
    stationary distribution of that matrix [1]_. Each virtual expert is then updated
    with the observed reward, importance-weighted and scaled by its own
    probability under that distribution.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to simulate in parallel.
    action_space : DiscreteSpace
        Finite action space of the environment. The batched expert is built on an
        equal copy of it.
    expert_cfg : dict of str to Any
        Configuration of the underlying expert, in the format accepted by
        :func:`~lama_lab.utils.build_from_config`. Its ``n_episodes``,
        ``action_space`` and ``name`` entries are overridden.
        The expert must expose an explicit policy through ``get_policy``
        and use the ``"action_index"`` and ``"action_probability"`` entries of its
        ``last_action_info`` in its update, as :class:`~lama_lab.agents.Exp3` does.
    name : str, optional
        Human-readable name of the agent. The expert is named
        ``"<name>_BatchedExpert"``.

    Attributes
    ----------
    expert : Agent
        Batched expert simulating one virtual expert per action in each episode,
        i.e. ``n_episodes * len(action_space)`` virtual episodes.
    stationary_dist : torch.Tensor
        Stationary distribution used to sample the last action, with shape
        ``(n_episodes, n_arms)``. It is uniform before the first call to
        :meth:`act`.
    warned_fallback : bool
        Whether the warning about the uniform fallback of the stationary
        distribution has been issued.
    last_action_info : dict of str to Any
        Metadata about the last selected action (inherited from :class:`Agent`).
        Besides ``"action"``, :meth:`act` stores ``"action_index"``, the indices
        of the sampled actions with shape ``(n_episodes,)``, and
        ``"action_probability"``, their probabilities under the stationary
        distribution with shape ``(n_episodes,)``.

    Raises
    ------
    ValueError
        If ``action_space`` is not an instance of
        :class:`~lama_lab.spaces.DiscreteSpace`.

    Notes
    -----
    The stationary distribution of each episode is computed with a batched linear
    solve. Episodes where it cannot be computed fall back to the uniform
    distribution, and a warning is logged once. Memory grows as
    ``n_episodes * n_arms**2`` and the cost of the solve as
    ``n_episodes * n_arms**3``.

    References
    ----------
    .. [1] A. Blum and Y. Mansour, "From external to internal regret," Journal of
       Machine Learning Research, vol. 8, pp. 1307-1324, 2007.
       https://jmlr.org/papers/v8/blum07a.html
    """

    def __init__(
        self,
        n_episodes: int,
        action_space: DiscreteSpace,
        expert_cfg: dict[str, Any],
        name: str = "BM",
    ) -> None:
        if not isinstance(action_space, DiscreteSpace):
            raise ValueError("'BlumMansour' requires a discrete action space.")

        super().__init__(n_episodes=n_episodes, action_space=action_space, name=name)

        # Build one expert that simulates all virtual experts of all episodes as a batch
        cfg = copy.deepcopy(expert_cfg)
        cfg.pop("_partial_", None)
        cfg["n_episodes"] = self.n_episodes * len(action_space)
        cfg["action_space"] = action_space
        cfg["name"] = f"{name}_BatchedExpert"

        self.expert: Agent = build_from_config(cfg)

        self.stationary_dist: torch.Tensor = torch.ones(
            (self.n_episodes, len(action_space))
        ) / len(action_space)
        self.warned_fallback = False
        return

    def _act(self) -> torch.Tensor:
        # The underlying expert returns one policy tensor for all virtual experts
        expert_probs = self.expert.get_policy()
        if expert_probs is None:
            raise ValueError(
                f"Expert '{self.expert.name}' returned None for 'get_policy()'. "
                "Blum-Mansour requires base experts with explicit probability "
                "distributions."
            )

        # Reshape the batched policy into one Markov matrix per episode
        markov_matrix = expert_probs.reshape(
            self.n_episodes, len(self.action_space), len(self.action_space)
        )

        # Solve the per-episode stationary distribution before sampling a global action
        self.stationary_dist = self._solve_stationary_distribution(markov_matrix)

        # Sample one action per episode from the stationary distribution
        action_index = torch.multinomial(self.stationary_dist, num_samples=1).squeeze(
            -1
        )

        # Keep the probability of the realized action for the importance-weight update
        action_probability = self.stationary_dist.gather(
            dim=1, index=action_index.unsqueeze(-1)
        ).squeeze(-1)

        self.last_action_info["action_index"] = action_index
        self.last_action_info["action_probability"] = action_probability
        return self.action_space[action_index]

    def _update(self, reward: torch.Tensor) -> None:
        if "action_index" not in self.last_action_info:
            raise RuntimeError("Cannot update before calling '_act()'.")

        action_index: torch.Tensor = self.last_action_info["action_index"]
        action_probability: torch.Tensor = self.last_action_info["action_probability"]

        # Clone the realized action across all virtual experts in the same episode
        expanded_action_index = (
            action_index.unsqueeze(1)
            .expand(self.n_episodes, len(self.action_space))
            .reshape(-1)
        )

        # Virtual expert i sees the probability pi(action) / pi(i) for the played
        # action, which scales its importance-weighted update by its own pi(i)
        virtual_action_probability = (
            action_probability.unsqueeze(1)
            / torch.clamp(self.stationary_dist, min=1e-12)
        ).reshape(-1)

        # Share the same environment reward across all virtual copies within an episode
        expanded_reward = (
            reward.unsqueeze(1)
            .expand(self.n_episodes, len(self.action_space))
            .reshape(-1)
        )

        # Pass the importance-weighted action metadata to the batched expert
        self.expert.last_action_info["action_index"] = expanded_action_index
        self.expert.last_action_info["action_probability"] = virtual_action_probability

        # Update the batched expert in one vectorized pass
        self.expert.update(expanded_reward)
        return

    def _solve_stationary_distribution(self, P: torch.Tensor) -> torch.Tensor:
        """Solve the stationary distribution of each Markov matrix in a batch.

        ``P`` has shape ``(n_episodes, n_arms, n_arms)`` and the result has shape
        ``(n_episodes, n_arms)``. Episodes whose system cannot be solved fall back
        to the uniform distribution.
        """
        batch_size, n, _ = P.shape

        I = torch.eye(n, dtype=P.dtype).expand(batch_size, n, n)
        A = (I - P).transpose(1, 2)
        A[:, -1, :] = 1.0

        b = torch.zeros(batch_size, n, 1, dtype=P.dtype)
        b[:, -1, 0] = 1.0

        # The extended solve reports singular matrices per episode instead of raising
        pi, info = torch.linalg.solve_ex(A, b)
        pi = torch.clamp(pi.squeeze(-1), min=0.0)
        total = pi.sum(dim=-1)

        # Near-singular systems do not set 'info', so non-finite or null solutions
        # are also treated as failures
        valid = (info == 0) & torch.isfinite(pi).all(dim=-1) & (total > 0)

        if not self.warned_fallback and not valid.all():
            logger.warning(
                f"{self.name} could not compute the stationary distribution for "
                f"{int((~valid).sum())} of {batch_size} episodes. Using the uniform "
                "distribution for them. This warning is shown only once."
            )
            self.warned_fallback = True

        # Only the failed episodes fall back to the uniform distribution
        uniform = torch.full_like(pi, 1.0 / n)
        return torch.where(valid.unsqueeze(-1), pi / total.unsqueeze(-1), uniform)

    def get_internal_state(self) -> dict[str, Any]:
        return self.expert.get_internal_state()

    def get_policy(self) -> torch.Tensor | None:
        return self.stationary_dist
