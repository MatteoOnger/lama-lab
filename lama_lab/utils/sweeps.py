import copy

import torch

from ..envs import MarketMakingEnvironment
from ..generators import BaseGenerator
from .buffers import RingBuffer
from .builder import build_from_config


def run_symmetric_duel(
    generator: BaseGenerator,
    n_episodes: int,
    n_rounds: int,
    agent_cfg: dict,
    window: int,
    epsilon: float = 1e-3,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Run two identical learners against each other and summarize the tail of play.

    Builds two independent instances of the agent described by `agent_cfg`
    and simulates `n_rounds` of :class:`~lama_lab.envs.MarketMakingEnvironment`,
    then reports bid/ask statistics over the last `window` rounds. Since both
    makers share the same learning algorithm and hyperparameters, their
    quotes are pooled together as samples of the same symmetric equilibrium.

    Parameters
    ----------
    generator : BaseGenerator
        Generator for the latent asset value.
    n_episodes : int
        Number of independent episodes to simulate in parallel.
    n_rounds : int
        Total number of rounds to simulate.
    agent_cfg : dict
        Hydra-style configuration of the learner, shared by both makers. Must
        not set ``n_episodes`` or ``name``, which are injected per maker.
    window : int
        Number of final rounds averaged over to summarize the learned quotes.
    epsilon : float, optional
        Numerical tolerance of the environment.
    seed : int, optional
        Seed set before building the agents, for reproducibility.

    Returns
    -------
    summary : dict of str to torch.Tensor
        Scalar tensors ``bid_mean``, ``bid_std``, ``ask_mean``, ``ask_std``,
        ``spread_mean`` and ``spread_std``, pooled over the last `window`
        rounds, both makers and all episodes.
    """
    if seed is not None:
        torch.manual_seed(seed)

    env = MarketMakingEnvironment(
        n_makers=2,
        n_episodes=n_episodes,
        n_rounds=n_rounds,
        generator_v=generator,
        epsilon=epsilon,
    )

    makers = []
    for i in range(2):
        cfg = copy.deepcopy(agent_cfg)
        cfg["n_episodes"] = n_episodes
        cfg["name"] = f"Maker {i}"
        makers.append(build_from_config(cfg))

    last_actions = RingBuffer(window, shape=(n_episodes, 2, 2), device="cpu")

    for round_idx in range(n_rounds):
        actions = torch.stack([maker.act() for maker in makers], dim=1)
        rewards = env.step(actions)

        for j, maker in enumerate(makers):
            maker.update(rewards[:, j])

        if round_idx >= n_rounds - window:
            last_actions.append(actions.cpu())

    # (window, n_episodes, n_makers, 2) -> (N, 2), pooling both makers together
    flat = last_actions.get_all().reshape(-1, 2)
    bid, ask = flat[:, 0], flat[:, 1]
    spread = ask - bid

    return {
        "bid_mean": bid.mean(),
        "bid_std": bid.std(),
        "ask_mean": ask.mean(),
        "ask_std": ask.std(),
        "spread_mean": spread.mean(),
        "spread_std": spread.std(),
    }
