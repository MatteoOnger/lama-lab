import torch

from ..generators import Generator


class MarketMakingEnvironment:
    """Market-making environment for batched independent episodes.

    At each round, every market maker submits a bid and an ask quote. A latent
    true value is sampled for each episode, and a trader executes on the side
    whose best quote has the smaller gap to that value. If the bid and ask gaps
    are within epsilon, the side is selected randomly. When multiple makers
    offer the selected best quote, the trade is split equally among them and
    each selected maker receives its share of the corresponding price gap.

    Parameters
    ----------
    n_makers : int
        Number of makers participating in each episode.
    n_episodes : int
        Number of independent episodes to process in a batch.
    n_rounds : int
        Total number of rounds to simulate.
    generator_v : Generator
        Generator used to sample the latent true values.
    epsilon : float, optional
        Numerical tolerance used when comparing prices and selecting the best
        offer.

    Attributes
    ----------
    round : int
        Current round index of the simulation.

    Notes
    -----
    The environment supports batched execution: multiple independent episodes
    are simulated simultaneously using tensors whose leading dimension is
    ``n_episodes``. Episodes do not interact with one another, enabling efficient
    parallel simulation with vectorized PyTorch operations.
    """

    def __init__(
        self,
        n_makers: int,
        n_episodes: int,
        n_rounds: int,
        generator_v: Generator,
        epsilon: float = 1e-8,
    ) -> None:
        self.n_makers = n_makers
        self.n_episodes = n_episodes
        self.n_rounds = n_rounds
        self.generator_v = generator_v
        self.epsilon = epsilon

        self.round = 0
        return

    def reset(self) -> None:
        """Reset the environment to its initial round state."""
        self.round = 0
        return

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """Advance the environment by one round using the provided actions.

        Parameters
        ----------
        actions : torch.Tensor
            Tensor of shape (n_episodes, n_makers, 2) containing the bid
            and ask quotes submitted by each maker for each episode.

        Returns
        -------
        rewards : torch.Tensor
            Tensor of shape (n_episodes, n_makers) containing the reward
            assigned to each maker for the current round.

        Raises
        ------
        ValueError
            If the input actions do not match the expected shape, or if
            the simulation has exceeded the maximum number of rounds.
        """
        if self.round >= self.n_rounds:
            raise ValueError(
                f"Simulation finished. Maximum number of rounds ({self.n_rounds}) reached. "
                "Call reset() to start a new simulation."
            )

        if actions.shape != (self.n_episodes, self.n_makers, 2):
            raise ValueError("actions must have shape (n_episodes, n_makers, 2).")

        true_values = self.generator_v.generate(self.n_episodes).to(
            device=actions.device
        )

        # Best bid and ask per episode
        best_bid = actions[:, :, 0].amax(dim=1)
        best_ask = actions[:, :, 1].amin(dim=1)

        # Measure each side's distance from the sampled true value
        bid_gap = true_values - best_bid
        ask_gap = best_ask - true_values

        # Prefer the smaller gap and randomize only numerically indistinguishable ties
        trader_prefers_ask = torch.randint(
            0, 2, (self.n_episodes,), dtype=torch.bool, device=actions.device
        )
        trader_prefers_ask[bid_gap > ask_gap + self.epsilon] = True
        trader_prefers_ask[ask_gap > bid_gap + self.epsilon] = False

        # Select each maker's quote on the side chosen by the trader
        chosen_side_prices = torch.where(
            trader_prefers_ask[:, None],
            actions[:, :, 1],
            actions[:, :, 0],
        )

        # Select the best quote on the chosen side
        chosen_price = torch.where(trader_prefers_ask, best_ask, best_bid)

        # Identify all makers whose quotes match the selected best quote
        selected_maker_indices = torch.where(
            torch.abs(chosen_side_prices - chosen_price[:, None]) < self.epsilon
        )

        # Count tied makers so the trade can be divided evenly
        n_selected_makers = torch.bincount(
            selected_maker_indices[0].reshape(-1), minlength=self.n_episodes
        )

        # Divide each episode's price gap among its selected makers
        reward_per_episode = (
            torch.where(trader_prefers_ask, ask_gap, bid_gap) / n_selected_makers
        )

        # Write each episode's shared reward to its selected makers
        rewards = torch.zeros(
            (self.n_episodes, self.n_makers),
            device=actions.device,
        )
        rewards[selected_maker_indices] = reward_per_episode[selected_maker_indices[0]]

        self.round += 1
        return rewards
