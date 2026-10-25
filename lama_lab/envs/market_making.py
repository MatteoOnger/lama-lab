import torch

from ..generators import Generator


class MarketMakingEnvironment:
    """Market-making environment for batched independent episodes.

    At each round, every market maker submits a bid and an ask quote. A latent
    true value is sampled for each episode, and a trader executes on the side
    whose best quote has the smaller gap to that value. If the bid and ask gaps
    are within ``epsilon``, the side is selected randomly. When multiple makers
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
        Generator used to sample the latent true values. Its closed-form
        functionals, if any, are used by :meth:`expected_payoff`.
    epsilon : float, optional
        Width of the trader's indifference band, and tolerance within which two
        quotes count as tied.

    Attributes
    ----------
    round : int
        Current round index of the simulation.

    Raises
    ------
    ValueError
        If ``epsilon`` is not strictly positive.
        If ``generator_v`` does not generate scalar values.

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
        if epsilon <= 0.0:
            raise ValueError(f"'epsilon' must be strictly positive. Got {epsilon}.")
        if generator_v.event_shape:
            raise ValueError(
                "'generator_v' must generate scalar values. "
                f"Got event shape {generator_v.event_shape}."
            )

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
            Tensor of shape ``(n_episodes, n_makers, 2)`` containing the bid
            and ask quotes submitted by each maker for each episode.

        Returns
        -------
        rewards : torch.Tensor
            Tensor of shape ``(n_episodes, n_makers)`` containing the reward
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
                "Call 'reset()' to start a new simulation."
            )

        if actions.shape != (self.n_episodes, self.n_makers, 2):
            raise ValueError(
                "'actions' must have shape (n_episodes, n_makers, 2). "
                f"Got {tuple(actions.shape)}."
            )

        true_values = self.generator_v.generate(self.n_episodes)

        # Best bid and ask per episode
        best_bid = actions[:, :, 0].amax(dim=1)
        best_ask = actions[:, :, 1].amin(dim=1)

        # Measure each side's distance from the sampled true value
        bid_gap = true_values - best_bid
        ask_gap = best_ask - true_values

        # Prefer the smaller gap and randomize only numerically indistinguishable ties
        trader_prefers_ask = torch.randint(0, 2, (self.n_episodes,), dtype=torch.bool)
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
        rewards = torch.zeros((self.n_episodes, self.n_makers))
        rewards[selected_maker_indices] = reward_per_episode[selected_maker_indices[0]]

        self.round += 1
        return rewards

    def expected_payoff(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute the expected reward of every maker for a batch of quote profiles.

        The expectation is over the latent value and the trader's coin flip, and
        follows the rules of :meth:`step`. It does not depend on ``n_episodes``
        and does not advance the round counter.

        Parameters
        ----------
        actions : torch.Tensor
            Tensor of shape ``(B, n_makers, 2)`` containing the bid and ask quotes
            of every maker in each of the ``B`` profiles.

        Returns
        -------
        payoff : torch.Tensor
            Tensor of shape ``(B, n_makers)``, in double precision.

        Raises
        ------
        ValueError
            If ``actions`` does not have shape ``(B, n_makers, 2)``.

        Notes
        -----
        The result is exact when the generator has closed-form functionals, and
        an estimate from a fixed sample otherwise (see
        :class:`~lama_lab.generators.Generator`).

        The trader buys at the best ask when the value exceeds the midpoint of
        the best quotes by more than ``epsilon / 2``, sells at the best bid when
        it falls below the midpoint by more than that, and flips a fair coin in
        between. The makers within ``epsilon`` of the best quote on the chosen
        side share the gap between that quote and the value equally.
        """
        if actions.ndim != 3 or actions.shape[1:] != (self.n_makers, 2):
            raise ValueError(
                f"'actions' must have shape (B, {self.n_makers}, 2). "
                f"Got {tuple(actions.shape)}."
            )

        actions = actions.double()
        bids, asks = actions[:, :, 0], actions[:, :, 1]
        best_bid = bids.amax(dim=1)
        best_ask = asks.amin(dim=1)

        # Indifference band of the trader around the midpoint of the best quotes
        mid = (best_bid + best_ask) / 2.0
        band_low = mid - self.epsilon / 2.0
        band_high = mid + self.epsilon / 2.0

        # Probability and partial mean of the value in each region of the decision
        gen = self.generator_v
        prob_bid = gen.cdf(band_low)
        mass_bid = gen.lower_partial_mean(band_low)
        prob_ask = gen.ccdf(band_high)
        mass_ask = gen.upper_partial_mean(band_high)
        prob_tie = gen.cdf(band_high) - prob_bid
        mass_tie = gen.lower_partial_mean(band_high) - mass_bid

        # Expected gap per unit share, with the coin splitting the band evenly
        profit_ask = best_ask * prob_ask - mass_ask
        profit_ask = profit_ask + 0.5 * (best_ask * prob_tie - mass_tie)
        profit_bid = mass_bid - best_bid * prob_bid
        profit_bid = profit_bid + 0.5 * (mass_tie - best_bid * prob_tie)

        # Makers within epsilon of the best quote on a side share its profit equally
        on_ask = ((asks - best_ask[:, None]).abs() < self.epsilon).double()
        on_bid = ((bids - best_bid[:, None]).abs() < self.epsilon).double()
        share_ask = on_ask / on_ask.sum(dim=1, keepdim=True)
        share_bid = on_bid / on_bid.sum(dim=1, keepdim=True)

        return share_ask * profit_ask[:, None] + share_bid * profit_bid[:, None]
