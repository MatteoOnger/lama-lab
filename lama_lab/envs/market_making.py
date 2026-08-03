import torch

from ..generators import BaseGenerator


class MarketMakingEnvironment:
    """Market-making environment for simulating maker-trader interactions.

    The environment batches multiple episodes and assigns rewards to makers based
    on how closely their submitted prices match the trader's chosen side and the
    latent true value sampled by the provided generator.

    Parameters
    ----------
    n_makers : int
        Number of makers participating in each episode.
    n_episodes : int
        Number of independent episodes to process in a batch.
    n_rounds : int
        Total number of rounds to simulate.
    generator_v : BaseGenerator
        Generator used to sample the latent true values.
    epsilon : float, optional
        Numerical tolerance used when comparing prices and selecting the best
        offer.
    device : torch.device, optional
        Device on which tensors should be allocated.
    """

    def __init__(
        self,
        n_makers: int,
        n_episodes: int,
        n_rounds: int,
        generator_v: BaseGenerator,
        epsilon: float = 1e-8,
        device: torch.device = None,
    ):
        self.n_makers = n_makers
        self.n_episodes = n_episodes
        self.n_rounds = n_rounds
        self.generator_v = generator_v
        self.epsilon = epsilon
        self.device = device if device is not None else torch.get_default_device()

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
            If the input actions do not match the expected shape.
        """
        if actions.shape != (self.n_episodes, self.n_makers, 2):
            raise ValueError("actions must have shape (n_episodes, n_makers, 2)")

        true_values = self.generator_v.generate(self.n_episodes)

        # Best bid and ask per episode
        best_bid = actions[:, :, 0].amax(dim=1)
        best_ask = actions[:, :, 1].amin(dim=1)

        # Price gaps relative to the true value
        bid_gap = true_values - best_bid
        ask_gap = best_ask - true_values

        # Trader's choice
        trader_prefers_ask = torch.randint(
            0, 2, (self.n_episodes,), dtype=torch.bool, device=self.device
        )
        trader_prefers_ask[bid_gap > ask_gap + self.epsilon] = True
        trader_prefers_ask[ask_gap > bid_gap + self.epsilon] = False

        # Prices offered on the chosen side
        chosen_side_prices = torch.where(
            trader_prefers_ask[:, None],
            actions[:, :, 1],
            actions[:, :, 0],
        )

        # Best price available to the trader
        chosen_price = torch.where(trader_prefers_ask, best_ask, best_bid)

        # Makers offering the chosen price
        selected_maker_indices = torch.where(
            torch.abs(chosen_side_prices - chosen_price[:, None]) < self.epsilon
        )

        # Makers chosen per episode
        n_selected_makers = torch.bincount(
            selected_maker_indices[0].reshape(-1), minlength=self.n_episodes
        )

        # Reward for each selected episode
        reward_per_episode = (
            torch.where(trader_prefers_ask, ask_gap, bid_gap) / n_selected_makers
        )

        # Assign rewards to the selected makers
        rewards = torch.zeros((self.n_episodes, self.n_makers), device=self.device)
        rewards[selected_maker_indices] = reward_per_episode[selected_maker_indices[0]]

        self.round += 1
        return rewards
