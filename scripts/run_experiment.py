import matplotlib.pyplot as plt
import torch

import lama_lab.analysis as analysis
import lama_lab.plotting as plotting
from lama_lab.agents import AgentPZOMD
from lama_lab.envs import MarketMakingEnvironment
from lama_lab.generators import GaussianMixtureGenerator
from lama_lab.projectors import MarketMakingProjector
from lama_lab.utils.buffers import RingBuffer

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Experiment configuration
# ============================================================================

n_makers = 2
n_episodes = 2
n_rounds = 10000

n_samples = 10_000

# ============================================================================
# Market value distribution
# ============================================================================

weights = torch.tensor([0.0, 1.0, 0.0])
means = torch.tensor([0.15, 0.5, 0.85])
stds = torch.tensor([0.03, 0.10, 0.03])

generator = GaussianMixtureGenerator(
    weights=weights,
    means=means,
    stds=stds,
    clamp_min=0.0,
    clamp_max=1.0,
)

samples = generator.generate(n_samples)

fixed_points = analysis.distributions.get_all_unique_fixed_points(
    initial_x_values=[i / 10 for i in range(1, 10)],
    samples=samples,
)

plotting.distributions.plot_distribution(samples, fixed_points)

# ============================================================================
# Environment
# ============================================================================

env = MarketMakingEnvironment(
    n_makers=n_makers,
    n_episodes=n_episodes,
    n_rounds=n_rounds,
    generator_v=generator,
    epsilon=0.001,
)

# ============================================================================
# Agents
# ============================================================================

projector = MarketMakingProjector(
    low=0.0,
    high=1.0,
    epsilon=0.001,
)

makers = [
    AgentPZOMD(
        n_episodes=n_episodes,
        init_x=[0.25, 0.75],
        project_fn=projector,
        eta_0=0.05,
        delta_0=1.0,
        min_eta=0.001,
        min_delta=0.001,
        name=f"Maker {i}",
    )
    for i in range(n_makers)
]

# ============================================================================
# Statistics buffers
# ============================================================================

mean_action_history = RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu")
min_action_history = RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu")
max_action_history = RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu")
std_action_history = RingBuffer(n_rounds, shape=(n_makers, 2), device="cpu")

last_action_history = RingBuffer(
    10,
    shape=(n_episodes, n_makers, 2),
    device="cpu",
)

mean_reward_history = RingBuffer(n_rounds, shape=(n_makers,), device="cpu")
min_reward_history = RingBuffer(n_rounds, shape=(n_makers,), device="cpu")
max_reward_history = RingBuffer(n_rounds, shape=(n_makers,), device="cpu")
std_reward_history = RingBuffer(n_rounds, shape=(n_makers,), device="cpu")

# ============================================================================
# Simulation
# ============================================================================

for _ in range(n_rounds):
    last_actions = torch.stack([maker.act() for maker in makers], dim=1)
    rewards = env.step(last_actions)

    for j, maker in enumerate(makers):
        maker.update(rewards[:, j])

    mean_action_history.append(last_actions.mean(dim=0))
    min_action_history.append(last_actions.amin(dim=0))
    max_action_history.append(last_actions.amax(dim=0))
    std_action_history.append(last_actions.std(dim=0))

    last_action_history.append(last_actions)

    mean_reward_history.append(rewards.mean(dim=0))
    min_reward_history.append(rewards.amin(dim=0))
    max_reward_history.append(rewards.amax(dim=0))
    std_reward_history.append(rewards.std(dim=0))

# ============================================================================
# Plots
# ============================================================================

last_actions = last_action_history.get_all().reshape(-1, n_makers, 2)
dispersion = analysis.actions.compute_action_dispersion(
    last_actions,
    reduce_features=False,
)

plotting.actions.plot_market_makers_actions_scatter(
    mean_action_history.get_all(),
    min_action_history.get_all(),
    max_action_history.get_all(),
    std_action_history.get_all(),
    fixed_points=fixed_points,
)
plotting.rewards.plot_rewards_scatter(
    mean_reward_history.get_all(),
    min_reward_history.get_all(),
    max_reward_history.get_all(),
    std_reward_history.get_all(),
)
plotting.actions.plot_market_makers_actions_histo2d(
    last_actions, title="Bid/Ask Actions at the End of Training"
)
plotting.actions.plot_market_makers_actions_dispersion_histo2d(
    dispersion, title="Market Makers Actions Dispersion at the End of Training"
)

plt.show()
