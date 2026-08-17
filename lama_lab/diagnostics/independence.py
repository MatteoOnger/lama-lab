import logging
import math

import torch

from ..analysis import get_exploitability

logger = logging.getLogger(__name__)


def build_log_checkpoints(
    n_rounds: int,
    start: int = 100,
    per_decade: int = 2,
) -> list[int]:
    """Build logarithmically spaced round indices at which to evaluate diagnostics.

    Parameters
    ----------
    n_rounds : int
        Total number of rounds of the simulation. Always included in the result.
    start : int, optional
        First checkpoint. Ignored if larger than `n_rounds`.
    per_decade : int, optional
        Number of checkpoints per decade, so that the default of two gives the
        sequence 100, 300, 1000, 3000 and so on.

    Returns
    -------
    checkpoints : list of int
        Strictly increasing round indices, one-based.

    Raises
    ------
    ValueError
        If `n_rounds`, `start` or `per_decade` are not strictly positive.

    Notes
    -----
    Checkpoints are rounded to one significant digit, which turns the geometric
    sequence into round numbers that read well in a table.
    """
    if n_rounds <= 0:
        raise ValueError(f"n_rounds must be strictly positive. Got {n_rounds}.")
    if start <= 0:
        raise ValueError(f"start must be strictly positive. Got {start}.")
    if per_decade <= 0:
        raise ValueError(f"per_decade must be strictly positive. Got {per_decade}.")

    checkpoints = set()

    step = 0
    while True:
        value = start * 10 ** (step / per_decade)
        magnitude = 10 ** int(math.floor(math.log10(value)))
        value = int(round(value / magnitude)) * magnitude

        if value > n_rounds:
            break
        checkpoints.add(value)
        step += 1

    checkpoints.add(n_rounds)
    return sorted(checkpoints)


class Exp3Diagnostics:
    r"""Online diagnostics for the empirical play of two independent learners.

    Vanishing external regret places the empirical distribution of play in the
    set of coarse correlated equilibria, which is weaker than convergence to a
    Nash equilibrium: a time average of product distributions need not itself be
    a product distribution. Since a coarse correlated equilibrium that *is* a
    product distribution is exactly a mixed Nash equilibrium, this class tracks
    both halves of the statement, and the exploitability that follows from them.

    All quantities are accumulated online and kept separately for each episode,
    so that the batch of independent episodes simulated by the environment acts
    as a batch of independent seeds.

    Parameters
    ----------
    payoff : torch.Tensor
        Payoff matrix of the first maker, of shape ``(n_arms, n_arms)``, as
        returned by :func:`~lama_lab.analysis.get_expected_payoff_matrix`. The
        payoff matrix of the second maker is its transpose.
    n_episodes : int
        Number of episodes tracked. May be smaller than the number of episodes
        simulated, since the joint distribution costs ``n_arms ** 2`` values per
        episode.
    device : torch.device, optional
        Device on which the accumulators are allocated. Defaults to the device
        of `payoff`.

    Attributes
    ----------
    METRIC_NAMES : tuple of str
        Names of the values returned by :meth:`snapshot`, in a fixed order.
    n_arms : int
        Number of arms of the shared action space.
    payoff_range : float
        Difference between the largest and the smallest expected payoff, used to
        bound the exploitability in terms of the independence defect.
    t : int
        Number of rounds recorded so far.

    Raises
    ------
    ValueError
        If `payoff` is not a square matrix, or if `n_episodes` is not strictly
        positive.

    Notes
    -----
    Independence must not be tested round by round: the learners sample
    independently at every round by construction, so the round-by-round joint
    distribution is a product by definition. The question is whether the time
    average

    .. math:: \bar\mu_T = \frac1T \sum_t p_{1,t} \otimes p_{2,t}

    approaches the product of the time-averaged marginals.
    """

    METRIC_NAMES = (
        "avg_regret_1",
        "avg_regret_2",
        "max_avg_regret",
        "avg_regret_realized_1",
        "avg_regret_realized_2",
        "independence_l1",
        "independence_tv",
        "realized_vs_expected_l1",
        "policy_drift_1",
        "policy_drift_2",
        "exploitability_1",
        "exploitability_2",
        "max_exploitability",
        "bound_1",
        "bound_2",
        "entropy_1",
        "entropy_2",
    )

    def __init__(
        self,
        payoff: torch.Tensor,
        n_episodes: int,
        device: torch.device | None = None,
    ):
        if payoff.ndim != 2 or payoff.shape[0] != payoff.shape[1]:
            raise ValueError(
                f"payoff must be a square matrix. Got shape {tuple(payoff.shape)}."
            )
        if n_episodes <= 0:
            raise ValueError(f"n_episodes must be strictly positive. Got {n_episodes}.")

        self.dtype = torch.float64
        self.device = payoff.device if device is None else device

        self.payoff = payoff.to(device=self.device, dtype=self.dtype)
        self.payoff_range = (self.payoff.amax() - self.payoff.amin()).item()

        self.n_arms = payoff.shape[0]
        self.n_episodes = n_episodes
        self.t = 0

        shape = (n_episodes, self.n_arms)
        kwargs = {"dtype": self.dtype, "device": self.device}

        self.mu_sum = torch.zeros((n_episodes, self.n_arms, self.n_arms), **kwargs)
        self.policy_sum = torch.zeros((2, *shape), **kwargs)
        self.joint_counts = torch.zeros((n_episodes, self.n_arms**2), **kwargs)
        self.arm_counts = torch.zeros((2, *shape), **kwargs)
        self.realized_sum = torch.zeros((2, n_episodes), **kwargs)

        self._prev_policy_sum = None
        self._prev_window = None
        self._prev_t = 0
        return

    def update(self, policies: torch.Tensor, arms: torch.Tensor) -> None:
        """Record one round of play.

        Parameters
        ----------
        policies : torch.Tensor
            Tensor of shape ``(2, n_episodes, n_arms)`` containing the mixed
            strategy of each maker, read *before* the actions were sampled.
        arms : torch.Tensor
            Tensor of shape ``(2, n_episodes)`` containing the index of the arm
            each maker then played.

        Raises
        ------
        ValueError
            If the inputs do not match the expected shapes.
        """
        expected_policies = (2, self.n_episodes, self.n_arms)
        if policies.shape != expected_policies:
            raise ValueError(
                f"Expected policies shape {expected_policies}, got {tuple(policies.shape)}."
            )
        if arms.shape != (2, self.n_episodes):
            raise ValueError(
                f"Expected arms shape {(2, self.n_episodes)}, got {tuple(arms.shape)}."
            )

        policies = policies.to(device=self.device, dtype=self.dtype)
        arms = arms.to(device=self.device)

        # Expected joint distribution of the round, free of sampling noise
        self.mu_sum += policies[0].unsqueeze(-1) * policies[1].unsqueeze(-2)
        self.policy_sum += policies

        ones = torch.ones((self.n_episodes, 1), dtype=self.dtype, device=self.device)
        self.joint_counts.scatter_add_(
            1, (arms[0] * self.n_arms + arms[1]).unsqueeze(-1), ones
        )
        self.arm_counts[0].scatter_add_(1, arms[0].unsqueeze(-1), ones)
        self.arm_counts[1].scatter_add_(1, arms[1].unsqueeze(-1), ones)

        self.realized_sum[0] += self.payoff[arms[0], arms[1]]
        self.realized_sum[1] += self.payoff[arms[1], arms[0]]

        self.t += 1
        return

    def get_distributions(self) -> dict[str, torch.Tensor]:
        """Return the time-averaged distributions accumulated so far.

        Returns
        -------
        distributions : dict of str to torch.Tensor
            The average marginals ``policy_1`` and ``policy_2`` of shape
            ``(n_episodes, n_arms)``, the average joint distribution ``joint``
            and the product of the average marginals ``product``, both of shape
            ``(n_episodes, n_arms, n_arms)``.

        Raises
        ------
        RuntimeError
            If no round has been recorded yet.
        """
        if self.t == 0:
            raise RuntimeError("No round has been recorded yet.")

        policy = self.policy_sum / self.t
        return {
            "policy_1": policy[0],
            "policy_2": policy[1],
            "joint": self.mu_sum / self.t,
            "product": policy[0].unsqueeze(-1) * policy[1].unsqueeze(-2),
        }

    def snapshot(self, tol: float = 1e-6) -> dict[str, torch.Tensor]:
        """Evaluate every diagnostic on the play recorded so far.

        Parameters
        ----------
        tol : float, optional
            Tolerance used when checking the exploitability against its bound.

        Returns
        -------
        metrics : dict of str to torch.Tensor
            One entry per name in :attr:`METRIC_NAMES`, each of shape
            ``(n_episodes,)``.

        Raises
        ------
        RuntimeError
            If no round has been recorded yet.

        Notes
        -----
        Consecutive calls delimit the windows over which the policy drift is
        measured, so this method is meant to be called once per checkpoint. The
        drift is undefined, and reported as ``nan``, at the first call.

        Writing $q_T$ for the product of the average marginals and $\\bar\\mu_T$
        for the average joint distribution, the exploitability and the average
        regret of a maker differ by $\\langle \\bar\\mu_T - q_T, U \\rangle$,
        because the best response term is common to both and depends only on the
        opponent marginal. Since the two distributions have the same total mass,
        that difference is at most the range of the payoffs times the total
        variation distance, so

        .. math:: \\mathrm{exploit}_i \\le r_i(T) + B_i \\, TV_T

        holds as an identity rather than as an approximation. It is checked
        numerically at every call, and a violation indicates that the payoffs
        and the policies are being measured inconsistently.
        """
        if self.t == 0:
            raise RuntimeError("No round has been recorded yet.")

        payoff = self.payoff
        n = float(self.t)

        joint = self.mu_sum / n
        policy = self.policy_sum / n
        product = policy[0].unsqueeze(-1) * policy[1].unsqueeze(-2)

        independence_l1 = (joint - product).abs().sum(dim=(-2, -1))
        independence_tv = 0.5 * independence_l1

        realized_joint = self.joint_counts.reshape(joint.shape) / n
        realized_vs_expected_l1 = (realized_joint - joint).abs().sum(dim=(-2, -1))

        # Value of a fixed deviation depends only on the opponent marginal,
        # which is shared by the joint distribution and by the product
        best = torch.stack(
            [
                (policy[1] @ payoff.T).amax(dim=-1),
                (policy[0] @ payoff.T).amax(dim=-1),
            ]
        )
        value = torch.stack(
            [
                (joint * payoff).sum(dim=(-2, -1)),
                (joint * payoff.T).sum(dim=(-2, -1)),
            ]
        )
        avg_regret = best - value

        frequency = self.arm_counts / n
        realized_best = torch.stack(
            [
                (frequency[1] @ payoff.T).amax(dim=-1),
                (frequency[0] @ payoff.T).amax(dim=-1),
            ]
        )
        avg_regret_realized = realized_best - self.realized_sum / n

        exploitability = torch.stack(get_exploitability(payoff, policy[0], policy[1]))
        bound = avg_regret + self.payoff_range * independence_tv

        excess = (exploitability - bound).amax().item()
        if excess > tol:
            logger.warning(
                "Exploitability exceeds its bound by %.3e at round %d. The "
                "payoff matrix and the recorded policies may be inconsistent.",
                excess,
                self.t,
            )

        drift = self._advance_window(policy)
        entropy = torch.special.entr(policy).sum(dim=-1)

        return {
            "avg_regret_1": avg_regret[0],
            "avg_regret_2": avg_regret[1],
            "max_avg_regret": avg_regret.amax(dim=0),
            "avg_regret_realized_1": avg_regret_realized[0],
            "avg_regret_realized_2": avg_regret_realized[1],
            "independence_l1": independence_l1,
            "independence_tv": independence_tv,
            "realized_vs_expected_l1": realized_vs_expected_l1,
            "policy_drift_1": drift[0],
            "policy_drift_2": drift[1],
            "exploitability_1": exploitability[0],
            "exploitability_2": exploitability[1],
            "max_exploitability": exploitability.amax(dim=0),
            "bound_1": bound[0],
            "bound_2": bound[1],
            "entropy_1": entropy[0],
            "entropy_2": entropy[1],
        }

    def _advance_window(self, policy: torch.Tensor) -> torch.Tensor:
        """Measure how much the average policy moved since the previous call.

        Individual updates stay noisy even once the learners have settled, so
        the drift compares the average policy over the rounds elapsed since the
        previous call with the average over the rounds before that. Calls are
        expected to be logarithmically spaced, which makes the two windows grow
        with the horizon instead of keeping a fixed width.

        Parameters
        ----------
        policy : torch.Tensor
            Time-averaged marginals, of shape ``(2, n_episodes, n_arms)``.

        Returns
        -------
        drift : torch.Tensor
            Tensor of shape ``(2, n_episodes)``, filled with ``nan`` when no
            previous window is available, or when no round elapsed since the
            previous call.
        """
        nan_drift = torch.full(
            (2, self.n_episodes),
            float("nan"),
            dtype=self.dtype,
            device=self.device,
        )

        # Two calls at the same round delimit an empty window
        if self._prev_policy_sum is not None and self.t == self._prev_t:
            return nan_drift

        if self._prev_policy_sum is None:
            window = policy
        else:
            window = (self.policy_sum - self._prev_policy_sum) / (self.t - self._prev_t)

        if self._prev_window is None:
            drift = nan_drift
        else:
            drift = (window - self._prev_window).abs().sum(dim=-1)

        self._prev_window = window
        self._prev_policy_sum = self.policy_sum.clone()
        self._prev_t = self.t
        return drift
