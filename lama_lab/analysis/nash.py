import logging

import torch

logger = logging.getLogger(__name__)


def get_nash_market_making(
    samples: torch.Tensor,
    fixed_points: torch.Tensor,
    grid_size: int = 5000,
    tol: float = 1e-4,
    verbose: bool = False,
) -> torch.Tensor:
    r"""Filter fixed points to retain only true Nash Equilibria.

    A fixed point $(b^*, x^*, a^*)$ is a Nash Equilibrium if no agent has a
    unilateral profitable deviation. In a competitive market, an agent can only
    capture volume by offering a better price than the market:
    - Ask deviation (Undercut): $a' < a^*$
    - Bid deviation (Outbid): $b' > b^*$

    If an agent quotes a worse price, they capture no volume and get 0 profit.
    Therefore, we only search for profitable deviations strictly within the
    bounds that improve the best market price.

    Parameters
    ----------
    samples : torch.Tensor
        One-dimensional tensor containing samples drawn from the target distribution.
    fixed_points : torch.Tensor
        Tensor of shape ``(N, 3)`` containing candidate fixed points.
    grid_size : int, optional
        Number of evaluation points to search for global deviations.
    tol : float, optional
        Numerical tolerance. A deviation is considered profitable only if the
        expected profit strictly exceeds ``tol``.
    verbose : bool, optional
        If True, prints detailed information whenever a candidate fixed point is
        invalidated by a profitable deviation. Default is True.

    Returns
    -------
    out : torch.Tensor
        Tensor containing only the rows from ``fixed_points`` that are verified
        Nash Equilibria.
    """
    if fixed_points.numel() == 0:
        return fixed_points

    samples_sorted = torch.sort(samples.flatten()).values
    n_samples = len(samples_sorted)
    x_min, x_max = samples_sorted[0].item(), samples_sorted[-1].item()

    # Precompute prefix sums for O(1) integral evaluations
    cum_sums = torch.cat(
        [
            torch.zeros(1, dtype=samples.dtype, device=samples.device),
            torch.cumsum(samples_sorted, dim=0),
        ]
    )

    nash_mask = []

    for i in range(fixed_points.shape[0]):
        b_star, x_star, a_star = fixed_points[i].tolist()

        # Evaluate Ask Deviations (Undercut: a' <= a*)
        # We only search for asks up to the current best ask a*
        a_grid = torch.linspace(x_min, a_star, steps=grid_size, device=samples.device)
        x_prime_a = (a_grid + b_star) / 2.0
        idx_a = torch.searchsorted(samples_sorted, x_prime_a)

        prob_upper = (n_samples - idx_a).to(samples.dtype) / n_samples
        upper_integral = (cum_sums[-1] - cum_sums[idx_a]) / n_samples
        profit_a = prob_upper * a_grid - upper_integral

        max_profit_a, best_a_idx = profit_a.max(dim=0)
        max_profit_a_val = max_profit_a.item()
        best_a_prime = a_grid[best_a_idx].item()

        # Check if the max profit is both positive and actually an undercut
        is_ask_profitable = (max_profit_a_val > tol) and (best_a_prime < a_star - 1e-5)

        # Evaluate Bid Deviations (Outbid: b' >= b*)
        # We only search for bids starting from the current best bid b*
        b_grid = torch.linspace(b_star, x_max, steps=grid_size, device=samples.device)
        x_prime_b = (a_star + b_grid) / 2.0
        idx_b = torch.searchsorted(samples_sorted, x_prime_b)

        prob_lower = idx_b.to(samples.dtype) / n_samples
        lower_integral = cum_sums[idx_b] / n_samples
        profit_b = lower_integral - prob_lower * b_grid

        max_profit_b, best_b_idx = profit_b.max(dim=0)
        max_profit_b_val = max_profit_b.item()
        best_b_prime = b_grid[best_b_idx].item()

        is_bid_profitable = (max_profit_b_val > tol) and (best_b_prime > b_star + 1e-5)

        # Check and Log Deviations
        if is_ask_profitable or is_bid_profitable:
            nash_mask.append(False)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "\n[X] Candidate Fixed Point (#%d) is NOT a Nash Equilibrium:\n"
                    "    Candidate: Bid=%.4f, Midpoint=%.4f, Ask=%.4f",
                    i + 1,
                    b_star,
                    x_star,
                    a_star,
                )
                if is_ask_profitable:
                    new_m_a = (best_a_prime + b_star) / 2.0
                    logger.debug(
                        "    --> Profitable ASK Deviation (Undercut):\n"
                        "        Deviating Ask (a'): %.4f\n"
                        "        New Midpoint (m'):  %.4f\n"
                        "        Expected Profit:    +%.6f",
                        best_a_prime,
                        new_m_a,
                        max_profit_a_val,
                    )
                if is_bid_profitable:
                    new_m_b = (a_star + best_b_prime) / 2.0
                    logger.debug(
                        "    --> Profitable BID Deviation (Outbid):\n"
                        "        Deviating Bid (b'): %.4f\n"
                        "        New Midpoint (m'):  %.4f\n"
                        "        Expected Profit:    +%.6f",
                        best_b_prime,
                        new_m_b,
                        max_profit_b_val,
                    )
        else:
            nash_mask.append(True)
            logger.debug(
                "\n[V] Candidate Fixed Point (#%d) IS a Nash Equilibrium:\n"
                "    Candidate: Bid=%.4f, Midpoint=%.4f, Ask=%.4f",
                i + 1,
                b_star,
                x_star,
                a_star,
            )

        valid_nash = fixed_points[nash_mask]
    return valid_nash
