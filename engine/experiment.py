"""Experiment design and read-out for a target vs control launch test.

Standard two-proportion formulas. scipy supplies the normal and chi-square
distributions; everything else is written out so each step can be explained.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from engine.unit_economics import adoption_probability


def sample_size_per_group(baseline: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Users needed in EACH group to detect an absolute lift of `mde`.

    baseline: expected conversion rate in the control group (0-1).
    mde: minimum detectable effect, absolute (0.02 means +2 percentage points).
    """
    if not 0 < baseline < 1:
        raise ValueError("baseline must be between 0 and 1")
    if mde <= 0 or baseline + mde >= 1:
        raise ValueError("mde must be positive and baseline + mde below 1")
    p1, p2 = baseline, baseline + mde
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    variance = p1 * (1 - p1) + p2 * (1 - p2)
    return math.ceil((z_alpha + z_power) ** 2 * variance / mde ** 2)


def minimum_detectable_effect(n_per_group: int, baseline: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest absolute lift a test of this size can reliably detect."""
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")
    low, high = 1e-6, 1 - baseline - 1e-6
    for _ in range(60):  # bisection: halve the range until it is tight
        mid = (low + high) / 2
        if sample_size_per_group(baseline, mid, alpha, power) > n_per_group:
            low = mid
        else:
            high = mid
    return high


def srm_check(n_target: int, n_control: int, expected_target_share: float = 0.5) -> dict:
    """Sample ratio mismatch: did the split come out as designed?

    A very small p-value (below 0.001) means the assignment is broken and the
    test result should not be trusted.
    """
    total = n_target + n_control
    expected = [total * expected_target_share, total * (1 - expected_target_share)]
    chi2, p_value = stats.chisquare([n_target, n_control], f_exp=expected)
    return {"chi2": float(chi2), "p_value": float(p_value), "mismatch": bool(p_value < 0.001)}


def two_proportion_test(x_target: int, n_target: int, x_control: int, n_control: int, alpha: float = 0.05) -> dict:
    """Difference in conversion between target and control, with a confidence interval."""
    p_t, p_c = x_target / n_target, x_control / n_control
    pooled = (x_target + x_control) / (n_target + n_control)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_target + 1 / n_control))
    z = (p_t - p_c) / se_pooled if se_pooled else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = math.sqrt(p_t * (1 - p_t) / n_target + p_c * (1 - p_c) / n_control)
    margin = stats.norm.ppf(1 - alpha / 2) * se_diff
    return {
        "target_rate": p_t,
        "control_rate": p_c,
        "absolute_lift": p_t - p_c,
        "relative_lift": (p_t - p_c) / p_c if p_c else float("inf"),
        "ci_low": float(p_t - p_c - margin),
        "ci_high": float(p_t - p_c + margin),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
    }


def simulate_test(users: pd.DataFrame, target_rate: float, control_rate: float,
                  n_per_group: int, seed: int = 7) -> dict:
    """Simulate one launch test on the synthetic base.

    Target users are offered `target_rate` cashback, control users `control_rate`.
    This is the only random step in the project; the seed makes it repeatable.
    """
    if 2 * n_per_group > len(users):
        raise ValueError("Not enough users for two groups of this size")
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(users))[: 2 * n_per_group]
    target, control = users.iloc[chosen[:n_per_group]], users.iloc[chosen[n_per_group:]]
    x_t = int((rng.random(n_per_group) < adoption_probability(target, target_rate)).sum())
    x_c = int((rng.random(n_per_group) < adoption_probability(control, control_rate)).sum())
    result = two_proportion_test(x_t, n_per_group, x_c, n_per_group)
    result.update({"n_per_group": n_per_group, "target_conversions": x_t, "control_conversions": x_c,
                   "srm": srm_check(n_per_group, n_per_group)})
    return result
