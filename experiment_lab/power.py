"""Two-sample design-stage normal-approximation power calculator.

This deliberately assumes independent Bernoulli outcomes, a fixed analysis
horizon and no sequential peeking, cluster effects, unequal allocation or
missing feedback. It is a planning exercise, not authorisation to deploy.
"""
from __future__ import annotations

from math import ceil, sqrt

from scipy.stats import norm


def sample_size_per_arm(
    baseline_rate: float = 0.65,
    minimum_detectable_effect: float = 0.05,
    alpha: float = 0.05,
    target_power: float = 0.80,
    anticipated_observation_rate: float = 1.0,
) -> dict:
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between zero and one")
    if not 0 < minimum_detectable_effect < 1 - baseline_rate:
        raise ValueError("effect must be positive and less than 1 - baseline")
    if not 0 < alpha < 1 or not 0 < target_power < 1:
        raise ValueError("alpha and target_power must be between zero and one")
    if not 0 < anticipated_observation_rate <= 1:
        raise ValueError("observation rate must be in (0,1]")
    p1, p2 = baseline_rate, baseline_rate + minimum_detectable_effect
    p_pool = (p1 + p2) / 2
    numerator = (
        norm.ppf(1 - alpha / 2) * sqrt(2 * p_pool * (1 - p_pool))
        + norm.ppf(target_power) * sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    n_observed = ceil(numerator / minimum_detectable_effect ** 2)
    n_randomised = ceil(n_observed / anticipated_observation_rate)
    return {
        "baseline_rate": baseline_rate,
        "minimum_detectable_effect": minimum_detectable_effect,
        "alpha_two_sided": alpha,
        "target_power": target_power,
        "anticipated_observation_rate": anticipated_observation_rate,
        "required_observed_per_arm": n_observed,
        "required_randomised_per_arm": n_randomised,
        "required_randomised_total": 2 * n_randomised,
        "limitations": "Normal approximation, equal allocation, independent outcomes, fixed horizon; missingness assumed unrelated to effect and safety",
    }
