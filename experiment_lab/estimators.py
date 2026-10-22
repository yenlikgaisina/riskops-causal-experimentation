"""Naive, IPW/Hajek and covariate-adjusted experiment estimators.

Normal-approximation uncertainty is pedagogical. In production, validate
propensity models and use robust inference appropriate for the randomisation
unit, clustering, repeated looks and the missingness mechanism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Estimate:
    method: str
    control_mean: float
    treatment_mean: float
    absolute_uplift: float
    standard_error: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> dict:
        return asdict(self)


def _weighted_mean_and_var(y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Hajek ratio estimate and first-order influence-function variance."""
    denominator = float(weights.sum())
    if denominator <= 0:
        raise ValueError("No positive observation weights")
    mean = float(np.dot(weights, y) / denominator)
    # Variance of sum(w*y) / sum(w), conditional on observed design.
    variance = float(np.dot(weights**2, (y - mean)**2) / denominator**2)
    return mean, variance


def estimate_difference(
    rows: pd.DataFrame, *, method: str = "ipw_cuped"
) -> Estimate:
    """Estimate treatment-control SLA uplift on eligible synthetic tickets.

    naive: observed-only difference (selection bias possible)
    ipw: self-normalised inverse observation-propensity weights
    ipw_cuped: IPW plus pooled pre-treatment covariate adjustment
    """
    if method not in {"naive", "ipw", "ipw_cuped"}:
        raise ValueError("Unsupported method")
    required = {"eligible", "arm", "observed", "sla_success", "baseline_score", "observation_propensity"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    population = rows.loc[rows["eligible"] & rows["arm"].isin(["control", "treatment"])]
    if population.empty:
        raise ValueError("No eligible experimental units")
    observed = population.loc[population["observed"]].copy()
    if observed["sla_success"].isna().any():
        raise ValueError("Observed outcomes must be present")
    if not observed["arm"].eq("treatment").any() or not observed["arm"].eq("control").any():
        raise ValueError("Both arms require observed outcomes")
    propensity = observed["observation_propensity"].to_numpy(dtype=float)
    if not np.isfinite(propensity).all() or (propensity <= 0).any() or (propensity > 1).any():
        raise ValueError("Observation propensities must be in (0, 1]")
    weights = np.ones(len(observed)) if method == "naive" else 1 / propensity
    outcomes = observed["sla_success"].to_numpy(dtype=float)
    if method == "ipw_cuped":
        x = observed["baseline_score"].to_numpy(dtype=float)
        xbar = float(population["baseline_score"].mean())
        x_obs_mean = float(np.average(x, weights=weights))
        y_obs_mean = float(np.average(outcomes, weights=weights))
        cov = float(np.average((x - x_obs_mean) * (outcomes - y_obs_mean), weights=weights))
        var_x = float(np.average((x - x_obs_mean) ** 2, weights=weights))
        theta = cov / var_x if var_x > 1e-12 else 0.0
        outcomes = outcomes - theta * (x - xbar)
    means, variances = {}, {}
    for arm in ("control", "treatment"):
        mask = (observed["arm"] == arm).to_numpy()
        means[arm], variances[arm] = _weighted_mean_and_var(outcomes[mask], weights[mask])
    se = sqrt(variances["control"] + variances["treatment"])
    uplift = means["treatment"] - means["control"]
    return Estimate(
        method=method, control_mean=round(means["control"], 6),
        treatment_mean=round(means["treatment"], 6),
        absolute_uplift=round(uplift, 6), standard_error=round(se, 6),
        ci_lower=round(uplift - 1.96 * se, 6),
        ci_upper=round(uplift + 1.96 * se, 6),
    )
