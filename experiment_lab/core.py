"""Deterministic randomisation and synthetic outcome generation.

No real case data are used. This is a design and statistical testing exercise,
not a safety-certified operational triage engine.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SENSITIVE_CATEGORIES = frozenset({
    "Child safety concern",
    "Self-harm concern",
    "Privacy concern",
    "Scam or fraud",
    "Billing safety escalation",
})
REQUIRED_COLUMNS = frozenset({
    "risk_category", "severity", "region", "channel", "language",
    "human_review_required", "model_confidence",
})
EXPERIMENT_SALT = "riskops-triage-policy-v1"


def is_eligible(row: pd.Series) -> bool:
    """Test only lower-risk cases; preserve human review across BOTH arms."""
    return (
        row["severity"] in {"Low", "Medium"}
        and row["risk_category"] not in SENSITIVE_CATEGORIES
        and float(row["model_confidence"]) >= 0.75
    )


def arm_for_unit(unit_id: str, salt: str = EXPERIMENT_SALT) -> str:
    """Stable 50/50 randomisation independent of run order or input features."""
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode("utf-8")).digest()
    return "treatment" if int.from_bytes(digest[:8], "big") < 2**63 else "control"


def simulate(
    source: str | Path,
    population: int = 20_000,
    seed: int = 42,
    treatment_effect: float = 0.05,
) -> pd.DataFrame:
    """Simulate a randomised experiment using synthetic template reports.

    treatment_effect is the **injected absolute uplift** in SLA success
    probability among eligible cases. Feedback is missing at random
    conditional on measured covariates, with its true observation propensity
    deliberately logged for IPW. Those propensities are NOT estimated here.
    """
    if population < 100:
        raise ValueError("population must be at least 100")
    if not 0 <= treatment_effect <= 0.20:
        raise ValueError("treatment_effect must be between 0 and 0.20")

    templates = pd.read_csv(source)
    missing = REQUIRED_COLUMNS - set(templates.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")
    if templates.empty:
        raise ValueError("Source CSV must contain at least one synthetic template")

    rng = np.random.default_rng(seed)
    sampled = templates.iloc[rng.integers(0, len(templates), size=population)].copy()
    sampled = sampled.reset_index(drop=True)
    sampled.insert(0, "experiment_unit_id", [f"EXP-{i:07d}" for i in range(population)])
    sampled["eligible"] = sampled.apply(is_eligible, axis=1)
    sampled["arm"] = [
        arm_for_unit(unit_id) if eligible else "excluded"
        for unit_id, eligible in zip(sampled["experiment_unit_id"], sampled["eligible"])
    ]
    # A pre-treatment, continuous risk/history proxy. This value cannot be
    # affected by treatment and is available even when follow-up is missing.
    sampled["baseline_score"] = rng.beta(2, 2, size=population)
    eligible = sampled["eligible"].to_numpy(dtype=bool)
    treated = (sampled["arm"] == "treatment").to_numpy()
    baseline = sampled["baseline_score"].to_numpy()

    base_rate = (
        0.43 + 0.46 * baseline
        + 0.075 * (sampled["severity"] == "Low").to_numpy()
        - 0.055 * (sampled["region"] == "Global / Unknown").to_numpy()
        - 0.045 * (sampled["language"] == "Other").to_numpy()
    )
    p_success = np.clip(base_rate + treatment_effect * treated, 0.04, 0.96)
    # Ineligible reports are *not* evaluated as an experimental outcome.
    success = rng.binomial(1, p_success, size=population).astype(float)
    success[~eligible] = np.nan

    # Recorded feedback / consent varies by observed attributes and arm.
    # This is MAR by construction, so inverse-probability weights are valid
    # here; real opt-out mechanisms may violate this assumption.
    propensity = (
        0.87
        - 0.25 * (sampled["language"] == "Other").to_numpy()
        - 0.12 * (sampled["region"] == "Global / Unknown").to_numpy()
        - 0.07 * (sampled["channel"] == "Mobile app").to_numpy()
        + 0.06 * treated
        - 0.09 * (sampled["severity"] == "Medium").to_numpy()
    )
    propensity = np.clip(propensity, 0.30, 0.97)
    observed = (rng.uniform(size=population) < propensity) & eligible
    success[~observed] = np.nan

    sampled["observed"] = observed
    sampled["observation_propensity"] = propensity
    sampled["sla_success"] = success
    # Hard policy constraint: eligibility does not override human review.
    sampled["human_review_preserved"] = True
    sampled["protected_exposure"] = treated & ~eligible
    return sampled[[
        "experiment_unit_id", "risk_category", "severity", "region",
        "channel", "language", "model_confidence", "human_review_required",
        "eligible", "arm", "baseline_score", "observed",
        "observation_propensity", "sla_success", "human_review_preserved",
        "protected_exposure",
    ]]
