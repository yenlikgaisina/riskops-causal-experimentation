"""Auditable routing decisions for the synthetic RiskOps experiment.

This module separates the *chosen action* from assertions about that action.
No real safety incident or user data belongs in this demonstration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSITIVE_CATEGORIES = frozenset({
    "Child safety concern", "Self-harm concern", "Privacy concern",
    "Scam or fraud", "Billing safety escalation",
})


def eligibility_mask(rows: pd.DataFrame) -> pd.Series:
    """Fail closed: exclude protected, low-confidence and mandated-review cases."""
    required = {"risk_category", "severity", "model_confidence", "human_review_required"}
    absent = required - set(rows.columns)
    if absent:
        raise ValueError(f"Missing policy inputs: {sorted(absent)}")
    # Strictly check the data type rather than treating a nonempty 'False' string as True.
    flag = rows["human_review_required"]
    if flag.isna().any() or not flag.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("human_review_required must be non-null booleans")
    confidence = pd.to_numeric(rows["model_confidence"], errors="coerce")
    return (
        rows["severity"].isin(["Low", "Medium"])
        & ~rows["risk_category"].isin(SENSITIVE_CATEGORIES)
        & confidence.between(0.75, 1.0, inclusive="both")
        & ~flag
    )


def choose_route(eligible: pd.Series, arms: pd.Series) -> pd.Series:
    """Return logged actions (not a post-hoc safety claim).

    Protected and mandated-review reports always go to human review. Every
    treatment exposure is represented by its real, observable route action.
    """
    if not eligible.index.equals(arms.index):
        raise ValueError("Eligible and arm indexes differ")
    actions = np.select(
        [~eligible.to_numpy(dtype=bool), arms.eq("treatment"), arms.eq("control")],
        ["human_review", "assisted_triage", "standard_triage"],
        default="invalid_arm",
    )
    return pd.Series(actions, index=eligible.index, name="route_action")


def audit_routing(rows: pd.DataFrame) -> dict:
    """Recompute invariant violations using actions and source attributes.

    Deliberately ignores the `human_review_preserved` and
    `protected_exposure` summary flags so callers cannot overwrite those to
    make a compromised route pass a safety gate.
    """
    required = {"eligible", "arm", "route_action", "human_review_required"}
    absent = required - set(rows.columns)
    if absent:
        raise ValueError(f"Missing route audit columns: {sorted(absent)}")
    expected_eligible = eligibility_mask(rows)
    claimed_eligible = rows["eligible"]
    invalid_eligibility = int((expected_eligible != claimed_eligible).sum())
    expected_arms = rows["arm"].isin(["control", "treatment"])
    invalid_assignments = int((expected_eligible != expected_arms).sum())
    assigned_excluded = int((~expected_eligible & rows["arm"].eq("treatment")).sum())
    exposed_protected = int((~expected_eligible & rows["route_action"].eq("assisted_triage")).sum())
    missing_review = int((rows["human_review_required"] & ~rows["route_action"].eq("human_review")).sum())
    expected_actions = choose_route(expected_eligible, rows["arm"])
    mismatched_actions = int((rows["route_action"] != expected_actions).sum())
    return {
        "invalid_eligibility": invalid_eligibility,
        "invalid_assignments": invalid_assignments,
        "assigned_excluded": assigned_excluded,
        "exposed_protected": exposed_protected,
        "missing_mandatory_review": missing_review,
        "mismatched_actions": mismatched_actions,
        "passed": not any((invalid_eligibility, invalid_assignments, assigned_excluded,
                           exposed_protected, missing_review, mismatched_actions)),
    }
