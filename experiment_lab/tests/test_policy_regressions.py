"""Checks that safety cannot be faked by flipping pre-computed status flags."""
from pathlib import Path

import pandas as pd
import pytest

from experiment_lab.core import simulate
from experiment_lab.decision import review_experiment
from experiment_lab.estimators import estimate_difference, weight_diagnostics
from experiment_lab.policy import audit_routing, eligibility_mask
from experiment_lab.power import sample_size_per_arm


def _example(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame([
        {"risk_category": "Policy confusion", "severity": "Low", "region": "UK", "channel": "Web form", "language": "English", "human_review_required": False, "model_confidence": 0.95},
        {"risk_category": "Child safety concern", "severity": "Critical", "region": "UK", "channel": "Web form", "language": "English", "human_review_required": True, "model_confidence": 0.95},
        {"risk_category": "Harassment", "severity": "Medium", "region": "UK", "channel": "Web form", "language": "English", "human_review_required": True, "model_confidence": 0.98},
    ]).to_csv(source, index=False)
    return simulate(source, population=10000, seed=42, treatment_effect=0.10)


def test_policy_fails_closed_for_mandatory_review(tmp_path):
    rows = _example(tmp_path)
    required = rows["human_review_required"]
    assert not rows.loc[required, "eligible"].any()
    assert rows.loc[required, "route_action"].eq("human_review").all()
    assert audit_routing(rows)["passed"]
    assert weight_diagnostics(rows)["control"]["effective_sample_size"] > 0


def test_independent_route_audit_catches_forged_flags(tmp_path):
    rows = _example(tmp_path)
    at_risk = rows.index[rows["human_review_required"]][0]
    rows.loc[at_risk, "route_action"] = "assisted_triage"
    rows.loc[at_risk, "protected_exposure"] = False  # forge the old summary flag
    rows.loc[at_risk, "human_review_preserved"] = True
    audit = audit_routing(rows)
    assert audit["exposed_protected"] >= 1
    assert audit["missing_mandatory_review"] >= 1
    estimate = estimate_difference(rows)
    verdict = review_experiment(rows, estimate)
    assert verdict.status == "simulation_blocked"
    assert any("Safety violation" in reason for reason in verdict.reasons)


def test_audit_catches_mislabelled_eligibility_and_arm(tmp_path):
    rows = _example(tmp_path)
    index = rows.index[rows["human_review_required"]][0]
    rows.loc[index, "eligible"] = True
    rows.loc[index, "arm"] = "treatment"
    assert not audit_routing(rows)["passed"]
    assert audit_routing(rows)["invalid_eligibility"] == 1


def test_source_human_review_requires_strict_boolean(tmp_path):
    malformed = pd.DataFrame([{
        "risk_category": "Policy confusion", "severity": "Low",
        "model_confidence": 0.99, "human_review_required": "False",
    }])
    with pytest.raises(ValueError):
        eligibility_mask(malformed)


def test_power_increases_when_effect_decreases_or_feedback_drops():
    base = sample_size_per_arm()
    smaller = sample_size_per_arm(minimum_detectable_effect=0.03)
    missing = sample_size_per_arm(anticipated_observation_rate=0.6)
    assert smaller["required_randomised_total"] > base["required_randomised_total"]
    assert missing["required_randomised_total"] > base["required_randomised_total"]
    with pytest.raises(ValueError):
        sample_size_per_arm(anticipated_observation_rate=0)
