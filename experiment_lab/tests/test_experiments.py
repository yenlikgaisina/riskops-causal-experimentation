"""Tests cover randomisation, sensitive exposure rules, inferential plumbing."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiment_lab.core import arm_for_unit, is_eligible, simulate
from experiment_lab.decision import analyse, review_experiment
from experiment_lab.estimators import estimate_difference


@pytest.fixture
def templates(tmp_path: Path) -> Path:
    data = pd.DataFrame([
        {"risk_category": "Policy confusion", "severity": "Low", "region": "UK", "channel": "Web form", "language": "English", "human_review_required": False, "model_confidence": 0.95},
        {"risk_category": "Harassment", "severity": "Medium", "region": "Global / Unknown", "channel": "Mobile app", "language": "Other", "human_review_required": True, "model_confidence": 0.90},
        {"risk_category": "Child safety concern", "severity": "Critical", "region": "UK", "channel": "Web form", "language": "English", "human_review_required": True, "model_confidence": 0.98},
        {"risk_category": "Scam or fraud", "severity": "Medium", "region": "UK", "channel": "Web form", "language": "English", "human_review_required": True, "model_confidence": 0.80},
        {"risk_category": "Policy confusion", "severity": "Low", "region": "UK", "channel": "Email support", "language": "English", "human_review_required": True, "model_confidence": 0.55},
    ])
    path = tmp_path / "tickets.csv"
    data.to_csv(path, index=False)
    return path


def test_assignment_is_stable_and_balanced():
    assert arm_for_unit("TKT-1") == arm_for_unit("TKT-1")
    arms = [arm_for_unit(str(i)) for i in range(10_000)]
    assert 0.48 < arms.count("treatment") / len(arms) < 0.52


def test_protected_categories_never_enter_treatment(templates):
    rows = simulate(templates, population=2_000, seed=11)
    assert not rows["protected_exposure"].any()
    assert not rows.loc[rows["severity"] == "Critical", "eligible"].any()
    assert not rows.loc[rows["risk_category"] == "Scam or fraud", "eligible"].any()
    assert rows["human_review_preserved"].all()
    assert rows.loc[~rows["observed"], "sla_success"].isna().all()


def test_repeatable_simulation_and_causal_report(templates):
    a = simulate(templates, population=16_000, seed=9, treatment_effect=0.10)
    b = simulate(templates, population=16_000, seed=9, treatment_effect=0.10)
    pd.testing.assert_frame_equal(a, b)
    report = analyse(a)
    assert report["data_provenance"] == "synthetic_only"
    assert report["simulation_decision"]["production_launch_authorised"] is False
    estimate = report["estimates"]["ipw_cuped"]
    assert abs(estimate["absolute_uplift"] - 0.10) < 0.05
    assert report["eligible_units"] + report["protected_units_excluded"] == 16_000


def test_guardrails_block_even_if_effect_positive(templates):
    rows = simulate(templates, population=16_000, seed=9, treatment_effect=0.15)
    rows.loc[rows.index[0], "protected_exposure"] = True
    decision = review_experiment(rows, estimate_difference(rows, method="ipw_cuped"))
    assert decision.status == "simulation_blocked"
    assert any("Safety violation" in reason for reason in decision.reasons)


def test_no_effect_fails_conservative_criterion(templates):
    rows = simulate(templates, population=10_000, seed=8, treatment_effect=0.0)
    est = estimate_difference(rows, method="ipw_cuped")
    assert abs(est.absolute_uplift) < 0.06
    if est.ci_lower <= 0:
        assert review_experiment(rows, est).status == "simulation_blocked"


def test_eligibility_requires_confidence_and_low_or_medium_risk():
    row = pd.Series({"risk_category": "Policy confusion", "severity": "Low", "model_confidence": 0.8})
    assert is_eligible(row)
    row["model_confidence"] = 0.7
    assert not is_eligible(row)
    row["model_confidence"] = 0.9
    row["severity"] = "High"
    assert not is_eligible(row)


def test_invalid_inputs_and_propensity(templates):
    with pytest.raises(ValueError):
        simulate(templates, population=99)
    with pytest.raises(ValueError):
        simulate(templates, population=100, treatment_effect=-0.1)
    rows = simulate(templates, population=1000, seed=2)
    ix = rows.index[rows["observed"]][0]
    rows.loc[ix, "observation_propensity"] = 0
    with pytest.raises(ValueError):
        estimate_difference(rows, method="ipw")
