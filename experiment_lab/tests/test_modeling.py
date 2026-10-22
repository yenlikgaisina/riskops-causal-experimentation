from pathlib import Path

import numpy as np
import pandas as pd

from experiment_lab.core import simulate
from experiment_lab.modeling import FEATURES, predict_sla_probability, train_sla_baseline
from experiment_lab.monitoring import monitor, population_stability_index


def test_model_training_only_control_and_scoring(tmp_path: Path):
    data = pd.DataFrame([{
        "risk_category": "Policy confusion", "severity": "Low", "region": "UK",
        "channel": "Web form", "language": "English",
        "human_review_required": False, "model_confidence": 0.94,
    }])
    path = tmp_path / "synthetic.csv"
    data.to_csv(path, index=False)
    rows = simulate(path, population=2500, seed=17)
    manifest = train_sla_baseline(rows, tmp_path / "model")
    assert manifest["synthetic_only"]
    assert 0 < manifest["holdout_brier_score_ipw"] < 1
    assert 0 <= manifest["holdout_roc_auc_ipw"] <= 1
    assert manifest["n_test"] > 100
    preds = predict_sla_probability(tmp_path / "model" / "sla_baseline.joblib", rows.head(10))
    assert len(preds) == 10
    assert np.all((preds >= 0) & (preds <= 1))


def test_drift_monitoring():
    assert population_stability_index(pd.Series(["UK"] * 100), pd.Series(["UK"] * 100)) == 0
    ref = pd.DataFrame({"region": ["UK"] * 90 + ["Europe"] * 10,
                        "channel": ["Web form"] * 100,
                        "language": ["English"] * 100,
                        "severity": ["Low"] * 100})
    new = ref.copy()
    new["region"] = "Europe"
    findings = monitor(ref, new)
    assert "region" in findings["alert_fields"]
    assert "channel" not in findings["alert_fields"]
