"""Synthetic SLA-risk baseline model, offline validation and artifact manifest.

Predictions are explanatory demonstrations and must not automate risk routing.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["severity", "region", "channel", "language", "risk_category"]
NUMERIC = ["baseline_score", "model_confidence"]
FEATURES = CATEGORICAL + NUMERIC


def train_sla_baseline(
    events: pd.DataFrame,
    directory: Path,
    *,
    random_state: int = 42,
) -> dict:
    """Train on observed CONTROL reports only; avoid experimental leakage.

    Offline scoring demonstrates train/holdout separation, outcome-label
    provenance, IPW sample weights, metric reporting and model versioning.
    """
    control = events.loc[
        (events["arm"] == "control") & events["observed"]
    ].copy()
    if len(control) < 200 or control["sla_success"].nunique() < 2:
        raise ValueError("At least 200 control observations and two outcome classes required")
    train, test = train_test_split(
        control, test_size=0.25, random_state=random_state,
        stratify=control["sla_success"],
    )
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", StandardScaler(), NUMERIC),
    ])
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    pipeline.fit(
        train[FEATURES], train["sla_success"].astype(int),
        classifier__sample_weight=1.0 / train["observation_propensity"].to_numpy(),
    )
    scores = pipeline.predict_proba(test[FEATURES])[:, 1]
    labels = test["sla_success"].astype(int).to_numpy()
    weights = 1.0 / test["observation_propensity"].to_numpy()
    report = {
        "model": "synthetic_sla_success_baseline_logistic_regression",
        "training_arm": "control_only",
        "synthetic_only": True,
        "n_train": len(train), "n_test": len(test),
        "holdout_brier_score_ipw": round(float(brier_score_loss(labels, scores, sample_weight=weights)), 5),
        "holdout_roc_auc_ipw": round(float(roc_auc_score(labels, scores, sample_weight=weights)), 5),
        "features": FEATURES,
        "limitations": "Synthetic model; no real-world calibration, privacy or safety validation",
    }
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "sla_baseline.joblib"
    joblib.dump(pipeline, model_path)
    report["artifact_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
    (directory / "sla_baseline_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def predict_sla_probability(model_path: Path, features: pd.DataFrame) -> np.ndarray:
    """Research-only scoring; only load locally trusted versioned artifacts."""
    model = joblib.load(model_path)
    return model.predict_proba(features[FEATURES])[:, 1]
