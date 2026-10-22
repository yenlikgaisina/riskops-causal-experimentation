"""Basic monitoring diagnostics for categorical population shift.

PSI is a heuristic signal, not a formal probability or drift hypothesis test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    if reference.empty or current.empty:
        raise ValueError("Both reference and current populations must be nonempty")
    ref = reference.astype(str).value_counts(normalize=True)
    cur = current.astype(str).value_counts(normalize=True)
    categories = ref.index.union(cur.index)
    ref_p = np.array([ref.get(key, 0.0) for key in categories], dtype=float)
    cur_p = np.array([cur.get(key, 0.0) for key in categories], dtype=float)
    # Smoothing preserves a usable, finite indicator with absent categories.
    eps = 1e-4
    ref_p = (ref_p + eps) / (1 + eps * len(categories))
    cur_p = (cur_p + eps) / (1 + eps * len(categories))
    return round(float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p))), 6)


def monitor(reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.2) -> dict:
    if not 0 < threshold < 5:
        raise ValueError("Threshold must be between 0 and 5")
    fields = ["region", "channel", "language", "severity"]
    results = {key: population_stability_index(reference[key], current[key]) for key in fields}
    return {
        "drift_psi": results,
        "alert_fields": [key for key, value in results.items() if value > threshold],
        "threshold": threshold,
        "not_a_production_alert": True,
    }
