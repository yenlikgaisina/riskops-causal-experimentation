"""Conservative, synthetic-only experiment review gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .estimators import Estimate, estimate_difference, weight_diagnostics
from .policy import audit_routing


@dataclass(frozen=True)
class Decision:
    status: str
    reasons: list[str]
    evidence_kind: str = "synthetic_simulation_only"
    production_launch_authorised: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def review_experiment(
    rows: pd.DataFrame,
    estimate: Estimate,
    *,
    min_observed_per_arm: int = 250,
    min_effect: float = 0.0,
) -> Decision:
    """Gate a demonstration run; NEVER grant approval for live deployment."""
    if min_observed_per_arm < 1:
        raise ValueError("min_observed_per_arm must be positive")
    reasons: list[str] = []
    audit = audit_routing(rows)
    for name, count in audit.items():
        if name != "passed" and count:
            reasons.append(f"Safety violation: {name} = {count}")
    if "protected_exposure" in rows and int(rows["protected_exposure"].sum()):
        reasons.append("Safety violation: reported protected exposure flag was raised")
    if "human_review_preserved" in rows and not bool(rows["human_review_preserved"].all()):
        reasons.append("Reported human-review preservation flag was violated")
    for arm in ("control", "treatment"):
        count = int(((rows["arm"] == arm) & rows["observed"]).sum())
        if count < min_observed_per_arm:
            reasons.append(f"Insufficient observed outcomes in {arm}: {count} < {min_observed_per_arm}")
    if estimate.ci_lower <= min_effect:
        reasons.append("Adjusted 95% CI does not exclude the minimum uplift threshold")
    if not reasons:
        reasons.append("Simulation passed predefined evidence and safety checks")
    return Decision(status="simulation_pass" if len(reasons) == 1 and reasons[0].startswith("Simulation passed") else "simulation_blocked", reasons=reasons)


def analyse(rows: pd.DataFrame) -> dict:
    """Produce a report containing both naive and bias-aware estimates."""
    estimates = {m: estimate_difference(rows, method=m).to_dict() for m in ("naive", "ipw", "ipw_cuped")}
    final = estimate_difference(rows, method="ipw_cuped")
    return {
        "experiment": "RiskOps A/B – assisted triage for eligible low/medium-risk reports",
        "data_provenance": "synthetic_only",
        "total_units": len(rows),
        "eligible_units": int(rows["eligible"].sum()),
        "protected_units_excluded": int((~rows["eligible"]).sum()),
        "observation_rate": {
            arm: round(float(rows.loc[rows["arm"] == arm, "observed"].mean()), 4)
            for arm in ("control", "treatment")
        },
        "estimates": estimates,
        "simulation_decision": review_experiment(rows, final).to_dict(),
        "routing_audit": audit_routing(rows),
        "weight_diagnostics": weight_diagnostics(rows),
        "limitations": [
            "Synthetic outcomes and injected treatment effect; no external validity",
            "IPW uses known synthetic observation propensities, not an estimated real-world model",
            "Missingness is MAR by construction; MNAR opt-outs would bias IPW",
            "Normal CIs do not handle continuous monitoring or multiple testing",
            "Safety checks are invariants, not measured proof of real-world safety",
            "No real deployment or launch is authorised by this report",
        ],
    }
