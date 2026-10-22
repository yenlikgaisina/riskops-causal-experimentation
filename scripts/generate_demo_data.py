"""Generate standalone templates; not derived from people or employer records."""
import csv
import random
from pathlib import Path

TEMPLATES = [
    ("Policy confusion", "Low"), ("Policy confusion", "Medium"),
    ("Harassment", "Medium"), ("Platform integrity", "Medium"),
    ("Misinformation", "Medium"), ("Account abuse", "Medium"),
    ("Scam or fraud", "High"), ("Child safety concern", "Critical"),
    ("Self-harm concern", "High"), ("Privacy concern", "Medium"),
    ("Billing safety escalation", "High"), ("Platform integrity", "High"),
]
REGIONS = ["UK", "Europe", "North America", "Global / Unknown"]
LANGUAGES = ["English", "French", "Arabic", "Other"]
CHANNELS = ["Web form", "Mobile app", "Email support"]


def generate(path: Path, n: int = 600, seed: int = 101) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ticket_id", "risk_category", "severity", "region", "channel",
            "language", "model_confidence", "human_review_required",
        ])
        writer.writeheader()
        for i in range(n):
            category, severity = rng.choice(TEMPLATES)
            confidence = round(rng.uniform(0.62, 0.98), 2)
            requires_review = (
                severity in ("High", "Critical") or confidence < 0.75
                or category in {"Scam or fraud", "Child safety concern", "Self-harm concern", "Privacy concern", "Billing safety escalation"}
            )
            writer.writerow({
                "ticket_id": f"DEMO-{i:04d}", "risk_category": category,
                "severity": severity, "region": rng.choice(REGIONS),
                "channel": rng.choice(CHANNELS), "language": rng.choice(LANGUAGES),
                "model_confidence": confidence, "human_review_required": requires_review,
            })


if __name__ == "__main__":
    generate(Path("data/synthetic_user_reports.csv"))
