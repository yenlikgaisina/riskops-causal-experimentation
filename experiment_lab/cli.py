"""Example: python -m experiment_lab.cli --source data/synthetic_user_reports.csv"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import simulate
from .decision import analyse
from .dashboard import render_dashboard
from .modeling import train_sla_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate and evaluate a safe triage-policy experiment")
    parser.add_argument("--source", type=Path, default=Path("data/synthetic_user_reports.csv"))
    parser.add_argument("--population", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--effect", type=float, default=0.05, help="Injected probability-point effect in synthetic data")
    parser.add_argument("--output", type=Path, default=Path("experiment_lab/reports/latest.json"))
    parser.add_argument("--dashboard", type=Path, default=Path("demo/index.html"))
    parser.add_argument("--train-model", action="store_true", help="Fit an offline synthetic SLA prediction model")
    args = parser.parse_args()
    rows = simulate(args.source, args.population, args.seed, args.effect)
    report = analyse(rows)
    if args.train_model:
        report["model_training"] = train_sla_baseline(rows, Path("experiment_lab/artifacts"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    render_dashboard(report, args.dashboard)
    print(json.dumps(report, indent=2))
    print(f"Report written to {args.output}")
    print(f"Dashboard written to {args.dashboard}")


if __name__ == "__main__":
    main()
