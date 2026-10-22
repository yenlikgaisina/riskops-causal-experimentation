# RiskOps Experimentation Lab

**A synthetic, safety-gated A/B experiment for trust-and-safety triage.**

This feature adds a measurable decision system to [Trust & Safety Risk Triage](../README.md). It asks whether a proposed assisted-routing workflow improves the share of *eligible* reports resolved within SLA, while retaining mandatory human review. It is a portfolio demonstration of experimentation architecture, causal inference, monitoring decisions and responsible AI.

**Important:** all data and effects are synthetic. The simulation inserts its own treatment effect. A `simulation_pass` is **not** proof of an effective workflow or permission to deploy any change.

## Architecture

`Synthetic CSV → eligibility & human-review gate → stable 50/50 randomisation → synthetic outcome + missing-feedback generation → naive/IPW/CUPED estimates → guardrail-gated experiment report.`

See [architecture, estimand, limitations, and roadmap](docs/ARCHITECTURE.md).

## What is implemented

- Stable SHA-256 unit assignment; no rollout to High/Critical, sensitive, or low-confidence reports.
- Reproducible event generation from this repository's synthetic tickets; explicit testable data schema.
- Naive observed mean, inverse-probability-weighted Hájek estimate, CUPED covariate adjustment and approximate 95% CIs.
- Feedback-selection mechanism depending on arm and observed attributes; true propensities available in **simulation only**.
- Conservative review gate that blocks on exposure of protected cases, changes to required human review, insufficient observations, or lack of statistical evidence.
- Offline SLA-success model training (scikit-learn, weighted training, held-out Brier/AUC, versioned artifact manifest).
- Population-stability monitoring diagnostics by region, language, channel and severity.
- Stateless FastAPI example, Dockerfile, pytest coverage, GitHub Actions CI, architecture/design notes.

## Run from the **repository root**

```bash
python -m pip install -r experiment_lab/requirements.txt
python -m pytest experiment_lab/tests -q
python -m experiment_lab.cli --population 20000 --seed 42 --effect 0.05 --train-model
```

The CLI writes a locally versioned model and metadata to `experiment_lab/artifacts/` when `--train-model` is set. It also writes `experiment_lab/reports/latest.json` (generated files should stay out of version control). The injected `--effect 0.05` means **+5 percentage points in synthetic SLA success probability**, not a measured real-world result. Set `--effect 0` to see a null scenario.

### Example API

```bash
uvicorn experiment_lab.api:app --reload
curl -X POST http://127.0.0.1:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{"population": 10000, "seed": 42, "effect": 0.05}'
```

The API only generates synthetic experiments; it cannot ingest real cases or authorise a rollout.

### Docker

Run from repository root:

```bash
docker build -f experiment_lab/Dockerfile -t riskops-experiment-lab .
docker run --rm -p 8000:8000 riskops-experiment-lab
```

## How this builds on my portfolio

The original repo covers risk taxonomy, SLA framework, SQL diagnostics, simulated triage and human review. This addition evaluates **whether a process change makes a difference** and designs a consistent evidence gate for deciding whether simulated results pass. It joins my interests in analytical engineering, causal inference, risk management, technical ownership and responsible ML.

## Limitations, transparency and next steps

- No real event data, live traffic, production ML serving or production deployment. An offline synthetic predictive model is included for demonstration.
- Outcomes are generated, not observed; the missingness mechanism is MAR by design, not verified externally.
- The IPW weights are oracle synthetic propensities and are not a trained estimator.
- Approximate confidence intervals should not be reused for sequential peeking, clusters or multiple metrics.
- An independent production launch would require security/privacy review, a pre-registered protocol, validated real outcomes, engineering monitoring, and approved human safety controls.
- A roadmap towards model registry, adaptive designs, sequential inference, drift monitoring and staging infrastructure is in the architecture doc.

**Development note:** This module was created with AI-assisted implementation and review. Claims of independent technical design or leadership should be made only for decisions and validation I personally complete and can defend.
