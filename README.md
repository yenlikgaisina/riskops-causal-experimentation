# RiskOps — Causal Experimentation & Decision Intelligence

[![Research-only](https://img.shields.io/badge/data-100%25_synthetic-50e0c0)](#data-and-ethics)

An end-to-end **portfolio engineering demonstration** combining randomised experiments, responsible machine learning, risk operations and reproducible deployment architecture. The system tests a proposed assisted-routing policy only on eligible lower-risk synthetic reports, measures SLA success, adjusts for missing feedback, monitors covariate drift, and enforces mandatory human review.

**[Live interactive demonstration](https://yenlikgaisina.github.io/riskops-causal-experimentation/)** (GitHub Pages; offline [demo report](demo/index.html)) · [System architecture and methods](experiment_lab/docs/ARCHITECTURE.md) · [ADR-001](experiment_lab/docs/ADR-001.md) · [Technical guide](experiment_lab/README.md)

## Quick start

```bash
python -m pip install -r experiment_lab/requirements.txt
python -m pytest experiment_lab/tests -q
python -m experiment_lab.cli --population 20000 --seed 42 --effect 0.05 --train-model
uvicorn experiment_lab.api:app --reload
```

On first use you can regenerate the entirely artificial template records:

```bash
python scripts/generate_demo_data.py
```

For the API: `POST /simulate` with `{"population":10000,"seed":42,"effect":0.05}`; `GET /health` provides status. The HTML report is regenerated at `demo/index.html` each CLI run. It includes an interactive browser-only **synthetic scenario explorer**, clearly separated from the fixed Python simulation report. No external services are called by the scenario controls.

## What to show recruiters

| Engineering area | Working implementation |
|---|---|
| Product analytics | Predefined randomised A/B experiment, interactive scenario explorer, primary SLA metric |
| Causal inference | IPW/Hájek missingness correction, CUPED, approximate uncertainty intervals, design-stage power calculation and weight diagnostics |
| ML lifecycle | Control-only offline logistic regression, weighted train/eval, version manifest and Brier/AUC |
| Responsible AI | Fail-closed exclusion for High/Critical, sensitive and mandated-review cases; independent audits of actual routing actions |
| MLOps | Stateless REST API, Docker build, unit tests, GitHub Actions, drift monitoring |
| Technical ownership | Architecture diagram, explicit assumptions, failure modes and roadmap |

## Data and ethics

**All reports and outcomes are synthetic.** A treatment effect is injected; it is not discovered in real product data. Inverse observation propensities are known because the simulator created them. `simulation_pass` does not approve a real rollout. No employer systems, case files, personal data or real incident reports are used. It is a credible engineering *prototype*, not production-scale experience or evidence of leading a team.

Created as a standalone extension inspired by a previous Trust & Safety Risk Triage portfolio case study. The project deliberately includes a minimal synthetic dataset generator so it can run in a new repository; it can also use the original parent's synthetic CSV schema without modifications.

**AI assistance disclosure:** Implementation and testing were AI-assisted (ChatGPT and Claude). The expanded local automated suite passes (17/17). Technical ownership claims should be backed by independent validation and the ability to explain and defend the design choices.

## Design-stage power planning

```python
from experiment_lab.power import sample_size_per_arm
print(sample_size_per_arm(baseline_rate=0.65, minimum_detectable_effect=0.05, anticipated_observation_rate=0.75))
```

This normal-approximation planner assumes a fixed horizon, independent outcomes and equal allocation. The conservative routing audit checks `route_action` against the original mandatory-review flag and policy eligibility, **not** a self-reported safety Boolean. The browser scenario explorer uses a separate seeded synthetic generator; its output is not a re-analysis of any operational data.
