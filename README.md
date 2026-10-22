# RiskOps — Causal Experimentation & Decision Intelligence

[![Research-only](https://img.shields.io/badge/data-100%25_synthetic-50e0c0)](#data-and-ethics)

An end-to-end **portfolio engineering demonstration** combining randomised experiments, responsible machine learning, risk operations and reproducible deployment architecture. The system tests a proposed assisted-routing policy only on eligible lower-risk synthetic reports, measures SLA success, adjusts for missing feedback, monitors covariate drift, and enforces mandatory human review.

**[Live demo report](https://yenlikgaisina.github.io/riskops-causal-experimentation/)** (GitHub Pages; also viewable locally at `demo/index.html`) · [System architecture and methods](experiment_lab/docs/ARCHITECTURE.md) · [ADR-001](experiment_lab/docs/ADR-001.md) · [Technical guide](experiment_lab/README.md)

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

For the API: `POST /simulate` with `{"population":10000,"seed":42,"effect":0.05}`; `GET /health` provides status. The HTML demo is regenerated at `demo/index.html` each CLI run. The dashboard is self-contained and can be published with GitHub Pages.

## What to show recruiters

| Engineering area | Working implementation |
|---|---|
| Product analytics | Predefined randomised A/B experiment and primary SLA metric |
| Causal inference | IPW/Hájek missingness correction, CUPED and uncertainty intervals |
| ML lifecycle | Control-only offline logistic regression, weighted train/eval, version manifest and Brier/AUC |
| Responsible AI | Exclusion gates for High/Critical and sensitive cases; human review preserved |
| MLOps | Stateless REST API, Docker build, unit tests, GitHub Actions, drift monitoring |
| Technical ownership | Architecture diagram, explicit assumptions, failure modes and roadmap |

## Data and ethics

**All reports and outcomes are synthetic.** A treatment effect is injected; it is not discovered in real product data. Inverse observation propensities are known because the simulator created them. `simulation_pass` does not approve a real rollout. No employer systems, case files, personal data or real incident reports are used. It is a credible engineering *prototype*, not production-scale experience or evidence of leading a team.

Created as a standalone extension inspired by a previous Trust & Safety Risk Triage portfolio case study. The project deliberately includes a minimal synthetic dataset generator so it can run in a new repository; it can also use the original parent's synthetic CSV schema without modifications.

**AI assistance disclosure:** Implementation and initial tests were AI-assisted (ChatGPT and Claude). The automated suite passes (11/11); any claim of personal technical ownership should be backed by independent validation and the ability to explain the design choices. This is a personal portfolio project, not production or team-led engineering experience.
