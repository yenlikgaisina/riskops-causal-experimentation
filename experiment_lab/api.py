"""Stateless example API; demo-only, not an operational risk triage service.

Run: uvicorn experiment_lab.api:app --reload
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core import simulate
from .decision import analyse

app = FastAPI(title="RiskOps Experimentation Lab", version="0.1.0", description="Synthetic-only experiment simulation; never authorises live launches")
SOURCE = Path(__file__).resolve().parents[1] / "data/synthetic_user_reports.csv"


class ExperimentRequest(BaseModel):
    population: int = Field(default=20_000, ge=100, le=100_000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    effect: float = Field(default=0.05, ge=0, le=0.20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "synthetic_only"}


@app.post("/simulate")
def run_experiment(req: ExperimentRequest) -> dict:
    if not SOURCE.is_file():
        raise HTTPException(status_code=503, detail="Synthetic source dataset is unavailable")
    try:
        return analyse(simulate(SOURCE, req.population, req.seed, req.effect))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
