
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.experiment.api_schema import ExperimentRequest
from app.experiment.experiment_pipeline import run_experiment
from app.llm.runner import LLMRunner
from app.llm.groq_client import GroqClient
from app.llm.openai_client import OpenAIClient
from app.rules.repository import get_rule_by_id
from app.database.connection import get_db
from app.database.repository import save_experiment, list_experiments, get_experiment


router = APIRouter(
    prefix="/api/experiments",
    tags=["Experiments"]
)


@router.get("")
def experiment_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return {"items": list_experiments(db, skip=skip, limit=limit), "skip": skip, "limit": limit}


@router.get("/{experiment_id}")
def experiment_detail(experiment_id: int, db: Session = Depends(get_db)):
    result = get_experiment(db, experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return result


def get_experiment_runner() -> LLMRunner:
    """Build a runner using configured LLM providers."""

    backend_dir = Path(__file__).resolve().parents[3]
    load_dotenv(backend_dir / ".env")

    clients = {}

    if os.getenv("GROQ_API_KEY"):
        clients["groq"] = GroqClient()

    if os.getenv("OPENAI_API_KEY"):
        clients["openai"] = OpenAIClient()

    if not clients:
        raise HTTPException(
            status_code=503,
            detail="No LLM provider is configured. Check backend/.env."
        )

    return LLMRunner(clients)


@router.post("/run")
def run_experiment_endpoint(
    request: ExperimentRequest,
    runner: LLMRunner = Depends(get_experiment_runner),
    db: Session = Depends(get_db),
):
    """Execute an experiment and return its results."""

    rule = get_rule_by_id(request.rule_id)

    if rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Clinical rule '{request.rule_id}' was not found."
        )

    for check in request.checks:
        if check.rule_id != rule.rule_id:
            raise HTTPException(
                status_code=400,
                detail="Every evaluation check must match the requested rule_id."
            )

    selected_providers = request.providers

    if selected_providers:
        unknown_providers = [
            provider
            for provider in selected_providers
            if provider not in runner.clients
        ]

        if unknown_providers:
            raise HTTPException(
                status_code=400,
                detail=f"Providers not configured: {unknown_providers}"
            )

    try:
        result = run_experiment(
            rule=rule,
            parameter_sets=request.parameter_sets,
            template=request.template,
            runner=runner,
            checks=request.checks,
            parameter_constraints=request.parameter_constraints,
            providers=selected_providers
        )

        result["rule_status"] = rule.status
        result["research_only"] = rule.status == "draft"

        try:
            save_experiment(
                db,
                request_config=request.model_dump(mode="json"),
                result=result,
            )
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail="Experiment completed but could not be persisted.",
            ) from error

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
