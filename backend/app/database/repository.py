import json
from datetime import timezone

from sqlalchemy import desc

from app.database.models import Experiment, Scenario, ModelResponse, Evaluation, RiskRecord, ExecutionError


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def save_experiment(session, request_config, result):
    """Persist a finished pipeline result atomically; caller owns the session."""
    try:
        experiment = Experiment(
            rule_id=result["rule_id"],
            experiment_status=result["experiment_status"],
            rule_status=result["rule_status"],
            research_only=result["research_only"],
            request_config=json_text(request_config),
            risk_summary=json_text(result.get("risk_summary", {})),
        )
        session.add(experiment)
        session.flush()

        scenario_rows = {}
        generated = result.get("generation", {})
        batch = result.get("batch_execution") or {}
        batch_results = {row.get("scenario_id"): row for row in batch.get("results", [])}
        for item in generated.get("scenarios", []):
            row = batch_results.get(item["scenario_id"], {})
            scenario = Scenario(
                experiment_id=experiment.id,
                scenario_id=item["scenario_id"],
                rule_id=item.get("rule_id", result["rule_id"]),
                patient_profile=json_text(item.get("patient_profile", {})),
                rendered_prompt=row.get("rendered_prompt"),
            )
            session.add(scenario)
            session.flush()
            scenario_rows[item["scenario_id"]] = scenario

            evaluation_data = row.get("evaluation", {})
            extracted_by_model = {
                (item.get("provider"), item.get("model")): item
                for item in evaluation_data.get("extracted_results", [])
            }
            # Only llm_query errors describe a failed model response. Errors
            # from extraction/evaluation are persisted below as execution
            # errors and must not change a successful response's status.
            errors_by_provider = {
                error.get("provider"): error
                for error in evaluation_data.get("errors", [])
                if error.get("stage") == "llm_query"
            }
            if "responses" in evaluation_data:
                response_records = evaluation_data["responses"]
            else:
                # Backward compatibility for results created before the
                # authoritative response list was added.
                response_records = [
                    {
                        "provider": provider,
                        "model": model,
                        "status": "success",
                        "response_text": extracted.get("raw_response"),
                    }
                    for (provider, model), extracted in extracted_by_model.items()
                ]
                response_records.extend(
                    {
                        "provider": provider,
                        "model": None,
                        "status": "error",
                    }
                    for provider in errors_by_provider
                )

            for response in response_records:
                provider = response.get("provider")
                model = response.get("model")
                extracted = extracted_by_model.get((provider, model))
                error = errors_by_provider.get(provider)
                response_status = response.get("status")
                error_message = response.get("error_message")
                if response_status == "error" and not error_message and error:
                    error_message = error.get("error")
                session.add(ModelResponse(
                    scenario_id=scenario.id,
                    provider=provider or "unknown",
                    model=model,
                    prompt=row.get("rendered_prompt"),
                    raw_response_text=(
                        response.get("response_text")
                        if response.get("response_text") is not None
                        else (extracted or {}).get("raw_response")
                    ),
                    response_status=response_status,
                    error_message=error_message,
                    extracted_data=json_text(extracted) if extracted else None,
                ))

            for item in evaluation_data.get("evaluation_results", []):
                session.add(Evaluation(
                    scenario_id=scenario.id,
                    provider=item["provider"], model=item["model"], rule_id=item["rule_id"],
                    status=item["status"], severity=item["severity"],
                    matched_prohibited_patterns=json_text(item.get("matched_prohibited_patterns", [])),
                    missing_required_patterns=json_text(item.get("missing_required_patterns", [])),
                    explanation=item["explanation"],
                    requires_human_review=item.get("requires_human_review", False),
                ))

        for item in result.get("risk_mapping", {}).get("risk_records", []):
            scenario = scenario_rows.get(item.get("scenario_id"))
            if scenario is None:
                continue
            session.add(RiskRecord(
                scenario_id=scenario.id,
                provider=item["provider"], model=item["model"], rule_id=item["rule_id"],
                status=item["status"], severity=item["severity"],
                patient_parameters=json_text(item.get("patient_parameters", {})),
                evidence=json_text(item.get("evidence", [])),
                matched_prohibited_patterns=json_text(item.get("matched_prohibited_patterns", [])),
                missing_required_patterns=json_text(item.get("missing_required_patterns", [])),
                explanation=item["explanation"],
                requires_human_review=item.get("requires_human_review", True),
            ))

        for error in batch.get("execution_errors", []):
            session.add(ExecutionError(
                experiment_id=experiment.id,
                scenario_label=error.get("scenario_id"),
                error_type="scenario_execution",
                message=error.get("error", "Unknown execution error"),
            ))
        for error in result.get("generation", {}).get("rejected_scenarios", []):
            session.add(ExecutionError(
                experiment_id=experiment.id,
                scenario_label=str(error.get("scenario_number", "")) or None,
                error_type="scenario_generation",
                message="; ".join(error.get("errors", [])),
            ))
        for scenario_row in batch.get("results", []):
            for error in scenario_row.get("evaluation", {}).get("errors", []):
                session.add(ExecutionError(
                    experiment_id=experiment.id,
                    scenario_label=scenario_row.get("scenario_id"),
                    error_type=error.get("stage", "evaluation"),
                    message=f"{error.get('provider', 'unknown')}: {error.get('error', '')}",
                ))
        for error in result.get("risk_mapping", {}).get("mapping_errors", []):
            session.add(ExecutionError(
                experiment_id=experiment.id,
                scenario_label=error.get("scenario_id"),
                error_type="risk_mapping",
                message=error.get("error", "Unknown mapping error"),
            ))
        session.commit()
        return experiment.id
    except Exception:
        session.rollback()
        raise


def list_experiments(session, skip=0, limit=20):
    rows = session.query(Experiment).order_by(desc(Experiment.created_at), desc(Experiment.id)).offset(skip).limit(limit).all()
    return [serialize_experiment(row, include_children=False) for row in rows]


def get_experiment(session, experiment_id):
    row = session.query(Experiment).filter(Experiment.id == experiment_id).first()
    return serialize_experiment(row, include_children=True) if row else None


def serialize_experiment(row, include_children):
    if row is None:
        return None
    created_at = row.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    value = {
        "id": row.id, "rule_id": row.rule_id, "experiment_status": row.experiment_status,
        "rule_status": row.rule_status, "research_only": row.research_only,
        "created_at": created_at.isoformat(), "request_config": json.loads(row.request_config),
        "risk_summary": json.loads(row.risk_summary),
    }
    if include_children:
        value["scenarios"] = []
        for scenario in row.scenarios:
            value["scenarios"].append({
                "id": scenario.id, "scenario_id": scenario.scenario_id, "rule_id": scenario.rule_id,
                "patient_profile": json.loads(scenario.patient_profile), "rendered_prompt": scenario.rendered_prompt,
                "model_responses": [{
                    "id": r.id, "provider": r.provider, "model": r.model, "prompt": r.prompt,
                    "raw_response_text": r.raw_response_text, "response_status": r.response_status,
                    "error_message": r.error_message,
                    "extracted_data": json.loads(r.extracted_data) if r.extracted_data else None,
                } for r in scenario.model_responses],
                "evaluations": [{
                    "id": e.id, "provider": e.provider, "model": e.model, "rule_id": e.rule_id,
                    "status": e.status, "severity": e.severity,
                    "matched_prohibited_patterns": json.loads(e.matched_prohibited_patterns),
                    "missing_required_patterns": json.loads(e.missing_required_patterns),
                    "explanation": e.explanation, "requires_human_review": e.requires_human_review,
                } for e in scenario.evaluations],
                "risk_records": [{
                    "id": r.id, "provider": r.provider, "model": r.model, "rule_id": r.rule_id,
                    "status": r.status, "severity": r.severity,
                    "patient_parameters": json.loads(r.patient_parameters), "evidence": json.loads(r.evidence),
                    "matched_prohibited_patterns": json.loads(r.matched_prohibited_patterns),
                    "missing_required_patterns": json.loads(r.missing_required_patterns),
                    "explanation": r.explanation, "requires_human_review": r.requires_human_review,
                } for r in scenario.risk_records],
            })
        value["errors"] = [{
            "id": error.id, "scenario_label": error.scenario_label,
            "error_type": error.error_type, "message": error.message,
        } for error in row.errors]
    return value
