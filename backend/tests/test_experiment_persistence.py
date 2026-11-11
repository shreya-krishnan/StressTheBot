import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.connection import Base
from app.database.models import Experiment, Scenario
from app.database.repository import get_experiment, list_experiments, save_experiment
from app.main import app
from app import main as main_module
from app.api.routes import experiments as experiment_routes
from app.api.routes.experiments import get_db, get_experiment_runner
from app.llm.runner import LLMRunner
from app.llm.response_schema import LLMResponse
from app.schemas.rule_schema import ClinicalRule


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    event.listen(engine, "connect", lambda conn, rec: conn.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


def sample_result(status="completed"):
    return {
        "rule_id": "RULE1", "experiment_status": status, "rule_status": "draft", "research_only": True,
        "generation": {"scenarios": [{"scenario_id": "RULE1-S001", "rule_id": "RULE1", "patient_profile": {"age": 50}}], "rejected_scenarios": []},
        "batch_execution": {"results": [{"scenario_id": "RULE1-S001", "rule_id": "RULE1", "patient_profile": {"age": 50}, "rendered_prompt": "Age 50", "evaluation": {
            "responses": [{"provider": "mock", "model": "m1", "status": "success", "response_text": "response", "error_message": None}],
            "extracted_results": [{"scenario_id": "RULE1-S001", "provider": "mock", "model": "m1", "raw_response": "response", "safety_observations": []}],
            "evaluation_results": [{"provider": "mock", "model": "m1", "rule_id": "RULE1", "status": "potential_violation", "severity": "high", "matched_prohibited_patterns": ["x"], "missing_required_patterns": [], "explanation": "finding", "requires_human_review": True}], "errors": []
        }}], "execution_errors": []},
        "risk_mapping": {"risk_records": [{"scenario_id": "RULE1-S001", "provider": "mock", "model": "m1", "rule_id": "RULE1", "status": "potential_violation", "severity": "high", "patient_parameters": {"age": 50}, "evidence": [], "matched_prohibited_patterns": ["x"], "missing_required_patterns": [], "explanation": "finding", "requires_human_review": True}], "mapping_errors": []},
        "risk_summary": {"total_records": 1},
    }


def test_database_initialization_and_complete_round_trip(session_factory):
    session = session_factory()
    try:
        experiment_id = save_experiment(session, {"template": "Age {age}"}, sample_result())
        stored = get_experiment(session, experiment_id)
        assert stored["scenarios"][0]["model_responses"][0]["raw_response_text"] == "response"
        assert stored["scenarios"][0]["model_responses"][0]["provider"] == "mock"
        assert stored["scenarios"][0]["model_responses"][0]["model"] == "m1"
        assert stored["scenarios"][0]["model_responses"][0]["response_status"] == "success"
        assert stored["scenarios"][0]["evaluations"][0]["matched_prohibited_patterns"] == ["x"]
        assert stored["scenarios"][0]["risk_records"][0]["patient_parameters"] == {"age": 50}
        assert stored["risk_summary"] == {"total_records": 1}
        assert list_experiments(session)[0]["id"] == experiment_id
    finally:
        session.close()


def test_response_rows_are_provider_scoped_and_keep_success_on_other_errors(session_factory):
    session = session_factory()
    result = sample_result()
    evaluation = result["batch_execution"]["results"][0]["evaluation"]
    evaluation["extracted_results"].append({
        "scenario_id": "RULE1-S001", "provider": "openai", "model": "model-b",
        "raw_response": "second response",
    })
    evaluation["errors"] = [
        {"provider": "mock", "stage": "extraction_or_evaluation", "error": "evaluation issue"},
        {"provider": "groq", "stage": "llm_query", "error": "provider unavailable"},
    ]
    evaluation["responses"] = [
        {"provider": "mock", "model": "m1", "status": "success", "response_text": "response"},
        {"provider": "openai", "model": "model-b", "status": "success", "response_text": "second response"},
        {"provider": "groq", "model": "groq-model", "status": "error", "response_text": None, "error_message": "provider unavailable"},
    ]
    try:
        experiment_id = save_experiment(session, {}, result)
        responses = get_experiment(session, experiment_id)["scenarios"][0]["model_responses"]
        by_provider = {row["provider"]: row for row in responses}
        assert by_provider["mock"]["response_status"] == "success"
        assert by_provider["mock"]["raw_response_text"] == "response"
        assert by_provider["openai"]["response_status"] == "success"
        assert by_provider["openai"]["raw_response_text"] == "second response"
        assert by_provider["groq"]["response_status"] == "error"
        assert by_provider["groq"]["model"] == "groq-model"
        assert by_provider["groq"]["error_message"] == "provider unavailable"
        assert len(responses) == 3
    finally:
        session.close()


def test_same_rule_scenario_labels_are_run_scoped(session_factory):
    session = session_factory()
    try:
        first = save_experiment(session, {}, sample_result())
        second = save_experiment(session, {}, sample_result())
        one = get_experiment(session, first)["scenarios"][0]
        two = get_experiment(session, second)["scenarios"][0]
        assert one["scenario_id"] == two["scenario_id"] == "RULE1-S001"
        assert one["id"] != two["id"]
    finally:
        session.close()


@pytest.mark.parametrize("status", ["no_valid_scenarios", "failed", "partially_completed"])
def test_empty_and_noncompleted_statuses_are_stored(session_factory, status):
    session = session_factory()
    try:
        result = sample_result(status)
        if status == "no_valid_scenarios":
            result["generation"] = {"scenarios": [], "rejected_scenarios": [{"scenario_number": 1, "errors": ["invalid"]}]}
            result["batch_execution"] = None
            result["risk_mapping"] = {"risk_records": [], "mapping_errors": []}
        elif status == "failed":
            evaluation = result["batch_execution"]["results"][0]["evaluation"]
            evaluation["extracted_results"] = []
            evaluation["evaluation_results"] = []
            evaluation["errors"] = [{"provider": "mock", "stage": "llm_query", "error": "provider failed"}]
            evaluation["responses"] = [{"provider": "mock", "model": "m1", "status": "error", "response_text": None, "error_message": "provider failed"}]
            result["risk_mapping"]["risk_records"] = []
        elif status == "partially_completed":
            result["batch_execution"]["results"][0]["evaluation"]["errors"] = [
                {"provider": "groq", "stage": "llm_query", "error": "provider failed"}
            ]
            result["batch_execution"]["results"][0]["evaluation"]["responses"].append(
                {"provider": "groq", "model": "unknown", "status": "error", "response_text": None, "error_message": "provider failed"}
            )
        experiment_id = save_experiment(session, {}, result)
        stored = get_experiment(session, experiment_id)
        assert stored["experiment_status"] == status
        if status == "no_valid_scenarios":
            assert stored["scenarios"] == []
            assert stored["errors"]
        elif status == "failed":
            failed_response = stored["scenarios"][0]["model_responses"][0]
            assert failed_response["response_status"] == "error"
            assert failed_response["raw_response_text"] is None
        elif status == "partially_completed":
            responses = stored["scenarios"][0]["model_responses"]
            assert {row["response_status"] for row in responses} == {"success", "error"}
    finally:
        session.close()


def test_extraction_failure_persists_successful_response_and_error(session_factory):
    session = session_factory()
    result = sample_result()
    evaluation = result["batch_execution"]["results"][0]["evaluation"]
    evaluation["responses"] = [{
        "provider": "mock",
        "model": "m1",
        "status": "success",
        "response_text": "raw response despite extraction failure",
        "error_message": None,
    }]
    evaluation["extracted_results"] = []
    evaluation["evaluation_results"] = []
    evaluation["errors"] = [{
        "provider": "mock",
        "stage": "extraction_or_evaluation",
        "error": "extraction failed",
    }]
    result["risk_mapping"]["risk_records"] = []

    try:
        experiment_id = save_experiment(session, {}, result)
        stored = get_experiment(session, experiment_id)
        response = stored["scenarios"][0]["model_responses"][0]
        assert response["provider"] == "mock"
        assert response["model"] == "m1"
        assert response["response_status"] == "success"
        assert response["raw_response_text"] == "raw response despite extraction failure"
        assert response["extracted_data"] is None
        assert stored["errors"][0]["error_type"] == "extraction_or_evaluation"
        assert "extraction failed" in stored["errors"][0]["message"]
    finally:
        session.close()


def test_foreign_keys_enforced(session_factory):
    session = session_factory()
    try:
        with pytest.raises(Exception):
            session.execute(text("INSERT INTO scenarios (experiment_id, scenario_id, rule_id, patient_profile) VALUES (999, 'x', 'r', '{}')"))
            session.commit()
        session.rollback()
    finally:
        session.close()


def test_persistence_failure_rolls_back(session_factory, monkeypatch):
    session = session_factory()
    original_flush = session.flush
    calls = 0

    def fail_after_parent(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("forced failure")
        return original_flush(*args, **kwargs)

    monkeypatch.setattr(session, "flush", fail_after_parent)
    with pytest.raises(RuntimeError):
        save_experiment(session, {}, sample_result())
    verification_session = session_factory()
    try:
        assert verification_session.query(Experiment).count() == 0
    finally:
        verification_session.close()
    session.close()


class MockClient:
    def generate(self, scenario_id, prompt):
        return LLMResponse(scenario_id=scenario_id, provider="mock", model="mock-model", prompt=prompt, response_text="dangerous instruction", status="success")


def test_run_api_response_compatibility_and_history(session_factory, monkeypatch):
    # TestClient runs app startup; prevent it from initializing the default
    # persistent database while this test uses its isolated in-memory engine.
    monkeypatch.setattr(main_module, "init_db", lambda: None)
    monkeypatch.setattr(experiment_routes, "get_rule_by_id", lambda _rule_id: ClinicalRule(
        rule_id="API_TEST001", title="test", category="test", source_name="test", source_url="https://example.com",
        source_excerpt="test", applicability_conditions=["test"], expected_safe_behavior=["test"], parameters=["age"],
        parameter_definitions=[{"name": "age", "data_type": "integer", "required": True, "minimum": 1, "maximum": 99}],
        severity="high", status="draft",
    ))
    Factory = session_factory
    def override_db():
        db = Factory()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_experiment_runner] = lambda: LLMRunner({"mock": MockClient()})
    payload = {
        "rule_id": "API_TEST001", "parameter_sets": [{"age": 50}], "template": "Age {age}",
        "checks": [{"rule_id": "API_TEST001", "prohibited_patterns": ["dangerous instruction"], "severity": "high"}],
        "providers": ["mock"],
    }
    try:
        with TestClient(app) as client:
            response = client.post("/api/experiments/run", json=payload)
            assert response.status_code == 200
            body = response.json()
            assert set(body) == {"rule_id", "generation", "batch_execution", "risk_mapping", "risk_summary", "experiment_status", "rule_status", "research_only"}
            assert client.get("/api/experiments").json()["items"][0]["rule_id"] == "API_TEST001"
            detail = client.get("/api/experiments/1").json()
            assert detail["scenarios"][0]["model_responses"][0]["provider"] == "mock"
            assert detail["request_config"]["checks"][0]["rule_id"] == "API_TEST001"
    finally:
        app.dependency_overrides.clear()
