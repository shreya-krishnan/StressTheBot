
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import experiments as experiment_routes
from app.api.routes.experiments import get_experiment_runner
from app.llm.runner import LLMRunner
from app.llm.response_schema import LLMResponse
from app.schemas.rule_schema import ClinicalRule


class MockLLMClient:
    def generate(self, scenario_id: str, prompt: str) -> LLMResponse:
        return LLMResponse(
            scenario_id=scenario_id,
            provider="mock",
            model="mock-model",
            prompt=prompt,
            response_text="This is a dangerous instruction.",
            status="success"
        )


def mock_rule():
    return ClinicalRule(
        rule_id="API_TEST001",
        title="API software test rule",
        category="test",
        source_name="Software Testing Placeholder",
        source_url="https://example.com/test",
        source_excerpt="Placeholder content.",

        applicability_conditions=[
            "Software testing only"
        ],
        expected_safe_behavior=[
            "Avoid the prohibited pattern"
        ],

        parameters=["age"],

        parameter_definitions=[
            {
                "name": "age",
                "data_type": "integer",
                "description": "Synthetic age used for API testing",
                "unit": "years",
                "required": True,
                "minimum": 18,
                "maximum": 100,
                "allowed_values": []
            }
        ],

        severity="high",
        severity_rationale="Placeholder severity for API testing.",
        status="approved",
        reviewed_by="Test Reviewer",
        reviewed_at="2026-11-03T10:00:00",
        review_notes="Approved for software testing only"
    )


def test_experiment_api_success(monkeypatch):
    monkeypatch.setattr(
        experiment_routes,
        "get_rule_by_id",
        lambda rule_id: mock_rule()
    )

    runner = LLMRunner({
        "mock": MockLLMClient()
    })

    app.dependency_overrides[get_experiment_runner] = lambda: runner

    payload = {
        "rule_id": "API_TEST001",
        "parameter_sets": [
            {"age": 45}
        ],
        "template": "Synthetic test profile. Age: {age}",
        "checks": [
            {
                "rule_id": "API_TEST001",
                "prohibited_patterns": ["dangerous instruction"],
                "required_patterns": [],
                "applicable": True,
                "severity": "high"
            }
        ],
        "providers": ["mock"]
    }

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/experiments/run",
                json=payload
            )

            assert response.status_code == 200

            data = response.json()

            assert data["experiment_status"] == "completed"
            assert data["risk_mapping"]["total_risk_records"] == 1

            assert (
                data["risk_mapping"]["risk_records"][0]["status"]
                == "potential_violation"
            )

    finally:
        app.dependency_overrides.clear()


def test_experiment_api_rejects_unapproved_rule(monkeypatch):
    draft_rule = mock_rule().model_copy(
        update={"status": "draft"}
    )

    monkeypatch.setattr(
        experiment_routes,
        "get_rule_by_id",
        lambda rule_id: draft_rule
    )

    runner = LLMRunner({
        "mock": MockLLMClient()
    })

    app.dependency_overrides[get_experiment_runner] = lambda: runner

    payload = {
        "rule_id": "API_TEST001",
        "parameter_sets": [{"age": 45}],
        "template": "Age: {age}",
        "checks": [
            {
                "rule_id": "API_TEST001",
                "prohibited_patterns": [],
                "required_patterns": [],
                "applicable": True,
                "severity": "high"
            }
        ]
    }

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/experiments/run",
                json=payload
            )

            assert response.status_code == 200

            data = response.json()

            assert data["rule_status"] == "draft"
            assert data["research_only"] is True

    finally:
        app.dependency_overrides.clear()
