from app.llm.base import BaseLLMClient
from app.llm.runner import LLMRunner
from app.llm.response_schema import LLMResponse

from app.evaluation.evaluation_schema import EvaluationCheck
from app.evaluation.pipeline import run_evaluation_pipeline


class MockLLMClient(BaseLLMClient):

    def generate(self, scenario_id, prompt):
        return LLMResponse(
            scenario_id=scenario_id,
            provider="mock",
            model="mock-model",
            prompt=prompt,
            response_text=(
                "Monitor renal function regularly. "
                "The medication should be used despite the warning."
            ),
            status="success"
        )


class FailingLLMClient(BaseLLMClient):

    def generate(self, scenario_id, prompt):
        raise RuntimeError("Simulated API failure")


def test_complete_evaluation_pipeline():

    runner = LLMRunner({
        "mock": MockLLMClient()
    })

    checks = [
        EvaluationCheck(
            rule_id="TEST001",
            prohibited_patterns=["used despite the warning"],
            severity="high"
        ),
        EvaluationCheck(
            rule_id="TEST002",
            required_patterns=["monitor renal function"],
            severity="medium"
        )
    ]

    result = run_evaluation_pipeline(
        runner=runner,
        scenario_id="TEST001-S001",
        prompt="Synthetic patient scenario",
        checks=checks
    )

    assert result["total_responses"] == 1
    assert result["total_extracted"] == 1
    assert result["total_evaluations"] == 2
    assert result["total_errors"] == 0
    assert result["responses"] == [{
        "provider": "mock",
        "model": "mock-model",
        "status": "success",
        "response_text": (
            "Monitor renal function regularly. "
            "The medication should be used despite the warning."
        ),
        "error_message": None,
    }]

    statuses = [
        item["status"]
        for item in result["evaluation_results"]
    ]

    assert statuses == [
        "potential_violation",
        "no_detected_issue"
    ]


def test_pipeline_handles_provider_failure():

    runner = LLMRunner({
        "mock": FailingLLMClient()
    })

    result = run_evaluation_pipeline(
        runner=runner,
        scenario_id="TEST001-S001",
        prompt="Synthetic patient scenario",
        checks=[]
    )

    assert result["total_responses"] == 1
    assert result["total_extracted"] == 0
    assert result["total_evaluations"] == 0
    assert result["total_errors"] == 1

    assert result["errors"][0]["stage"] == "llm_query"
    assert result["responses"] == [{
        "provider": "mock",
        "model": "unknown",
        "status": "error",
        "response_text": None,
        "error_message": "Simulated API failure",
    }]


def test_pipeline_preserves_successful_response_when_extraction_fails(monkeypatch):
    runner = LLMRunner({"mock": MockLLMClient()})

    def fail_extraction(_response):
        raise RuntimeError("Simulated extraction failure")

    monkeypatch.setattr("app.evaluation.pipeline.extract_response", fail_extraction)

    result = run_evaluation_pipeline(
        runner=runner,
        scenario_id="TEST001-S001",
        prompt="Synthetic patient scenario",
        checks=[],
    )

    assert result["total_responses"] == 1
    assert result["total_extracted"] == 0
    assert result["responses"][0]["provider"] == "mock"
    assert result["responses"][0]["model"] == "mock-model"
    assert result["responses"][0]["status"] == "success"
    assert result["responses"][0]["response_text"].startswith("Monitor renal")
    assert result["errors"] == [{
        "provider": "mock",
        "stage": "extraction_or_evaluation",
        "error": "Simulated extraction failure",
    }]
