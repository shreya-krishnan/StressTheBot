from app.llm.runner import LLMRunner
from app.response_extractor.extractor import extract_response

from app.evaluation.evaluation_schema import EvaluationCheck
from app.evaluation.rule_evaluator import evaluate_response


def run_evaluation_pipeline(
    runner: LLMRunner,
    scenario_id: str,
    prompt: str,
    checks: list[EvaluationCheck],
    providers: list[str] | None = None
) -> dict:

    # Step 1: Query the selected LLM providers.
    responses = runner.run(
        scenario_id=scenario_id,
        prompt=prompt,
        providers=providers
    )

    extracted_results = []
    evaluation_results = []
    errors = []

    # Step 2: Process each model response.
    for response in responses:

        if response.status != "success":
            errors.append({
                "provider": response.provider,
                "stage": "llm_query",
                "error": response.error_message
            })
            continue

        try:
            # Step 3: Extract structured information.
            extracted = extract_response(response)

            extracted_results.append(extracted.model_dump())

            # Step 4: Evaluate against configured rules.
            for check in checks:
                result = evaluate_response(
                    response=extracted,
                    check=check
                )

                evaluation_results.append(result.model_dump())

        except Exception as error:
            errors.append({
                "provider": response.provider,
                "stage": "extraction_or_evaluation",
                "error": str(error)
            })

    return {
        "scenario_id": scenario_id,
        "total_responses": len(responses),
        "total_extracted": len(extracted_results),
        "total_evaluations": len(evaluation_results),
        "total_errors": len(errors),
        "responses": [
            {
                "provider": response.provider,
                "model": response.model,
                "status": response.status,
                "response_text": response.response_text,
                "error_message": response.error_message,
            }
            for response in responses
        ],
        "extracted_results": extracted_results,
        "evaluation_results": evaluation_results,
        "errors": errors
    }
