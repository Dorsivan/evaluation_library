from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.genai.evaluation.entities import EvaluationResult

from llm_evaluation._config import get_config
from llm_evaluation._excel import write_results_to_excel
from llm_evaluation._judges import create_judge


def _make_http_predict(
    url: str,
    body: dict[str, Any] | None = None,
    api_key: str | None = None,
    response_key: str | None = None,
    timeout: int = 120,
) -> Callable[..., Any]:
    def predict(**kwargs: Any) -> Any:
        request_body = {**(body or {}), **kwargs}
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.post(url, json=request_body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        if response_key:
            return result[response_key]
        return result

    return predict


def evaluate(
    data: Any,
    scorers: list[Any] | None = None,
    predict_fn: Callable[..., Any] | None = None,
    predict_url: str | None = None,
    predict_body: dict[str, Any] | None = None,
    predict_api_key: str | None = None,
    response_key: str | None = None,
    judge_prompt: str | None = None,
    judge_name: str = "llm_judge",
    model_id: str | None = None,
    excel_path: str | Path | None = None,
    skip_excel: bool = False,
    log_excel_to_mlflow: bool = True,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> EvaluationResult:
    """Run ``mlflow.genai.evaluate`` and export an Excel report.

    ``run_name``/``tags`` are applied to the evaluation run. When
    ``log_excel_to_mlflow`` is true the Excel report is attached to that run
    as an artifact (under ``reports/``).
    """
    if predict_fn is None and predict_url is not None:
        predict_fn = _make_http_predict(
            url=predict_url,
            body=predict_body,
            api_key=predict_api_key,
            response_key=response_key,
        )

    all_scorers = list(scorers or [])

    if judge_prompt is not None:
        all_scorers.append(create_judge(name=judge_name, instructions=judge_prompt))

    get_config()  # make sure tracking URI / workspace / experiment are set

    if mlflow.active_run() is not None:
        # Caller manages the run (e.g. nested in a pipeline step).
        if tags:
            mlflow.set_tags(tags)
        result = mlflow.genai.evaluate(
            data=data, scorers=all_scorers, predict_fn=predict_fn, model_id=model_id
        )
    else:
        with mlflow.start_run(run_name=run_name, tags=tags):
            result = mlflow.genai.evaluate(
                data=data, scorers=all_scorers, predict_fn=predict_fn, model_id=model_id
            )

    if not skip_excel and result.result_df is not None:
        cfg = get_config()
        if excel_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = cfg.excel_filename or f"eval_results_{timestamp}.xlsx"
            excel_path = Path(cfg.excel_output_dir) / filename

        excel_path = write_results_to_excel(result.result_df, result.metrics, excel_path)
        print(f"Results exported to {excel_path}")

        if log_excel_to_mlflow and result.run_id:
            MlflowClient().log_artifact(result.run_id, str(excel_path), artifact_path="reports")
            print(f"Excel report logged to MLflow run {result.run_id} (reports/)")

    return result
