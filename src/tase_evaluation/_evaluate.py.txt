from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
import mlflow
from mlflow.genai.evaluation.entities import EvaluationResult

from tase_evaluation._config import get_config
from tase_evaluation._excel import write_results_to_excel
from tase_evaluation._judges import create_judge


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
) -> EvaluationResult:
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

    result = mlflow.genai.evaluate(
        data=data,
        scorers=all_scorers,
        predict_fn=predict_fn,
        model_id=model_id,
    )

    if not skip_excel and result.result_df is not None:
        cfg = get_config()
        if excel_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = cfg.excel_filename or f"eval_results_{timestamp}.xlsx"
            excel_path = Path(cfg.excel_output_dir) / filename

        excel_path = write_results_to_excel(result.result_df, result.metrics, excel_path)
        print(f"Results exported to {excel_path}")

    return result
