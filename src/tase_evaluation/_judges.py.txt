from __future__ import annotations

from typing import Any

from mlflow.genai.judges import make_judge

from tase_evaluation._config import get_config


def create_judge(
    name: str,
    instructions: str,
    feedback_value_type: Any = bool,
    description: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
    inference_params: dict[str, Any] | None = None,
    generate_rationale_first: bool = False,
):
    cfg = get_config()

    return make_judge(
        name=name,
        instructions=instructions,
        model=model or cfg.judge_model_uri,
        base_url=base_url or cfg.judge_model_url,
        extra_headers=extra_headers if extra_headers is not None else cfg.judge_extra_headers,
        description=description,
        feedback_value_type=feedback_value_type,
        inference_params=inference_params,
        generate_rationale_first=generate_rationale_first,
    )
