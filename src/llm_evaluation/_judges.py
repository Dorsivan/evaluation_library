from __future__ import annotations

import inspect
from typing import Any

from mlflow.genai.judges import make_judge

from llm_evaluation._config import get_config

# MLflow versions differ: newer make_judge() accepts base_url / extra_headers /
# generate_rationale_first, while 3.10 (RHOAI 3.4) does not and instead reads the
# OpenAI-compatible endpoint from OPENAI_API_BASE / OPENAI_API_KEY (set by configure()).
_MAKE_JUDGE_PARAMS = set(inspect.signature(make_judge).parameters)


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
    kwargs: dict[str, Any] = {
        "name": name,
        "instructions": instructions,
        "model": model or cfg.judge_model_uri,
        "description": description,
        "feedback_value_type": feedback_value_type,
        "inference_params": inference_params,
        "base_url": base_url or cfg.judge_model_url,
        "extra_headers": extra_headers if extra_headers is not None else cfg.judge_extra_headers,
        "generate_rationale_first": generate_rationale_first,
    }
    return make_judge(**{k: v for k, v in kwargs.items() if k in _MAKE_JUDGE_PARAMS})
