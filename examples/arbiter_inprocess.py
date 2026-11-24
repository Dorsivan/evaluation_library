#!/usr/bin/env python3
"""Evaluate arbiter *in-process* so each evaluation row carries the full
arbiter trace (k8s_manifest_pipeline -> analyze/generate/review -> LLM calls).

Requires arbiter (branch ``workshop``) importable, e.g.::

    pip install "git+https://github.com/Dorsivan/arbiter@workshop"   # or PYTHONPATH=/path/to/arbiter

Environment: OPENAI_API_BASE / OPENAI_API_KEY / MODEL_NAME for arbiter,
JUDGE_MODEL_URL / JUDGE_MODEL_NAME / JUDGE_API_KEY for the judges,
MLFLOW_TRACKING_URI / MLFLOW_WORKSPACE, ARBITER_PROMPT_ALIAS (production|staging).
"""

import os
import sys

from llm_evaluation import (
    Correctness,
    RelevanceToQuery,
    Safety,
    configure,
)

sys.path.insert(0, os.path.dirname(__file__))
from k8s_evaluation import (  # noqa: E402  (reuse dataset + scorers)
    build_dataset,
    has_deployment,
    has_health_probes,
    has_resource_limits,
    has_security_context,
    has_service,
    is_valid_yaml,
    uses_correct_namespace,
)

try:
    import mlflow.openai
    from app.models import GenerateRequest
    from app.pipeline import run_pipeline
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "arbiter is not importable. Install the arbiter 'workshop' branch or set PYTHONPATH.\n"
        f"Original error: {exc}"
    )


def predict_fn(description: str) -> str:
    response = run_pipeline(
        GenerateRequest(description=description, namespace="evaluation", replicas=2)
    )
    return response.final_manifests


def main():
    alias = os.environ.get("ARBITER_PROMPT_ALIAS", "production")
    cfg = configure(experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "arbiter-eval"))
    mlflow.openai.autolog()

    from llm_evaluation import evaluate

    judge_kwargs = dict(model=cfg.judge_model_uri, extra_headers=cfg.judge_extra_headers)
    scorers = [
        is_valid_yaml,
        has_deployment,
        has_service,
        has_resource_limits,
        has_security_context,
        has_health_probes,
        uses_correct_namespace,
        Correctness(**judge_kwargs),
        Safety(**judge_kwargs),
        RelevanceToQuery(**judge_kwargs),
    ]
    results = evaluate(
        data=build_dataset(),
        predict_fn=predict_fn,
        scorers=scorers,
        run_name=f"arbiter-{alias}",
        tags={"arbiter.prompt_alias": alias, "predict": "inprocess"},
    )
    print(results.metrics)


if __name__ == "__main__":
    main()
