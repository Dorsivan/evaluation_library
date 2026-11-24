"""Plain-Python helpers for evaluation pipelines (e.g. Kubeflow / Data Science
Pipelines components). They contain no KFP code so they can be unit tested and
reused from notebooks.

Typical flow::

    baseline = run_evaluation(dataset, predict_fn, scorers, run_name="production")
    candidate = run_evaluation(dataset, predict_fn2, scorers, run_name="staging")
    ok, reasons = compare_and_gate(candidate["metrics"], baseline["metrics"],
                                   thresholds={"is_valid_yaml/mean": 0.95},
                                   must_not_regress=["k8s_manifest_quality/mean"])
"""

from __future__ import annotations

from typing import Any, Callable

import mlflow

from llm_evaluation._config import configure, get_config
from llm_evaluation._evaluate import evaluate


def _load_dataset(dataset: Any) -> Any:
    """Accept an MLflow dataset object, a list of records, or a dataset name."""
    if isinstance(dataset, str):
        from mlflow.genai.datasets import search_datasets

        matches = [d for d in search_datasets() if d.name == dataset]
        if not matches:
            raise ValueError(f"MLflow dataset '{dataset}' not found")
        return matches[0]
    return dataset


def run_evaluation(
    dataset: Any,
    predict_fn: Callable[..., Any],
    scorers: list[Any],
    *,
    experiment_name: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    judge_prompt: str | None = None,
    excel_path: str | None = None,
    log_excel_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Evaluate ``predict_fn`` on ``dataset`` and return ``{"metrics", "run_id"}``.

    ``dataset`` may be an MLflow dataset, a list of records, or a dataset name.
    Scorer selection (e.g. a "quick" vs "full" profile, in-process vs HTTP
    prediction) is the caller's choice: pass the matching ``scorers`` and
    ``predict_fn``.
    """
    if experiment_name and experiment_name != get_config().experiment_name:
        configure(experiment_name=experiment_name)
    result = evaluate(
        data=_load_dataset(dataset),
        scorers=scorers,
        predict_fn=predict_fn,
        judge_prompt=judge_prompt,
        run_name=run_name,
        tags=tags,
        excel_path=excel_path,
        log_excel_to_mlflow=log_excel_to_mlflow,
    )
    metrics = {k: float(v) for k, v in (result.metrics or {}).items() if _is_number(v)}
    return {"metrics": metrics, "run_id": result.run_id}


def compare_and_gate(
    candidate_metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    must_not_regress: list[str] | None = None,
    tolerance: float = 0.0,
) -> tuple[bool, list[str]]:
    """Decide whether a candidate may be promoted.

    - every metric in ``thresholds`` must be ``>=`` its threshold;
    - every metric in ``must_not_regress`` must be ``>= baseline - tolerance``.

    Returns ``(passed, reasons)``; ``reasons`` lists each failed check (or the
    checks that passed, when everything passes).
    """
    failures: list[str] = []
    passes: list[str] = []
    for name, minimum in (thresholds or {}).items():
        value = candidate_metrics.get(name)
        if value is None:
            failures.append(f"{name}: missing from candidate metrics")
        elif value < minimum:
            failures.append(f"{name}: {value:.3f} < threshold {minimum:.3f}")
        else:
            passes.append(f"{name}: {value:.3f} >= {minimum:.3f}")
    for name in must_not_regress or []:
        value = candidate_metrics.get(name)
        base = (baseline_metrics or {}).get(name)
        if value is None or base is None:
            failures.append(f"{name}: missing (candidate={value}, baseline={base})")
        elif value < base - tolerance:
            failures.append(f"{name}: regressed {value:.3f} < baseline {base:.3f}")
        else:
            passes.append(f"{name}: {value:.3f} >= baseline {base:.3f}")
    return (not failures, failures or passes)


def log_gate_decision(passed: bool, reasons: list[str], run_id: str | None = None) -> None:
    """Record a gate decision as tags on an MLflow run (the candidate run by default)."""
    client = mlflow.tracking.MlflowClient()
    target = run_id or (mlflow.active_run().info.run_id if mlflow.active_run() else None)
    if target is None:
        return
    client.set_tag(target, "gate.passed", str(passed).lower())
    client.set_tag(target, "gate.reasons", "; ".join(reasons)[:5000])


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
