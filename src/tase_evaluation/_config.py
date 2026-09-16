from __future__ import annotations

import os
from dataclasses import dataclass

import mlflow


@dataclass
class EvalConfig:
    tracking_uri: str = ""
    workspace: str = ""
    experiment_name: str = ""

    judge_model_name: str = ""
    judge_model_url: str = ""
    judge_api_key: str = ""

    excel_output_dir: str = "."
    excel_filename: str = ""

    def __post_init__(self):
        self.tracking_uri = self.tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.workspace = self.workspace or os.environ.get("MLFLOW_WORKSPACE", "")
        self.experiment_name = self.experiment_name or os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", "evaluation"
        )
        self.judge_model_name = self.judge_model_name or os.environ.get(
            "JUDGE_MODEL_NAME", "judge"
        )
        self.judge_model_url = self.judge_model_url or os.environ.get(
            "JUDGE_MODEL_URL", "https://vllm-judge.apps.ocp.example.com/v1"
        )
        self.judge_api_key = self.judge_api_key or os.environ.get(
            "JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", "")
        )

    @property
    def judge_model_uri(self) -> str:
        return f"openai:/{self.judge_model_name}"

    @property
    def judge_extra_headers(self) -> dict[str, str] | None:
        if self.judge_api_key:
            return {"Authorization": f"Bearer {self.judge_api_key}"}
        return None


_active_config: EvalConfig | None = None


def configure(config: EvalConfig | None = None, **overrides) -> EvalConfig:
    global _active_config
    if config is None:
        config = EvalConfig(**overrides)
    else:
        for k, v in overrides.items():
            setattr(config, k, v)

    _active_config = config
    _apply_mlflow_settings(config)
    return config


def get_config() -> EvalConfig:
    global _active_config
    if _active_config is None:
        _active_config = EvalConfig()
        _apply_mlflow_settings(_active_config)
    return _active_config


def _apply_mlflow_settings(cfg: EvalConfig) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    if cfg.workspace:
        mlflow.set_workspace(cfg.workspace)
    mlflow.set_experiment(cfg.experiment_name)
