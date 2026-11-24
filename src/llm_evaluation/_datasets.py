from __future__ import annotations

from typing import Any

import pandas as pd
from mlflow.genai.datasets import create_dataset as _mlflow_create_dataset


def create_dataset(
    name: str,
    records: list[dict[str, Any]],
    tags: dict[str, str] | None = None,
):
    dataset = _mlflow_create_dataset(name=name, tags=tags)
    if records:
        dataset.merge_records(records)
    return dataset


def create_dataset_from_df(
    name: str,
    df: pd.DataFrame,
    tags: dict[str, str] | None = None,
):
    dataset = _mlflow_create_dataset(name=name, tags=tags)
    dataset.merge_records(df)
    return dataset
