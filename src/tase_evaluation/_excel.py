from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_results_to_excel(
    result_df: pd.DataFrame,
    metrics: dict[str, float],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="Results", index=False)

        if metrics:
            metrics_df = pd.DataFrame(
                list(metrics.items()),
                columns=["Metric", "Value"],
            )
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    return path
