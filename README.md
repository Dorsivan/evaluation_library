# tase_evaluation

Internal evaluation library wrapping MLflow GenAI. Provides simple function calls to create datasets, run evaluations with custom and built-in scorers, and execute LLM-as-judge steps. Every evaluation run automatically produces an Excel report.

## Installation

```bash
# From the repo (editable/development install)
pip install -e .

# From git
pip install git+https://github.com/Dorsivan/evaluation_library

# If published to a PyPI registry
pip install tase_evaluation
```

## Quick Start

```python
from tase_evaluation import scorer, evaluate, create_dataset

# 1. Create a dataset
dataset = create_dataset("my_eval_v1", records=[
    {
        "inputs": {"question": "What is Python?"},
        "expectations": {"expected_response": "A programming language"},
    },
    {
        "inputs": {"question": "What is 2+2?"},
        "expectations": {"expected_response": "4"},
    },
])

# 2. Add custom scorers (optional)
@scorer
def is_not_empty(outputs: str) -> bool:
    return bool(outputs and outputs.strip())

# 3. Run evaluation — Excel report is generated automatically
results = evaluate(
    data=dataset,
    predict_url="http://my-service/ask",
    response_key="answer",
    scorers=[is_not_empty],
    judge_prompt="Is this answer correct?\n{{ inputs }}\n{{ outputs }}",
)

print(results.metrics)
```

## Configuration

The library reads configuration from environment variables by default. You can also configure explicitly.

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URL | `http://localhost:5000` |
| `MLFLOW_WORKSPACE` | MLflow workspace | (none) |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment name | `evaluation` |
| `JUDGE_MODEL_NAME` | Judge LLM model name | `judge` |
| `JUDGE_MODEL_URL` | Judge LLM base URL | `https://vllm-judge.apps.ocp.example.com/v1` |
| `JUDGE_API_KEY` | Judge LLM API key | Falls back to `OPENAI_API_KEY` |

### Explicit Configuration

```python
from tase_evaluation import configure, EvalConfig

# Option 1: keyword overrides (env vars fill the rest)
configure(
    experiment_name="my-experiment",
    judge_model_name="gpt-4o",
    judge_api_key="sk-...",
    excel_output_dir="./results",
)

# Option 2: full config object
config = EvalConfig(
    tracking_uri="http://mlflow.internal:5000",
    experiment_name="my-experiment",
    judge_model_name="gpt-4o",
    judge_model_url="https://api.openai.com/v1",
    judge_api_key="sk-...",
    excel_output_dir="./results",
)
configure(config)
```

If you never call `configure()`, the library auto-initializes from environment variables on first use.

## API Reference

### `create_dataset(name, records, tags=None)`

Create or update a named evaluation dataset.

```python
dataset = create_dataset(
    name="my_dataset",
    records=[
        {"inputs": {"prompt": "..."}, "expectations": {"expected_response": "..."}},
    ],
    tags={"version": "1.0", "team": "ml-platform"},
)
```

Each record must have an `inputs` dict. The `expectations` dict is optional but required for some built-in scorers (e.g., `Correctness` needs `expected_response`).

### `create_dataset_from_df(name, df, tags=None)`

Create a dataset from a pandas DataFrame. The DataFrame must have an `inputs` column.

```python
import pandas as pd

df = pd.DataFrame({
    "inputs": [{"question": "What is AI?"}, {"question": "What is ML?"}],
    "expectations": [{"expected_response": "..."}, {"expected_response": "..."}],
})
dataset = create_dataset_from_df("my_dataset", df)
```

### `evaluate(data, scorers, ...)`

Run an evaluation and export results to Excel.

**Option 1: URL-based prediction (default)**

Provide a URL and the library handles the HTTP call. Each dataset record's `inputs` dict is merged with `predict_body` and POSTed to the URL.

```python
results = evaluate(
    data=dataset,
    scorers=[my_scorer],
    predict_url="http://my-service/generate",
    predict_body={"namespace": "default", "replicas": 2},  # extra fields merged with inputs
    predict_api_key="sk-...",                               # optional Bearer token
    response_key="output",                                  # optional: extract this key from the JSON response
)
```

**Option 2: Custom predict function (overrides URL)**

If you need full control (custom auth, response parsing, non-JSON APIs), provide a function instead. When `predict_fn` is provided, `predict_url` is ignored.

```python
def my_predict(question: str) -> str:
    resp = requests.post("http://my-service/ask", json={"q": question})
    return resp.json()["answer"]

results = evaluate(
    data=dataset,
    predict_fn=my_predict,
    scorers=[my_scorer],
    excel_path="./results/run_42.xlsx",  # optional, auto-generated if omitted
)
```

**LLM-as-judge (optional)**

Pass a `judge_prompt` string to automatically run an LLM judge alongside your scorers. If omitted, no LLM judge runs. The prompt must use `{{ inputs }}` and/or `{{ outputs }}` template variables.

```python
results = evaluate(
    data=dataset,
    predict_url="http://my-service/generate",
    scorers=[my_scorer],
    judge_prompt=(
        "Is the following response accurate and complete?\n\n"
        "User request: {{ inputs }}\n"
        "Response: {{ outputs }}\n"
    ),
)
```

**Accessing results:**

```python
print(results.metrics)      # aggregated scores
print(results.result_df)    # per-row results DataFrame
print(results.run_id)       # MLflow run ID
```

The Excel output has two sheets:
- **Results**: full per-row scorer outputs
- **Metrics**: aggregate metric summary

### Custom Scorers

Decorate any function with `@scorer` to create a custom scorer:

```python
from tase_evaluation import scorer

@scorer
def my_check(outputs: str) -> bool:
    return "error" not in outputs.lower()

@scorer
def length_check(outputs: str, expectations: dict) -> bool:
    max_len = expectations.get("max_length", 1000)
    return len(outputs) <= max_len

@scorer(name="custom_name", aggregations=["mean", "min"])
def quality_score(inputs: dict, outputs: str) -> float:
    return compute_quality(inputs, outputs)
```

Valid parameter names: `inputs`, `outputs`, `expectations`, `trace`. The decorator inspects the function signature and only passes what you declare.

Return types: `bool`, `int`, `float`, `str`.

### Built-in LLM Scorers

Re-exported from MLflow for convenience:

```python
from tase_evaluation import Correctness, Safety, RelevanceToQuery, get_config

cfg = get_config()

scorers = [
    Correctness(model=cfg.judge_model_uri, extra_headers=cfg.judge_extra_headers),
    Safety(model=cfg.judge_model_uri, extra_headers=cfg.judge_extra_headers),
    RelevanceToQuery(model=cfg.judge_model_uri, extra_headers=cfg.judge_extra_headers),
]
```

## Building and Publishing the pip Package

### Build the package

```bash
# Install build tools
pip install build

# Build the package (creates dist/ with .whl and .tar.gz)
python -m build
```

This produces two files in `dist/`:
- `tase_evaluation-0.1.0-py3-none-any.whl` (wheel)
- `tase_evaluation-0.1.0.tar.gz` (source distribution)

### Upload to PyPI

```bash
# Install twine
pip install twine

# Upload to PyPI
twine upload dist/*

# Upload to TestPyPI first (recommended for first-time publishing)
twine upload --repository testpypi dist/*
```

### Upload to an internal/private registry

```bash
# Upload to a private PyPI registry
twine upload --repository-url https://your-internal-pypi.com/simple/ dist/*

# Or configure the registry in ~/.pypirc:
# [your-registry]
# repository = https://your-internal-pypi.com/simple/
# username = __token__
# password = your-api-token
twine upload --repository your-registry dist/*
```

### Versioning

Update the version in `pyproject.toml` before building a new release:

```toml
[project]
version = "0.2.0"
```

Then rebuild and upload:

```bash
python -m build
twine upload dist/*
```

## Examples

See [`examples/k8s_evaluation.py`](examples/k8s_evaluation.py) for a complete example evaluating Kubernetes manifest generation with custom code scorers and an LLM judge.
