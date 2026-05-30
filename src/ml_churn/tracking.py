import re
from pathlib import Path

import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator

_INVALID = re.compile(r"[^a-zA-Z0-9_\-\. :/]")


def _sanitize(key: str) -> str:
    return _INVALID.sub("_", key)


def log_run(
    run_name: str,
    metrics: dict,
    params: dict | None = None,
    model: BaseEstimator | None = None,
    df: pd.DataFrame | None = None,
    data_source: str | Path | None = None,
) -> None:
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics({_sanitize(k): v for k, v in metrics.items()})
        if model is not None:
            mlflow.sklearn.log_model(model, artifact_path="model")
        if df is not None:
            dataset = mlflow.data.from_pandas(
                df,
                source=str(data_source) if data_source else None,
                name="telco-churn",
                targets="Churn",
            )
            mlflow.log_input(dataset, context="training")
