import time
from pathlib import Path

import mlflow
import mlflow.data
import mlflow.entities
import mlflow.sklearn
from mlflow.entities import DatasetInput
from mlflow.tracking import MlflowClient

from mlflow.tracking.context.registry import resolve_tags
import pandas as pd
from sklearn.base import BaseEstimator


def create_run(experiment: str, run_name: str, tracking_uri: str | None = None) -> str:
    """Open a run and return its run_id. The id is what gets passed between steps."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment_id = _get_or_create_experiment_id(client, experiment)
    run = client.create_run(experiment_id, run_name=run_name, tags=resolve_tags())
    return run.info.run_id


def end_run(run_id: str, status: str = "FINISHED") -> None:
    MlflowClient().set_terminated(run_id, status=status)


def log_dataset(
    run_id: str,
    df: pd.DataFrame,
    source: Path | str,
    name: str = "telco-churn",
    targets: str = "Churn",
) -> None:
    dataset = mlflow.data.from_pandas(df, source=str(source), name=name, targets=targets)
    MlflowClient().log_inputs(run_id, [DatasetInput(dataset=dataset._to_mlflow_entity())])


def log_params(run_id: str, params: dict) -> None:
    MlflowClient().log_batch(
        run_id,
        params=[mlflow.entities.Param(k, str(v)) for k, v in params.items()],
    )


def log_metrics(run_id: str, metrics: dict) -> None:
    ts = int(time.time() * 1000)
    MlflowClient().log_batch(
        run_id,
        metrics=[mlflow.entities.Metric(k, v, ts, 0) for k, v in metrics.items()],
    )


def log_model(run_id: str, clf: BaseEstimator, model_name: str | None = None) -> None:
    client = MlflowClient()
    mlflow.set_experiment(experiment_id=client.get_run(run_id).info.experiment_id)
    mlflow.sklearn.log_model(
        clf,
        name=model_name or "model",
        registered_model_name=model_name,
        run_id=run_id
    )
    if model_name:
        client.set_tag(run_id, "model_name", model_name)


def _get_or_create_experiment_id(client: MlflowClient, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return client.create_experiment(name)
