import os
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task
from airflow.models.param import Param
from ml_churn import data, evaluate as eval_mod, model, registry, tracking
from ml_churn import tune as tune_mod


@dag(
    dag_id="churn_training",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "dataset_name": "telco-customer-churn-kaggle-dataset_2026-05-27.csv",
        "run_name": "lgbm-baseline",
        "experiment": "churn-prediction",
        "do_tune": Param(False, type="boolean"),
        "n_trials": Param(30, type="integer"),
        "model_name": "churn-lgbm",
        "test_size": Param(0.2, type="number"),
        "random_state": Param(42, type="integer"),
    },
)
def churn_training():

    @task
    def pull_data(**context) -> str:
        p = context["params"]
        dataset_name = p["dataset_name"]
        project_dir = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
        data_path = f"{project_dir}/data/raw/{dataset_name}"

        subprocess.run(
            ["dvc", "pull", f"{data_path}.dvc"],
            cwd=project_dir,
            check=True,
        )
        return data_path

    @task
    def preprocess_data(**context) -> str:
        data_path = context["ti"].xcom_pull(task_ids="pull_data")

        df = data.load_data(data_path)
        df = data.clean(df)
        df = data.encode(df)

        tmp_dir = Path("/tmp/churn_dag") / context["run_id"]
        tmp_dir.mkdir(parents=True, exist_ok=True)

        return str(df.to_pickle(tmp_dir / "df.pkl"))

    @task
    def split_data(**context) -> dict:
        p = context["params"]
        df_path = context["ti"].xcom_pull(task_ids="preprocess_data")

        df = pd.read_pickle(df_path)

        split = data.make_split(df, test_size=p["test_size"], random_state=p["random_state"])
        base = Path(df_path).parent
        paths = {
            "X_train": str(base / "X_train.pkl"),
            "X_test": str(base / "X_test.pkl"),
            "y_train": str(base / "y_train.pkl"),
            "y_test": str(base / "y_test.pkl"),
        }

        split.X_train.to_pickle(paths["X_train"])
        split.X_test.to_pickle(paths["X_test"])
        split.y_train.to_pickle(paths["y_train"])
        split.y_test.to_pickle(paths["y_test"])

        return paths

    @task
    def create_run(**context) -> str:
        p = context["params"]
        return tracking.create_run(p["experiment"], p["run_name"])

    @task
    def log_dataset(**context) -> None:
        p = context["params"]
        ti = context["ti"]
        df_path = ti.xcom_pull(task_ids="preprocess_data")
        mlflow_run_id = ti.xcom_pull(task_ids="create_run")

        df = pd.read_pickle(df_path)

        project_dir = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
        source = f"{project_dir}/data/raw/{p['dataset_name']}"
        tracking.log_dataset(mlflow_run_id, df, source=source)
        tracking.log_params(mlflow_run_id, {"test_size": p["test_size"], "random_state": p["random_state"]})

    @task
    def tune_hyperparams(**context) -> dict:
        p = context["params"]
        if not p["do_tune"]:
            return {}

        ti = context["ti"]
        split_paths = ti.xcom_pull(task_ids="split_data")
        mlflow_run_id = ti.xcom_pull(task_ids="create_run")

        X_train = pd.read_pickle(split_paths["X_train"])
        y_train = pd.read_pickle(split_paths["y_train"])

        study = tune_mod.run_optuna_study(
            X_train, y_train, n_trials=p["n_trials"], random_state=p["random_state"]
        )

        best_params = study.best_params
        tracking.log_params(mlflow_run_id, {"tune.n_trials": p["n_trials"], **best_params})
        tracking.log_metrics(mlflow_run_id, {"tune.best_cv_roc_auc": study.best_value})

        return best_params

    @task
    def train(**context) -> str:
        p = context["params"]
        ti = context["ti"]
        split_paths = ti.xcom_pull(task_ids="split_data")
        mlflow_run_id = ti.xcom_pull(task_ids="create_run")
        best_params = ti.xcom_pull(task_ids="tune_hyperparams")

        X_train = pd.read_pickle(split_paths["X_train"])
        y_train = pd.read_pickle(split_paths["y_train"])

        clf = model.build_lgbm(random_state=p["random_state"], **best_params)

        clf.fit(X_train, y_train)

        tracking.log_model(mlflow_run_id, clf, model_name=p["model_name"])

        model_path = str(Path(split_paths["X_train"]).parent / "model.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        return model_path

    @task
    def evaluate(**context) -> dict:
        ti = context["ti"]
        split_paths = ti.xcom_pull(task_ids="split_data")
        model_path = ti.xcom_pull(task_ids="train")
        mlflow_run_id = ti.xcom_pull(task_ids="create_run")

        X_test = pd.read_pickle(split_paths["X_test"])
        y_test = pd.read_pickle(split_paths["y_test"])
        X_train = pd.read_pickle(split_paths["X_train"])
        y_train = pd.read_pickle(split_paths["y_train"])

        with open(model_path, "rb") as f:
            clf = pickle.load(f)
    
        metrics = eval_mod.compute_metrics(clf, X_test, y_test, X_train, y_train)

        tracking.log_metrics(mlflow_run_id, metrics)

        return metrics

    @task
    def register_model(**context) -> str:
        p = context["params"]
        if not p["model_name"]:
            return ""

        return registry.verify_model_registered(p["model_name"])

    # Need to ensure that mlflow run ended
    @task(trigger_rule="all_done")
    def end_run(**context) -> None:
        mlflow_run_id = context["ti"].xcom_pull(task_ids="create_run")
        if not mlflow_run_id:
            return

        tracking.end_run(mlflow_run_id)

    (
        pull_data()
        >> preprocess_data()
        >> split_data()
        >> create_run()
        >> log_dataset()
        >> tune_hyperparams()
        >> train()
        >> evaluate()
        >> register_model()
        >> end_run()
    )


churn_training()
