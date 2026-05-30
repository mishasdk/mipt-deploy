import argparse
from pathlib import Path

from ml_churn import data, evaluate, model, registry, tracking, tune


def run(
    data_path: Path,
    run_name: str,
    experiment: str,
    tracking_uri: str | None,
    do_tune: bool,
    n_trials: int,
    model_name: str | None,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict:
    df = data.load_data(data_path)
    df = data.clean(df)
    df = data.encode(df)
    split = data.make_split(df, test_size=test_size, random_state=random_state)

    run_id = tracking.create_run(experiment, run_name, tracking_uri)
    try:
        tracking.log_dataset(run_id, df, source=data_path)
        tracking.log_params(run_id, {"test_size": test_size, "random_state": random_state})

        params: dict = {}
        if do_tune:
            study = tune.run_optuna_study(
                split.X_train, split.y_train, n_trials=n_trials, random_state=random_state
            )
            params = study.best_params
            tracking.log_params(run_id, {"tune.n_trials": n_trials, **params})
            tracking.log_metrics(run_id, {"tune.best_cv_roc_auc": study.best_value})

        clf = model.build_lgbm(random_state=random_state, **params)
        clf.fit(split.X_train, split.y_train)
        tracking.log_model(run_id, clf, model_name=model_name)

        metrics = evaluate.compute_metrics(
            clf, split.X_test, split.y_test, split.X_train, split.y_train
        )
        tracking.log_metrics(run_id, metrics)
    except Exception:
        tracking.end_run(run_id, status="FAILED")
        raise
    tracking.end_run(run_id)

    if model_name:
        registry.verify_model_registered(model_name, tracking_uri=tracking_uri)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate churn model")
    parser.add_argument("--data", required=True, type=Path, help="Path to CSV dataset")
    parser.add_argument("--run-name", default="lgbm-baseline", help="MLflow run name")
    parser.add_argument("--experiment", default="churn-prediction", help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--model-name", default=None, help="Register model under this name in MLflow Model Registry")
    args = parser.parse_args()

    metrics = run(
        data_path=args.data,
        run_name=args.run_name,
        experiment=args.experiment,
        tracking_uri=args.tracking_uri,
        do_tune=args.tune,
        n_trials=args.n_trials,
        model_name=args.model_name,
    )

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
