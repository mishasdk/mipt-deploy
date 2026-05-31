import mlflow


def verify_model_registered(model_name: str) -> str:
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name)
    if not versions:
        raise RuntimeError(f"Model '{model_name}' not found in MLflow Model Registry")
    version = versions[0].version
    print(f"[registry] '{model_name}' v{version} confirmed in registry")
    return version
