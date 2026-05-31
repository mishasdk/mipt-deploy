import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_name: str
    tracking_uri: str
    feature_store_uri: str | None
    host: str
    port: int


def load_settings() -> Settings:
    return Settings(
        model_name=os.environ.get("MODEL_NAME", "churn-lgbm"),
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000/mlflow"),
        feature_store_uri=os.environ.get("FEATURE_STORE_URI"),
        host=os.environ.get("SERVICE_HOST", "0.0.0.0"),
        port=int(os.environ.get("SERVICE_PORT", "8000")),
    )


settings = load_settings()
