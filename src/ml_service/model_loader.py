import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class ModelMetadata:
    """Snapshot of the active model's registry + run info."""

    name: str
    version: str
    run_id: str | None
    stage: str | None
    aliases: list[str]
    creation_timestamp: int
    features: list[str]
    metrics: dict[str, float]
    params: dict[str, str]
    loaded_at: str


@dataclass
class LoadedModel:
    model: BaseEstimator
    metadata: ModelMetadata


class ModelService:
    """Lazily loads the latest registered model into memory and caches it.

    The model is fetched from the MLflow Model Registry on the first call to
    `ensure_loaded()` and kept in process memory afterwards. Loading is guarded
    by a lock so concurrent requests trigger only a single load.
    """

    def __init__(self, model_name: str, tracking_uri: str | None = None) -> None:
        self._model_name = model_name
        self._tracking_uri = tracking_uri
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def ensure_loaded(self) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded
        with self._lock:
            # Another thread may have loaded it while we waited for the lock.
            if self._loaded is None:
                self._loaded = self._load()
        return self._loaded

    def get_model(self) -> BaseEstimator:
        return self.ensure_loaded().model

    def get_metadata(self) -> ModelMetadata:
        return self.ensure_loaded().metadata

    def reload(self) -> None:
        """Drop the cached model so the next request reloads the latest version."""
        with self._lock:
            self._loaded = None

    def _client(self) -> MlflowClient:
        return MlflowClient(tracking_uri=self._tracking_uri)

    def _load(self) -> LoadedModel:
        client = self._client()
        # Same "latest registered version" selection as ml_churn.registry.
        versions = client.get_latest_versions(self._model_name)
        if not versions:
            raise RuntimeError(
                f"Model '{self._model_name}' not found in MLflow Model Registry"
            )
        mv = versions[0]

        model_uri = f"models:/{self._model_name}/{mv.version}"
        model = mlflow.sklearn.load_model(model_uri)

        metadata = self._build_metadata(client, mv, model)
        print(f"[model_loader] loaded '{self._model_name}' v{mv.version} into memory")
        return LoadedModel(model=model, metadata=metadata)

    def _build_metadata(self, client: MlflowClient, mv, model) -> ModelMetadata:
        metrics: dict[str, float] = {}
        params: dict[str, str] = {}
        if mv.run_id:
            run = client.get_run(mv.run_id)
            metrics = dict(run.data.metrics)
            params = dict(run.data.params)

        return ModelMetadata(
            name=self._model_name,
            version=str(mv.version),
            run_id=mv.run_id,
            stage=getattr(mv, "current_stage", None),
            aliases=list(getattr(mv, "aliases", []) or []),
            creation_timestamp=getattr(mv, "creation_timestamp", 0),
            features=_input_feature_names(model),
            metrics=metrics,
            params=params,
            loaded_at=datetime.now(timezone.utc).isoformat(),
        )


def _input_feature_names(model) -> list[str]:
    """Best-effort recovery of the feature column order the model expects."""
    names = getattr(model, "feature_name_", None)  # LGBMClassifier
    if names is not None:
        return list(names)
    names = getattr(model, "feature_names_in_", None)  # generic sklearn
    if names is not None:
        return list(names)
    return []
