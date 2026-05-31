import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from ml_churn import feature_store
from ml_service.config import settings
from ml_service.features import FeatureStoreProvider, RequestFeatureProvider
from ml_service.metrics import record_prediction, setup_metrics, update_model_quality
from ml_service.model_loader import LoadedModel, ModelService
from ml_service.schemas import (
    HealthResponse,
    ModelMetadataResponse,
    PredictByIdRequest,
    PredictRequest,
    PredictResponse,
    Prediction,
)

app = FastAPI(title="Churn ML Service", version="0.1.0")

setup_metrics(app)

model_service = ModelService(
    model_name=settings.model_name,
    tracking_uri=settings.tracking_uri,
    on_load=update_model_quality,
)

feature_store_engine = (
    feature_store.get_engine(settings.feature_store_uri)
    if settings.feature_store_uri
    else None
)


def _predict(loaded: LoadedModel, frame: pd.DataFrame) -> PredictResponse:
    model = loaded.model
    probabilities = model.predict_proba(frame)[:, 1]
    labels = model.predict(frame)
    record_prediction(loaded.metadata.version, labels.tolist(), probabilities.tolist())
    predictions = [
        Prediction(churn=int(label), churn_probability=float(prob))
        for label, prob in zip(labels, probabilities)
    ]
    return PredictResponse(
        model_name=loaded.metadata.name,
        model_version=loaded.metadata.version,
        predictions=predictions,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Does not force a model load."""
    return HealthResponse(status="ok", model_loaded=model_service.is_loaded)


@app.get("/model", response_model=ModelMetadataResponse)
def model_metadata() -> ModelMetadataResponse:
    """Metadata of the currently active model (lazy-loads on first call)."""
    try:
        meta = model_service.get_metadata()
    except Exception as exc:  # registry/load failure
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModelMetadataResponse(**meta.__dict__)


@app.post("/reload", response_model=ModelMetadataResponse)
def reload_model() -> ModelMetadataResponse:
    """Drop the cached model and load the latest registered version."""
    model_service.reload()
    try:
        meta = model_service.get_metadata()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModelMetadataResponse(**meta.__dict__)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Batch inference. Lazily loads the model into memory on first call."""
    if not request.instances:
        raise HTTPException(status_code=422, detail="`instances` must be non-empty")

    try:
        loaded = model_service.ensure_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    provider = RequestFeatureProvider(feature_order=loaded.metadata.features)
    try:
        frame = provider.to_frame(request.instances)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _predict(loaded, frame)


@app.post("/predict_by_id", response_model=PredictResponse)
def predict_by_id(request: PredictByIdRequest) -> PredictResponse:
    """Online inference by entity key. Features are read from the feature store."""
    if not request.customer_ids:
        raise HTTPException(status_code=422, detail="`customer_ids` must be non-empty")
    if feature_store_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Feature store is not configured (FEATURE_STORE_URI unset)",
        )

    try:
        loaded = model_service.ensure_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    provider = FeatureStoreProvider(
        feature_order=loaded.metadata.features, engine=feature_store_engine
    )
    try:
        frame = provider.to_frame(request.customer_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _predict(loaded, frame)


def main() -> None:
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
