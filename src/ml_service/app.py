import uvicorn
from fastapi import FastAPI, HTTPException

from ml_service.config import settings
from ml_service.features import RequestFeatureProvider
from ml_service.metrics import setup_metrics
from ml_service.model_loader import ModelService
from ml_service.schemas import (
    HealthResponse,
    ModelMetadataResponse,
    PredictRequest,
    PredictResponse,
    Prediction,
)

app = FastAPI(title="Churn ML Service", version="0.1.0")

setup_metrics(app)

model_service = ModelService(
    model_name=settings.model_name,
    tracking_uri=settings.tracking_uri,
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

    model = loaded.model
    provider = RequestFeatureProvider(feature_order=loaded.metadata.features)
    try:
        frame = provider.to_frame(request.instances)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    probabilities = model.predict_proba(frame)[:, 1]
    labels = model.predict(frame)

    predictions = [
        Prediction(churn=int(label), churn_probability=float(prob))
        for label, prob in zip(labels, probabilities)
    ]
    return PredictResponse(
        model_name=loaded.metadata.name,
        model_version=loaded.metadata.version,
        predictions=predictions,
    )


def main() -> None:
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
