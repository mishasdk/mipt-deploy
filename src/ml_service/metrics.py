import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from ml_service.model_loader import ModelMetadata

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
# Predicted churn probability spans [0, 1]
PROBABILITY_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Infrastructure / request metrics
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("endpoint", "method"),
    buckets=LATENCY_BUCKETS,
)

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    labelnames=("endpoint", "method", "status_code"),
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently being processed.",
    labelnames=("endpoint", "method"),
)

# Model quality / prediction metrics

MODEL_QUALITY = Gauge(
    "model_quality_score",
    "Offline quality metric of the deployed model, sourced from its MLflow run "
    "(e.g. roc_auc, f1, precision, recall, accuracy).",
    labelnames=("metric", "model_version"),
)

MODEL_INFO = Gauge(
    "model_info",
    "Active model identity exposed as labels; the value is always 1.",
    labelnames=("model_version", "run_id"),
)

PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total individual predictions served.",
    labelnames=("model_version",),
)

PREDICTED_CHURN_TOTAL = Counter(
    "model_predicted_churn_total",
    "Predictions with a positive (churn=1) label.",
    labelnames=("model_version",),
)

CHURN_PROBABILITY = Histogram(
    "model_churn_probability",
    "Distribution of predicted churn probabilities.",
    buckets=PROBABILITY_BUCKETS,
)

METRICS_PATH = "/metrics"


def update_model_quality(metadata: "ModelMetadata") -> None:
    """Publish the deployed model's MLflow metrics as Prometheus gauges.

    Called whenever a model is (re)loaded. Old series are cleared first so a
    version switch does not leave stale ``model_version`` labels behind.
    """
    MODEL_QUALITY.clear()
    MODEL_INFO.clear()
    for name, value in metadata.metrics.items():
        MODEL_QUALITY.labels(metric=name, model_version=metadata.version).set(value)
    MODEL_INFO.labels(
        model_version=metadata.version, run_id=metadata.run_id or ""
    ).set(1)


def record_prediction(
    model_version: str,
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> None:
    """Record online prediction volume, positive rate, and score distribution."""
    PREDICTIONS_TOTAL.labels(model_version).inc(len(labels))
    PREDICTED_CHURN_TOTAL.labels(model_version).inc(sum(int(x) for x in labels))
    for prob in probabilities:
        CHURN_PROBABILITY.observe(float(prob))


def _endpoint_label(request: Request) -> str:
    """Route template (e.g. ``/predict``), not the raw path, to bound cardinality."""
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    return template or request.url.path


def setup_metrics(app: FastAPI) -> None:
    """Attach the metrics middleware and the ``/metrics`` endpoint to ``app``."""

    @app.middleware("http")
    async def _record_metrics(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path == METRICS_PATH:
            return await call_next(request)

        start = time.perf_counter()
        endpoint = _endpoint_label(request)
        REQUESTS_IN_PROGRESS.labels(endpoint, request.method).inc()
        try:
            response = await call_next(request)
        except Exception:
            _observe(request, endpoint, "500", time.perf_counter() - start)
            raise
        finally:
            REQUESTS_IN_PROGRESS.labels(endpoint, request.method).dec()

        _observe(request, endpoint, str(response.status_code), time.perf_counter() - start)
        return response

    @app.get(METRICS_PATH, include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def _observe(request: Request, endpoint: str, status_code: str, elapsed: float) -> None:
    REQUEST_DURATION.labels(endpoint, request.method).observe(elapsed)
    REQUEST_COUNT.labels(endpoint, request.method, status_code).inc()
