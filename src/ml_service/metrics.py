import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)

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

METRICS_PATH = "/metrics"


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
        try:
            response = await call_next(request)
        except Exception:
            # An unhandled error surfaces as a 500 to the client; record it as such.
            _observe(request, "500", time.perf_counter() - start)
            raise

        _observe(request, str(response.status_code), time.perf_counter() - start)
        return response

    @app.get(METRICS_PATH, include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def _observe(request: Request, status_code: str, elapsed: float) -> None:
    endpoint = _endpoint_label(request)
    REQUEST_DURATION.labels(endpoint, request.method).observe(elapsed)
    REQUEST_COUNT.labels(endpoint, request.method, status_code).inc()
