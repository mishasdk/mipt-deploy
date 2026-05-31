FROM python:3.12-slim

# libgomp1 is required by LightGBM at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml-churn

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[service]"

EXPOSE 8000

CMD ["churn-service"]
