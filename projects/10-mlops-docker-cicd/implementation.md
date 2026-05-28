# Project 10: MLOps — Docker + GitHub Actions + AWS ECS — Implementation Guide

## Prerequisites
- Project 01 complete (trained model exists)
- Docker Desktop running
- AWS CLI configured, ECR access
- GitHub repository for this project
- `conda activate aiarch`
- Packages: `mlflow`, `prometheus-client`, `kafka-python`

---

## Project Structure
```
10-mlops-docker-cicd/
├── train.py                   # MLflow experiment tracking
├── api.py                     # FastAPI + Prometheus metrics
├── kafka_producer.py          # Kafka prediction stream producer
├── kafka_consumer.py          # Kafka prediction stream consumer
├── Dockerfile                 # Production container
├── docker-compose.yml         # Local dev stack
├── grafana/
│   └── dashboards/
│       └── predictions.json   # Grafana dashboard config
├── .github/
│   └── workflows/
│       └── deploy.yml         # CI/CD pipeline
├── tests/
│   └── test_api.py
├── docs/
│   └── adr/
│       └── ADR-001-deployment-target.md
└── .env
```

---

## Phase 1: MLflow Experiment Tracking

### Step 1.1 — Create train.py

```python
# train.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "housing-price-prediction"
MODEL_NAME = "HousingPricePredictor"


def train_with_mlflow(
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"gbr-{n_estimators}est"):
        # Log parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "algorithm": "GradientBoostingRegressor",
        })

        # Load data + feature engineering (same as P01)
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        df["rooms_per_person"] = df["AveRooms"] / df["Population"].clip(lower=1)
        df["bedrooms_ratio"]   = df["AveBedrms"] / df["AveRooms"].clip(lower=1)

        features = [c for c in df.columns if c != "MedHouseVal"]
        X = df[features]
        y = df["MedHouseVal"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )),
        ])
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        # Log metrics
        mlflow.log_metrics({
            "test_r2":   round(r2, 4),
            "test_mae":  round(mae, 4),
            "test_rmse": round(rmse, 4),
        })

        # Log model + register it
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # Also save locally for the API
        Path("models").mkdir(exist_ok=True)
        joblib.dump(pipeline, "models/best_model.joblib")

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        print(f"Model registered as '{MODEL_NAME}'")

    return r2


if __name__ == "__main__":
    train_with_mlflow()
```

---

## Phase 2: FastAPI with Prometheus Metrics

### Step 2.1 — Create api.py

```python
# api.py
import os
import time
import joblib
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response

MODEL_PATH = Path("models/best_model.joblib")
model = None

# Prometheus metrics
prediction_counter = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["status"],
)
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
model_loaded_gauge = Gauge(
    "model_loaded",
    "Whether the ML model is currently loaded (1=yes, 0=no)",
)
prediction_value_histogram = Histogram(
    "prediction_value",
    "Distribution of predicted house values (hundreds of thousands USD)",
    buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        model_loaded_gauge.set(1)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        model_loaded_gauge.set(0)
        print(f"WARNING: Model not found at {MODEL_PATH}. Run train.py first.")
    yield
    model = None
    model_loaded_gauge.set(0)


app = FastAPI(title="Housing Price Predictor MLOps", lifespan=lifespan)


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., example=3.5)
    HouseAge: float = Field(..., example=20.0)
    AveRooms: float = Field(..., example=5.0)
    AveBedrms: float = Field(..., example=1.0)
    Population: float = Field(..., example=1000.0)
    AveOccup: float = Field(..., example=3.0)
    Latitude: float = Field(..., example=37.5)
    Longitude: float = Field(..., example=-122.0)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: HouseFeatures):
    if model is None:
        prediction_counter.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        data = features.model_dump()
        rooms_per_person = data["AveRooms"] / max(data["Population"], 1)
        bedrooms_ratio   = data["AveBedrms"] / max(data["AveRooms"], 1)

        X = np.array([[
            data["MedInc"], data["HouseAge"], data["AveRooms"], data["AveBedrms"],
            data["Population"], data["AveOccup"], data["Latitude"], data["Longitude"],
            rooms_per_person, bedrooms_ratio,
        ]])

        prediction = float(model.predict(X)[0])

        latency = time.perf_counter() - start
        prediction_latency.observe(latency)
        prediction_counter.labels(status="success").inc()
        prediction_value_histogram.observe(prediction)

        return {
            "predicted_value": round(prediction, 4),
            "unit": "hundreds of thousands USD",
            "latency_ms": round(latency * 1000, 2),
        }
    except Exception as e:
        prediction_counter.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

---

## Phase 3: Docker Setup

### Step 3.1 — Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py train.py ./
COPY models/ models/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 3.2 — Create requirements.txt
```bash
cat > requirements.txt << 'EOF'
fastapi>=0.111
uvicorn[standard]>=0.29
scikit-learn>=1.4
numpy>=1.26
pandas>=2.0
joblib>=1.4
prometheus-client>=0.20
pydantic>=2.0
kafka-python>=2.0
mlflow>=2.13
EOF
```

### Step 3.3 — Create docker-compose.yml

```yaml
# docker-compose.yml
version: "3.9"

services:
  # FastAPI prediction service
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5001
    volumes:
      - ./models:/app/models
    depends_on:
      - mlflow
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MLflow tracking server
  mlflow:
    image: python:3.11-slim
    command: >
      bash -c "pip install mlflow && mlflow server
        --backend-store-uri sqlite:///mlflow.db
        --default-artifact-root /mlflow-artifacts
        --host 0.0.0.0
        --port 5001"
    ports:
      - "5001:5001"
    volumes:
      - mlflow-data:/mlflow-artifacts
      - mlflow-db:/app

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=7d"

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_SECURITY_ADMIN_USER=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

  # Zookeeper (for Kafka)
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  # Kafka
  kafka:
    image: confluentinc/cp-kafka:7.6.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"

volumes:
  mlflow-data:
  mlflow-db:
  grafana-data:
```

### Step 3.4 — Create prometheus.yml
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "housing-api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: "/metrics"
```

### Step 3.5 — Build and run
```bash
# Train model first (outside Docker)
python train.py

# Build and start all services
docker compose up --build -d

# Verify
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Access:
- API: http://localhost:8000/docs
- MLflow: http://localhost:5001
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

---

## Phase 4: Grafana Dashboard

### Step 4.1 — Create grafana/dashboards/predictions.json
```bash
mkdir -p grafana/dashboards
cat > grafana/dashboards/predictions.json << 'DASHBOARD'
{
  "title": "Housing Price Predictor",
  "panels": [
    {
      "title": "Predictions per Second",
      "type": "graph",
      "targets": [{"expr": "rate(predictions_total[1m])", "legendFormat": "{{status}}"}]
    },
    {
      "title": "P95 Latency (ms)",
      "type": "stat",
      "targets": [{"expr": "histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) * 1000"}]
    },
    {
      "title": "Model Loaded",
      "type": "stat",
      "targets": [{"expr": "model_loaded"}]
    },
    {
      "title": "Prediction Value Distribution",
      "type": "heatmap",
      "targets": [{"expr": "rate(prediction_value_bucket[5m])"}]
    }
  ]
}
DASHBOARD
```

### Step 4.2 — Configure Grafana data source
1. Open http://localhost:3000 → Login (admin/admin)
2. Settings → Data Sources → Add data source → Prometheus
3. URL: `http://prometheus:9090`
4. Click **Save & Test**
5. Import dashboard: Dashboards → Import → Upload JSON → select `grafana/dashboards/predictions.json`

---

## Phase 5: Kafka Streaming

### Step 5.1 — Create kafka_producer.py

```python
# kafka_producer.py
"""Sends predictions to Kafka topic for real-time streaming."""
import json
import time
import random
import requests
from kafka import KafkaProducer

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "predictions"
API_URL = "http://localhost:8000/predict"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def generate_prediction():
    payload = {
        "MedInc":     round(random.uniform(1.0, 15.0), 2),
        "HouseAge":   round(random.uniform(1, 52), 1),
        "AveRooms":   round(random.uniform(2, 10), 1),
        "AveBedrms":  round(random.uniform(0.8, 2.5), 1),
        "Population": random.randint(100, 10000),
        "AveOccup":   round(random.uniform(1.5, 8.0), 1),
        "Latitude":   round(random.uniform(32.0, 42.0), 4),
        "Longitude":  round(random.uniform(-124.0, -114.0), 4),
    }
    response = requests.post(API_URL, json=payload)
    result = response.json()
    return {"input": payload, **result, "timestamp": time.time()}


if __name__ == "__main__":
    print(f"Streaming predictions to Kafka topic '{TOPIC}'...")
    while True:
        try:
            record = generate_prediction()
            producer.send(TOPIC, record)
            print(f"Predicted: ${record['predicted_value'] * 100_000:,.0f}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(2)
```

### Step 5.2 — Create kafka_consumer.py

```python
# kafka_consumer.py
"""Consumes predictions from Kafka and logs them."""
import json
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "predictions"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP,
    auto_offset_reset="latest",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    group_id="prediction-logger",
)

if __name__ == "__main__":
    print(f"Listening on Kafka topic '{TOPIC}'...")
    for message in consumer:
        record = message.value
        val = record.get("predicted_value", 0) * 100_000
        print(f"[{record.get('timestamp', '')}] Prediction: ${val:,.0f} | Latency: {record.get('latency_ms', 0):.1f}ms")
```

---

## Phase 6: GitHub Actions CI/CD

### Step 6.1 — Create .github/workflows/deploy.yml

```yaml
# .github/workflows/deploy.yml
name: CI/CD — Test, Build, Deploy to AWS ECS

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: housing-predictor
  ECS_SERVICE: housing-predictor-service
  ECS_CLUSTER: aiarch-cluster
  CONTAINER_NAME: housing-predictor

jobs:
  # Job 1: Test
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements.txt pytest httpx
      - name: Train model for tests
        run: python train.py
        env:
          MLFLOW_TRACKING_URI: sqlite:///mlflow_test.db
      - name: Run tests
        run: pytest tests/ -v

  # Job 2: Build and push to ECR
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    outputs:
      image: ${{ steps.build-image.outputs.image }}
    steps:
      - uses: actions/checkout@v4
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      - name: Build, tag, and push image
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

  # Job 3: Deploy to ECS
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      - name: Download task definition
        run: |
          aws ecs describe-task-definition \
            --task-definition ${{ env.CONTAINER_NAME }} \
            --query taskDefinition > task-definition.json
      - name: Update ECS task definition with new image
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: task-definition.json
          container-name: ${{ env.CONTAINER_NAME }}
          image: ${{ needs.build.outputs.image }}
      - name: Deploy to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: ${{ steps.task-def.outputs.task-definition }}
          service: ${{ env.ECS_SERVICE }}
          cluster: ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true
```

### Step 6.2 — Set up ECR and ECS (one-time)
```bash
# Create ECR repository
aws ecr create-repository \
  --repository-name housing-predictor \
  --region us-east-1

# Create ECS cluster
aws ecs create-cluster --cluster-name aiarch-cluster

# Add GitHub secrets to your repo:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Settings → Secrets → Actions → New repository secret
```

---

## Phase 7: Architecture Decision Record

Create `docs/adr/ADR-001-deployment-target.md`:

```markdown
# ADR-001: Deployment Target — ECS Fargate vs Lambda vs EKS

**Date:** 2026-05-27
**Status:** Accepted

## Context
We need to host a FastAPI ML inference service that:
- Starts in <30 seconds (model load time)
- Handles sporadic traffic (no constant load)
- Has <$20/month budget
- Supports Prometheus metrics scraping
- Integrates with GitHub Actions CI/CD

## Decision
Use **AWS ECS Fargate** with Application Load Balancer.

## Alternatives Considered
| Option | Cost | Cold Start | Prometheus | Notes |
|--------|------|-----------|-----------|-------|
| **ECS Fargate** | ~$10-15/mo (0.25 vCPU, 0.5 GB) | <5s (container running) | Yes (ALB target) | Good balance |
| Lambda | <$1/mo | 2-5s + model load | Complex (custom layer) | 3MB model makes cold start slow |
| EC2 t3.micro | ~$8/mo | N/A (always on) | Yes | Manual management |
| EKS | ~$73/mo base | N/A | Yes | Too expensive for this scale |

## Rationale
- Fargate runs containers without managing EC2 instances
- ALB health checks align with our `/health` endpoint
- Prometheus can scrape via service discovery or static config
- GitHub Actions ECS deploy action is well-maintained
- Free tier: 750 hours/month of t2.micro equivalent on ECS (Fargate has no free tier, but cost is predictable)

## Consequences
- Must push to ECR before deploying (no local image support)
- Model artifact must be baked into the Docker image (or mounted via EFS)
- Scaling requires ALB + ECS Auto Scaling policy (future work)
```

---

## Verification Checklist
- [ ] `python train.py` completes and registers model in MLflow
- [ ] MLflow UI at http://localhost:5001 shows the experiment run
- [ ] `docker compose up --build` starts all 6 services
- [ ] `curl http://localhost:8000/health` returns `{"status":"ok",...}`
- [ ] `curl http://localhost:8000/predict` returns a valid price
- [ ] `curl http://localhost:8000/metrics` returns Prometheus format
- [ ] Grafana at http://localhost:3000 shows live prediction rates
- [ ] `python kafka_producer.py` streams predictions to Kafka
- [ ] `python kafka_consumer.py` prints received predictions
- [ ] GitHub Actions workflow runs green (all 3 jobs pass)
- [ ] Docker image pushed to ECR (visible in AWS Console)
- [ ] ECS service deployed and healthy
- [ ] ADR-001 written

---

## Troubleshooting

**Docker build fails: `models/` directory not found**
Run `python train.py` first to create `models/best_model.joblib`, then build.

**Prometheus scrape fails**
In `prometheus.yml`, the target `api:8000` uses the Docker service name. On localhost, change to `localhost:8000`.

**Grafana "No data" in dashboard**
- Confirm Prometheus data source URL is `http://prometheus:9090` (not localhost)
- Check `/metrics` endpoint returns data: `curl http://localhost:8000/metrics | grep prediction`

**Kafka connection refused**
Kafka takes ~30 seconds to start. Wait and retry, or run: `docker compose logs kafka`

**ECS deployment fails: `task definition not found`**
Create a task definition in AWS Console first, then the workflow can update it.

---

## Next Steps → Project 11: Multi-Cloud GenAI Architecture
```bash
conda activate aiarch
pip install anthropic google-cloud-aiplatform openai redis
mkdir -p ~/Documents/ai-journey/projects/11-multicloud-architecture/{providers,docs/{adr,c4},terraform}
```
