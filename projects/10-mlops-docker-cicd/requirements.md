# Project 10: MLOps — Docker + GitHub Actions + AWS ECS

**XP:** 400 | **Cost:** ~$5–15 (ECS, ECR, S3) | **Duration:** Weeks 19–20

## Objective
Implement a full MLOps lifecycle: containerize the ML model from Project 01 with Docker, set up a CI/CD pipeline with GitHub Actions that runs tests, builds the Docker image, pushes to AWS ECR, and deploys to AWS ECS Fargate. Add MLflow for experiment tracking, Prometheus + Grafana for monitoring, and a Kafka mini-module for streaming predictions.

## Tools
- **Docker** — containerization
- **GitHub Actions** — CI/CD pipeline
- **AWS ECR** — container registry
- **AWS ECS Fargate** — serverless container deployment
- **AWS ALB** — load balancer
- **MLflow** — experiment tracking and model registry
- **Prometheus + Grafana** — metrics and dashboards (Docker Compose local)
- **Kafka** — streaming predictions (Docker Compose local)

## Project Structure
```
10-mlops-docker-cicd/
├── app/
│   ├── main.py               ← FastAPI prediction service
│   ├── model_loader.py       ← MLflow model loading
│   └── metrics.py            ← Prometheus metrics export
├── models/                   ← Trained model artifacts
│   └── .gitkeep
├── training/
│   ├── train.py              ← MLflow-tracked training script
│   └── evaluate.py
├── kafka/
│   ├── producer.py           ← Stream prediction requests
│   └── consumer.py           ← Process predictions from stream
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboard.json
├── Dockerfile
├── docker-compose.yml        ← Local: app + mlflow + prometheus + grafana + kafka
├── .github/
│   └── workflows/
│       └── mlops-ci.yml      ← Full CI/CD pipeline
├── terraform/
│   ├── main.tf               ← ECS, ECR, ALB
│   └── variables.tf
├── docs/adr/
│   └── 001-deployment-target.md
└── requirements.txt
```

## Phase 1: MLflow Experiment Tracking (Days 1–2)

`training/train.py`:
```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import argparse

def train(n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.1):
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        })
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        
        # Log metrics
        mlflow.log_metrics({"r2": r2, "mae": mae, "rmse": rmse})
        
        # Log model with signature
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name="california-housing-gbm",
        )
        
        print(f"R²: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        return mlflow.active_run().info.run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("california-housing")
    
    run_id = train(args.n_estimators, args.max_depth, args.learning_rate)
    print(f"Run ID: {run_id}")
```

## Phase 2: FastAPI with Prometheus Metrics (Days 3–4)

`app/metrics.py`:
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

PREDICTION_COUNTER = Counter("predictions_total", "Total predictions made", ["status"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", buckets=[.01, .05, .1, .25, .5, 1])
MODEL_LOADED = Gauge("model_loaded", "Whether model is loaded (1=yes)")
PREDICTION_VALUE = Histogram("prediction_value_usd", "Distribution of predicted house prices", buckets=[50000, 100000, 200000, 300000, 500000, 1000000])

class MetricsTimer:
    def __enter__(self):
        self._start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
```

`app/main.py`:
```python
from fastapi import FastAPI, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import mlflow
import numpy as np
from metrics import (PREDICTION_COUNTER, PREDICTION_LATENCY, MODEL_LOADED,
                     PREDICTION_VALUE, MetricsTimer, generate_latest, CONTENT_TYPE_LATEST)
import os

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(
        f"models:/california-housing-gbm/Production"
    )
    MODEL_LOADED.set(1)
    yield
    MODEL_LOADED.set(0)

app = FastAPI(title="Housing Price Predictor", lifespan=lifespan)

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(features: HouseFeatures):
    with MetricsTimer() as t:
        try:
            X = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                           features.AveBedrms, features.Population, features.AveOccup,
                           features.Latitude, features.Longitude]])
            price = float(model.predict(X)[0]) * 100_000
            PREDICTION_COUNTER.labels(status="success").inc()
            PREDICTION_VALUE.observe(price)
            result = {"predicted_price_usd": round(price, 2)}
        except Exception as e:
            PREDICTION_COUNTER.labels(status="error").inc()
            raise
    
    PREDICTION_LATENCY.observe(t.elapsed)
    return result
```

## Phase 3: Dockerfile and Docker Compose (Day 5)

`Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`:
```yaml
version: "3.9"
services:
  app:
    build: .
    ports: ["8000:8000"]
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on: [mlflow]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.12.1
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow-artifacts
    volumes: ["mlflow-data:/mlflow-artifacts"]

  prometheus:
    image: prom/prometheus:v2.51.0
    ports: ["9090:9090"]
    volumes: ["./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml"]

  grafana:
    image: grafana/grafana:10.3.0
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes: ["grafana-data:/var/lib/grafana"]

  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment: {ZOOKEEPER_CLIENT_PORT: 2181}

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    ports: ["9092:9092"]
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    depends_on: [zookeeper]

volumes:
  mlflow-data:
  grafana-data:
```

## Phase 4: GitHub Actions CI/CD (Days 6–8)

`.github/workflows/mlops-ci.yml`:
```yaml
name: MLOps CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: housing-predictor
  ECS_SERVICE: housing-predictor-service
  ECS_CLUSTER: housing-cluster

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Login to ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      
      - name: Build, tag, push Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster ${{ env.ECS_CLUSTER }} \
            --service ${{ env.ECS_SERVICE }} \
            --force-new-deployment
          aws ecs wait services-stable \
            --cluster ${{ env.ECS_CLUSTER }} \
            --services ${{ env.ECS_SERVICE }}
```

## Phase 5: Kafka Streaming Module (Days 9–10)

`kafka/producer.py`:
```python
from confluent_kafka import Producer
import json
import time
import random

conf = {"bootstrap.servers": "localhost:9092"}
p = Producer(conf)

sample_houses = [
    {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984, "AveBedrms": 1.024,
     "Population": 322.0, "AveOccup": 2.556, "Latitude": 37.88, "Longitude": -122.23},
    {"MedInc": 4.2, "HouseAge": 20.0, "AveRooms": 5.2, "AveBedrms": 1.1,
     "Population": 1200.0, "AveOccup": 3.1, "Latitude": 34.05, "Longitude": -118.24},
]

def delivery_report(err, msg):
    if err:
        print(f"Delivery failed: {err}")
    else:
        print(f"Delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

print("Streaming prediction requests to Kafka...")
for i in range(20):
    house = random.choice(sample_houses)
    house["request_id"] = f"req-{i:04d}"
    house["timestamp"] = time.time()
    
    p.produce("prediction-requests", json.dumps(house).encode(), callback=delivery_report)
    p.poll(0)
    time.sleep(0.5)

p.flush()
```

`kafka/consumer.py`:
```python
from confluent_kafka import Consumer
import requests
import json

conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "prediction-consumer",
    "auto.offset.reset": "earliest",
}
c = Consumer(conf)
c.subscribe(["prediction-requests"])

API_URL = "http://localhost:8000/predict"

print("Consuming prediction requests...")
try:
    while True:
        msg = c.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        
        data = json.loads(msg.value().decode())
        request_id = data.pop("request_id", "unknown")
        data.pop("timestamp", None)
        
        response = requests.post(API_URL, json=data)
        price = response.json().get("predicted_price_usd", 0)
        print(f"[{request_id}] Predicted price: ${price:,.2f}")
finally:
    c.close()
```

## Monitoring Setup

`monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "housing-predictor"
    static_configs:
      - targets: ["app:8000"]
    metrics_path: "/metrics"
```

Import the Grafana dashboard JSON to visualize:
- Predictions per minute
- Latency percentiles (p50, p95, p99)
- Error rate
- Price distribution histogram
- Model loaded status

## Acceptance Criteria
- [ ] `docker compose up` starts all services without errors
- [ ] `/predict` endpoint responds with valid price
- [ ] `/metrics` endpoint returns Prometheus-format metrics
- [ ] Grafana dashboard shows prediction rate and latency
- [ ] GitHub Actions pipeline runs all tests on PR
- [ ] Docker image pushed to ECR on merge to main
- [ ] ECS deployment completes (service stable)
- [ ] MLflow UI shows experiment runs with metrics
- [ ] Kafka producer/consumer streams predictions through the API
- [ ] ADR-001 written: ECS Fargate vs Lambda vs EKS for ML serving

## ADR Template
Create `docs/adr/001-deployment-target.md`:
```markdown
# ADR-001: Deployment Target for ML API

## Status: Accepted

## Context
FastAPI prediction service needs to be deployed to AWS. Options: Lambda, ECS Fargate, EKS.
Model size: ~50MB (scikit-learn). Expected traffic: 10-100 req/min.

## Decision
ECS Fargate with ALB.

## Consequences
+ Always-warm — no cold start latency (critical for sub-100ms SLA)
+ Easy container deployment — same Docker image as local dev
+ ALB handles HTTPS, health checks, and load distribution
- More expensive than Lambda for low traffic (<100 req/day)
- No auto-scaling to zero (minimum 1 task running)
- Overkill vs Lambda for pure REST inference with no startup deps
```
