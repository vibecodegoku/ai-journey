# Project 01: ML Foundations & API Deployment

**XP:** 200 | **Cost:** Free | **Duration:** Weeks 1–2

## Objective
Build a complete ML pipeline: raw data → EDA → feature engineering → train → evaluate → serialize → deploy as REST API. Get comfortable with the core ML workflow before moving to cloud.

## Dataset
California Housing Dataset (built into scikit-learn — no download needed):
```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
```

## Project Structure
```
01-ml-foundations/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── app/
│   ├── main.py
│   └── models/best_model.pkl
├── tests/
│   └── test_api.py
├── docs/adr/
│   └── 001-model-selection.md
└── requirements.txt
```

## Phase 1: Exploratory Data Analysis (Days 1–2)
In `notebooks/01_eda.ipynb`:
- Load dataset, inspect shape/dtypes/stats
- Plot distributions for all 8 features
- Correlation heatmap
- Identify outliers using IQR method
- Write 3-sentence EDA summary

## Phase 2: Feature Engineering (Days 3–4)
- Handle any missing values (strategy: median imputation)
- Scale features: compare StandardScaler vs MinMaxScaler
- Create 2 derived features (e.g., `rooms_per_person = AveRooms / AveOccup`)
- 80/20 train/test split
- Package into a scikit-learn `Pipeline`

## Phase 3: Model Training & Evaluation (Days 5–7)
Train and compare these 3 models:

| Model | Expected R² | Notes |
|---|---|---|
| Linear Regression | ~0.60 | Baseline |
| Random Forest | ~0.80 | Main model |
| Gradient Boosting | ~0.83 | Best performer |

Metrics to report: RMSE, MAE, R² with 5-fold cross-validation.

## Phase 4: FastAPI Deployment (Days 8–10)
```python
# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np
from contextlib import asynccontextmanager

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load("models/best_model.pkl")
    yield

app = FastAPI(title="House Price Predictor", lifespan=lifespan)

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
    return {"status": "ok"}

@app.post("/predict")
def predict(features: HouseFeatures):
    X = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                   features.AveBedrms, features.Population, features.AveOccup,
                   features.Latitude, features.Longitude]])
    prediction = model.predict(X)
    return {"predicted_price_usd": round(float(prediction[0]) * 100_000, 2)}
```

Run: `uvicorn app.main:app --reload`  
Test: visit `http://localhost:8000/docs`

## Phase 5: Testing (Days 11–14)
```python
# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_predict_returns_valid_price():
    payload = {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984,
               "AveBedrms": 1.024, "Population": 322.0, "AveOccup": 2.556,
               "Latitude": 37.88, "Longitude": -122.23}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["predicted_price_usd"] > 0
```

Run: `pytest tests/ -v`

## ADR Template
Create `docs/adr/001-model-selection.md`:
```markdown
# ADR-001: Model Selection for House Price Prediction

## Status: Accepted

## Context
We need a regression model for the California Housing dataset...

## Decision
Random Forest Regressor with 100 estimators

## Consequences
+ Handles non-linear relationships
+ Feature importance built-in
- Larger model size vs linear regression (5MB vs 50KB)
```

## Acceptance Criteria
- [ ] Best model R² > 0.80 on test set
- [ ] FastAPI `/predict` returns valid price
- [ ] FastAPI `/docs` UI accessible at localhost:8000
- [ ] `pytest` passes all tests (0 failures)
- [ ] ADR-001 written and committed
- [ ] Feature importance plot saved as PNG
