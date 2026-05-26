# Project 07: GCP Vertex AI End-to-End Pipeline

**XP:** 300 | **Cost:** Free tier + ~$5 | **Duration:** Weeks 13–14

## Objective
Build and orchestrate a complete ML pipeline on GCP Vertex AI Pipelines using Kubeflow Pipelines (KFP) SDK. The pipeline covers data validation, preprocessing, training, evaluation, and conditional model deployment to a Vertex AI Endpoint. Store artifacts in GCS, track experiments in Vertex AI Experiments, and use Vertex AI Model Registry for versioning.

## GCP Services Used
- **Vertex AI Pipelines** — Kubeflow-based ML orchestration
- **Vertex AI Training** — managed training jobs
- **Vertex AI Endpoints** — online prediction serving
- **Vertex AI Experiments** — experiment tracking (MLflow-like)
- **Vertex AI Model Registry** — versioned model store
- **Cloud Storage (GCS)** — artifacts and datasets
- **Artifact Registry** — custom container storage
- **BigQuery** (optional) — large-scale data source

## Project Structure
```
07-vertex-ai-pipeline/
├── pipelines/
│   ├── components/
│   │   ├── data_validation.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── deploy.py
│   └── pipeline.py              ← Assembled KFP pipeline
├── notebooks/
│   ├── 01_setup_and_explore.ipynb
│   └── 02_run_pipeline.ipynb
├── src/
│   └── trainer/
│       ├── task.py              ← Training script for custom job
│       └── model.py
├── cloudbuild.yaml              ← CI for container building
├── docs/adr/
│   └── 001-pipeline-orchestration.md
└── requirements.txt
```

## Phase 1: GCP Setup (Day 1)

```bash
# Set project and enable APIs
gcloud config set project YOUR_PROJECT_ID

gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# Create GCS bucket
BUCKET="gs://vertex-pipeline-$(gcloud config get-value project)"
gsutil mb -l us-central1 $BUCKET
echo "Bucket: $BUCKET"

# Create service account for pipeline
gcloud iam service-accounts create vertex-pipeline-sa \
  --display-name="Vertex AI Pipeline SA"

gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:vertex-pipeline-sa@$(gcloud config get-value project).iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:vertex-pipeline-sa@$(gcloud config get-value project).iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

## Phase 2: Define KFP Components (Days 2–4)

Install:
```bash
pip install google-cloud-aiplatform kfp google-cloud-storage pandas scikit-learn
```

### Data Validation Component
`pipelines/components/data_validation.py`:
```python
from kfp import dsl
from kfp.dsl import Dataset, Output, Metrics

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "great-expectations"],
)
def validate_data(
    dataset_uri: str,
    metrics: Output[Metrics],
    validated_dataset: Output[Dataset],
) -> bool:
    import pandas as pd
    import json
    
    df = pd.read_csv(dataset_uri)
    
    # Basic validation checks
    checks = {
        "no_null_target": df["MedHouseVal"].isna().sum() == 0,
        "positive_values": (df["MedHouseVal"] > 0).all(),
        "expected_columns": len(df.columns) == 9,
        "min_rows": len(df) >= 1000,
    }
    
    passed = all(checks.values())
    
    metrics.log_metric("validation_passed", int(passed))
    metrics.log_metric("row_count", len(df))
    metrics.log_metric("null_count", df.isna().sum().sum())
    
    if passed:
        df.to_csv(validated_dataset.path, index=False)
    
    return passed
```

### Preprocess Component
`pipelines/components/preprocess.py`:
```python
from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Artifact

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "joblib"],
)
def preprocess(
    validated_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    preprocessor: Output[Artifact],
    test_size: float = 0.2,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
    
    df = pd.read_csv(validated_dataset.path)
    
    # Feature engineering
    df["rooms_per_person"] = df["AveRooms"] / df["AveOccup"]
    df["bedrooms_ratio"] = df["AveBedrms"] / df["AveRooms"]
    
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    pipe = Pipeline([("scaler", StandardScaler())])
    X_train_scaled = pipe.fit_transform(X_train)
    X_test_scaled = pipe.transform(X_test)
    
    # Save as CSV (label in first column)
    pd.concat([y_train.reset_index(drop=True), pd.DataFrame(X_train_scaled, columns=X_train.columns)], axis=1).to_csv(train_dataset.path, index=False)
    pd.concat([y_test.reset_index(drop=True), pd.DataFrame(X_test_scaled, columns=X_test.columns)], axis=1).to_csv(test_dataset.path, index=False)
    
    joblib.dump(pipe, preprocessor.path)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
```

### Train Component
`pipelines/components/train.py`:
```python
from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Model, Metrics

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "joblib"],
)
def train_model(
    train_dataset: Input[Dataset],
    model_artifact: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100,
    max_depth: int = 10,
) -> float:
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np
    import joblib
    
    df = pd.read_csv(train_dataset.path)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)
    
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    metrics.log_metric("train_r2", r2)
    metrics.log_metric("train_rmse", rmse)
    
    joblib.dump(model, model_artifact.path + ".pkl")
    model_artifact.metadata["framework"] = "scikit-learn"
    model_artifact.metadata["r2"] = r2
    
    return r2
```

### Evaluate Component
`pipelines/components/evaluate.py`:
```python
from kfp import dsl
from kfp.dsl import Dataset, Model, Input, Metrics

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "joblib"],
)
def evaluate_model(
    test_dataset: Input[Dataset],
    model_artifact: Input[Model],
    metrics: Output[Metrics],
    r2_threshold: float = 0.80,
) -> bool:
    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np
    import joblib
    
    df = pd.read_csv(test_dataset.path)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    model = joblib.load(model_artifact.path + ".pkl")
    preds = model.predict(X)
    
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    metrics.log_metric("test_r2", r2)
    metrics.log_metric("test_mae", mae)
    metrics.log_metric("test_rmse", rmse)
    
    passed = r2 >= r2_threshold
    metrics.log_metric("deployment_approved", int(passed))
    
    return passed
```

## Phase 3: Assemble Pipeline (Days 5–6)

`pipelines/pipeline.py`:
```python
import kfp
from kfp import dsl
from kfp.dsl import pipeline
from components.data_validation import validate_data
from components.preprocess import preprocess
from components.train import train_model
from components.evaluate import evaluate_model

@pipeline(name="california-housing-pipeline", description="E2E housing price prediction")
def housing_pipeline(
    dataset_uri: str,
    project_id: str,
    region: str = "us-central1",
    n_estimators: int = 100,
    max_depth: int = 10,
    r2_threshold: float = 0.80,
):
    validate_task = validate_data(dataset_uri=dataset_uri)
    
    preprocess_task = preprocess(
        validated_dataset=validate_task.outputs["validated_dataset"],
    ).after(validate_task)
    
    train_task = train_model(
        train_dataset=preprocess_task.outputs["train_dataset"],
        n_estimators=n_estimators,
        max_depth=max_depth,
    ).after(preprocess_task)
    
    evaluate_task = evaluate_model(
        test_dataset=preprocess_task.outputs["test_dataset"],
        model_artifact=train_task.outputs["model_artifact"],
        r2_threshold=r2_threshold,
    ).after(train_task)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=housing_pipeline,
        package_path="housing_pipeline.yaml",
    )
    print("Pipeline compiled to housing_pipeline.yaml")
```

## Phase 4: Run on Vertex AI (Days 7–9)

In `notebooks/02_run_pipeline.ipynb`:
```python
from google.cloud import aiplatform
import os

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
REGION = "us-central1"
BUCKET = f"gs://vertex-pipeline-{PROJECT_ID}"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

# Upload dataset to GCS
from google.cloud import storage
client = storage.Client()
bucket = client.bucket(f"vertex-pipeline-{PROJECT_ID}")
blob = bucket.blob("data/california_housing.csv")

from sklearn.datasets import fetch_california_housing
import pandas as pd
df = fetch_california_housing(as_frame=True).frame
df.to_csv("/tmp/california_housing.csv", index=False)
blob.upload_from_filename("/tmp/california_housing.csv")
dataset_uri = f"{BUCKET}/data/california_housing.csv"

# Submit pipeline run
job = aiplatform.PipelineJob(
    display_name="housing-pipeline-run",
    template_path="housing_pipeline.yaml",
    pipeline_root=f"{BUCKET}/pipeline_root",
    parameter_values={
        "dataset_uri": dataset_uri,
        "project_id": PROJECT_ID,
        "n_estimators": 150,
        "r2_threshold": 0.82,
    },
)
job.submit()
print(f"Pipeline running: {job.resource_name}")
print(f"View at: https://console.cloud.google.com/vertex-ai/pipelines")
```

## Acceptance Criteria
- [ ] Pipeline compiles to valid YAML without errors
- [ ] All 4 components run successfully on Vertex AI
- [ ] Validation component catches bad data (manually test with corrupted CSV)
- [ ] Model achieves test R² > 0.82
- [ ] Pipeline artifacts visible in GCS bucket
- [ ] Metrics tracked in Vertex AI Experiments
- [ ] ADR-001 written: KFP vs Vertex AI AutoML vs custom training

## ADR Template
Create `docs/adr/001-pipeline-orchestration.md`:
```markdown
# ADR-001: Pipeline Orchestration Tool Selection

## Status: Accepted

## Context
Need to orchestrate multi-step ML pipeline with artifact tracking, conditional execution,
and managed compute on GCP. Options: Vertex AI Pipelines (KFP), AutoML, Cloud Composer (Airflow).

## Decision
Vertex AI Pipelines with KFP SDK v2.

## Consequences
+ Serverless — no cluster to manage vs Cloud Composer
+ Native GCP integration — GCS artifacts, IAM, Monitoring
+ KFP components are reusable across pipelines
+ Visual DAG in Cloud Console for debugging
- Steeper learning curve than AutoML
- Component containerization adds iteration time
- Minimum 5min pipeline overhead (container startup)
```
