# Project 07: GCP Vertex AI End-to-End Pipeline — Implementation Guide

## Prerequisites
- GCP account with $300 free credit active
- `gcloud` CLI installed and authenticated
- `conda activate aiarch`
- Packages: `google-cloud-aiplatform`, `kfp`, `scikit-learn`, `pandas`
- GCS bucket created for artifact storage

---

## Project Structure
```
07-vertex-ai-pipeline/
├── components/
│   ├── data_validation.py     # KFP component 1
│   ├── preprocess.py          # KFP component 2
│   ├── train.py               # KFP component 3
│   └── evaluate.py            # KFP component 4
├── pipeline.py                # Pipeline assembly and compile
├── run_pipeline.py            # Submit pipeline to Vertex AI
├── housing_pipeline.yaml      # Compiled pipeline (auto-generated)
├── docs/
│   └── adr/
│       └── ADR-001-pipeline-orchestration.md
└── .env
```

---

## Phase 1: GCP Setup

### Step 1.1 — Create GCP project and enable APIs
```bash
# Set your project ID
PROJECT_ID="aiarch-vertex-$(date +%Y)"
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  containerregistry.googleapis.com

echo "APIs enabled for project: $PROJECT_ID"
```

### Step 1.2 — Create GCS bucket
```bash
REGION="us-central1"
BUCKET_NAME="${PROJECT_ID}-pipeline-artifacts"

gsutil mb -l $REGION gs://$BUCKET_NAME
echo "Bucket created: gs://$BUCKET_NAME"
```

### Step 1.3 — Set up authentication
```bash
gcloud auth application-default login
```

### Step 1.4 — Create .env
```bash
cat > .env << EOF
PROJECT_ID=$PROJECT_ID
REGION=$REGION
GCS_BUCKET=$BUCKET_NAME
PIPELINE_ROOT=gs://$BUCKET_NAME/pipelines
EXPERIMENT_NAME=housing-price-pipeline
EOF
```

### Step 1.5 — Install KFP
```bash
conda activate aiarch
pip install kfp google-cloud-aiplatform
kfp --version  # Should be v2.x
```

---

## Phase 2: KFP Components

Each component is a self-contained Python function decorated with `@dsl.component`.

### Step 2.1 — Create components/data_validation.py

```python
# components/data_validation.py
from kfp.dsl import component, Output, Dataset


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn"],
)
def data_validation(
    output_dataset: Output[Dataset],
    min_rows: int = 1000,
) -> dict:
    """
    Component 1: Validate raw California Housing data.
    Checks: no null target, positive values, expected columns, minimum rows.
    """
    import json
    import pandas as pd
    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    validation_results = {}

    # Check 1: Minimum rows
    validation_results["row_count"] = len(df)
    assert len(df) >= min_rows, f"Too few rows: {len(df)} < {min_rows}"

    # Check 2: No null target
    null_target = df["MedHouseVal"].isnull().sum()
    validation_results["null_target"] = int(null_target)
    assert null_target == 0, f"Target has {null_target} nulls"

    # Check 3: Expected columns present
    expected_cols = {"MedInc", "HouseAge", "AveRooms", "AveBedrms",
                     "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"}
    missing = expected_cols - set(df.columns)
    validation_results["missing_columns"] = list(missing)
    assert not missing, f"Missing columns: {missing}"

    # Check 4: Positive population and rooms
    neg_pop = (df["Population"] <= 0).sum()
    neg_rooms = (df["AveRooms"] <= 0).sum()
    validation_results["negative_population"] = int(neg_pop)
    validation_results["negative_rooms"] = int(neg_rooms)
    assert neg_pop == 0, f"{neg_pop} rows with Population <= 0"
    assert neg_rooms == 0, f"{neg_rooms} rows with AveRooms <= 0"

    # Save validated data
    df.to_csv(output_dataset.path, index=False)

    print(f"Validation PASSED: {len(df)} rows, all checks passed")
    print(f"Results: {json.dumps(validation_results, indent=2)}")
    return validation_results
```

### Step 2.2 — Create components/preprocess.py

```python
# components/preprocess.py
from kfp.dsl import component, Input, Output, Dataset


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "numpy"],
)
def preprocess(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Component 2: Feature engineering + train/test split.
    Adds: rooms_per_person, bedrooms_ratio.
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_dataset.path)

    # Feature engineering (same as Project 01)
    df["rooms_per_person"] = df["AveRooms"] / df["Population"].clip(lower=1)
    df["bedrooms_ratio"]   = df["AveBedrms"] / df["AveRooms"].clip(lower=1)

    # Remove outliers (cap extreme values)
    df["AveOccup"] = df["AveOccup"].clip(upper=20)
    df["rooms_per_person"] = df["rooms_per_person"].clip(upper=5)

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_df = X_train.copy()
    train_df["MedHouseVal"] = y_train.values
    test_df = X_test.copy()
    test_df["MedHouseVal"] = y_test.values

    train_df.to_csv(train_dataset.path, index=False)
    test_df.to_csv(test_dataset.path, index=False)

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Features: {list(X_train.columns)}")
```

### Step 2.3 — Create components/train.py

```python
# components/train.py
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"],
)
def train_model(
    train_dataset: Input[Dataset],
    model_artifact: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 200,
    max_depth: int = 5,
    random_state: int = 42,
):
    """
    Component 3: Train GradientBoostingRegressor and log metrics.
    """
    import pandas as pd
    import joblib
    from pathlib import Path
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    df = pd.read_csv(train_dataset.path)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )),
    ])
    pipeline.fit(X, y)

    # Training metrics (in-sample)
    y_pred_train = pipeline.predict(X)
    train_r2 = r2_score(y, y_pred_train)
    train_mae = mean_absolute_error(y, y_pred_train)

    # Log metrics to Vertex AI Experiments
    metrics.log_metric("train_r2", round(train_r2, 4))
    metrics.log_metric("train_mae", round(train_mae, 4))
    metrics.log_metric("n_estimators", n_estimators)
    metrics.log_metric("max_depth", max_depth)

    # Save model
    model_path = Path(model_artifact.path) / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    model_artifact.metadata["framework"] = "scikit-learn"
    model_artifact.metadata["algorithm"] = "GradientBoostingRegressor"

    print(f"Training complete. Train R²={train_r2:.4f}, MAE={train_mae:.4f}")
    print(f"Model saved to {model_path}")
```

### Step 2.4 — Create components/evaluate.py

```python
# components/evaluate.py
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, ClassificationMetrics


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"],
)
def evaluate_model(
    test_dataset: Input[Dataset],
    model_artifact: Input[Model],
    metrics: Output[Metrics],
    r2_threshold: float = 0.80,
) -> bool:
    """
    Component 4: Evaluate model on test set.
    Returns True if R² > threshold (deployment gate).
    """
    import pandas as pd
    import joblib
    import numpy as np
    from pathlib import Path
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # Load model
    model_path = Path(model_artifact.path) / "model.joblib"
    pipeline = joblib.load(model_path)

    # Load test data
    df = pd.read_csv(test_dataset.path)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    y_pred = pipeline.predict(X)
    r2   = r2_score(y, y_pred)
    mae  = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred) ** 0.5

    # Log test metrics
    metrics.log_metric("test_r2", round(r2, 4))
    metrics.log_metric("test_mae", round(mae, 4))
    metrics.log_metric("test_rmse", round(rmse, 4))
    metrics.log_metric("r2_threshold", r2_threshold)
    metrics.log_metric("deploy_approved", int(r2 >= r2_threshold))

    passed = r2 >= r2_threshold
    print(f"Test R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    print(f"Deployment gate (R² ≥ {r2_threshold}): {'PASS' if passed else 'FAIL'}")
    return passed
```

---

## Phase 3: Pipeline Assembly

### Step 3.1 — Create pipeline.py

```python
# pipeline.py
import os
from dotenv import load_dotenv
from kfp import dsl, compiler
from components.data_validation import data_validation
from components.preprocess import preprocess
from components.train import train_model
from components.evaluate import evaluate_model

load_dotenv()

PIPELINE_ROOT = os.environ["PIPELINE_ROOT"]
PIPELINE_NAME = "california-housing-pipeline"
OUTPUT_FILE = "housing_pipeline.yaml"


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="California Housing Price Prediction: validate → preprocess → train → evaluate",
    pipeline_root=PIPELINE_ROOT,
)
def housing_pipeline(
    min_rows: int = 1000,
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: int = 5,
    r2_threshold: float = 0.80,
):
    # Step 1: Validate
    validation_task = data_validation(min_rows=min_rows)
    validation_task.set_display_name("Data Validation")

    # Step 2: Preprocess (depends on validation)
    preprocess_task = preprocess(
        input_dataset=validation_task.outputs["output_dataset"],
        test_size=test_size,
    )
    preprocess_task.set_display_name("Feature Engineering")
    preprocess_task.after(validation_task)

    # Step 3: Train
    train_task = train_model(
        train_dataset=preprocess_task.outputs["train_dataset"],
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    train_task.set_display_name("Train GradientBoosting")
    train_task.set_cpu_limit("2")
    train_task.set_memory_limit("4G")

    # Step 4: Evaluate (conditional deployment gate)
    evaluate_task = evaluate_model(
        test_dataset=preprocess_task.outputs["test_dataset"],
        model_artifact=train_task.outputs["model_artifact"],
        r2_threshold=r2_threshold,
    )
    evaluate_task.set_display_name("Evaluate & Gate")

    # Conditional: only deploy if evaluation passes
    with dsl.If(evaluate_task.output == True, name="deploy-if-approved"):
        from kfp.dsl import importer
        # In a real pipeline, add a deploy component here
        # For this project, the gate output serves as the approval signal
        pass


def compile_pipeline():
    compiler.Compiler().compile(
        pipeline_func=housing_pipeline,
        package_path=OUTPUT_FILE,
    )
    print(f"Pipeline compiled to {OUTPUT_FILE}")


if __name__ == "__main__":
    compile_pipeline()
```

### Step 3.2 — Compile the pipeline
```bash
conda activate aiarch
python pipeline.py
# Expected: housing_pipeline.yaml created
ls -lh housing_pipeline.yaml
```

---

## Phase 4: Submit to Vertex AI

### Step 4.1 — Create run_pipeline.py

```python
# run_pipeline.py
import os
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

PROJECT_ID     = os.environ["PROJECT_ID"]
REGION         = os.environ.get("REGION", "us-central1")
PIPELINE_ROOT  = os.environ["PIPELINE_ROOT"]
EXPERIMENT     = os.environ.get("EXPERIMENT_NAME", "housing-price-pipeline")
PIPELINE_FILE  = "housing_pipeline.yaml"


def run():
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        experiment=EXPERIMENT,
    )

    job = aiplatform.PipelineJob(
        display_name="housing-pipeline-run",
        template_path=PIPELINE_FILE,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "min_rows": 1000,
            "test_size": 0.2,
            "n_estimators": 200,
            "max_depth": 5,
            "r2_threshold": 0.80,
        },
        enable_caching=True,
    )

    job.submit(
        experiment=EXPERIMENT,
        experiment_run=f"run-{__import__('time').strftime('%Y%m%d-%H%M%S')}",
    )

    print(f"Pipeline submitted!")
    print(f"View in console: https://console.cloud.google.com/vertex-ai/pipelines?project={PROJECT_ID}")
    print(f"Job resource name: {job.resource_name}")

    # Wait for completion (optional — removes for async execution)
    job.wait()
    print(f"Pipeline completed with status: {job.state}")


if __name__ == "__main__":
    run()
```

### Step 4.2 — Submit the pipeline
```bash
python run_pipeline.py
```

> The pipeline runs in Vertex AI. Monitor progress at:
> https://console.cloud.google.com/vertex-ai/pipelines

---

## Phase 5: View Results in Vertex AI Experiments

### Step 5.1 — Check experiment metrics via SDK
```python
# check_experiment.py
import os
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()

aiplatform.init(project=os.environ["PROJECT_ID"], location=os.environ["REGION"])
experiment = aiplatform.Experiment(experiment_name=os.environ["EXPERIMENT_NAME"])

runs = aiplatform.ExperimentRun.list(experiment=experiment)
for run in runs:
    print(f"\nRun: {run.name}")
    print(f"  Metrics: {run.get_metrics()}")
```

```bash
python check_experiment.py
```

---

## Phase 6: Architecture Decision Record

Create `docs/adr/ADR-001-pipeline-orchestration.md`:

```markdown
# ADR-001: Pipeline Orchestration Tool Selection

**Date:** 2026-05-27
**Status:** Accepted

## Context
We need to orchestrate a 4-step ML pipeline (validate → preprocess → train → evaluate)
with artifact tracking, conditional deployment gates, and experiment logging.
The pipeline must run on GCP.

## Decision
Use **Kubeflow Pipelines v2 (KFP) on Vertex AI**.

## Alternatives Considered
| Tool | Pros | Cons |
|------|------|------|
| **KFP v2 on Vertex AI** | Native GCP, experiment tracking, artifact lineage, conditional steps | GCP lock-in |
| Apache Airflow (Cloud Composer) | DAG-based, mature, cloud-agnostic | Not ML-native, no artifact lineage |
| Prefect / Dagster | Modern Python-native, good DX | Extra setup on GCP, no native Vertex Experiments |
| SageMaker Pipelines | Great for AWS | We're on GCP for this project |

## Rationale
- KFP v2 compiles to YAML — portable and reproducible
- Vertex AI Experiments auto-tracks metrics per component run
- Conditional gates (`dsl.If`) enable clean deploy/no-deploy logic
- Component caching avoids re-running expensive steps on reruns
- Free tier covers pipeline runs at this scale

## Consequences
- Components are containerized Python functions — each runs in isolation
- Artifact URIs (GCS paths) provide lineage between steps
- Pipeline YAML can be reused across environments with different parameter values
```

---

## Verification Checklist
- [ ] `gcloud services enable aiplatform.googleapis.com` succeeded
- [ ] GCS bucket created: `gsutil ls gs://YOUR_BUCKET`
- [ ] `python pipeline.py` generates `housing_pipeline.yaml` without errors
- [ ] `python run_pipeline.py` submits job successfully
- [ ] Pipeline visible in Vertex AI Console → Pipelines → Runs
- [ ] All 4 components show green checkmarks in pipeline graph
- [ ] Data validation catches bad data (test: pass `min_rows=99999` to see it fail)
- [ ] Test R² > 0.82 (visible in evaluate component logs)
- [ ] Metrics visible in Vertex AI Experiments console
- [ ] Artifacts (model.joblib) stored in GCS
- [ ] ADR-001 written

---

## Troubleshooting

**`google.api_core.exceptions.PermissionDenied`**
Run `gcloud auth application-default login` and ensure the account has `roles/aiplatform.user`.

**`ModuleNotFoundError` in component**
Each component runs in isolation. All imports must be inside the function body, and packages listed in `packages_to_install`.

**Pipeline compilation error: `dsl.If` output type mismatch**
Ensure `evaluate_model` returns `bool`, not `str`. Check the return type annotation.

**Vertex AI pipeline stuck in PENDING**
Ensure the Vertex AI API is enabled and the service account has GCS read/write access.

**Cost concern**
Set `enable_caching=True` (already set) so reruns reuse cached component outputs.
The pipeline costs ~$0.50–2.00 to run depending on machine type and duration.

---

## Next Steps → Project 08: LangGraph Research Agent
```bash
conda activate aiarch
mkdir -p ~/Documents/ai-journey/projects/08-langgraph-agent
cd ~/Documents/ai-journey/projects/08-langgraph-agent
pip install langgraph tavily-python anthropic
```
