# Project 02: AWS SageMaker Classification Pipeline

**XP:** 250 | **Cost:** Free tier (~$0–5) | **Duration:** Weeks 3–4

## Objective
Build an end-to-end ML classification pipeline on AWS using SageMaker: upload data to S3, train a model using a SageMaker built-in algorithm, deploy to a real-time endpoint, run batch inference, and set up Model Monitor to detect data drift.

## Dataset
UCI Adult Income Dataset (binary classification: income >50K or ≤50K)
```python
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ["age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"]
df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
```

## AWS Services Used
- **S3** — data storage and model artifacts
- **SageMaker Training Jobs** — managed training
- **SageMaker Endpoints** — real-time inference
- **SageMaker Batch Transform** — offline batch inference
- **SageMaker Model Monitor** — data drift detection
- **IAM** — SageMaker execution role
- **CloudWatch** — logs and metrics

## Project Structure
```
02-cloud-ml-aws/
├── notebooks/
│   ├── 01_data_prep_s3.ipynb
│   ├── 02_sagemaker_training.ipynb
│   └── 03_endpoint_monitoring.ipynb
├── src/
│   ├── preprocess.py
│   └── evaluate.py
├── docs/adr/
│   └── 001-algorithm-selection.md
└── requirements.txt
```

## Phase 1: IAM & Environment Setup (Day 1)

### Create SageMaker Execution Role
```python
import boto3, sagemaker
from sagemaker import get_execution_role

sess = sagemaker.Session()
role = get_execution_role()           # or create via IAM console
bucket = sess.default_bucket()        # auto-creates sagemaker-{region}-{account}
prefix = "adult-income"
print(f"Role: {role}")
print(f"Bucket: {bucket}")
```

### Verify AWS credentials locally
```bash
aws sts get-caller-identity
aws s3 ls                             # should list your buckets
```

## Phase 2: Data Preparation & S3 Upload (Days 2–3)

In `notebooks/01_data_prep_s3.ipynb`:

### Clean and encode
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Drop missing values
df = df.dropna()

# Encode target
df["label"] = (df["income"].str.strip() == ">50K").astype(int)
df = df.drop("income", axis=1)

# One-hot encode categoricals
cat_cols = ["workclass","education","marital-status","occupation",
            "relationship","race","sex","native-country"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# XGBoost expects label in first column
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = pd.concat([train_df["label"], train_df.drop("label", axis=1)], axis=1)
test_df  = pd.concat([test_df["label"],  test_df.drop("label",  axis=1)], axis=1)

# Save without header (SageMaker XGBoost requirement)
train_df.to_csv("train.csv", index=False, header=False)
test_df.to_csv("test.csv", index=False, header=False)
```

### Upload to S3
```python
import boto3

s3 = boto3.client("s3")
s3.upload_file("train.csv", bucket, f"{prefix}/data/train.csv")
s3.upload_file("test.csv",  bucket, f"{prefix}/data/test.csv")

train_uri = f"s3://{bucket}/{prefix}/data/train.csv"
test_uri  = f"s3://{bucket}/{prefix}/data/test.csv"
print(f"Train: {train_uri}")
```

## Phase 3: SageMaker Training Job (Days 4–6)

In `notebooks/02_sagemaker_training.ipynb`:

### Train with Built-in XGBoost
```python
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

xgb_image = sagemaker.image_uris.retrieve("xgboost", sess.boto_region_name, "1.7-1")

xgb = Estimator(
    image_uri=xgb_image,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",         # ~$0.115/hr — use spot to save 70%
    use_spot_instances=True,
    max_run=3600,
    max_wait=7200,
    output_path=f"s3://{bucket}/{prefix}/output",
    sagemaker_session=sess,
)

xgb.set_hyperparameters(
    objective="binary:logistic",
    num_round=100,
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    eval_metric="auc",
)

xgb.fit({
    "train": TrainingInput(train_uri, content_type="text/csv"),
    "validation": TrainingInput(test_uri, content_type="text/csv"),
})
```

### Check training metrics
```python
# Get AUC from CloudWatch metrics
analytics = xgb.training_job_analytics()
analytics.dataframe().tail(5)
```

## Phase 4: Deploy Real-Time Endpoint (Days 7–8)

```python
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",       # cheapest inference instance
    serializer=CSVSerializer(),
)

# Test with a single row (no label column)
sample = test_df.drop(columns=[0]).iloc[0].values.tolist()
csv_str = ",".join(str(v) for v in sample)
result = predictor.predict(csv_str)
print(f"Prediction: {result}")          # probability score

# IMPORTANT: Delete endpoint after testing to avoid charges
# predictor.delete_endpoint()
```

## Phase 5: Batch Transform (Day 9)

```python
transformer = xgb.transformer(
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{bucket}/{prefix}/batch-output",
)

transformer.transform(
    data=test_uri,
    data_type="S3Prefix",
    content_type="text/csv",
    split_type="Line",
)
transformer.wait()

# Download and evaluate
import io
s3 = boto3.client("s3")
obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/batch-output/test.csv.out")
preds = pd.read_csv(io.BytesIO(obj["Body"].read()), header=None)
```

## Phase 6: Model Monitor (Days 10–12)

```python
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

# Create baseline from training data
monitor.suggest_baseline(
    baseline_dataset=train_uri,
    dataset_format=DatasetFormat.csv(header=False),
    output_s3_uri=f"s3://{bucket}/{prefix}/baseline",
    wait=True,
)

# Schedule hourly monitoring
from sagemaker.model_monitor import CronExpressionGenerator

monitor.create_monitoring_schedule(
    monitor_schedule_name="adult-income-monitor",
    endpoint_input=predictor.endpoint_name,
    output_s3_uri=f"s3://{bucket}/{prefix}/monitor-reports",
    statistics=monitor.baseline_statistics(),
    constraints=monitor.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly(),
)
```

## Evaluation Metrics to Report

| Metric | Target | Notes |
|---|---|---|
| AUC (validation) | > 0.87 | XGBoost built-in metric |
| Accuracy | > 85% | Post-inference evaluation |
| Precision | > 0.80 | For >50K class |
| Recall | > 0.65 | For >50K class |
| Training cost | < $1.00 | Using spot instances |

## Cost Management

Track costs in the Cost Explorer:
```bash
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-12-31 \
  --granularity MONTHLY \
  --filter '{"Dimensions": {"Key": "SERVICE", "Values": ["Amazon SageMaker"]}}' \
  --metrics "BlendedCost"
```

**Always delete endpoints when not in use:**
```python
# List all endpoints
sm = boto3.client("sagemaker")
endpoints = sm.list_endpoints()["Endpoints"]
for ep in endpoints:
    print(ep["EndpointName"], ep["EndpointStatus"])

# Delete
predictor.delete_endpoint()
```

## ADR Template

Create `docs/adr/001-algorithm-selection.md`:
```markdown
# ADR-001: Algorithm Selection for Income Classification

## Status: Accepted

## Context
Binary classification on tabular data with mixed numeric/categorical features.
Need high accuracy with low inference latency on AWS managed infrastructure.

## Decision
SageMaker built-in XGBoost 1.7 with spot instance training.

## Consequences
+ No custom container needed — faster iteration
+ Native S3/CloudWatch integration
+ Spot instances cut training cost by ~70%
- Less control over preprocessing vs custom script mode
- Locked to SageMaker's XGBoost version cadence
```

## Acceptance Criteria
- [ ] Training job completes with validation AUC > 0.87
- [ ] Real-time endpoint returns valid probability score (0–1)
- [ ] Batch transform produces output file in S3
- [ ] Model Monitor baseline created and schedule active
- [ ] All endpoints deleted after testing (cost check)
- [ ] ADR-001 committed to `docs/adr/`
- [ ] Training cost stays under $2 total (spot + inference)
