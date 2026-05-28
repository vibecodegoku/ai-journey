# Project 02: AWS SageMaker Classification Pipeline — Implementation Guide

## Prerequisites
- Project 01 complete
- AWS account with free tier active (12 months)
- AWS CLI configured (`aws configure`)
- `conda activate aiarch`
- Packages: `boto3`, `sagemaker`, `pandas`, `scikit-learn`

---

## Project Structure
```
02-cloud-ml-aws/
├── data/
│   └── adult_income.csv       # Downloaded UCI dataset
├── data_prep.py               # Feature engineering + S3 upload
├── train_sagemaker.py         # Launch SageMaker XGBoost training job
├── deploy.py                  # Deploy real-time endpoint
├── batch_transform.py         # Run batch inference
├── monitor.py                 # Set up Model Monitor
├── cleanup.py                 # Delete all resources (avoid charges)
├── predict.py                 # Test the live endpoint
├── docs/
│   └── adr/
│       └── ADR-001-algorithm-selection.md
└── requirements.txt
```

---

## Phase 1: AWS Setup

### Step 1.1 — Create IAM Role for SageMaker

In the AWS Console (or via CLI):
```bash
# Create the trust policy file
cat > /tmp/sagemaker-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "sagemaker.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create the role
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document file:///tmp/sagemaker-trust.json

# Attach managed policies
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Get your account ID and role ARN (save this!)
aws sts get-caller-identity --query Account --output text
aws iam get-role --role-name SageMakerExecutionRole --query Role.Arn --output text
```

Save the Role ARN — it looks like: `arn:aws:iam::123456789012:role/SageMakerExecutionRole`

### Step 1.2 — Create S3 bucket
```bash
# Replace with a globally unique name (e.g., your-name-aiarch-2026)
BUCKET_NAME="your-name-aiarch-$(date +%Y)"
REGION="us-east-1"

aws s3 mb s3://$BUCKET_NAME --region $REGION
echo "Bucket: $BUCKET_NAME"
```

### Step 1.3 — Set environment variables
Create a `.env` file in the project root:
```bash
cat > .env << EOF
AWS_REGION=us-east-1
S3_BUCKET=$BUCKET_NAME
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
EOF
```

---

## Phase 2: Data Preparation

### Step 2.1 — Download UCI Adult Income dataset
```bash
mkdir -p data
curl -o data/adult_income.csv \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
```

If the URL is unavailable, create the data programmatically — see data_prep.py below.

### Step 2.2 — Create data_prep.py

```python
# data_prep.py
import os
import pandas as pd
import numpy as np
import boto3
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.environ["S3_BUCKET"]
REGION = os.environ.get("AWS_REGION", "us-east-1")
PREFIX = "adult-income"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

CATEGORICAL = [
    "workclass", "education", "marital-status",
    "occupation", "relationship", "race", "sex", "native-country",
]


def load_data():
    try:
        df = pd.read_csv("data/adult_income.csv", names=COLUMNS, header=None,
                         skipinitialspace=True)
    except FileNotFoundError:
        # Fallback: generate synthetic data for testing
        print("WARNING: Dataset not found, generating synthetic data")
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "age": np.random.randint(18, 90, n),
            "workclass": np.random.choice(["Private", "Self-emp", "Government"], n),
            "fnlwgt": np.random.randint(10000, 1000000, n),
            "education": np.random.choice(["Bachelors", "HS-grad", "Masters", "Some-college"], n),
            "education-num": np.random.randint(1, 16, n),
            "marital-status": np.random.choice(["Married", "Single", "Divorced"], n),
            "occupation": np.random.choice(["Tech-support", "Craft-repair", "Sales", "Exec-managerial"], n),
            "relationship": np.random.choice(["Husband", "Wife", "Own-child", "Not-in-family"], n),
            "race": np.random.choice(["White", "Black", "Asian-Pac-Islander", "Other"], n),
            "sex": np.random.choice(["Male", "Female"], n),
            "capital-gain": np.random.randint(0, 100000, n),
            "capital-loss": np.random.randint(0, 4356, n),
            "hours-per-week": np.random.randint(1, 99, n),
            "native-country": np.random.choice(["United-States", "Mexico", "India", "Philippines"], n),
            "income": np.random.choice(["<=50K", ">50K"], n),
        })
    return df


def preprocess(df):
    # Target: binary label (1 = income >50K)
    df = df.copy()
    df["income"] = (df["income"].str.strip().str.rstrip(".") == ">50K").astype(int)

    # Drop fnlwgt (sampling weight, not a feature)
    df = df.drop(columns=["fnlwgt"])

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)

    # Ensure all values are numeric
    df = df.astype(float)

    # SageMaker XGBoost expects label in the FIRST column
    cols = ["income"] + [c for c in df.columns if c != "income"]
    df = df[cols]

    return df


def split_and_upload(df):
    train = df.sample(frac=0.8, random_state=42)
    test = df.drop(train.index)
    val = train.sample(frac=0.1, random_state=42)
    train = train.drop(val.index)

    print(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

    s3 = boto3.client("s3", region_name=REGION)

    def upload_df(data, key):
        buf = StringIO()
        data.to_csv(buf, index=False, header=False)
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        uri = f"s3://{BUCKET}/{key}"
        print(f"Uploaded {key}: {uri}")
        return uri

    train_uri = upload_df(train, f"{PREFIX}/train/train.csv")
    val_uri   = upload_df(val,   f"{PREFIX}/validation/validation.csv")
    test_uri  = upload_df(test,  f"{PREFIX}/test/test.csv")

    return train_uri, val_uri, test_uri


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    df = preprocess(df)
    print(f"After preprocessing: {df.shape[1]} columns")
    train_uri, val_uri, test_uri = split_and_upload(df)
    print("\nData prep complete!")
    print(f"Train URI: {train_uri}")
    print(f"Validation URI: {val_uri}")
    print(f"Test URI: {test_uri}")
```

### Step 2.3 — Run data prep
```bash
conda activate aiarch
pip install python-dotenv
python data_prep.py
```

---

## Phase 3: SageMaker Training with XGBoost + Spot Instances

### Step 3.1 — Create train_sagemaker.py

```python
# train_sagemaker.py
import os
import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ["S3_BUCKET"]
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
PREFIX = "adult-income"

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

# Retrieve the XGBoost built-in image URI
xgb_image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=REGION,
    version="1.7-1",
)
print(f"XGBoost image URI: {xgb_image_uri}")


def launch_training():
    estimator = Estimator(
        image_uri=xgb_image_uri,
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.large",
        # Spot instances — up to 90% cheaper
        use_spot_instances=True,
        max_run=3600,           # Max 1 hour
        max_wait=7200,          # Wait up to 2 hours for spot capacity
        output_path=f"s3://{BUCKET}/{PREFIX}/model-artifacts",
        sagemaker_session=session,
        hyperparameters={
            "objective":        "binary:logistic",
            "num_round":        100,
            "max_depth":        5,
            "eta":              0.2,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "eval_metric":      "auc",
            "early_stopping_rounds": 10,
        },
    )

    train_input = TrainingInput(
        s3_data=f"s3://{BUCKET}/{PREFIX}/train/",
        content_type="text/csv",
    )
    val_input = TrainingInput(
        s3_data=f"s3://{BUCKET}/{PREFIX}/validation/",
        content_type="text/csv",
    )

    print("Starting training job (spot instance)...")
    estimator.fit({"train": train_input, "validation": val_input})
    print(f"\nTraining complete. Job name: {estimator.latest_training_job.name}")
    print(f"Model artifacts: {estimator.model_data}")
    return estimator


if __name__ == "__main__":
    estimator = launch_training()
```

### Step 3.2 — Run training
```bash
python train_sagemaker.py
```
> Training takes 5–15 minutes. Spot instances may need to wait for capacity.

---

## Phase 4: Deploy Real-Time Endpoint

### Step 4.1 — Create deploy.py

```python
# deploy.py
import os
import json
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ["S3_BUCKET"]
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
ENDPOINT_NAME = "adult-income-xgboost"

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))


def deploy_endpoint(model_data_uri: str):
    """Deploy XGBoost model to a real-time endpoint."""
    from sagemaker.xgboost import XGBoostModel

    xgb_model = XGBoostModel(
        model_data=model_data_uri,
        role=ROLE_ARN,
        framework_version="1.7-1",
        sagemaker_session=session,
    )

    predictor = xgb_model.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium",  # Cheapest option for free tier
        endpoint_name=ENDPOINT_NAME,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
    )

    print(f"Endpoint deployed: {ENDPOINT_NAME}")
    return predictor


def test_endpoint():
    """Send a test CSV row to the endpoint."""
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    # A sample row (without the label column)
    # Must match the feature columns produced by data_prep.py
    sample = "39,13,0,2401,0,0,40" + ",0" * 50  # Simplified placeholder

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=sample,
    )
    result = json.loads(response["Body"].read())
    print(f"Prediction (probability of >50K income): {result}")


if __name__ == "__main__":
    # You'll need the model_data URI from train_sagemaker.py output
    model_uri = input("Enter model artifact URI (s3://...): ").strip()
    predictor = deploy_endpoint(model_uri)
    print("\nEndpoint is live. Testing...")
    # test_endpoint()  # Uncomment after preparing a proper test row
```

### Step 4.2 — Deploy
```bash
python deploy.py
# Paste the s3:// URI printed by train_sagemaker.py
```
Deployment takes 5–8 minutes.

---

## Phase 5: Batch Transform

### Step 5.1 — Create batch_transform.py

```python
# batch_transform.py
import os
import boto3
import sagemaker
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ["S3_BUCKET"]
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
PREFIX = "adult-income"

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))


def run_batch_transform(model_name: str):
    """Run batch inference on the test split."""
    transformer = sagemaker.transformer.Transformer(
        model_name=model_name,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=f"s3://{BUCKET}/{PREFIX}/batch-output/",
        sagemaker_session=session,
        accept="text/csv",
        assemble_with="Line",
    )

    transformer.transform(
        data=f"s3://{BUCKET}/{PREFIX}/test/test.csv",
        content_type="text/csv",
        split_type="Line",
        join_source="Input",   # Append predictions to input rows
    )

    transformer.wait()
    print(f"Batch transform complete. Output at: s3://{BUCKET}/{PREFIX}/batch-output/")


if __name__ == "__main__":
    model_name = input("Enter SageMaker model name: ").strip()
    run_batch_transform(model_name)
```

---

## Phase 6: Model Monitor

### Step 6.1 — Create monitor.py

```python
# monitor.py
import os
import boto3
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor, MonitoringOutput
from sagemaker.model_monitor.dataset_format import DatasetFormat
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ["S3_BUCKET"]
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
ENDPOINT_NAME = "adult-income-xgboost"
PREFIX = "adult-income"

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))


def setup_monitoring():
    monitor = DefaultModelMonitor(
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=session,
    )

    # Suggest baseline from training data
    print("Running baseline job (takes ~5 min)...")
    monitor.suggest_baseline(
        baseline_dataset=f"s3://{BUCKET}/{PREFIX}/train/train.csv",
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=f"s3://{BUCKET}/{PREFIX}/monitor-baseline/",
        wait=True,
    )

    print("Baseline complete. Scheduling hourly monitoring...")

    # Schedule monitoring every hour (minimum frequency)
    monitor.create_monitoring_schedule(
        monitor_schedule_name=f"{ENDPOINT_NAME}-monitor",
        endpoint_input=ENDPOINT_NAME,
        output=MonitoringOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{BUCKET}/{PREFIX}/monitor-output/",
        ),
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression="cron(0 * ? * * *)",  # Hourly
    )

    print(f"Monitoring schedule active for endpoint: {ENDPOINT_NAME}")
    return monitor


if __name__ == "__main__":
    setup_monitoring()
```

### Step 6.2 — Run monitor setup
```bash
python monitor.py
```

---

## Phase 7: Cost Tracking & Cleanup

### Step 7.1 — Create cleanup.py

```python
# cleanup.py
"""
Run this when done to avoid ongoing charges.
Deletes endpoint, model, and monitoring schedule.
"""
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT_NAME = "adult-income-xgboost"

sm = boto3.client("sagemaker", region_name=REGION)


def cleanup():
    # Delete monitoring schedule
    try:
        sm.delete_monitoring_schedule(
            MonitoringScheduleName=f"{ENDPOINT_NAME}-monitor"
        )
        print("Deleted monitoring schedule")
    except sm.exceptions.ResourceNotFound:
        print("No monitoring schedule found")

    # Delete endpoint
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Deleted endpoint: {ENDPOINT_NAME}")
    except sm.exceptions.ClientError:
        print("Endpoint not found or already deleted")

    # List and delete associated endpoint configs
    configs = sm.list_endpoint_configs(NameContains=ENDPOINT_NAME)
    for cfg in configs.get("EndpointConfigs", []):
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        print(f"Deleted endpoint config: {cfg['EndpointConfigName']}")

    print("\nCleanup complete. Check AWS Cost Explorer to confirm no remaining resources.")


if __name__ == "__main__":
    confirm = input("This will delete all project resources. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        cleanup()
    else:
        print("Aborted.")
```

### Step 7.2 — Check estimated cost
```bash
aws ce get-cost-and-usage \
  --time-period Start=2026-05-01,End=2026-06-01 \
  --granularity MONTHLY \
  --metrics "UnblendedCost" \
  --filter '{"Dimensions": {"Key": "SERVICE", "Values": ["Amazon SageMaker"]}}'
```

---

## Phase 8: Architecture Decision Record

Create `docs/adr/ADR-001-algorithm-selection.md`:

```markdown
# ADR-001: Algorithm Selection for Income Classification

**Date:** 2026-05-27
**Status:** Accepted

## Context
Binary classification on UCI Adult Income dataset (predict whether income >$50K).
We need AUC > 0.87, accuracy > 85%, and must use AWS SageMaker's built-in algorithms
to benefit from managed infrastructure and spot instance training.

## Decision
Use **SageMaker built-in XGBoost** (binary:logistic objective).

## Alternatives Considered
| Algorithm | Pros | Cons |
|-----------|------|------|
| **XGBoost** | High AUC on tabular data, native SageMaker support, spot instances | Less interpretable than logistic regression |
| Linear Learner | Fast, interpretable | Lower AUC (~0.82) on non-linear patterns |
| Random Cut Forest | Good for anomaly detection | Not designed for classification |

## Rationale
- XGBoost consistently achieves AUC > 0.87 on Adult Income dataset benchmarks
- Native SageMaker image eliminates container management
- Spot instance support reduces training cost by ~70%
- Hyperparameter tuning via SageMaker Automatic Model Tuning available if needed

## Consequences
- Model is not easily exportable to non-XGBoost runtimes
- Monitoring requires SageMaker Model Monitor (additional cost)
- Local testing requires installing xgboost package separately
```

---

## Verification Checklist
- [ ] S3 bucket created with train/validation/test CSV files
- [ ] `python data_prep.py` completes without errors
- [ ] SageMaker training job shows status `Completed` in AWS Console
- [ ] Training job AUC metric > 0.87 (visible in CloudWatch Logs)
- [ ] `python deploy.py` creates endpoint in `InService` state
- [ ] `POST` to endpoint `/invocations` returns probabilities (0–1)
- [ ] Batch transform output file exists in S3
- [ ] Monitoring schedule active (visible in SageMaker Console → Endpoints → Monitor)
- [ ] Total SageMaker cost < $2
- [ ] `python cleanup.py` deletes endpoint successfully
- [ ] ADR-001 written

---

## Troubleshooting

**Training job stuck in `Starting` state**
Spot instances may wait for capacity. Set `use_spot_instances=False` to use on-demand (slightly more expensive) to proceed faster.

**`ResourceLimitExceeded` error**
You may have hit a SageMaker quota. In the AWS Console → Service Quotas → SageMaker, request a limit increase for `ml.m5.large` training instances.

**Endpoint deployment fails with `FailedToLoadModel`**
Make sure the model URI points to a `.tar.gz` in S3, not just the folder.

**Billing concerns**
Always run `cleanup.py` when done. Real-time endpoints accrue cost per hour even when idle.

---

## Next Steps → Project 03: Azure Document Intelligence
```bash
mkdir -p ~/Documents/ai-journey/projects/03-azure-ai-services
cd ~/Documents/ai-journey/projects/03-azure-ai-services
```
