# Project 04: AWS Bedrock Multi-Modal Chat — Implementation Guide

## Prerequisites
- Project 02 complete (AWS CLI configured, boto3 installed)
- AWS account with Bedrock model access enabled
- `conda activate aiarch`
- Packages: `boto3`, `streamlit`, `python-dotenv`

---

## Project Structure
```
04-aws-bedrock/
├── bedrock_client.py          # Multi-model Bedrock wrapper
├── memory.py                  # DynamoDB conversation history
├── cost_tracker.py            # Per-session cost calculation
├── app.py                     # Streamlit chat UI
├── notebooks/
│   └── model_comparison.ipynb # Latency/cost/quality benchmark
├── docs/
│   └── adr/
│       └── ADR-001-model-selection.md
└── .env
```

---

## Phase 1: Enable Bedrock Model Access

### Step 1.1 — Request model access in AWS Console
1. Open AWS Console → Amazon Bedrock → Model access (left sidebar)
2. Click **Manage model access**
3. Enable these models:
   - **Anthropic Claude 3 Haiku** (claude-3-haiku-20240307-v1:0)
   - **Anthropic Claude 3 Sonnet** (claude-3-sonnet-20240229-v1:0)
   - **Amazon Titan Text G1 - Express** (amazon.titan-text-express-v1)
   - **Mistral 7B Instruct** (mistral.mistral-7b-instruct-v0:2)
4. Click **Save changes** — takes 1–5 minutes to activate

### Step 1.2 — Create DynamoDB table
```bash
aws dynamodb create-table \
  --table-name "BedrockConversations" \
  --attribute-definitions \
    AttributeName=session_id,AttributeType=S \
    AttributeName=timestamp,AttributeType=S \
  --key-schema \
    AttributeName=session_id,KeyType=HASH \
    AttributeName=timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

### Step 1.3 — Create .env
```bash
cat > .env << 'EOF'
AWS_REGION=us-east-1
DYNAMODB_TABLE=BedrockConversations
EOF
```

---

## Phase 2: Bedrock Client

### Step 2.1 — Create bedrock_client.py

```python
# bedrock_client.py
import os
import json
import base64
import time
from pathlib import Path
from typing import Iterator
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Model IDs and their pricing (per 1K tokens)
MODEL_CONFIGS = {
    "claude-3-haiku": {
        "model_id":    "anthropic.claude-3-haiku-20240307-v1:0",
        "input_cost":  0.00025,   # $0.25 per 1M input tokens
        "output_cost": 0.00125,   # $1.25 per 1M output tokens
        "supports_vision": True,
    },
    "claude-3-sonnet": {
        "model_id":    "anthropic.claude-3-sonnet-20240229-v1:0",
        "input_cost":  0.003,
        "output_cost": 0.015,
        "supports_vision": True,
    },
    "titan-express": {
        "model_id":    "amazon.titan-text-express-v1",
        "input_cost":  0.0008,
        "output_cost": 0.0016,
        "supports_vision": False,
    },
    "mistral-7b": {
        "model_id":    "mistral.mistral-7b-instruct-v0:2",
        "input_cost":  0.00015,
        "output_cost": 0.0002,
        "supports_vision": False,
    },
}


class BedrockClient:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=REGION)

    def _build_claude_body(
        self,
        messages: list[dict],
        system_prompt: str,
        image_path: str | None = None,
        max_tokens: int = 1024,
    ) -> dict:
        """Build Anthropic Claude API request body."""
        # If image provided, add it to the last user message
        if image_path:
            image_data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
            ext = Path(image_path).suffix.lower().lstrip(".")
            media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": messages[-1]["content"]},
            ]
            messages = messages[:-1] + [{"role": "user", "content": content}]

        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }

    def _build_titan_body(self, prompt: str, max_tokens: int = 1024) -> dict:
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": 0.7,
                "topP": 0.9,
            },
        }

    def _build_mistral_body(self, messages: list[dict], max_tokens: int = 1024) -> dict:
        # Mistral uses chat completions format
        prompt = "\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in messages])
        return {
            "prompt": f"<s>[INST] {prompt} [/INST]",
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

    def invoke(
        self,
        model_key: str,
        messages: list[dict],
        system_prompt: str = "You are a helpful AI assistant.",
        image_path: str | None = None,
        max_tokens: int = 1024,
    ) -> dict:
        """Invoke a model and return response + usage metrics."""
        config = MODEL_CONFIGS[model_key]
        model_id = config["model_id"]

        start = time.perf_counter()

        if model_key in ("claude-3-haiku", "claude-3-sonnet"):
            body = self._build_claude_body(messages, system_prompt, image_path, max_tokens)
        elif model_key == "titan-express":
            prompt = "\n".join([m["content"] for m in messages])
            body = self._build_titan_body(prompt, max_tokens)
        elif model_key == "mistral-7b":
            body = self._build_mistral_body(messages, max_tokens)
        else:
            raise ValueError(f"Unknown model: {model_key}")

        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        latency_ms = (time.perf_counter() - start) * 1000
        response_body = json.loads(response["body"].read())

        # Parse output based on model
        if model_key in ("claude-3-haiku", "claude-3-sonnet"):
            text = response_body["content"][0]["text"]
            input_tokens = response_body["usage"]["input_tokens"]
            output_tokens = response_body["usage"]["output_tokens"]
        elif model_key == "titan-express":
            text = response_body["results"][0]["outputText"]
            input_tokens = response_body.get("inputTextTokenCount", 0)
            output_tokens = response_body["results"][0].get("tokenCount", 0)
        elif model_key == "mistral-7b":
            text = response_body["outputs"][0]["text"]
            input_tokens = 0   # Mistral doesn't return token counts
            output_tokens = 0

        cost = (
            input_tokens / 1000 * config["input_cost"] +
            output_tokens / 1000 * config["output_cost"]
        )

        return {
            "text": text,
            "model": model_key,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": round(cost, 6),
        }

    def stream(
        self,
        model_key: str,
        messages: list[dict],
        system_prompt: str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream response tokens from a Claude model."""
        if model_key not in ("claude-3-haiku", "claude-3-sonnet"):
            # Fall back to non-streaming for other models
            result = self.invoke(model_key, messages, system_prompt, max_tokens=max_tokens)
            yield result["text"]
            return

        config = MODEL_CONFIGS[model_key]
        body = self._build_claude_body(messages, system_prompt, max_tokens=max_tokens)

        response = self.client.invoke_model_with_response_stream(
            modelId=config["model_id"],
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta.get("text", "")
```

---

## Phase 3: DynamoDB Conversation Memory

### Step 3.1 — Create memory.py

```python
# memory.py
import os
from datetime import datetime, timezone
import boto3
from boto3.dynamodb.conditions import Key
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "BedrockConversations")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_table():
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    return dynamodb.Table(TABLE_NAME)


def save_message(session_id: str, role: str, content: str, model: str = ""):
    """Persist a chat message to DynamoDB."""
    table = get_table()
    table.put_item(Item={
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "content": content,
        "model": model,
    })


def get_history(session_id: str, limit: int = 20) -> list[dict]:
    """Retrieve recent messages for a session."""
    table = get_table()
    response = table.query(
        KeyConditionExpression=Key("session_id").eq(session_id),
        Limit=limit,
        ScanIndexForward=True,  # Oldest first
    )
    return [
        {"role": item["role"], "content": item["content"]}
        for item in response["Items"]
    ]


def clear_session(session_id: str):
    """Delete all messages for a session."""
    table = get_table()
    response = table.query(
        KeyConditionExpression=Key("session_id").eq(session_id),
    )
    with table.batch_writer() as batch:
        for item in response["Items"]:
            batch.delete_item(Key={
                "session_id": item["session_id"],
                "timestamp": item["timestamp"],
            })
```

---

## Phase 4: Cost Tracker

### Step 4.1 — Create cost_tracker.py

```python
# cost_tracker.py
from collections import defaultdict


class CostTracker:
    """Track token usage and cost per session."""

    def __init__(self):
        self._sessions: dict[str, list[dict]] = defaultdict(list)

    def record(self, session_id: str, model: str, input_tokens: int,
               output_tokens: int, cost_usd: float, latency_ms: float):
        self._sessions[session_id].append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        })

    def session_summary(self, session_id: str) -> dict:
        records = self._sessions.get(session_id, [])
        if not records:
            return {"total_cost_usd": 0, "total_tokens": 0, "calls": 0}
        return {
            "total_cost_usd": round(sum(r["cost_usd"] for r in records), 6),
            "total_input_tokens": sum(r["input_tokens"] for r in records),
            "total_output_tokens": sum(r["output_tokens"] for r in records),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in records) / len(records), 1),
            "calls": len(records),
        }

    def all_sessions_total(self) -> float:
        return round(sum(
            r["cost_usd"]
            for records in self._sessions.values()
            for r in records
        ), 6)
```

---

## Phase 5: Streamlit Chat UI

### Step 5.1 — Create app.py

```python
# app.py
import uuid
import streamlit as st
from bedrock_client import BedrockClient, MODEL_CONFIGS
from memory import save_message, get_history, clear_session
from cost_tracker import CostTracker

st.set_page_config(page_title="Bedrock Chat", page_icon="🤖", layout="wide")
st.title("AWS Bedrock Multi-Model Chat")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "cost_tracker" not in st.session_state:
    st.session_state.cost_tracker = CostTracker()
if "client" not in st.session_state:
    st.session_state.client = BedrockClient()

client = st.session_state.client
tracker = st.session_state.cost_tracker
session_id = st.session_state.session_id

# Sidebar controls
with st.sidebar:
    st.header("Settings")

    model_key = st.selectbox(
        "Model",
        options=list(MODEL_CONFIGS.keys()),
        format_func=lambda k: f"{k} ({'vision' if MODEL_CONFIGS[k]['supports_vision'] else 'text'})",
    )

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant.",
        height=100,
    )

    use_streaming = st.toggle("Streaming", value=True)
    max_tokens = st.slider("Max tokens", 128, 4096, 1024, step=128)

    uploaded_image = st.file_uploader(
        "Upload image (vision models only)",
        type=["jpg", "jpeg", "png"],
        disabled=not MODEL_CONFIGS[model_key]["supports_vision"],
    )

    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        session_id = st.session_state.session_id
        st.rerun()

    st.divider()
    st.subheader(f"Session: {session_id}")
    summary = tracker.session_summary(session_id)
    st.metric("Session Cost", f"${summary.get('total_cost_usd', 0):.6f}")
    st.metric("Calls", summary.get("calls", 0))
    st.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f} ms")

# Load history from DynamoDB
history = get_history(session_id)

# Display chat history
for msg in history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything..."):
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)
    save_message(session_id, "user", prompt)

    # Build messages list
    messages = get_history(session_id)

    # Save uploaded image temporarily
    image_path = None
    if uploaded_image and MODEL_CONFIGS[model_key]["supports_vision"]:
        import tempfile, os
        suffix = os.path.splitext(uploaded_image.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded_image.read())
            image_path = f.name

    # Generate response
    with st.chat_message("assistant"):
        if use_streaming and model_key in ("claude-3-haiku", "claude-3-sonnet"):
            # Streaming
            response_text = ""
            placeholder = st.empty()
            for chunk in client.stream(model_key, messages, system_prompt, max_tokens):
                response_text += chunk
                placeholder.markdown(response_text + "▌")
            placeholder.markdown(response_text)
            # Record cost (approximate for streaming)
            tracker.record(session_id, model_key, 0, 0, 0.0, 0.0)
        else:
            # Non-streaming
            with st.spinner("Thinking..."):
                result = client.invoke(
                    model_key, messages, system_prompt, image_path, max_tokens
                )
            response_text = result["text"]
            st.write(response_text)
            tracker.record(
                session_id, model_key,
                result["input_tokens"], result["output_tokens"],
                result["cost_usd"], result["latency_ms"],
            )
            st.caption(
                f"Tokens: {result['input_tokens']} in / {result['output_tokens']} out | "
                f"Cost: ${result['cost_usd']:.6f} | Latency: {result['latency_ms']:.0f}ms"
            )

    save_message(session_id, "assistant", response_text, model_key)

    if image_path:
        import os; os.unlink(image_path)
```

### Step 5.2 — Run Streamlit app
```bash
conda activate aiarch
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## Phase 6: Model Comparison Notebook

### Step 6.1 — Create notebooks/model_comparison.ipynb

Open JupyterLab and create a new notebook with these cells:

**Cell 1 — Setup**
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
from bedrock_client import BedrockClient, MODEL_CONFIGS
import pandas as pd
import matplotlib.pyplot as plt
import time

client = BedrockClient()
```

**Cell 2 — Benchmark prompts**
```python
test_prompts = [
    "Explain transformer architecture in 2 sentences.",
    "Write a Python function to reverse a linked list.",
    "What are the trade-offs between SQL and NoSQL databases?",
]

results = []
for model_key in MODEL_CONFIGS.keys():
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            result = client.invoke(model_key, messages)
            results.append({
                "model": model_key,
                "prompt": prompt[:40] + "...",
                "latency_ms": result["latency_ms"],
                "cost_usd": result["cost_usd"],
                "output_tokens": result["output_tokens"],
                "response_preview": result["text"][:100],
            })
            print(f"[OK] {model_key}: {result['latency_ms']:.0f}ms, ${result['cost_usd']:.6f}")
        except Exception as e:
            print(f"[SKIP] {model_key}: {e}")

df = pd.DataFrame(results)
df
```

**Cell 3 — Visualize results**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Avg latency per model
latency_by_model = df.groupby("model")["latency_ms"].mean()
axes[0].bar(latency_by_model.index, latency_by_model.values)
axes[0].set_title("Average Latency (ms)")
axes[0].tick_params(axis='x', rotation=45)

# Avg cost per model
cost_by_model = df.groupby("model")["cost_usd"].mean()
axes[1].bar(cost_by_model.index, cost_by_model.values)
axes[1].set_title("Average Cost per Call (USD)")
axes[1].tick_params(axis='x', rotation=45)

# Output tokens
tokens_by_model = df.groupby("model")["output_tokens"].mean()
axes[2].bar(tokens_by_model.index, tokens_by_model.values)
axes[2].set_title("Average Output Tokens")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("Saved model_comparison.png")
```

---

## Phase 7: Architecture Decision Record

Create `docs/adr/ADR-001-model-selection.md`:

```markdown
# ADR-001: Bedrock Model Selection Strategy

**Date:** 2026-05-27
**Status:** Accepted

## Context
We need to support multiple LLM models via AWS Bedrock for a multi-modal chat application.
Models differ in cost, latency, quality, and vision capabilities.

## Decision
Default model: **Claude 3 Haiku** for cost-efficiency. Allow user to switch to Claude 3 Sonnet
for complex tasks. Include Titan and Mistral as cost-effective alternatives for simple queries.

## Model Comparison

| Model | Avg Latency | Cost/1K tokens | Vision | Notes |
|-------|------------|----------------|--------|-------|
| Claude 3 Haiku | ~800ms | $0.00025/$0.00125 | Yes | Best cost/quality ratio |
| Claude 3 Sonnet | ~2000ms | $0.003/$0.015 | Yes | Best quality, 12x costlier |
| Titan Express | ~600ms | $0.0008/$0.0016 | No | Amazon-native, moderate quality |
| Mistral 7B | ~700ms | $0.00015/$0.0002 | No | Cheapest, lower quality |

## Rationale
Claude 3 Haiku provides the best balance: fast enough for real-time chat (~800ms),
supports vision, and costs <$0.001 per typical conversation turn.

## Consequences
- Streaming only supported for Claude models via `invoke_model_with_response_stream`
- Vision requires image base64 encoding before API call
- DynamoDB session cost: ~$0.00025 per 1K read/write units (negligible)
```

---

## Verification Checklist
- [ ] All 4 models accessible in Bedrock console (Model access: Enabled)
- [ ] DynamoDB table `BedrockConversations` created
- [ ] `python bedrock_client.py` (add a `__main__` test) returns text from Claude 3 Haiku
- [ ] `streamlit run app.py` opens UI at localhost:8501
- [ ] Image upload works with Claude 3 Haiku (vision model)
- [ ] Switching models in dropdown works
- [ ] Streaming toggle shows text appearing progressively
- [ ] Session cost updates in sidebar after each message
- [ ] DynamoDB: messages appear in AWS Console → DynamoDB → Explore items
- [ ] `model_comparison.ipynb` runs and saves comparison chart
- [ ] Total Bedrock cost < $15
- [ ] ADR-001 written

---

## Troubleshooting

**`AccessDeniedException` from Bedrock**
The model isn't enabled. Go to AWS Console → Bedrock → Model access and enable it. Takes 1–5 min.

**`ThrottlingException`**
Bedrock has per-minute rate limits on free tier. Add `time.sleep(1)` between requests in the notebook.

**Streaming not working**
Ensure you're using `invoke_model_with_response_stream`, not `invoke_model`. Only Claude models support streaming via Bedrock.

**DynamoDB `ResourceNotFoundException`**
Table not created yet. Run the `aws dynamodb create-table` command from Phase 1.

---

## Next Steps → Project 05: LLM Fine-Tuning with QLoRA
```bash
# Project 05 runs in Google Colab (T4 GPU) — open colab.research.google.com
# Clone or upload the project 05 notebooks there
mkdir -p ~/Documents/ai-journey/projects/05-llm-finetuning/notebooks
```
