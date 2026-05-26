# Project 04: AWS Bedrock Multi-Modal Chat

**XP:** 300 | **Cost:** ~$5–15 (Bedrock pay-per-token) | **Duration:** Weeks 7–8

## Objective
Build a production-grade multi-modal chat application using AWS Bedrock that supports text, images, and documents. Implement conversation memory with DynamoDB, streaming responses, cost tracking per session, and a Streamlit UI. Compare 3 different foundation models on the same prompt set.

## AWS Services Used
- **Amazon Bedrock** — foundation model inference (Claude 3, Titan, Mistral)
- **DynamoDB** — conversation history storage
- **S3** — image/document uploads for multi-modal input
- **AWS Lambda** — chat API handler
- **CloudWatch** — cost and latency metrics
- **Bedrock Model Evaluation** — side-by-side model comparison

## Project Structure
```
04-aws-bedrock/
├── app/
│   ├── main.py               ← Streamlit UI
│   ├── bedrock_client.py     ← Bedrock wrapper with streaming
│   ├── memory.py             ← DynamoDB conversation store
│   └── cost_tracker.py       ← Per-session token cost calculator
├── notebooks/
│   ├── 01_model_comparison.ipynb
│   └── 02_multimodal_demo.ipynb
├── lambda/
│   └── chat_handler.py       ← Optional Lambda deployment
├── docs/adr/
│   └── 001-model-selection.md
└── requirements.txt
```

## Phase 1: Enable Bedrock Models (Day 1)

### Request model access (AWS Console)
1. Go to **AWS Console → Bedrock → Model access**
2. Request access to:
   - `anthropic.claude-3-haiku-20240307-v1:0` (fastest, cheapest)
   - `anthropic.claude-3-sonnet-20240229-v1:0` (balanced)
   - `amazon.titan-text-express-v1` (AWS native)
   - `mistral.mistral-7b-instruct-v0:2` (open-source)
3. Access is usually granted within minutes for most models

```python
import boto3

bedrock = boto3.client("bedrock", region_name="us-east-1")
models = bedrock.list_foundation_models()["modelSummaries"]
for m in models:
    if m["modelLifecycle"]["status"] == "ACTIVE":
        print(m["modelId"], "-", m["providerName"])
```

## Phase 2: Bedrock Client with Streaming (Days 2–3)

`app/bedrock_client.py`:
```python
import boto3
import json
import base64
from typing import Generator

MODELS = {
    "claude-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "titan": "amazon.titan-text-express-v1",
    "mistral": "mistral.mistral-7b-instruct-v0:2",
}

TOKEN_COSTS = {
    "anthropic.claude-3-haiku-20240307-v1:0":   {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-sonnet-20240229-v1:0":  {"input": 0.003,   "output": 0.015},
    "amazon.titan-text-express-v1":              {"input": 0.0002,  "output": 0.0006},
    "mistral.mistral-7b-instruct-v0:2":          {"input": 0.00015, "output": 0.0002},
}

class BedrockClient:
    def __init__(self, region: str = "us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def invoke(self, model_key: str, messages: list, system: str = "") -> dict:
        model_id = MODELS[model_key]
        
        if "claude" in model_id:
            body = self._claude_body(messages, system)
        elif "titan" in model_id:
            body = self._titan_body(messages)
        else:
            body = self._mistral_body(messages)

        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        
        result = json.loads(response["body"].read())
        return self._parse_response(model_id, result)

    def stream(self, model_key: str, messages: list, system: str = "") -> Generator:
        model_id = MODELS[model_key]
        body = self._claude_body(messages, system)
        
        response = self.client.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(body),
        )
        
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                yield chunk["delta"].get("text", "")

    def invoke_with_image(self, model_key: str, text: str, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        
        ext = image_path.rsplit(".", 1)[-1].lower()
        media_type = f"image/{ext if ext != 'jpg' else 'jpeg'}"
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": text},
            ],
        }]
        return self.invoke(model_key, messages)

    def _claude_body(self, messages: list, system: str) -> dict:
        body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 2048, "messages": messages}
        if system:
            body["system"] = system
        return body

    def _titan_body(self, messages: list) -> dict:
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return {"inputText": prompt, "textGenerationConfig": {"maxTokenCount": 2048, "temperature": 0.7}}

    def _mistral_body(self, messages: list) -> dict:
        prompt = "".join(f"[INST]{m['content']}[/INST]" if m["role"] == "user" else m["content"] for m in messages)
        return {"prompt": prompt, "max_tokens": 2048, "temperature": 0.7}

    def _parse_response(self, model_id: str, result: dict) -> dict:
        if "claude" in model_id:
            text = result["content"][0]["text"]
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        elif "titan" in model_id:
            text = result["results"][0]["outputText"]
            input_tokens = result.get("inputTextTokenCount", 0)
            output_tokens = result["results"][0].get("tokenCount", 0)
        else:
            text = result["outputs"][0]["text"]
            input_tokens = result.get("prompt_token_count", 0)
            output_tokens = result.get("generation_token_count", 0)

        costs = TOKEN_COSTS.get(model_id, {"input": 0, "output": 0})
        cost_usd = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000

        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 6),
        }
```

## Phase 3: DynamoDB Conversation Memory (Day 4)

`app/memory.py`:
```python
import boto3
from datetime import datetime, timezone
from decimal import Decimal
import json

class ConversationMemory:
    def __init__(self, table_name: str = "bedrock-conversations"):
        self.ddb = boto3.resource("dynamodb")
        self.table = self.ddb.Table(table_name)

    @classmethod
    def create_table(cls, table_name: str = "bedrock-conversations"):
        ddb = boto3.client("dynamodb")
        ddb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

    def save_turn(self, session_id: str, role: str, content: str, metadata: dict = None):
        self.table.put_item(Item={
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {},
        })

    def get_history(self, session_id: str, limit: int = 20) -> list:
        response = self.table.query(
            KeyConditionExpression="session_id = :sid",
            ExpressionAttributeValues={":sid": session_id},
            ScanIndexForward=True,
            Limit=limit,
        )
        return [{"role": item["role"], "content": item["content"]} for item in response["Items"]]

    def get_session_cost(self, session_id: str) -> float:
        response = self.table.query(
            KeyConditionExpression="session_id = :sid",
            ExpressionAttributeValues={":sid": session_id},
        )
        return sum(
            float(item.get("metadata", {}).get("cost_usd", 0))
            for item in response["Items"]
        )
```

## Phase 4: Streamlit UI (Days 5–7)

`app/main.py`:
```python
import streamlit as st
import uuid
from bedrock_client import BedrockClient, MODELS
from memory import ConversationMemory

st.set_page_config(page_title="Bedrock Multi-Model Chat", layout="wide")
st.title("AWS Bedrock Multi-Modal Chat")

# Initialize session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

client = BedrockClient()
memory = ConversationMemory()

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_key = st.selectbox("Model", list(MODELS.keys()), index=0)
    system_prompt = st.text_area("System Prompt", value="You are a helpful AI assistant.")
    use_streaming = st.checkbox("Stream responses", value=True)
    uploaded_image = st.file_uploader("Upload image (multi-modal)", type=["jpg", "png", "jpeg"])
    
    st.divider()
    st.metric("Session Cost", f"${st.session_state.total_cost:.4f}")
    st.caption(f"Session: {st.session_state.session_id[:8]}...")
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.total_cost = 0.0
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        if use_streaming and not uploaded_image:
            placeholder = st.empty()
            full_response = ""
            for chunk in client.stream(model_key, st.session_state.messages, system_prompt):
                full_response += chunk
                placeholder.write(full_response + "▌")
            placeholder.write(full_response)
            response_text = full_response
            cost = 0.0
        elif uploaded_image:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_image.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_image.read())
                tmp_path = tmp.name
            result = client.invoke_with_image(model_key, prompt, tmp_path)
            os.unlink(tmp_path)
            st.write(result["text"])
            response_text = result["text"]
            cost = result["cost_usd"]
        else:
            result = client.invoke(model_key, st.session_state.messages, system_prompt)
            st.write(result["text"])
            response_text = result["text"]
            cost = result["cost_usd"]
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.total_cost += cost
    memory.save_turn(st.session_state.session_id, "user", prompt)
    memory.save_turn(st.session_state.session_id, "assistant", response_text, {"cost_usd": cost})
```

Run:
```bash
streamlit run app/main.py
```

## Phase 5: Model Comparison Notebook (Days 8–10)

In `notebooks/01_model_comparison.ipynb`:
```python
import time
import pandas as pd

test_prompts = [
    "Explain transformer attention in 3 sentences.",
    "Write a Python function to detect SQL injection.",
    "What are the trade-offs between RAG and fine-tuning?",
    "Summarize the EU AI Act's key requirements for high-risk systems.",
]

results = []
for model_key in ["claude-haiku", "claude-sonnet", "titan", "mistral"]:
    for prompt in test_prompts:
        start = time.time()
        messages = [{"role": "user", "content": prompt}]
        result = client.invoke(model_key, messages)
        latency = time.time() - start
        
        results.append({
            "model": model_key,
            "prompt_preview": prompt[:40],
            "latency_s": round(latency, 2),
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost_usd": result["cost_usd"],
            "response_preview": result["text"][:100],
        })

df = pd.DataFrame(results)
print(df.groupby("model")[["latency_s", "cost_usd", "output_tokens"]].mean())
```

## Acceptance Criteria
- [ ] Text chat works with at least 3 different models
- [ ] Multi-modal image input works (describe an uploaded image)
- [ ] Conversation history persists in DynamoDB across page refreshes
- [ ] Session cost tracked and displayed in sidebar
- [ ] Model comparison notebook shows latency/cost/quality table
- [ ] Streaming responses work without errors
- [ ] ADR-001 written: Claude Haiku vs Sonnet cost-quality trade-off
- [ ] Total project cost under $15

## ADR Template
Create `docs/adr/001-model-selection.md`:
```markdown
# ADR-001: Primary Model for Production Chat

## Status: Accepted

## Context
Multi-modal chat app needs to balance response quality, latency, and cost.
Expected load: 50–100 messages/day during development.

## Decision
Claude 3 Haiku as default model; Sonnet available as premium option.

## Consequences
+ Haiku: 3x cheaper than Sonnet, 2x faster, acceptable quality for most queries
+ Both support vision for multi-modal input
- Haiku may miss nuance on complex reasoning tasks
- Cannot use Haiku for extended thinking (requires Sonnet/Opus)
```
