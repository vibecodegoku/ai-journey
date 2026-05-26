# Project 11: Multi-Cloud GenAI Architecture

**XP:** 500 | **Cost:** ~$20–40 (multi-cloud API calls) | **Duration:** Weeks 21–22

## Objective
Design and implement a production-grade multi-cloud GenAI architecture with an intelligent LLM Router that fails over between AWS Bedrock, Azure OpenAI, and GCP Vertex AI. Implement cost optimization, circuit breakers, semantic caching, and document the full architecture with C4 diagrams and Architecture Decision Records.

## Architecture Overview

```
Client Request
     │
     ▼
┌─────────────────────┐
│   LLM Router        │  ← Semantic cache check → early return
│   (FastAPI)         │  ← Rate limit check
│                     │  ← Circuit breaker state check
└────────┬────────────┘
         │  Route to cheapest available provider
    ┌────┴─────────────────────────┐
    │           │                  │
    ▼           ▼                  ▼
┌────────┐ ┌─────────┐   ┌──────────────┐
│ AWS    │ │ Azure   │   │ GCP Vertex   │
│Bedrock │ │ OpenAI  │   │ AI Gemini    │
│Claude 3│ │ GPT-4o  │   │ Gemini 1.5   │
└────────┘ └─────────┘   └──────────────┘
    │           │                  │
    └─────┬─────┘                  │
          │    ← response          │
          ▼                        │
  ┌──────────────┐                 │
  │  Cache +     │ ◄───────────────┘
  │  Cost Track  │
  │  (Redis)     │
  └──────────────┘
```

## Project Structure
```
11-multicloud-architecture/
├── router/
│   ├── main.py               ← FastAPI LLM Router
│   ├── providers/
│   │   ├── base.py           ← Abstract provider interface
│   │   ├── bedrock.py        ← AWS Bedrock adapter
│   │   ├── azure_openai.py   ← Azure OpenAI adapter
│   │   └── vertex.py         ← GCP Vertex AI adapter
│   ├── circuit_breaker.py    ← Circuit breaker state machine
│   ├── cache.py              ← Semantic cache (Redis + embeddings)
│   └── cost_tracker.py       ← Per-request cost accounting
├── terraform/
│   ├── aws/
│   │   └── bedrock.tf
│   ├── azure/
│   │   └── openai.tf
│   └── gcp/
│       └── vertex.tf
├── docs/
│   ├── adr/
│   │   ├── 001-routing-strategy.md
│   │   └── 002-caching-strategy.md
│   └── c4/
│       ├── context.md
│       ├── container.md
│       └── component.md
├── notebooks/
│   └── 01_cost_analysis.ipynb
├── tests/
│   └── test_router.py
└── requirements.txt
```

## Phase 1: Provider Interface (Days 1–2)

`router/providers/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator

@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float

class LLMProvider(ABC):
    name: str
    models: dict[str, dict]     # model_name → {"input_cost_per_1k", "output_cost_per_1k"}
    
    @abstractmethod
    async def generate(self, messages: list[dict], model: str, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    async def stream(self, messages: list[dict], model: str, **kwargs) -> AsyncGenerator[str, None]:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        m = self.models.get(model, {})
        return (input_tokens * m.get("input_cost_per_1k", 0) + output_tokens * m.get("output_cost_per_1k", 0)) / 1000
```

`router/providers/bedrock.py`:
```python
import boto3
import json
import time
from .base import LLMProvider, LLMResponse

class BedrockProvider(LLMProvider):
    name = "aws_bedrock"
    models = {
        "claude-3-haiku":  {"id": "anthropic.claude-3-haiku-20240307-v1:0",  "input_cost_per_1k": 0.00025, "output_cost_per_1k": 0.00125},
        "claude-3-sonnet": {"id": "anthropic.claude-3-sonnet-20240229-v1:0", "input_cost_per_1k": 0.003,   "output_cost_per_1k": 0.015},
        "titan-express":   {"id": "amazon.titan-text-express-v1",            "input_cost_per_1k": 0.0002,  "output_cost_per_1k": 0.0006},
    }
    
    def __init__(self, region: str = "us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
    
    async def generate(self, messages: list[dict], model: str = "claude-3-haiku", **kwargs) -> LLMResponse:
        model_info = self.models[model]
        start = time.time()
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": messages,
        }
        
        response = self.client.invoke_model(
            modelId=model_info["id"],
            body=json.dumps(body),
        )
        
        result = json.loads(response["body"].read())
        latency = (time.time() - start) * 1000
        
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]
        
        return LLMResponse(
            text=result["content"][0]["text"],
            provider=self.name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.calculate_cost(model, input_tokens, output_tokens),
            latency_ms=latency,
        )
    
    async def health_check(self) -> bool:
        try:
            await self.generate([{"role": "user", "content": "ping"}], model="titan-express")
            return True
        except:
            return False
    
    async def stream(self, messages, model="claude-3-haiku", **kwargs):
        model_id = self.models[model]["id"]
        body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 1024, "messages": messages}
        response = self.client.invoke_model_with_response_stream(modelId=model_id, body=json.dumps(body))
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                yield chunk["delta"].get("text", "")
```

## Phase 2: Circuit Breaker (Day 3)

`router/circuit_breaker.py`:
```python
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import asyncio

class State(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing — reject all requests
    HALF_OPEN = "half_open" # Testing if recovered

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60, success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        
        self._state = State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._recent_errors = deque(maxlen=100)
    
    @property
    def state(self) -> State:
        if self._state == State.OPEN:
            if datetime.now() - self._last_failure_time > timedelta(seconds=self.timeout_seconds):
                self._state = State.HALF_OPEN
                self._success_count = 0
        return self._state
    
    def record_success(self):
        self._failure_count = 0
        if self._state == State.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = State.CLOSED
    
    def record_failure(self, error: str = ""):
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        self._recent_errors.append({"time": datetime.now().isoformat(), "error": error})
        
        if self._failure_count >= self.failure_threshold:
            self._state = State.OPEN
    
    def is_available(self) -> bool:
        return self.state in (State.CLOSED, State.HALF_OPEN)
    
    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "recent_errors": list(self._recent_errors)[-5:],
        }
```

## Phase 3: LLM Router (Days 4–6)

`router/main.py`:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import time
import hashlib

from providers.bedrock import BedrockProvider
from providers.azure_openai import AzureOpenAIProvider
from providers.vertex import VertexAIProvider
from circuit_breaker import CircuitBreaker
from cost_tracker import CostTracker

app = FastAPI(title="Multi-Cloud LLM Router")

# Initialize providers
providers = {
    "bedrock": BedrockProvider(),
    "azure":   AzureOpenAIProvider(),
    "vertex":  VertexAIProvider(),
}

# Circuit breakers per provider
breakers = {name: CircuitBreaker(failure_threshold=3, timeout_seconds=30) for name in providers}

# Cost tracking
tracker = CostTracker()

# Simple in-memory cache (use Redis in production)
response_cache: dict = {}

class ChatRequest(BaseModel):
    messages: list[dict]
    preferred_provider: Optional[str] = None
    max_cost_usd: Optional[float] = None
    model_tier: str = "fast"    # "fast" | "smart"
    cache_ttl: int = 300        # Cache TTL in seconds

ROUTING_PRIORITY = {
    "fast":  ["bedrock:claude-3-haiku", "azure:gpt-4o-mini", "vertex:gemini-flash"],
    "smart": ["vertex:gemini-pro", "azure:gpt-4o", "bedrock:claude-3-sonnet"],
}

@app.post("/chat")
async def chat(req: ChatRequest):
    # Cache check
    cache_key = hashlib.sha256(str(req.messages).encode()).hexdigest()
    if cache_key in response_cache:
        cached = response_cache[cache_key]
        if time.time() - cached["timestamp"] < req.cache_ttl:
            cached["cache_hit"] = True
            return cached
    
    # Determine routing order
    if req.preferred_provider and breakers[req.preferred_provider].is_available():
        route_order = [req.preferred_provider]
    else:
        route_order = [r.split(":")[0] for r in ROUTING_PRIORITY[req.model_tier]]
    
    # Try each provider in order
    last_error = None
    for provider_name in route_order:
        if not breakers[provider_name].is_available():
            continue
        
        provider = providers[provider_name]
        model = ROUTING_PRIORITY[req.model_tier][
            next(i for i, r in enumerate(ROUTING_PRIORITY[req.model_tier]) if r.startswith(provider_name))
        ].split(":")[1]
        
        try:
            result = await asyncio.wait_for(
                provider.generate(req.messages, model=model),
                timeout=30.0,
            )
            
            breakers[provider_name].record_success()
            tracker.record(provider_name, model, result.cost_usd, result.input_tokens, result.output_tokens)
            
            response = {
                "text": result.text,
                "provider": provider_name,
                "model": model,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
                "cache_hit": False,
                "timestamp": time.time(),
            }
            response_cache[cache_key] = response
            return response
        
        except Exception as e:
            last_error = str(e)
            breakers[provider_name].record_failure(last_error)
            continue
    
    raise HTTPException(status_code=503, detail=f"All providers failed. Last error: {last_error}")

@app.get("/status")
async def status():
    return {
        "providers": {
            name: {
                "available": breakers[name].is_available(),
                "circuit_breaker": breakers[name].get_status(),
                "costs_today": tracker.get_provider_costs(name),
            }
            for name in providers
        },
        "total_cost_today_usd": tracker.get_total_cost(),
    }
```

## Phase 4: C4 Architecture Documentation (Days 7–8)

`docs/c4/context.md`:
```markdown
# C4 Level 1: System Context

## Multi-Cloud LLM Router

**Users:**
- Application developers calling the unified LLM API
- Internal services that need LLM capabilities

**System:**
- LLM Router — single entry point for all LLM requests

**External Systems:**
- AWS Bedrock (Claude 3, Titan)
- Azure OpenAI (GPT-4o, GPT-4o-mini)
- GCP Vertex AI (Gemini 1.5 Pro, Flash)
- Redis (semantic cache)
- PostgreSQL (cost tracking, audit log)

**Key relationships:**
- Applications → Router: REST API (JSON, streaming SSE)
- Router → Cloud LLMs: Provider-specific SDKs
- Router → Redis: Cache check/set per request
- Router → PostgreSQL: Cost and usage logging
```

`docs/c4/container.md`:
```markdown
# C4 Level 2: Container Diagram

## Containers in the LLM Router System

| Container | Technology | Responsibility |
|---|---|---|
| API Gateway | FastAPI + uvicorn | Route requests, auth, rate limiting |
| Provider Adapters | Python classes | Normalize provider-specific APIs |
| Circuit Breaker | In-memory state machine | Detect provider failures, enable fallback |
| Semantic Cache | Redis + embeddings | Cache semantically similar queries |
| Cost Tracker | PostgreSQL | Per-request cost logging, budget alerts |
| Provider Monitor | Background task | Health-check each provider every 30s |

## Data flows
1. Request arrives → Cache check (Redis) → Cache hit: return immediately
2. Cache miss → Route selector → Try providers in priority order
3. Provider responds → Record cost → Update cache → Return to client
4. Provider fails → Circuit breaker trips → Try next provider
```

## Phase 5: Terraform Infrastructure (Days 9–10)

`terraform/aws/bedrock.tf`:
```hcl
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = var.aws_region }

# IAM role for Bedrock access
resource "aws_iam_role" "bedrock_role" {
  name = "llm-router-bedrock-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_policy" {
  name = "bedrock-invoke"
  role = aws_iam_role.bedrock_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
      Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/*"
    }]
  })
}
```

## Acceptance Criteria
- [ ] Router successfully calls at least 2 of 3 cloud providers
- [ ] Circuit breaker opens after 3 consecutive failures and falls back to next provider
- [ ] Cache returns identical response for duplicate queries (verify with `/status` hit count)
- [ ] `/status` endpoint shows circuit breaker state and per-provider cost
- [ ] C4 context and container diagrams written in `docs/c4/`
- [ ] ADR-001 written: routing strategy trade-offs
- [ ] ADR-002 written: caching strategy
- [ ] Cost tracker shows accurate per-provider USD spend

## ADR Template
Create `docs/adr/001-routing-strategy.md`:
```markdown
# ADR-001: LLM Routing Strategy

## Status: Accepted

## Context
Multi-cloud LLM router must select provider for each request based on availability,
cost, latency, and capability. Providers have different pricing, SLAs, and uptime.

## Decision
Priority-based routing with circuit breakers: Fast tier defaults to Claude 3 Haiku →
GPT-4o-mini → Gemini Flash. Smart tier: Gemini Pro → GPT-4o → Claude 3 Sonnet.

## Consequences
+ Deterministic fallback order — easy to reason about
+ Circuit breakers prevent cascading failures
+ Different tiers allow cost/quality trade-off at call time
- Priority order may not be optimal for all query types
- No dynamic load balancing based on real-time latency
```
