# Project 11: Multi-Cloud GenAI Architecture — Implementation Guide

## Prerequisites
- Projects 04 (Bedrock), 07 (Vertex AI), active Azure account
- AWS CLI, gcloud, Azure CLI configured
- `conda activate aiarch`
- Packages: `anthropic`, `google-cloud-aiplatform`, `openai`, `fastapi`, `redis`

---

## Project Structure
```
11-multicloud-architecture/
├── providers/
│   ├── base.py            # Abstract LLMProvider interface
│   ├── bedrock.py         # AWS Bedrock provider
│   ├── azure.py           # Azure OpenAI provider
│   └── vertex.py          # GCP Vertex AI provider
├── circuit_breaker.py     # Circuit breaker state machine
├── router.py              # LLM routing logic
├── cache.py               # Semantic caching (in-memory + Redis)
├── cost_tracker.py        # Per-provider cost tracking
├── api.py                 # FastAPI app
├── terraform/
│   └── main.tf            # AWS Bedrock IAM setup
├── docs/
│   ├── adr/
│   │   ├── ADR-001-routing-strategy.md
│   │   └── ADR-002-caching-strategy.md
│   └── c4/
│       ├── c4-level1-system-context.md
│       └── c4-level2-containers.md
└── .env
```

---

## Phase 1: Setup

### Step 1.1 — Install dependencies
```bash
conda activate aiarch
pip install anthropic google-cloud-aiplatform openai fastapi uvicorn \
            redis python-dotenv boto3
```

### Step 1.2 — Azure OpenAI setup
1. Azure Portal → Create Resource → Azure OpenAI
2. Deploy a model: Azure OpenAI Studio → Deployments → Create → gpt-35-turbo
3. Get endpoint, key, and deployment name

### Step 1.3 — Create .env
```bash
cat > .env << 'EOF'
# AWS Bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_DEPLOYMENT=gpt-35-turbo
AZURE_OPENAI_API_VERSION=2024-02-01

# GCP Vertex AI
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
VERTEX_MODEL_ID=gemini-1.0-pro

# Redis (optional — uses in-memory cache if not set)
REDIS_URL=redis://localhost:6379

# Cache TTL
CACHE_TTL_SECONDS=300

# Routing
DEFAULT_TIER=fast
EOF
```

---

## Phase 2: Abstract Provider Interface

### Step 2.1 — Create providers/base.py

```python
# providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'bedrock', 'azure', 'vertex')."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The specific model being used."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a response. Raises on failure."""

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream response tokens. Raises on failure."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if provider is reachable. Does NOT raise."""
```

---

## Phase 3: Provider Implementations

### Step 3.1 — Create providers/bedrock.py

```python
# providers/bedrock.py
import os
import json
import time
from typing import Iterator
import boto3
from providers.base import LLMProvider, LLMResponse

# Cost per 1K tokens (input, output)
COSTS = {
    "anthropic.claude-3-haiku-20240307-v1:0": (0.00025, 0.00125),
    "amazon.titan-text-express-v1":            (0.0008, 0.0016),
}


class BedrockProvider(LLMProvider):
    def __init__(self):
        self._model_id = os.environ.get(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
        )
        self._region = os.environ.get("AWS_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=self._region)

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def model_id(self) -> str:
        return self._model_id

    def _build_body(self, messages, system_prompt, max_tokens, temperature) -> dict:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": messages,
        }

    def generate(self, messages, system_prompt="You are a helpful assistant.",
                 max_tokens=1024, temperature=0.7) -> LLMResponse:
        start = time.perf_counter()
        body = self._build_body(messages, system_prompt, max_tokens, temperature)
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        latency = (time.perf_counter() - start) * 1000
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]
        in_tok  = result["usage"]["input_tokens"]
        out_tok = result["usage"]["output_tokens"]
        in_cost, out_cost = COSTS.get(self._model_id, (0, 0))
        cost = (in_tok / 1000 * in_cost) + (out_tok / 1000 * out_cost)
        return LLMResponse(
            text=text, provider=self.name, model=self._model_id,
            input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=round(latency, 1), cost_usd=round(cost, 6),
        )

    def stream(self, messages, system_prompt="You are a helpful assistant.",
               max_tokens=1024) -> Iterator[str]:
        body = self._build_body(messages, system_prompt, max_tokens, 0.7)
        response = self._client.invoke_model_with_response_stream(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                yield chunk.get("delta", {}).get("text", "")

    def health_check(self) -> bool:
        try:
            self.generate([{"role": "user", "content": "ping"}], max_tokens=5)
            return True
        except Exception:
            return False
```

### Step 3.2 — Create providers/azure.py

```python
# providers/azure.py
import os
import time
from typing import Iterator
from openai import AzureOpenAI
from providers.base import LLMProvider, LLMResponse

# GPT-3.5-turbo pricing
COSTS = {"gpt-35-turbo": (0.0015, 0.002)}


class AzureOpenAIProvider(LLMProvider):
    def __init__(self):
        self._deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")
        self._client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )

    @property
    def name(self) -> str:
        return "azure"

    @property
    def model_id(self) -> str:
        return self._deployment

    def generate(self, messages, system_prompt="You are a helpful assistant.",
                 max_tokens=1024, temperature=0.7) -> LLMResponse:
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=all_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content
        in_tok  = response.usage.prompt_tokens
        out_tok = response.usage.completion_tokens
        in_cost, out_cost = COSTS.get(self._deployment, (0, 0))
        cost = (in_tok / 1000 * in_cost) + (out_tok / 1000 * out_cost)
        return LLMResponse(
            text=text, provider=self.name, model=self._deployment,
            input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=round(latency, 1), cost_usd=round(cost, 6),
        )

    def stream(self, messages, system_prompt="You are a helpful assistant.",
               max_tokens=1024) -> Iterator[str]:
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        stream = self._client.chat.completions.create(
            model=self._deployment,
            messages=all_messages,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def health_check(self) -> bool:
        try:
            self.generate([{"role": "user", "content": "ping"}], max_tokens=5)
            return True
        except Exception:
            return False
```

### Step 3.3 — Create providers/vertex.py

```python
# providers/vertex.py
import os
import time
from typing import Iterator
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part
from providers.base import LLMProvider, LLMResponse


class VertexAIProvider(LLMProvider):
    def __init__(self):
        self._project = os.environ["GCP_PROJECT_ID"]
        self._region  = os.environ.get("GCP_REGION", "us-central1")
        self._model_id_str = os.environ.get("VERTEX_MODEL_ID", "gemini-1.0-pro")
        vertexai.init(project=self._project, location=self._region)
        self._model = GenerativeModel(self._model_id_str)

    @property
    def name(self) -> str:
        return "vertex"

    @property
    def model_id(self) -> str:
        return self._model_id_str

    def _format_messages(self, messages: list[dict]) -> list[Content]:
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append(Content(role=role, parts=[Part.from_text(m["content"])]))
        return contents

    def generate(self, messages, system_prompt="You are a helpful assistant.",
                 max_tokens=1024, temperature=0.7) -> LLMResponse:
        contents = self._format_messages(messages)
        start = time.perf_counter()
        response = self._model.generate_content(
            contents,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        latency = (time.perf_counter() - start) * 1000
        text = response.text
        in_tok  = response.usage_metadata.prompt_token_count
        out_tok = response.usage_metadata.candidates_token_count
        # Gemini 1.0 Pro: $0.0005/1K input, $0.0015/1K output
        cost = (in_tok / 1000 * 0.0005) + (out_tok / 1000 * 0.0015)
        return LLMResponse(
            text=text, provider=self.name, model=self._model_id_str,
            input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=round(latency, 1), cost_usd=round(cost, 6),
        )

    def stream(self, messages, system_prompt="You are a helpful assistant.",
               max_tokens=1024) -> Iterator[str]:
        contents = self._format_messages(messages)
        for chunk in self._model.generate_content(contents, stream=True):
            if chunk.text:
                yield chunk.text

    def health_check(self) -> bool:
        try:
            self.generate([{"role": "user", "content": "ping"}], max_tokens=5)
            return True
        except Exception:
            return False
```

---

## Phase 4: Circuit Breaker

### Step 4.1 — Create circuit_breaker.py

```python
# circuit_breaker.py
import time
from enum import Enum


class State(Enum):
    CLOSED    = "CLOSED"       # Normal operation
    OPEN      = "OPEN"         # Failures threshold reached, rejecting calls
    HALF_OPEN = "HALF_OPEN"    # Testing if service recovered


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = State.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None

    def call_allowed(self) -> bool:
        """Check if a call is allowed given current state."""
        if self.state == State.CLOSED:
            return True
        if self.state == State.OPEN:
            if time.time() - (self.last_failure_time or 0) >= self.recovery_timeout:
                self.state = State.HALF_OPEN
                self.success_count = 0
                return True
            return False
        if self.state == State.HALF_OPEN:
            return self.success_count < self.half_open_max_calls
        return False

    def record_success(self):
        self.failure_count = 0
        if self.state == State.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = State.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = State.OPEN

    @property
    def status(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
        }
```

---

## Phase 5: Cache

### Step 5.1 — Create cache.py

```python
# cache.py
import os
import hashlib
import json
import time
from typing import Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", 300))


def _make_key(messages: list[dict], system_prompt: str) -> str:
    content = json.dumps({"messages": messages, "system": system_prompt}, sort_keys=True)
    return "rag:" + hashlib.sha256(content.encode()).hexdigest()[:16]


class InMemoryCache:
    """Simple TTL-based in-memory cache."""
    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> Optional[str]:
        if key in self._store:
            value, expires_at = self._store[key]
            if time.time() < expires_at:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: str, ttl: int = CACHE_TTL):
        self._store[key] = (value, time.time() + ttl)

    def clear(self):
        self._store.clear()

    @property
    def size(self) -> int:
        # Clean expired entries
        now = time.time()
        self._store = {k: v for k, v in self._store.items() if v[1] > now}
        return len(self._store)


class SemanticCache:
    """Wraps either Redis or in-memory cache with a consistent interface."""
    def __init__(self):
        redis_url = os.environ.get("REDIS_URL")
        if redis_url and REDIS_AVAILABLE:
            try:
                self._backend = redis.from_url(redis_url, decode_responses=True)
                self._backend.ping()
                self._use_redis = True
                print("SemanticCache: using Redis")
            except Exception:
                self._backend = InMemoryCache()
                self._use_redis = False
                print("SemanticCache: Redis unavailable, using in-memory")
        else:
            self._backend = InMemoryCache()
            self._use_redis = False
            print("SemanticCache: using in-memory")

    def get(self, messages: list[dict], system_prompt: str = "") -> Optional[str]:
        key = _make_key(messages, system_prompt)
        if self._use_redis:
            return self._backend.get(key)
        return self._backend.get(key)

    def set(self, messages: list[dict], system_prompt: str, value: str):
        key = _make_key(messages, system_prompt)
        if self._use_redis:
            self._backend.setex(key, CACHE_TTL, value)
        else:
            self._backend.set(key, value)

    @property
    def backend_type(self) -> str:
        return "redis" if self._use_redis else "in-memory"
```

---

## Phase 6: Router + Cost Tracker

### Step 6.1 — Create cost_tracker.py

```python
# cost_tracker.py
from collections import defaultdict
from threading import Lock


class CostTracker:
    def __init__(self):
        self._data: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def record(self, provider: str, cost_usd: float):
        with self._lock:
            self._data[provider].append(cost_usd)

    def summary(self) -> dict:
        with self._lock:
            return {
                provider: {
                    "total_cost_usd": round(sum(costs), 6),
                    "call_count": len(costs),
                    "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else 0,
                }
                for provider, costs in self._data.items()
            }

    def grand_total(self) -> float:
        with self._lock:
            return round(sum(c for costs in self._data.values() for c in costs), 6)
```

### Step 6.2 — Create router.py

```python
# router.py
import os
from dotenv import load_dotenv
from providers.base import LLMProvider, LLMResponse
from providers.bedrock import BedrockProvider
from providers.azure import AzureOpenAIProvider
from providers.vertex import VertexAIProvider
from circuit_breaker import CircuitBreaker, State
from cache import SemanticCache
from cost_tracker import CostTracker

load_dotenv()

# Routing tiers: order defines priority
TIERS = {
    "fast":  ["bedrock", "azure", "vertex"],   # Prioritize speed
    "smart": ["azure", "vertex", "bedrock"],   # Prioritize quality
}


class LLMRouter:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._cache = SemanticCache()
        self._cost_tracker = CostTracker()
        self._init_providers()

    def _init_providers(self):
        """Initialize available providers (skip if credentials missing)."""
        candidates = {
            "bedrock": BedrockProvider,
            "azure":   AzureOpenAIProvider,
            "vertex":  VertexAIProvider,
        }
        for name, cls in candidates.items():
            try:
                provider = cls()
                self._providers[name] = provider
                self._breakers[name] = CircuitBreaker(
                    failure_threshold=3,
                    recovery_timeout=60,
                )
                print(f"[Router] Provider loaded: {name}")
            except Exception as e:
                print(f"[Router] Skipping {name}: {e}")

        if not self._providers:
            raise RuntimeError("No LLM providers available. Check credentials in .env")

    def generate(
        self,
        messages: list[dict],
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        tier: str = "fast",
        max_cost_usd: float | None = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        # Check cache
        if use_cache:
            cached = self._cache.get(messages, system_prompt)
            if cached:
                return LLMResponse(
                    text=cached, provider="cache", model="cache",
                    input_tokens=0, output_tokens=0, latency_ms=0, cost_usd=0,
                )

        # Check budget
        if max_cost_usd is not None:
            current_spend = self._cost_tracker.grand_total()
            if current_spend >= max_cost_usd:
                raise ValueError(f"Budget exceeded: ${current_spend:.4f} >= ${max_cost_usd:.4f}")

        # Try providers in tier priority order
        priority = TIERS.get(tier, TIERS["fast"])
        available = [p for p in priority if p in self._providers]

        for provider_name in available:
            breaker = self._breakers[provider_name]
            if not breaker.call_allowed():
                print(f"[Router] {provider_name} circuit breaker OPEN — skipping")
                continue

            try:
                provider = self._providers[provider_name]
                response = provider.generate(messages, system_prompt, max_tokens)
                breaker.record_success()
                self._cost_tracker.record(provider_name, response.cost_usd)
                if use_cache:
                    self._cache.set(messages, system_prompt, response.text)
                return response
            except Exception as e:
                print(f"[Router] {provider_name} failed: {e}")
                breaker.record_failure()

        raise RuntimeError("All providers failed or circuit breakers open")

    @property
    def status(self) -> dict:
        return {
            "providers": {
                name: {
                    "available": True,
                    "model": p.model_id,
                    "circuit_breaker": self._breakers[name].status,
                }
                for name, p in self._providers.items()
            },
            "cost_summary": self._cost_tracker.summary(),
            "grand_total_usd": self._cost_tracker.grand_total(),
            "cache_backend": self._cache.backend_type,
        }
```

---

## Phase 7: FastAPI App

### Step 7.1 — Create api.py

```python
# api.py
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from router import LLMRouter

router: LLMRouter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router
    router = LLMRouter()
    yield


app = FastAPI(title="Multi-Cloud LLM Router", lifespan=lifespan)


class ChatRequest(BaseModel):
    messages: list[dict]
    system_prompt: str = "You are a helpful assistant."
    max_tokens: int = 1024
    tier: str = "fast"
    max_cost_usd: Optional[float] = None
    use_cache: bool = True
    stream: bool = False


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        if request.stream:
            # Get provider for streaming
            priority = ["bedrock", "azure", "vertex"]
            for name in priority:
                if name in router._providers and router._breakers[name].call_allowed():
                    provider = router._providers[name]
                    def gen():
                        try:
                            for chunk in provider.stream(
                                request.messages, request.system_prompt, request.max_tokens
                            ):
                                yield chunk
                        except Exception as e:
                            yield f"\n[Error: {e}]"
                    return StreamingResponse(gen(), media_type="text/plain")
            raise HTTPException(503, "No streaming provider available")

        response = router.generate(
            messages=request.messages,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            tier=request.tier,
            max_cost_usd=request.max_cost_usd,
            use_cache=request.use_cache,
        )
        return {
            "text": response.text,
            "provider": response.provider,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
        }
    except ValueError as e:
        raise HTTPException(status_code=402, detail=str(e))  # Budget exceeded
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))  # All providers failed


@app.get("/status")
def status():
    return router.status


@app.get("/health")
def health():
    return {"status": "ok", "providers": list(router._providers.keys())}
```

### Step 7.2 — Run and test
```bash
uvicorn api:app --reload --port 8002

# Test chat
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "tier": "fast"
  }'

# Status
curl http://localhost:8002/status | python3 -m json.tool
```

---

## Phase 8: Terraform + C4 Diagrams

### Step 8.1 — Create terraform/main.tf

```hcl
# terraform/main.tf
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = "us-east-1"
}

# IAM policy for Bedrock access
resource "aws_iam_policy" "bedrock_access" {
  name        = "BedrockLLMRouterPolicy"
  description = "Allow LLM Router to call Bedrock models"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
      ]
      Resource = "arn:aws:bedrock:us-east-1::foundation-model/*"
    }]
  })
}
```

```bash
cd terraform
terraform init
terraform apply
```

### Step 8.2 — Create docs/c4/c4-level1-system-context.md

```markdown
# C4 Level 1 — System Context: Multi-Cloud GenAI Router

```
[User / Client Application]
         |
         | HTTPS (REST)
         v
[LLM Router FastAPI Service]
    |           |          |
    | AWS       | Azure    | GCP
    v           v          v
[Bedrock]  [Azure OpenAI] [Vertex AI]
```

**Users:** Developers and applications consuming the unified LLM API
**LLM Router:** Provides a single endpoint abstracting 3 cloud LLM providers
**Bedrock:** AWS-hosted Claude 3 Haiku
**Azure OpenAI:** GPT-3.5-turbo via Azure
**Vertex AI:** Gemini 1.0 Pro on GCP
```

### Step 8.3 — Create docs/c4/c4-level2-containers.md

```markdown
# C4 Level 2 — Container Diagram

```
[Client]
    |
    v
[FastAPI API (api.py)]
    |
    |---> [LLMRouter (router.py)]
    |         |---> [CircuitBreaker x3 (circuit_breaker.py)]
    |         |---> [SemanticCache (cache.py)] -----> [Redis / In-Memory]
    |         |---> [CostTracker (cost_tracker.py)]
    |         |
    |         |---> [BedrockProvider] ----> [AWS Bedrock]
    |         |---> [AzureProvider]   ----> [Azure OpenAI]
    |         |---> [VertexProvider]  ----> [GCP Vertex AI]
```
```

---

## Verification Checklist
- [ ] `uvicorn api:app --port 8002` starts without errors
- [ ] At least 2 providers initialize successfully (visible in startup logs)
- [ ] `POST /chat` returns a response from one provider
- [ ] `GET /status` shows circuit breaker states and cost summary
- [ ] Cache hit: send same message twice, second response has `provider: cache`
- [ ] Circuit breaker test: set wrong API key for one provider → it opens after 3 failures
- [ ] Budget control: `max_cost_usd: 0.001` raises 402 after exceeding limit
- [ ] `terraform apply` creates IAM policy (or runs plan without error)
- [ ] C4 diagrams written in `docs/c4/`
- [ ] ADR-001 and ADR-002 written
- [ ] Total cloud cost < $40

---

## Troubleshooting

**Provider initialization fails**
Check `.env` has correct keys. Run with only one provider first (comment others out in `_init_providers`).

**Circuit breaker opens unexpectedly**
Lower `failure_threshold` is too small. Default is 3 — adjust to 5 for flaky networks.

**Cache always misses**
Ensure messages are structurally identical between calls (same dict keys/values). The cache key is SHA256 of the serialized messages.

**Azure 401 errors**
Verify the deployment name matches exactly what's in Azure OpenAI Studio.

---

## Next Steps → Project 12: Capstone Platform
```bash
conda activate aiarch
mkdir -p ~/Documents/ai-journey/projects/12-capstone/{backend,frontend,docs/{adr,c4}}
# The capstone reuses: P11 (router), P06 (RAG), P08 (agent)
```
