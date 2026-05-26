# Project 12: Capstone — Full-Stack AI Platform

**XP:** 1000 | **Cost:** ~$30–60 | **Duration:** Weeks 23–24

## Objective
Build a production-ready Full-Stack AI Platform that integrates all skills from the previous 11 projects: a multi-cloud LLM backend, RAG knowledge base, autonomous agent, real-time monitoring, CI/CD pipeline, and a polished React-based frontend. This is your portfolio centerpiece — it should be deployable, demonstrable in interviews, and documented with a full ADR history.

## Platform Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Platform                               │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │   Frontend   │    │   API Gateway    │    │   Agent      │  │
│  │  (React/     │───▶│   (FastAPI)      │───▶│   Engine     │  │
│  │   Next.js)   │    │   + Auth (JWT)   │    │  (LangGraph) │  │
│  └──────────────┘    └────────┬─────────┘    └──────────────┘  │
│                               │                                  │
│               ┌───────────────┼───────────────┐                 │
│               ▼               ▼               ▼                 │
│     ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│     │  LLM Router  │  │  RAG Engine  │  │  Document Store │   │
│     │  (Multi-     │  │  (ChromaDB   │  │  (PostgreSQL +  │   │
│     │   Cloud)     │  │   + RAGAS)   │  │   S3/GCS/Azure) │   │
│     └──────────────┘  └──────────────┘  └─────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Observability: LangSmith + Prometheus + Grafana        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure
```
12-capstone/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── hooks/
│   ├── package.json
│   └── next.config.js
├── backend/
│   ├── api/
│   │   ├── main.py             ← FastAPI app
│   │   ├── auth.py             ← JWT authentication
│   │   ├── routes/
│   │   │   ├── chat.py         ← Chat + streaming
│   │   │   ├── documents.py    ← Document upload/ingest
│   │   │   └── agent.py        ← Agent execution
│   │   └── models.py           ← Pydantic schemas
│   ├── core/
│   │   ├── llm_router.py       ← From Project 11
│   │   ├── rag_engine.py       ← From Project 06
│   │   └── agent_engine.py     ← From Project 08
│   └── Dockerfile
├── infra/
│   ├── docker-compose.yml      ← Full local stack
│   ├── terraform/              ← Cloud deployment
│   └── k8s/                    ← Optional Kubernetes manifests
├── .github/
│   └── workflows/
│       └── deploy.yml
├── docs/
│   ├── adr/                    ← 5+ ADRs for major decisions
│   ├── c4/                     ← All 4 C4 diagram levels
│   └── api.md                  ← API documentation
└── README.md                   ← Portfolio-ready project README
```

## Phase 1: Backend API (Days 1–3)

`backend/api/main.py`:
```python
from fastapi import FastAPI, Depends, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from auth import get_current_user, create_access_token
from routes import chat, documents, agent
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize all engines on startup
    from core.llm_router import LLMRouter
    from core.rag_engine import RAGEngine
    app.state.router = LLMRouter()
    app.state.rag = RAGEngine()
    yield

app = FastAPI(title="AI Platform API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,      prefix="/api/v1/chat",      tags=["Chat"])
app.include_router(documents.router, prefix="/api/v1/documents",  tags=["Documents"])
app.include_router(agent.router,     prefix="/api/v1/agent",      tags=["Agent"])

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
```

`backend/api/routes/chat.py`:
```python
from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from auth import get_current_user
import json

router = APIRouter()

class ChatMessage(BaseModel):
    messages: list[dict]
    use_rag: bool = False
    provider: str = "auto"
    stream: bool = True

@router.post("/")
async def chat(req: ChatMessage, request: Request, user=Depends(get_current_user)):
    rag_engine = request.app.state.rag
    llm_router = request.app.state.router
    
    if req.use_rag:
        # Augment with RAG context
        query = req.messages[-1]["content"]
        context = await rag_engine.retrieve(query)
        augmented_messages = req.messages.copy()
        augmented_messages[-1]["content"] = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        augmented_messages = req.messages
    
    if req.stream:
        async def generator():
            async for chunk in llm_router.stream(augmented_messages, provider=req.provider):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generator(), media_type="text/event-stream")
    else:
        result = await llm_router.generate(augmented_messages, provider=req.provider)
        return result
```

## Phase 2: Authentication (Day 4)

`backend/api/auth.py`:
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
import secrets

SECRET_KEY = secrets.token_hex(32)     # Generate once, store in env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

security = HTTPBearer()

# Simple demo user store (use a real DB in production)
DEMO_USERS = {
    "demo": {"password": "password123", "role": "user"},
    "admin": {"password": "admin123", "role": "admin"},
}

def create_access_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload["sub"], "role": payload["role"]}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
```

## Phase 3: Frontend (Days 5–7)

`frontend/src/pages/index.tsx` (Next.js):
```tsx
import { useState } from "react"
import ChatPanel from "@/components/ChatPanel"
import DocumentPanel from "@/components/DocumentPanel"
import AgentPanel from "@/components/AgentPanel"
import Sidebar from "@/components/Sidebar"

export default function Home() {
  const [activeTab, setActiveTab] = useState<"chat" | "documents" | "agent">("chat")

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="flex-1 overflow-hidden">
        {activeTab === "chat" && <ChatPanel />}
        {activeTab === "documents" && <DocumentPanel />}
        {activeTab === "agent" && <AgentPanel />}
      </main>
    </div>
  )
}
```

`frontend/src/components/ChatPanel.tsx`:
```tsx
import { useState, useRef, useEffect } from "react"

interface Message { role: "user" | "assistant"; content: string }

export default function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [useRag, setUseRag] = useState(false)
  const [streaming, setStreaming] = useState(false)

  const sendMessage = async () => {
    if (!input.trim()) return
    const userMsg: Message = { role: "user", content: input }
    setMessages(prev => [...prev, userMsg])
    setInput("")
    setStreaming(true)
    
    const response = await fetch("/api/v1/chat/", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${localStorage.getItem("token")}` },
      body: JSON.stringify({ messages: [...messages, userMsg], use_rag: useRag, stream: true }),
    })
    
    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let assistantMsg = ""
    
    setMessages(prev => [...prev, { role: "assistant", content: "" }])
    
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const chunk = decoder.decode(value)
      const lines = chunk.split("\n").filter(l => l.startsWith("data: "))
      for (const line of lines) {
        const data = line.replace("data: ", "")
        if (data === "[DONE]") break
        const parsed = JSON.parse(data)
        assistantMsg += parsed.text
        setMessages(prev => [...prev.slice(0, -1), { role: "assistant", content: assistantMsg }])
      }
    }
    setStreaming(false)
  }

  return (
    <div className="flex flex-col h-full p-4">
      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.map((m, i) => (
          <div key={i} className={`p-3 rounded-lg ${m.role === "user" ? "bg-blue-800 ml-8" : "bg-gray-700 mr-8"}`}>
            <span className="text-xs text-gray-400">{m.role}</span>
            <p className="mt-1">{m.content}</p>
          </div>
        ))}
      </div>
      <div className="flex gap-2 items-center">
        <label className="flex items-center gap-1 text-sm">
          <input type="checkbox" checked={useRag} onChange={e => setUseRag(e.target.checked)} />
          Use RAG
        </label>
        <input
          className="flex-1 bg-gray-800 rounded-lg p-3 text-white"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && !e.shiftKey && sendMessage()}
          placeholder="Ask anything..."
        />
        <button
          className="bg-blue-600 px-4 py-3 rounded-lg disabled:opacity-50"
          onClick={sendMessage}
          disabled={streaming || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  )
}
```

## Phase 4: Full Docker Compose Stack (Day 8)

`infra/docker-compose.yml`:
```yaml
version: "3.9"
services:
  frontend:
    build: ../frontend
    ports: ["3000:3000"]
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on: [backend]

  backend:
    build: ../backend
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql://aiplatform:password@db:5432/aiplatform
      - REDIS_URL=redis://redis:6379
      - CHROMA_HOST=chroma
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
    depends_on: [db, redis, chroma]

  db:
    image: postgres:16-alpine
    environment: {POSTGRES_DB: aiplatform, POSTGRES_USER: aiplatform, POSTGRES_PASSWORD: password}
    volumes: [pgdata:/var/lib/postgresql/data]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  chroma:
    image: chromadb/chroma:0.5.0
    ports: ["8001:8000"]
    volumes: [chromadata:/chroma/chroma]

  prometheus:
    image: prom/prometheus:v2.51.0
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:10.3.0
    ports: ["3001:3000"]
    environment: {GF_SECURITY_ADMIN_PASSWORD: admin}

volumes:
  pgdata:
  chromadata:
```

## Phase 5: Portfolio README + ADRs (Days 9–10)

Write a comprehensive `README.md` with:
- Live demo URL (GitHub Pages or Vercel)
- Architecture diagram (ASCII or image)
- Quick start (`docker compose up`)
- Feature list with screenshots
- Tech stack table
- API documentation link

Write 5 ADRs covering:
1. Frontend framework (Next.js vs Vite + React)
2. Vector store (ChromaDB vs Pinecone vs pgvector)
3. Authentication strategy (JWT vs Auth0 vs Cognito)
4. Deployment target (ECS vs Lambda vs Render)
5. LLM provider selection for each use case

## Interview Demo Script

Prepare a 5-minute demo covering:
1. **Chat with RAG** — upload a PDF, ask questions about it
2. **Multi-cloud fallover** — demonstrate circuit breaker by temporarily disabling a provider
3. **Agent task** — run a research query showing multi-step tool use
4. **Monitoring** — show Grafana dashboard with live metrics
5. **Cost tracking** — show `/status` endpoint with per-provider costs

## Acceptance Criteria
- [ ] `docker compose up` starts all 7 services cleanly
- [ ] Frontend loads at `localhost:3000` with all 3 tabs working
- [ ] Streaming chat response visible in frontend
- [ ] RAG toggle augments responses with document context
- [ ] Agent tab shows multi-step research task execution
- [ ] Grafana shows live request metrics
- [ ] GitHub Actions CI/CD pipeline green on main branch
- [ ] 5+ ADRs committed to `docs/adr/`
- [ ] C4 diagrams (all 4 levels) in `docs/c4/`
- [ ] `README.md` is portfolio-quality (someone unfamiliar can run it)
- [ ] Total cloud cost under $60 for the 2-week project

## Portfolio Deployment Options

| Option | Cost | Effort | Best For |
|---|---|---|---|
| GitHub Pages (frontend only) | Free | Low | Static portfolio |
| Render (full stack) | ~$7/mo | Medium | Demo environment |
| AWS ECS + Vercel | ~$20/mo | High | Production quality |
| Fly.io | ~$5/mo | Medium | Quick full-stack |

**Minimum viable portfolio deployment**: Push frontend to Vercel (free), backend to Render (free tier), add live URL to README.
