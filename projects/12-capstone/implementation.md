# Project 12: Capstone — Full-Stack AI Platform — Implementation Guide

## Prerequisites
- Projects 06 (RAG), 08 (LangGraph), 11 (Multi-Cloud Router) complete
- Docker Desktop running
- AWS CLI, gcloud, Azure CLI configured
- Node.js 18+ installed (`brew install node`)
- `conda activate aiarch`

---

## Project Structure
```
12-capstone/
├── backend/
│   ├── main.py                  # FastAPI entrypoint
│   ├── auth.py                  # JWT authentication
│   ├── routers/
│   │   ├── chat.py              # /api/v1/chat
│   │   ├── documents.py         # /api/v1/documents
│   │   └── agent.py             # /api/v1/agent
│   ├── engines/
│   │   ├── llm_router.py        # Reuse from P11
│   │   ├── rag_engine.py        # Reuse from P06
│   │   └── agent_engine.py      # Reuse from P08
│   ├── models/
│   │   └── schemas.py           # Pydantic models
│   └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   └── app/
│       ├── layout.tsx
│       ├── page.tsx
│       └── components/
│           ├── ChatTab.tsx
│           ├── DocumentsTab.tsx
│           └── AgentTab.tsx
├── docker-compose.yml           # 7 services
├── .github/
│   └── workflows/
│       └── deploy.yml
├── docs/
│   ├── adr/
│   │   ├── ADR-001-architecture.md
│   │   ├── ADR-002-auth.md
│   │   ├── ADR-003-streaming.md
│   │   ├── ADR-004-storage.md
│   │   └── ADR-005-deployment.md
│   └── c4/
│       ├── level1-system-context.md
│       ├── level2-containers.md
│       ├── level3-components.md
│       └── level4-code.md
└── README.md
```

---

## Phase 1: Backend Foundation

### Step 1.1 — Create backend/requirements.txt
```bash
mkdir -p backend/routers backend/engines backend/models
cat > backend/requirements.txt << 'EOF'
fastapi>=0.111
uvicorn[standard]>=0.29
pydantic>=2.0
python-jose[cryptography]>=3.3
passlib[bcrypt]>=1.7
python-multipart>=0.0.9
httpx>=0.27
sse-starlette>=1.6
langchain>=0.2
langchain-community>=0.2
langgraph>=0.1
chromadb>=0.5
anthropic>=0.28
openai>=1.30
google-cloud-aiplatform>=1.50
boto3>=1.34
prometheus-client>=0.20
psycopg2-binary>=2.9
redis>=5.0
sqlalchemy>=2.0
alembic>=1.13
python-dotenv>=1.0
pypdf>=4.0
EOF
pip install -r backend/requirements.txt
```

### Step 1.2 — Create backend/auth.py

```python
# backend/auth.py
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me-in-production-use-long-random-string")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Demo users (in production, use PostgreSQL)
DEMO_USERS = {
    "admin": {"username": "admin", "hashed_password": pwd_context.hash("admin123"), "role": "admin"},
    "user":  {"username": "user",  "hashed_password": pwd_context.hash("user123"),  "role": "user"},
}


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in DEMO_USERS:
            raise HTTPException(status_code=401, detail="Invalid token")
        return DEMO_USERS[username]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Step 1.3 — Create backend/models/schemas.py

```python
# backend/models/schemas.py
from pydantic import BaseModel
from typing import Optional


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system_prompt: str = "You are a helpful AI assistant."
    use_rag: bool = False
    provider_tier: str = "fast"
    max_tokens: int = 1024
    stream: bool = False


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    rag_used: bool = False
    sources: list[dict] = []
    cost_usd: float = 0.0


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_vectors: int
    status: str


class AgentRequest(BaseModel):
    topic: str
    max_iterations: int = 8


class AgentStatusResponse(BaseModel):
    task_id: str
    status: str
    topic: str
    iteration_count: int = 0
    final_report: Optional[str] = None
    awaiting_approval: bool = False
```

---

## Phase 2: Backend Engines

### Step 2.1 — Create backend/engines/rag_engine.py

```python
# backend/engines/rag_engine.py
"""RAG engine — reuses patterns from Project 06."""
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


class RAGEngine:
    def __init__(self):
        self._embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
        self._vectorstore = None
        self._load_or_create_store()

    def _load_or_create_store(self):
        self._vectorstore = Chroma(
            embedding_function=self._embeddings,
            persist_directory=CHROMA_DIR,
            collection_name="capstone_docs",
        )

    def ingest_pdf(self, file_path: str) -> int:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        self._vectorstore.add_documents(chunks)
        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def augment_prompt(self, query: str) -> tuple[str, list[dict]]:
        """Return augmented context and source list."""
        docs = self.retrieve(query)
        context_parts = []
        sources = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page   = doc.metadata.get("page", "")
            context_parts.append(f"[Source {i}: {source} p.{page}]\n{doc.page_content}")
            sources.append({"source": source, "page": page, "preview": doc.page_content[:150]})
        context = "\n\n---\n\n".join(context_parts)
        augmented = f"Context:\n{context}\n\nQuestion: {query}"
        return augmented, sources

    @property
    def vector_count(self) -> int:
        return self._vectorstore._collection.count()
```

### Step 2.2 — Create backend/engines/agent_engine.py

```python
# backend/engines/agent_engine.py
"""Agent engine — reuses LangGraph patterns from Project 08."""
import uuid
import threading
from typing import Optional
from langchain_core.messages import HumanMessage

# Re-import from P08 patterns
# In production, these would be in a shared package
ACTIVE_TASKS: dict[str, dict] = {}


class AgentEngine:
    """Manages research agent tasks with persistence."""

    def start_task(self, topic: str, max_iterations: int = 8) -> str:
        """Start a research task and return its ID."""
        task_id = str(uuid.uuid4())[:8]
        ACTIVE_TASKS[task_id] = {
            "status": "starting",
            "topic": topic,
            "iteration_count": 0,
            "final_report": None,
            "awaiting_approval": False,
        }

        def run():
            try:
                # Simplified agent loop for capstone
                # In production, use the full LangGraph agent from P08
                import time
                ACTIVE_TASKS[task_id]["status"] = "running"
                for i in range(min(max_iterations, 3)):
                    time.sleep(2)  # Simulate work
                    ACTIVE_TASKS[task_id]["iteration_count"] = i + 1
                ACTIVE_TASKS[task_id]["status"] = "awaiting_approval"
                ACTIVE_TASKS[task_id]["awaiting_approval"] = True
            except Exception as e:
                ACTIVE_TASKS[task_id]["status"] = f"error: {e}"

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return task_id

    def get_status(self, task_id: str) -> Optional[dict]:
        return ACTIVE_TASKS.get(task_id)

    def approve(self, task_id: str) -> bool:
        task = ACTIVE_TASKS.get(task_id)
        if not task or task["status"] != "awaiting_approval":
            return False
        task["status"] = "complete"
        task["awaiting_approval"] = False
        task["final_report"] = f"Research complete on: {task['topic']}\n\n[Full report would appear here]"
        return True
```

---

## Phase 3: API Routers

### Step 3.1 — Create backend/routers/chat.py

```python
# backend/routers/chat.py
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.auth import get_current_user
from backend.models.schemas import ChatRequest, ChatResponse
from backend.engines.rag_engine import RAGEngine
from backend.engines.llm_router import LLMRouter  # From P11

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
_rag_engine: RAGEngine = None
_llm_router: LLMRouter = None


def get_rag() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def get_router() -> LLMRouter:
    global _llm_router
    if _llm_router is None:
        _llm_router = LLMRouter()
    return _llm_router


@router.post("")
def chat(
    request: ChatRequest,
    user=Depends(get_current_user),
    rag: RAGEngine = Depends(get_rag),
    llm: LLMRouter = Depends(get_router),
):
    messages = [m.model_dump() for m in request.messages]
    system_prompt = request.system_prompt
    sources = []
    rag_used = False

    # Augment with RAG if requested
    if request.use_rag and messages:
        last_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)
        if last_user_msg:
            augmented, sources = rag.augment_prompt(last_user_msg["content"])
            messages = messages[:-1] + [{"role": "user", "content": augmented}]
            system_prompt = "You are a helpful assistant. Always cite sources using [Source N] notation."
            rag_used = True

    if request.stream:
        def generate():
            for chunk in llm._providers.get("bedrock", list(llm._providers.values())[0]).stream(
                messages, system_prompt, request.max_tokens
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    response = llm.generate(messages, system_prompt, request.max_tokens, request.provider_tier)
    return ChatResponse(
        text=response.text,
        provider=response.provider,
        model=response.model,
        rag_used=rag_used,
        sources=sources,
        cost_usd=response.cost_usd,
    )
```

### Step 3.2 — Create backend/routers/documents.py

```python
# backend/routers/documents.py
import os
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from backend.auth import get_current_user
from backend.models.schemas import DocumentUploadResponse
from backend.engines.rag_engine import RAGEngine

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def get_rag() -> RAGEngine:
    return RAGEngine()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    rag: RAGEngine = Depends(get_rag),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = rag.ingest_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    return DocumentUploadResponse(
        filename=file.filename,
        chunks_added=chunks,
        total_vectors=rag.vector_count,
        status="success",
    )


@router.get("")
def list_documents(user=Depends(get_current_user), rag: RAGEngine = Depends(get_rag)):
    return {"total_vectors": rag.vector_count, "status": "ok"}
```

### Step 3.3 — Create backend/routers/agent.py

```python
# backend/routers/agent.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from backend.auth import get_current_user
from backend.models.schemas import AgentRequest, AgentStatusResponse
from backend.engines.agent_engine import AgentEngine

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])
_engine = AgentEngine()


@router.post("", response_model=AgentStatusResponse)
def start_agent(
    request: AgentRequest,
    user=Depends(get_current_user),
):
    task_id = _engine.start_task(request.topic, request.max_iterations)
    return AgentStatusResponse(task_id=task_id, status="started", topic=request.topic)


@router.get("/{task_id}", response_model=AgentStatusResponse)
def get_agent_status(task_id: str, user=Depends(get_current_user)):
    status = _engine.get_status(task_id)
    if not status:
        raise HTTPException(404, f"Task {task_id} not found")
    return AgentStatusResponse(task_id=task_id, **status)


@router.post("/{task_id}/approve")
def approve_agent(task_id: str, user=Depends(get_current_user)):
    success = _engine.approve(task_id)
    if not success:
        raise HTTPException(400, "Task not awaiting approval")
    return {"message": f"Task {task_id} approved"}
```

### Step 3.4 — Create backend/main.py

```python
# backend/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from passlib.context import CryptContext
from jose import jwt
from backend.auth import pwd_context, create_access_token, DEMO_USERS
from backend.routers import chat, documents, agent
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel

request_counter = Counter("api_requests_total", "Total API requests", ["endpoint"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI Platform starting up...")
    yield
    print("AI Platform shutting down...")


app = FastAPI(
    title="Full-Stack AI Platform",
    description="Capstone: Multi-cloud LLM + RAG + Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(agent.router)


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/v1/auth/login")
def login(request: LoginRequest):
    user = DEMO_USERS.get(request.username)
    if not user or not pwd_context.verify(request.password, user["hashed_password"]):
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": request.username})
    return {"access_token": token, "token_type": "bearer", "username": request.username}


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### Step 3.5 — Run backend
```bash
cd backend
uvicorn main:app --reload --port 8000
# Visit http://localhost:8000/docs
```

---

## Phase 4: Frontend (Next.js)

### Step 4.1 — Initialize Next.js app
```bash
cd ~/Documents/ai-journey/projects/12-capstone
npx create-next-app@latest frontend \
  --typescript \
  --tailwind \
  --app \
  --no-src-dir \
  --import-alias "@/*"
cd frontend
npm install axios
```

### Step 4.2 — Create frontend/app/page.tsx

```tsx
// frontend/app/page.tsx
"use client";
import { useState } from "react";
import ChatTab from "./components/ChatTab";
import DocumentsTab from "./components/DocumentsTab";
import AgentTab from "./components/AgentTab";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"chat" | "documents" | "agent">("chat");
  const [token, setToken] = useState<string>("");

  const handleLogin = async () => {
    const res = await fetch("http://localhost:8000/api/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: "admin", password: "admin123" }),
    });
    const data = await res.json();
    setToken(data.access_token);
  };

  if (!token) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-gray-900">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-white mb-4">AI Platform</h1>
          <p className="text-gray-400 mb-6">Full-Stack GenAI Capstone Project</p>
          <button
            onClick={handleLogin}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded"
          >
            Login as Admin
          </button>
        </div>
      </main>
    );
  }

  const tabs = [
    { id: "chat", label: "Chat" },
    { id: "documents", label: "Documents" },
    { id: "agent", label: "Agent" },
  ] as const;

  return (
    <main className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex justify-between items-center">
          <h1 className="text-xl font-bold">AI Platform</h1>
          <div className="flex gap-2">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded ${
                  activeTab === tab.id
                    ? "bg-blue-600"
                    : "bg-gray-700 hover:bg-gray-600"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="p-6">
        {activeTab === "chat" && <ChatTab token={token} />}
        {activeTab === "documents" && <DocumentsTab token={token} />}
        {activeTab === "agent" && <AgentTab token={token} />}
      </div>
    </main>
  );
}
```

### Step 4.3 — Create frontend/app/components/ChatTab.tsx

```tsx
// frontend/app/components/ChatTab.tsx
"use client";
import { useState } from "react";

interface Message { role: "user" | "assistant"; content: string; }

export default function ChatTab({ token }: { token: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [useRag, setUseRag] = useState(false);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg: Message = { role: "user", content: input };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/v1/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          messages: newMessages,
          use_rag: useRag,
          stream: false,
        }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, { role: "assistant", content: data.text }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm text-gray-300">
          <input
            type="checkbox"
            checked={useRag}
            onChange={e => setUseRag(e.target.checked)}
          />
          Use RAG (search documents)
        </label>
      </div>

      <div className="bg-gray-800 rounded-lg p-4 h-96 overflow-y-auto mb-4 space-y-3">
        {messages.length === 0 && (
          <p className="text-gray-500 text-center mt-16">Ask anything...</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg text-sm ${
                msg.role === "user" ? "bg-blue-600" : "bg-gray-700"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {loading && <div className="text-gray-400 text-sm">Thinking...</div>}
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && sendMessage()}
          placeholder="Type your message..."
          className="flex-1 bg-gray-700 rounded px-4 py-2 text-sm focus:outline-none"
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 px-4 py-2 rounded text-sm"
        >
          Send
        </button>
      </div>
    </div>
  );
}
```

### Step 4.4 — Create minimal DocumentsTab and AgentTab

```tsx
// frontend/app/components/DocumentsTab.tsx
"use client";
import { useState } from "react";

export default function DocumentsTab({ token }: { token: string }) {
  const [status, setStatus] = useState("");
  const [uploading, setUploading] = useState(false);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("http://localhost:8000/api/v1/documents/upload", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });
      const data = await res.json();
      setStatus(`Uploaded: ${data.filename} (${data.chunks_added} chunks, total: ${data.total_vectors})`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Upload Documents</h2>
      <div className="bg-gray-800 rounded-lg p-6">
        <input type="file" accept=".pdf" onChange={handleUpload} disabled={uploading}
          className="block w-full text-sm text-gray-400 mb-4" />
        {uploading && <p className="text-blue-400 text-sm">Uploading and ingesting...</p>}
        {status && <p className="text-green-400 text-sm mt-2">{status}</p>}
      </div>
    </div>
  );
}
```

```tsx
// frontend/app/components/AgentTab.tsx
"use client";
import { useState, useEffect } from "react";

export default function AgentTab({ token }: { token: string }) {
  const [topic, setTopic] = useState("");
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<any>(null);

  const startAgent = async () => {
    const res = await fetch("http://localhost:8000/api/v1/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ topic, max_iterations: 6 }),
    });
    const data = await res.json();
    setTaskId(data.task_id);
  };

  const approve = async () => {
    await fetch(`http://localhost:8000/api/v1/agent/${taskId}/approve`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
  };

  useEffect(() => {
    if (!taskId) return;
    const interval = setInterval(async () => {
      const res = await fetch(`http://localhost:8000/api/v1/agent/${taskId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      setTaskStatus(data);
      if (data.status === "complete") clearInterval(interval);
    }, 2000);
    return () => clearInterval(interval);
  }, [taskId]);

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-lg font-semibold mb-4">Research Agent</h2>
      <div className="flex gap-2 mb-4">
        <input value={topic} onChange={e => setTopic(e.target.value)}
          placeholder="Research topic..." className="flex-1 bg-gray-700 rounded px-4 py-2 text-sm" />
        <button onClick={startAgent} className="bg-blue-600 px-4 py-2 rounded text-sm">Start</button>
      </div>

      {taskStatus && (
        <div className="bg-gray-800 rounded-lg p-4 space-y-2">
          <p className="text-sm">Task: <code>{taskStatus.task_id}</code></p>
          <p className="text-sm">Status: <span className="text-yellow-400">{taskStatus.status}</span></p>
          <p className="text-sm">Iterations: {taskStatus.iteration_count}</p>

          {taskStatus.awaiting_approval && (
            <button onClick={approve} className="bg-green-600 px-4 py-2 rounded text-sm">
              Approve & Finalize
            </button>
          )}

          {taskStatus.final_report && (
            <div className="mt-4 p-3 bg-gray-700 rounded text-sm whitespace-pre-wrap">
              {taskStatus.final_report}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

### Step 4.5 — Run frontend
```bash
cd frontend
npm run dev
# Open http://localhost:3000
```

---

## Phase 5: Docker Compose (7 Services)

### Step 5.1 — Create docker-compose.yml

```yaml
# docker-compose.yml
version: "3.9"

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/aiplatform
      - REDIS_URL=redis://redis:6379
      - CHROMA_DIR=/app/chroma_db
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    env_file:
      - .env
    depends_on:
      - postgres
      - redis
      - chromadb
    volumes:
      - ./chroma_db:/app/chroma_db

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: aiplatform
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma-data:/chroma/.chroma

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  postgres-data:
  chroma-data:
  grafana-data:
```

### Step 5.2 — Run the stack
```bash
docker compose up --build -d
docker compose ps  # All 7 should be running
```

---

## Phase 6: CI/CD, C4 Diagrams, and ADRs

### Step 6.1 — Create .github/workflows/deploy.yml
Reuse the GitHub Actions workflow from Project 10, updated for this project's ECR repository and ECS service names.

### Step 6.2 — Write the 5 ADRs

Create each file in `docs/adr/`:

- **ADR-001-architecture.md**: Monorepo with separate backend/frontend containers — enables independent scaling
- **ADR-002-auth.md**: JWT stateless tokens — compatible with SSE streaming (no session cookies needed)
- **ADR-003-streaming.md**: Server-Sent Events (SSE) for chat — simpler than WebSockets for unidirectional streaming
- **ADR-004-storage.md**: PostgreSQL for user data + ChromaDB for vectors + Redis for cache — purpose-built storage for each data type
- **ADR-005-deployment.md**: ECS Fargate + ALB — consistent with P10, avoids EKS overhead at this scale

### Step 6.3 — Write C4 diagrams

Create each file in `docs/c4/`:

- **level1-system-context.md**: User → AI Platform → [Bedrock, Azure OpenAI, Vertex AI, Ollama]
- **level2-containers.md**: Frontend, Backend API, PostgreSQL, Redis, ChromaDB, Prometheus, Grafana
- **level3-components.md**: Backend components: Auth, Chat Router, Doc Router, Agent Router, LLM Router, RAG Engine, Agent Engine
- **level4-code.md**: Key classes: `LLMRouter.generate()`, `RAGEngine.augment_prompt()`, `AgentEngine.start_task()`

---

## Verification Checklist
- [ ] `docker compose up` starts all 7 services cleanly (`docker compose ps`)
- [ ] Frontend loads at http://localhost:3000 — login button visible
- [ ] Login with admin/admin123 succeeds, all 3 tabs visible
- [ ] Chat tab: sends message, gets LLM response
- [ ] Chat tab: RAG toggle enabled, response includes source citations
- [ ] Documents tab: PDF upload succeeds, chunk count shown
- [ ] Agent tab: Start research, status updates every 2s, Approve button appears
- [ ] Approve button finalizes and shows report
- [ ] Grafana at http://localhost:3001 shows API request metrics
- [ ] GitHub Actions workflow runs green (test → build → push)
- [ ] 5 ADRs written in `docs/adr/`
- [ ] 4 C4 diagrams written in `docs/c4/`
- [ ] Total cloud cost < $60

---

## Troubleshooting

**CORS errors in browser**
Check `allow_origins` in `backend/main.py` matches frontend URL (e.g., `http://localhost:3000`).

**Ollama not reachable from Docker**
Use `http://host.docker.internal:11434` (macOS/Windows) or your host IP (Linux) in `OLLAMA_BASE_URL`.

**ChromaDB embedding fails**
Ensure `nomic-embed-text` model is pulled and Ollama is running before starting Docker.

**`docker compose up` port conflict**
If port 3000 is in use (e.g., by a previous Next.js dev server): `kill $(lsof -ti:3000)`

**JWT token expired**
Re-login to get a new token. Token expiry is 60 minutes by default.

---

## Congratulations — AI Architect Journey Complete!

You've built:
- 12 hands-on projects across AWS, Azure, GCP
- A full-stack AI platform with RAG, agents, and multi-cloud LLMs
- CI/CD pipelines, Dockerized services, and monitoring dashboards
- Architecture documentation (ADRs + C4 diagrams)

**Total XP earned:** 5,250 points → Architect level unlocked

Next steps for your portfolio:
1. Deploy the capstone to ECS and share the live URL
2. Write a blog post walkthrough of the journey
3. Prepare for certification exams using the interview prep in `README.md`
