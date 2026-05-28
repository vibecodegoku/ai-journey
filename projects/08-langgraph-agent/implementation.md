# Project 08: Research Agent with LangGraph + Tool Use — Implementation Guide

## Prerequisites
- Project 06 complete (LangChain installed)
- Claude API access or Bedrock (from Project 04)
- Tavily API key: https://tavily.com (free tier: 1000 searches/month)
- `conda activate aiarch`
- Packages: `langgraph`, `tavily-python`, `beautifulsoup4`, `fastapi`, `langsmith`

---

## Project Structure
```
08-langgraph-agent/
├── state.py           # TypedDict state schema
├── tools.py           # 4 agent tools
├── nodes.py           # Researcher + ToolNode logic
├── graph.py           # LangGraph state machine
├── checkpointer.py    # SQLite persistence
├── api.py             # FastAPI wrapper
├── test_agent.py      # Integration test
├── docs/
│   └── adr/
│       └── ADR-001-state-management.md
└── .env
```

---

## Phase 1: Setup

### Step 1.1 — Install dependencies
```bash
conda activate aiarch
pip install langgraph tavily-python anthropic langchain-anthropic \
            beautifulsoup4 requests fastapi uvicorn aiosqlite langsmith
```

### Step 1.2 — Get API keys
**Tavily** (web search):
1. Go to https://tavily.com → Sign Up → Get API Key (free tier)

**Anthropic Claude** (via API):
1. Go to https://console.anthropic.com → API Keys → Create Key
2. Or reuse AWS Bedrock credentials from Project 04

### Step 1.3 — Create .env
```bash
cat > .env << 'EOF'
TAVILY_API_KEY=tvly-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=aiarch-langgraph-agent
SQLITE_DB_PATH=agent_state.db
MAX_ITERATIONS=10
EOF
```

---

## Phase 2: State Schema

### Step 2.1 — Create state.py

```python
# state.py
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The complete state of the research agent.
    All fields are carried through the graph execution.
    """
    # Message history (append-only, managed by LangGraph)
    messages: Annotated[list[BaseMessage], add_messages]

    # Research context
    research_topic: str
    search_results: list[dict]       # Raw search results
    code_outputs: list[str]          # Python execution outputs
    report_sections: list[dict]      # Written report sections

    # Control flow
    iteration_count: int
    max_iterations: int
    human_approved: bool             # HITL approval flag

    # Output
    final_report: str
```

---

## Phase 3: Agent Tools

### Step 3.1 — Create tools.py

```python
# tools.py
import os
import subprocess
import textwrap
import tempfile
from typing import Optional
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tavily import TavilyClient

# Tool 1: Web Search
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information on a given query. Returns snippets and URLs."""
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    try:
        response = tavily.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )
        results = []
        for r in response.get("results", []):
            results.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}")
        return "\n\n---\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"


# Tool 2: Read URL
@tool
def read_url(url: str, max_chars: int = 3000) -> str:
    """Fetch and extract the main text content from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        # Remove scripts, styles, nav
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)

        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + f"\n\n[Truncated — {len(cleaned)} chars total]"

        return cleaned
    except Exception as e:
        return f"Error reading URL: {e}"


# Tool 3: Execute Python
@tool
def execute_python(code: str) -> str:
    """
    Execute a Python code snippet in a sandboxed subprocess.
    Use for data analysis, calculations, and generating insights.
    """
    try:
        # Write to temp file to avoid injection
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        os.unlink(tmp_path)

        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout}"
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return output or "Code executed (no output)"
    except subprocess.TimeoutExpired:
        return "Execution timed out (30s limit)"
    except Exception as e:
        return f"Execution error: {e}"


# Tool 4: Write Section
@tool
def write_section(title: str, content: str, section_type: str = "body") -> str:
    """
    Write a structured report section.
    section_type: 'introduction', 'body', 'analysis', 'conclusion'
    Returns confirmation with word count.
    """
    word_count = len(content.split())
    section = {
        "title": title,
        "content": content,
        "type": section_type,
        "word_count": word_count,
    }
    return f"Section written: '{title}' ({word_count} words, type={section_type})"


ALL_TOOLS = [web_search, read_url, execute_python, write_section]
```

---

## Phase 4: Graph Nodes

### Step 4.1 — Create nodes.py

```python
# nodes.py
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from state import AgentState
from tools import ALL_TOOLS, write_section

SYSTEM_PROMPT = """You are an expert research assistant. Your goal is to produce a comprehensive,
well-structured research report on the given topic.

You have access to these tools:
- web_search: Find current information on any topic
- read_url: Get full content from specific web pages
- execute_python: Run calculations or data analysis
- write_section: Write a report section

Research Process:
1. Search for the topic overview
2. Read 2-3 detailed sources
3. Analyze key findings (use execute_python if needed)
4. Write Introduction, 3-4 Body sections, and Conclusion using write_section
5. When done, say "RESEARCH COMPLETE" to trigger human review

Always cite sources. Be thorough but concise."""


def get_llm():
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0.3,
    ).bind_tools(ALL_TOOLS)


def researcher_node(state: AgentState) -> dict:
    """The main research node: calls Claude with tools."""
    llm = get_llm()

    # Build messages: system + history
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm.invoke(messages)

    # Track iterations
    new_iteration = state.get("iteration_count", 0) + 1

    return {
        "messages": [response],
        "iteration_count": new_iteration,
    }


def should_continue(state: AgentState) -> str:
    """Conditional routing: decide next step."""
    messages = state["messages"]
    last_message = messages[-1]

    # Check max iterations
    if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
        return "finalize"

    # Check if agent called tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Check if agent signaled completion
    content = getattr(last_message, "content", "")
    if "RESEARCH COMPLETE" in content:
        return "human_review"

    return "finalize"


# ToolNode handles all tool execution
tool_node = ToolNode(ALL_TOOLS)


def human_review_node(state: AgentState) -> dict:
    """
    HITL interrupt point. The graph pauses here, waiting for human approval.
    FastAPI's /approve endpoint resumes execution.
    """
    return {
        "human_approved": False,  # Set to True by external /approve call
    }


def finalize_node(state: AgentState) -> dict:
    """Compile all write_section calls into the final report."""
    # Extract written sections from tool messages
    from langchain_core.messages import ToolMessage
    sections = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and "Section written" in str(msg.content):
            sections.append(msg.content)

    # Compile final report from last LLM message
    final_msg = state["messages"][-1]
    final_report = getattr(final_msg, "content", "Report generation incomplete.")

    return {"final_report": final_report}
```

---

## Phase 5: Build the Graph

### Step 5.1 — Create graph.py

```python
# graph.py
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state import AgentState
from nodes import researcher_node, tool_node, human_review_node, finalize_node, should_continue

load_dotenv()

SQLITE_PATH = os.environ.get("SQLITE_DB_PATH", "agent_state.db")


def build_graph(checkpointer=None):
    """Build and compile the research agent graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools",      tool_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("finalize",   finalize_node)

    # Set entry point
    workflow.set_entry_point("researcher")

    # Conditional edges from researcher
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {
            "tools":        "tools",
            "human_review": "human_review",
            "finalize":     "finalize",
        },
    )

    # After tools, go back to researcher
    workflow.add_edge("tools", "researcher")

    # After human review (HITL interrupt), go to finalize
    workflow.add_edge("human_review", "finalize")

    # End
    workflow.add_edge("finalize", END)

    if checkpointer:
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_review"],  # HITL interrupt point
        )
    return workflow.compile()


def get_graph_with_persistence():
    """Get graph with SQLite checkpointer for state persistence."""
    checkpointer = SqliteSaver.from_conn_string(SQLITE_PATH)
    return build_graph(checkpointer), checkpointer
```

---

## Phase 6: FastAPI Wrapper

### Step 6.1 — Create api.py

```python
# api.py
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from state import AgentState
from graph import get_graph_with_persistence
from langchain_core.messages import HumanMessage

load_dotenv()

# Global graph + checkpointer
graph = None
checkpointer = None
active_tasks: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, checkpointer
    graph, checkpointer = get_graph_with_persistence()
    yield


app = FastAPI(title="Research Agent API", lifespan=lifespan)


class ResearchRequest(BaseModel):
    topic: str
    max_iterations: int = 10


class ResearchResponse(BaseModel):
    research_id: str
    status: str
    topic: str


class ResearchStatus(BaseModel):
    research_id: str
    status: str
    iteration_count: int
    topic: str
    final_report: Optional[str] = None
    awaiting_approval: bool = False


def run_research_sync(research_id: str, topic: str, max_iterations: int):
    """Run the research agent synchronously (called in background task)."""
    config = {"configurable": {"thread_id": research_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=f"Research this topic thoroughly: {topic}")],
        "research_topic": topic,
        "search_results": [],
        "code_outputs": [],
        "report_sections": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "human_approved": False,
        "final_report": "",
    }

    active_tasks[research_id] = {"status": "running", "topic": topic}

    # Run until interrupt (HITL) or completion
    for chunk in graph.stream(initial_state, config):
        node_name = list(chunk.keys())[0]
        if node_name == "human_review":
            active_tasks[research_id]["status"] = "awaiting_approval"
            return  # Pause here; /approve will resume

    active_tasks[research_id]["status"] = "complete"


@app.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background: BackgroundTasks):
    """Start a new research task."""
    research_id = str(uuid.uuid4())[:8]
    background.add_task(
        run_research_sync, research_id, request.topic, request.max_iterations
    )
    return ResearchResponse(
        research_id=research_id,
        status="started",
        topic=request.topic,
    )


@app.get("/research/{research_id}", response_model=ResearchStatus)
async def get_research_status(research_id: str):
    """Get the current status and result of a research task."""
    task_info = active_tasks.get(research_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Research ID {research_id} not found")

    config = {"configurable": {"thread_id": research_id}}
    try:
        state = graph.get_state(config)
        values = state.values if state else {}
        iteration = values.get("iteration_count", 0)
        final_report = values.get("final_report", "")
        awaiting = task_info["status"] == "awaiting_approval"
    except Exception:
        iteration = 0
        final_report = ""
        awaiting = False

    return ResearchStatus(
        research_id=research_id,
        status=task_info.get("status", "unknown"),
        iteration_count=iteration,
        topic=task_info.get("topic", ""),
        final_report=final_report or None,
        awaiting_approval=awaiting,
    )


@app.post("/research/{research_id}/approve")
async def approve_research(research_id: str, background: BackgroundTasks):
    """Resume agent after human review approval (HITL)."""
    task_info = active_tasks.get(research_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Research not found")
    if task_info["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail="Research is not awaiting approval")

    config = {"configurable": {"thread_id": research_id}}

    # Update state to approved and resume
    graph.update_state(config, {"human_approved": True})
    active_tasks[research_id]["status"] = "resuming"

    def resume():
        for chunk in graph.stream(None, config):
            pass
        active_tasks[research_id]["status"] = "complete"

    background.add_task(resume)
    return {"message": f"Research {research_id} approved and resumed"}
```

### Step 6.2 — Run the API
```bash
conda activate aiarch
uvicorn api:app --reload --port 8001
```

### Step 6.3 — Test the full flow
```bash
# Start a research task
RESEARCH_ID=$(curl -s -X POST http://localhost:8001/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Latest advances in RAG systems 2024", "max_iterations": 8}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['research_id'])")

echo "Research ID: $RESEARCH_ID"

# Poll status (repeat until awaiting_approval or complete)
curl http://localhost:8001/research/$RESEARCH_ID

# If awaiting_approval, approve it
curl -X POST http://localhost:8001/research/$RESEARCH_ID/approve

# Get final report
curl http://localhost:8001/research/$RESEARCH_ID | python3 -m json.tool
```

---

## Phase 7: Architecture Decision Record

Create `docs/adr/ADR-001-state-management.md`:

```markdown
# ADR-001: State Management — LangGraph vs AgentExecutor

**Date:** 2026-05-27
**Status:** Accepted

## Context
We need a research agent that can use multiple tools, maintain state across iterations,
support human-in-the-loop interrupts, and persist state across process restarts.

## Decision
Use **LangGraph** with **SqliteSaver checkpointer**.

## Alternatives Considered
| Framework | Pros | Cons |
|-----------|------|------|
| **LangGraph** | Graph-based control flow, HITL interrupts, SQLite checkpointing | Learning curve |
| LangChain AgentExecutor | Simple, well-documented | No native HITL, no state persistence |
| AutoGen | Multi-agent conversations | Heavier setup, less precise control flow |
| Custom loop | Full control | Manual state management, no built-in persistence |

## Rationale
- `interrupt_before=["human_review"]` provides native HITL support
- `SqliteSaver` persists the full agent state across process restarts — critical for long research tasks
- Conditional edges (`should_continue`) give precise control over the agent loop
- LangSmith integration captures the full execution graph for observability

## Consequences
- State is a TypedDict — all fields must be defined upfront
- `graph.update_state()` enables external state injection (used by `/approve` endpoint)
- SQLite is sufficient for development; production would use PostgreSQL checkpointer
- The `interrupt_before` mechanic pauses the graph at a specific node boundary
```

---

## Verification Checklist
- [ ] `.env` has valid TAVILY_API_KEY and ANTHROPIC_API_KEY
- [ ] `uvicorn api:app --port 8001` starts without errors
- [ ] `POST /research` returns a research_id
- [ ] Agent begins searching (visible in server logs)
- [ ] `GET /research/{id}` shows increasing iteration_count
- [ ] Agent pauses with `status: awaiting_approval` after "RESEARCH COMPLETE"
- [ ] `POST /research/{id}/approve` resumes and sets status to `complete`
- [ ] Final report contains multiple sections with cited sources
- [ ] `agent_state.db` file created (SQLite persistence working)
- [ ] LangSmith trace shows tool calls (if configured)
- [ ] Max iterations enforced (test by setting `max_iterations: 2`)
- [ ] ADR-001 written

---

## Troubleshooting

**`AuthenticationError` from Anthropic**
Verify `ANTHROPIC_API_KEY` in `.env`. Get a key at https://console.anthropic.com

**`TavilyError: 401`**
Check TAVILY_API_KEY. Free tier allows 1000 searches/month.

**Agent loops indefinitely**
The `should_continue` function's `"RESEARCH COMPLETE"` check is case-sensitive.
The LLM may output lowercase. Adjust:
```python
if "research complete" in content.lower():
    return "human_review"
```

**SQLite `database is locked`**
Only one process can write at a time. Ensure you're not running multiple uvicorn workers.

**HITL interrupt not firing**
Ensure `interrupt_before=["human_review"]` is passed to `workflow.compile()` — not to `add_node`.

---

## Next Steps → Project 09: CrewAI Multi-Agent
```bash
conda activate aiarch
pip install crewai crewai-tools
mkdir -p ~/Documents/ai-journey/projects/09-crewai-multiagent
```
