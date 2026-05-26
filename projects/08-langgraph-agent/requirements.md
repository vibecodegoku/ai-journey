# Project 08: Research Agent with LangGraph + Tool Use

**XP:** 400 | **Cost:** ~$3–8 (LLM API calls) | **Duration:** Weeks 15–16

## Objective
Build a stateful, multi-step research agent using LangGraph that can search the web, read URLs, write code, execute it, and produce structured research reports. Implement human-in-the-loop (HITL) checkpoints for sensitive actions, persist state across sessions, and add LangSmith tracing for full observability.

## Tools
- **LangGraph** — stateful agent graph
- **LangChain** — tool definitions, LLM wrappers
- **Tavily Search API** — web search (free tier: 1000 searches/month)
- **Claude 3 Haiku or GPT-4o-mini** — LLM backbone
- **LangSmith** — full execution trace
- **FastAPI** — HTTP API wrapper for the agent

## Project Structure
```
08-langgraph-agent/
├── agent/
│   ├── graph.py           ← LangGraph state machine definition
│   ├── nodes.py           ← Node functions (search, read, code, report)
│   ├── tools.py           ← Tool definitions
│   ├── state.py           ← TypedDict state schema
│   └── checkpointer.py    ← Persistence (SQLite or Redis)
├── app/
│   └── api.py             ← FastAPI wrapper
├── notebooks/
│   └── 01_agent_walkthrough.ipynb
├── tests/
│   └── test_agent.py
├── docs/adr/
│   └── 001-state-management.md
└── requirements.txt
```

## Setup

```bash
pip install langgraph langchain langchain-anthropic langchain-community
pip install tavily-python               # web search
pip install langsmith fastapi uvicorn
pip install pytest httpx

# Set env vars
export ANTHROPIC_API_KEY="your-key"
export TAVILY_API_KEY="your-key"        # get at app.tavily.com
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="lsv2_pt_..."
export LANGCHAIN_PROJECT="research-agent"
```

## Phase 1: Define State Schema (Day 1)

`agent/state.py`:
```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    # Conversation messages (append-only via reducer)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Research context
    research_topic: str
    search_queries: list[str]
    search_results: list[dict]
    urls_visited: list[str]
    
    # Code execution
    code_snippets: list[str]
    code_outputs: list[str]
    
    # Report generation
    report_sections: dict[str, str]
    final_report: str
    
    # Control flow
    iterations: int
    max_iterations: int
    human_approved: bool
    status: str              # "researching" | "writing" | "awaiting_approval" | "done"
```

## Phase 2: Define Tools (Days 2–3)

`agent/tools.py`:
```python
from langchain_core.tools import tool
from tavily import TavilyClient
import subprocess
import sys
import requests
from bs4 import BeautifulSoup

tavily = TavilyClient()

@tool
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for current information on a topic."""
    results = tavily.search(query=query, max_results=max_results)
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"][:500]}
        for r in results.get("results", [])
    ]

@tool
def read_url(url: str) -> str:
    """Fetch and extract main text content from a URL."""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove scripts, styles, nav
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        text = soup.get_text(separator="\n", strip=True)
        # Limit to 3000 chars to avoid token overflow
        return text[:3000]
    except Exception as e:
        return f"Error reading URL: {e}"

@tool
def execute_python(code: str) -> str:
    """Execute Python code and return stdout/stderr. Use for calculations and data analysis."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        return output[:2000] if output else "Code executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"

@tool
def write_section(section_name: str, content: str) -> str:
    """Write a named section to the research report. Call multiple times for different sections."""
    return f"Section '{section_name}' recorded ({len(content)} chars)"

TOOLS = [web_search, read_url, execute_python, write_section]
```

## Phase 3: Build LangGraph (Days 4–6)

`agent/nodes.py`:
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from state import ResearchState
from tools import TOOLS

llm = ChatAnthropic(model="claude-3-haiku-20240307").bind_tools(TOOLS)

SYSTEM_PROMPT = """You are an expert research agent. Your job is to:
1. Search for relevant information using web_search
2. Read detailed content from promising URLs using read_url
3. Run calculations or data processing using execute_python
4. Write structured report sections using write_section
5. Produce a comprehensive final report

Be systematic: search broadly first, then dive deep on the most relevant sources.
Always cite your sources. Stop after writing all report sections."""

def researcher_node(state: ResearchState) -> ResearchState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1,
        "status": "researching",
    }

def tools_node(state: ResearchState) -> ResearchState:
    tool_node = ToolNode(TOOLS)
    result = tool_node.invoke(state)
    return result

def should_continue(state: ResearchState) -> str:
    messages = state["messages"]
    last = messages[-1]
    
    if state["iterations"] >= state["max_iterations"]:
        return "end"
    
    # If last message has tool calls, continue to tools
    if hasattr(last, "tool_calls") and last.tool_calls:
        # Check if any tool call is write_section — may need human approval
        for tc in last.tool_calls:
            if tc["name"] == "write_section":
                return "tools"
        return "tools"
    
    return "end"
```

`agent/graph.py`:
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from state import ResearchState
from nodes import researcher_node, should_continue
from tools import TOOLS

def build_graph(checkpoint_db: str = "checkpoints.db"):
    # SQLite checkpointer for persistence
    memory = SqliteSaver.from_conn_string(checkpoint_db)
    
    builder = StateGraph(ResearchState)
    
    # Add nodes
    builder.add_node("researcher", researcher_node)
    builder.add_node("tools", ToolNode(TOOLS))
    
    # Set entry point
    builder.set_entry_point("researcher")
    
    # Conditional routing
    builder.add_conditional_edges(
        "researcher",
        should_continue,
        {"tools": "tools", "end": END},
    )
    
    # Tools always return to researcher
    builder.add_edge("tools", "researcher")
    
    return builder.compile(checkpointer=memory)

graph = build_graph()

def run_research(topic: str, thread_id: str = "default", max_iterations: int = 10) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [HumanMessage(content=f"Research this topic comprehensively: {topic}")],
        "research_topic": topic,
        "search_queries": [],
        "search_results": [],
        "urls_visited": [],
        "code_snippets": [],
        "code_outputs": [],
        "report_sections": {},
        "final_report": "",
        "iterations": 0,
        "max_iterations": max_iterations,
        "human_approved": False,
        "status": "researching",
    }
    
    result = graph.invoke(initial_state, config)
    return result
```

## Phase 4: Human-in-the-Loop Checkpoint (Day 7)

Add HITL interrupt before publishing final report:
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

def build_hitl_graph(checkpoint_db: str = "checkpoints_hitl.db"):
    memory = SqliteSaver.from_conn_string(checkpoint_db)
    
    builder = StateGraph(ResearchState)
    builder.add_node("researcher", researcher_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("human_review", lambda s: s)   # passthrough — interrupt happens here
    builder.add_node("finalize", finalize_report)
    
    builder.set_entry_point("researcher")
    builder.add_conditional_edges("researcher", should_continue, {"tools": "tools", "end": "human_review"})
    builder.add_edge("tools", "researcher")
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)
    
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],    # PAUSE here for human approval
    )

# Usage pattern:
# 1. Run graph → hits interrupt at "human_review"
# 2. Human reviews state via graph.get_state(config)
# 3. Human resumes with graph.invoke(None, config)  ← None continues from checkpoint
```

## Phase 5: FastAPI Wrapper (Days 8–10)

`app/api.py`:
```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from agent.graph import run_research, graph
import uuid

app = FastAPI(title="Research Agent API")

class ResearchRequest(BaseModel):
    topic: str
    max_iterations: int = 10

class ResearchStatus(BaseModel):
    thread_id: str
    status: str
    iterations: int
    message_count: int

@app.post("/research", response_model=dict)
async def start_research(request: ResearchRequest, background: BackgroundTasks):
    thread_id = str(uuid.uuid4())
    background.add_task(run_research, request.topic, thread_id, request.max_iterations)
    return {"thread_id": thread_id, "status": "started"}

@app.get("/research/{thread_id}")
async def get_research_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state.values:
        return {"error": "Thread not found"}
    
    return {
        "thread_id": thread_id,
        "status": state.values.get("status", "unknown"),
        "iterations": state.values.get("iterations", 0),
        "report": state.values.get("final_report", ""),
        "next": list(state.next),
    }

@app.post("/research/{thread_id}/approve")
async def approve_and_continue(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(None, config)     # resume from interrupt
    return {"status": "resumed", "thread_id": thread_id}
```

Run:
```bash
uvicorn app.api:app --reload
```

## Acceptance Criteria
- [ ] Agent successfully researches a topic and produces a multi-section report
- [ ] Web search, URL reading, and code execution tools all work
- [ ] State persists across process restarts (SQLite checkpoint)
- [ ] HITL interrupt pauses execution; `/approve` endpoint resumes it
- [ ] LangSmith shows full execution trace with tool calls and token counts
- [ ] Agent stops at `max_iterations` without infinite loops
- [ ] FastAPI `/research` and `/research/{id}` endpoints respond correctly
- [ ] ADR-001 written: LangGraph vs LangChain AgentExecutor

## ADR Template
Create `docs/adr/001-state-management.md`:
```markdown
# ADR-001: Agent State Management Approach

## Status: Accepted

## Context
Need a stateful agent that can pause for human review, resume sessions, and track
multi-step research progress. LangChain AgentExecutor lacks checkpointing.

## Decision
LangGraph with SQLite checkpointer.

## Consequences
+ Full state persistence — restart agent mid-run after crashes
+ HITL interrupt pattern — pause any node for human review
+ Explicit graph topology — easier to debug than opaque AgentExecutor loops
- More boilerplate than AgentExecutor for simple use cases
- LangGraph API changes frequently (v0.1 → v0.2 breaking changes)
```
