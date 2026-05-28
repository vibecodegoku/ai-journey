# Project 09: Multi-Agent Content Team with CrewAI — Implementation Guide

## Prerequisites
- Project 08 complete (LangChain, Tavily installed)
- Anthropic API key or Bedrock configured
- `conda activate aiarch`
- Packages: `crewai`, `crewai-tools`, `streamlit`, `tavily-python`

---

## Project Structure
```
09-crewai-multiagent/
├── agents.py              # 4 specialized agents
├── tasks.py               # 4 sequential tasks
├── tools.py               # 3 custom tools
├── crew.py                # Crew assembly + run
├── app.py                 # Streamlit UI
├── outputs/               # Generated articles (.md) and metadata (.json)
├── docs/
│   └── adr/
│       └── ADR-001-agent-orchestration.md
└── .env
```

---

## Phase 1: Setup

### Step 1.1 — Install dependencies
```bash
conda activate aiarch
pip install crewai crewai-tools streamlit tavily-python anthropic
```

### Step 1.2 — Create .env
```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-your-key-here
TAVILY_API_KEY=tvly-your-key-here
OPENAI_API_KEY=not-needed   # Required by CrewAI even if using Anthropic
OLLAMA_BASE_URL=http://localhost:11434
EOF
```

> CrewAI requires an `OPENAI_API_KEY` env var to not crash on import, even when using other LLMs.
> Set it to a dummy value if using Anthropic or Ollama.

---

## Phase 2: Custom Tools

### Step 2.1 — Create tools.py

```python
# tools.py
import os
import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool
from tavily import TavilyClient


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for current information on a topic. "
        "Returns titles, URLs, and summaries of top results."
    )

    def _run(self, query: str) -> str:
        try:
            client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            response = client.search(query=query, max_results=5, search_depth="advanced")
            results = []
            for r in response.get("results", []):
                results.append(
                    f"Title: {r['title']}\n"
                    f"URL: {r['url']}\n"
                    f"Summary: {r['content'][:400]}"
                )
            return "\n\n---\n\n".join(results) or "No results found."
        except Exception as e:
            return f"Search error: {e}"


class ReadURLTool(BaseTool):
    name: str = "read_url"
    description: str = (
        "Read the full content of a web page given its URL. "
        "Returns the main text content, useful for in-depth research."
    )

    def _run(self, url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; ContentBot/1.0)"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            result = "\n".join(lines)[:4000]
            return result or "No readable content found."
        except Exception as e:
            return f"Error reading URL: {e}"


class WordCountTool(BaseTool):
    name: str = "word_count"
    description: str = (
        "Count the number of words in a given text. "
        "Use this to verify article length before submission."
    )

    def _run(self, text: str) -> str:
        words = len(text.split())
        chars = len(text)
        sentences = text.count(".") + text.count("!") + text.count("?")
        return (
            f"Word count: {words}\n"
            f"Character count: {chars}\n"
            f"Approximate sentences: {sentences}"
        )
```

---

## Phase 3: Agents

### Step 3.1 — Create agents.py

```python
# agents.py
import os
from crewai import Agent
from langchain_anthropic import ChatAnthropic
from tools import WebSearchTool, ReadURLTool, WordCountTool

search_tool = WebSearchTool()
read_tool   = ReadURLTool()
count_tool  = WordCountTool()


def get_llm(temperature: float = 0.3):
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=temperature,
    )


def create_researcher() -> Agent:
    return Agent(
        role="Senior Research Analyst",
        goal=(
            "Conduct thorough research on the given topic. "
            "Find credible sources, gather statistics, and identify key trends. "
            "Provide at least 5 distinct sources with URLs."
        ),
        backstory=(
            "You are an expert research analyst with 15 years of experience "
            "in content research. You excel at finding authoritative sources "
            "and synthesizing complex information."
        ),
        tools=[search_tool, read_tool],
        llm=get_llm(temperature=0.1),
        max_iter=5,
        verbose=True,
    )


def create_writer() -> Agent:
    return Agent(
        role="Senior Content Writer",
        goal=(
            "Write a comprehensive, engaging article of exactly 1400-1600 words "
            "based on the research findings. Include a compelling introduction, "
            "3-4 body sections with headers, and a strong conclusion."
        ),
        backstory=(
            "You are an award-winning content writer known for transforming "
            "complex research into engaging, readable articles. You always "
            "cite your sources inline and hit precise word count targets."
        ),
        tools=[count_tool],
        llm=get_llm(temperature=0.7),
        max_iter=3,
        verbose=True,
    )


def create_editor() -> Agent:
    return Agent(
        role="Chief Editor",
        goal=(
            "Review and improve the article for clarity, flow, accuracy, and readability. "
            "Ensure all claims have citations, fix grammar issues, and verify "
            "the word count is between 1400-1600 words."
        ),
        backstory=(
            "You are a meticulous editor with decades of experience at major "
            "publications. You have an eye for clarity and accuracy, and you "
            "never let an article through without proper citation."
        ),
        tools=[count_tool],
        llm=get_llm(temperature=0.2),
        max_iter=3,
        verbose=True,
    )


def create_seo_specialist() -> Agent:
    return Agent(
        role="SEO Specialist",
        goal=(
            "Generate comprehensive SEO metadata for the article including: "
            "meta title (50-60 chars), meta description (150-160 chars), "
            "5 primary keywords, 5 secondary keywords, and a URL slug. "
            "Return valid JSON."
        ),
        backstory=(
            "You are an SEO expert who has optimized content for major websites. "
            "You understand search intent, keyword density, and how to craft "
            "metadata that drives organic traffic."
        ),
        tools=[],
        llm=get_llm(temperature=0.3),
        max_iter=2,
        verbose=True,
    )
```

---

## Phase 4: Tasks

### Step 4.1 — Create tasks.py

```python
# tasks.py
from pydantic import BaseModel
from crewai import Task


# Pydantic output models
class ResearchFindings(BaseModel):
    topic: str
    key_points: list[str]
    sources: list[dict]        # [{"title": ..., "url": ..., "summary": ...}]
    statistics: list[str]
    expert_opinions: list[str]


class ArticleDraft(BaseModel):
    title: str
    content: str               # Full article text 1400-1600 words
    word_count: int
    sections: list[str]        # Section headings


class SEOMetadata(BaseModel):
    meta_title: str            # 50-60 chars
    meta_description: str      # 150-160 chars
    primary_keywords: list[str]    # 5 keywords
    secondary_keywords: list[str]  # 5 keywords
    url_slug: str
    schema_type: str           # e.g., "Article", "BlogPosting"


def create_research_task(researcher, topic: str) -> Task:
    return Task(
        description=(
            f"Research the topic: '{topic}'\n\n"
            "Your research must include:\n"
            "1. At least 5 credible sources with URLs\n"
            "2. Key statistics and data points\n"
            "3. Current trends and developments (2024-2025)\n"
            "4. Expert opinions or quotes\n"
            "5. Any controversies or debates in this space\n\n"
            "Use web_search to find sources, then read_url to get details."
        ),
        expected_output=(
            "A structured research report with: topic summary, "
            "5+ key points, 5+ sources with URLs, statistics, and expert insights."
        ),
        agent=researcher,
        output_pydantic=ResearchFindings,
    )


def create_writing_task(writer, research_task: Task) -> Task:
    return Task(
        description=(
            "Write a comprehensive article based on the research findings.\n\n"
            "Requirements:\n"
            "- Length: EXACTLY 1400-1600 words (use word_count tool to verify)\n"
            "- Structure: Introduction + 3-4 body sections with H2 headers + Conclusion\n"
            "- Cite at least 5 sources inline (e.g., 'According to [Source]...')\n"
            "- Write in an engaging, informative style for a general audience\n"
            "- Include a compelling headline\n\n"
            "Use the word_count tool to check your draft. Revise if outside 1400-1600 words."
        ),
        expected_output=(
            "A complete article with title, 1400-1600 words, "
            "structured sections, and inline source citations."
        ),
        agent=writer,
        context=[research_task],
        output_pydantic=ArticleDraft,
    )


def create_editing_task(editor, writing_task: Task) -> Task:
    return Task(
        description=(
            "Review and edit the article draft.\n\n"
            "Checklist:\n"
            "1. Verify word count is 1400-1600 (use word_count tool)\n"
            "2. Check all claims have citations\n"
            "3. Fix grammar, punctuation, and spelling errors\n"
            "4. Improve sentence variety and flow\n"
            "5. Ensure the introduction hooks the reader\n"
            "6. Verify the conclusion summarizes key points\n\n"
            "Return the COMPLETE edited article text, not just a summary of changes."
        ),
        expected_output=(
            "The fully edited article with the same structure, "
            "corrected grammar, improved flow, and verified word count."
        ),
        agent=editor,
        context=[writing_task],
        output_pydantic=ArticleDraft,
    )


def create_seo_task(seo_specialist, editing_task: Task) -> Task:
    return Task(
        description=(
            "Generate SEO metadata for the final article.\n\n"
            "Requirements:\n"
            "- meta_title: 50-60 characters (include primary keyword)\n"
            "- meta_description: 150-160 characters (compelling, includes call to action)\n"
            "- primary_keywords: 5 high-value keywords from the article\n"
            "- secondary_keywords: 5 related/LSI keywords\n"
            "- url_slug: lowercase, hyphenated, max 60 chars\n"
            "- schema_type: 'Article' or 'BlogPosting'\n\n"
            "Return ONLY valid JSON matching the SEOMetadata schema."
        ),
        expected_output=(
            "Valid JSON with meta_title, meta_description, primary_keywords (5), "
            "secondary_keywords (5), url_slug, and schema_type."
        ),
        agent=seo_specialist,
        context=[editing_task],
        output_pydantic=SEOMetadata,
    )
```

---

## Phase 5: Crew Assembly

### Step 5.1 — Create crew.py

```python
# crew.py
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from crewai import Crew, Process
from agents import create_researcher, create_writer, create_editor, create_seo_specialist
from tasks import (
    create_research_task, create_writing_task,
    create_editing_task, create_seo_task,
)

load_dotenv()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_content_crew(topic: str) -> dict:
    """Run the 4-agent content creation crew for a given topic."""
    print(f"\n{'='*60}")
    print(f"Starting Content Crew for: {topic}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Create agents
    researcher    = create_researcher()
    writer        = create_writer()
    editor        = create_editor()
    seo_specialist = create_seo_specialist()

    # Create tasks
    research_task = create_research_task(researcher, topic)
    writing_task  = create_writing_task(writer, research_task)
    editing_task  = create_editing_task(editor, writing_task)
    seo_task      = create_seo_task(seo_specialist, editing_task)

    # Assemble crew with sequential process
    crew = Crew(
        agents=[researcher, writer, editor, seo_specialist],
        tasks=[research_task, writing_task, editing_task, seo_task],
        process=Process.sequential,
        verbose=True,
        memory=True,                    # Shared memory via Ollama embeddings
        embedder={
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            },
        },
        max_rpm=10,                     # Rate limit: 10 requests/minute
    )

    result = crew.kickoff()
    elapsed = time.time() - start_time

    # Extract outputs
    article_text = ""
    seo_metadata = {}

    # Get task outputs
    for task_output in crew.tasks:
        if hasattr(task_output, "output") and task_output.output:
            output = task_output.output
            if hasattr(output, "pydantic") and output.pydantic:
                pydantic_obj = output.pydantic
                class_name = type(pydantic_obj).__name__
                if class_name == "ArticleDraft":
                    article_text = pydantic_obj.content
                elif class_name == "SEOMetadata":
                    seo_metadata = pydantic_obj.model_dump()

    # Fallback: use raw result
    if not article_text and hasattr(result, "raw"):
        article_text = str(result.raw)

    # Save outputs
    topic_slug = topic.lower().replace(" ", "-")[:40]
    article_path = OUTPUT_DIR / f"{topic_slug}.md"
    seo_path     = OUTPUT_DIR / f"{topic_slug}-seo.json"

    article_path.write_text(article_text)
    seo_path.write_text(json.dumps(seo_metadata, indent=2))

    word_count = len(article_text.split())
    print(f"\n{'='*60}")
    print(f"Crew complete in {elapsed:.1f}s")
    print(f"Article: {word_count} words → {article_path}")
    print(f"SEO metadata → {seo_path}")
    print(f"{'='*60}\n")

    return {
        "topic": topic,
        "article": article_text,
        "seo_metadata": seo_metadata,
        "word_count": word_count,
        "runtime_seconds": round(elapsed, 1),
        "article_path": str(article_path),
        "seo_path": str(seo_path),
    }


if __name__ == "__main__":
    topic = input("Enter article topic: ").strip()
    if not topic:
        topic = "The Impact of Large Language Models on Software Development"
    result = run_content_crew(topic)
```

### Step 5.2 — Run the crew (CLI test)
```bash
conda activate aiarch
# Make sure Ollama is running for memory embeddings
ollama serve &

python crew.py
# Enter: The Impact of Large Language Models on Software Development
```
Expected runtime: 3–8 minutes.

---

## Phase 6: Streamlit UI

### Step 6.1 — Create app.py

```python
# app.py
import json
import time
import threading
import streamlit as st
from crew import run_content_crew

st.set_page_config(page_title="AI Content Crew", page_icon="✍️", layout="wide")
st.title("AI Content Creation Crew")
st.caption("4 specialized agents: Researcher → Writer → Editor → SEO Specialist")

# Session state
if "result" not in st.session_state:
    st.session_state.result = None
if "running" not in st.session_state:
    st.session_state.running = False

# Input
with st.form("topic_form"):
    topic = st.text_input(
        "Article Topic",
        placeholder="e.g., The Future of Quantum Computing in Cybersecurity",
    )
    submit = st.form_submit_button("Generate Article", type="primary")

if submit and topic.strip():
    st.session_state.running = True
    st.session_state.result = None

    with st.spinner("Crew is working... (3-8 minutes)"):
        try:
            result = run_content_crew(topic.strip())
            st.session_state.result = result
        except Exception as e:
            st.error(f"Error: {e}")
    st.session_state.running = False

# Display results
if st.session_state.result:
    result = st.session_state.result

    # Stats header
    col1, col2, col3 = st.columns(3)
    col1.metric("Word Count", result["word_count"])
    col2.metric("Runtime", f"{result['runtime_seconds']:.0f}s")
    col3.metric("Sources", "5+")

    # Tabs for outputs
    tab1, tab2, tab3 = st.tabs(["Article", "SEO Metadata", "Statistics"])

    with tab1:
        st.markdown(result["article"])
        st.download_button(
            "Download Article (.md)",
            data=result["article"],
            file_name=f"article-{int(time.time())}.md",
            mime="text/markdown",
        )

    with tab2:
        seo = result["seo_metadata"]
        if seo:
            st.json(seo)
            st.download_button(
                "Download SEO JSON",
                data=json.dumps(seo, indent=2),
                file_name=f"seo-{int(time.time())}.json",
                mime="application/json",
            )
        else:
            st.warning("SEO metadata not available.")

    with tab3:
        st.subheader("Generation Statistics")
        st.write(f"**Topic:** {result['topic']}")
        st.write(f"**Word count:** {result['word_count']} words")
        st.write(f"**Runtime:** {result['runtime_seconds']} seconds")
        st.write(f"**Article saved:** `{result['article_path']}`")
        st.write(f"**SEO saved:** `{result['seo_path']}`")

        # Word count check
        if 1400 <= result["word_count"] <= 1600:
            st.success(f"Word count {result['word_count']} is within target range (1400-1600)")
        else:
            st.warning(f"Word count {result['word_count']} outside target range (1400-1600)")
```

### Step 6.2 — Run Streamlit
```bash
streamlit run app.py
```
Open **http://localhost:8501**

---

## Phase 7: Architecture Decision Record

Create `docs/adr/ADR-001-agent-orchestration.md`:

```markdown
# ADR-001: Agent Orchestration — CrewAI vs AutoGen vs LangGraph

**Date:** 2026-05-27
**Status:** Accepted

## Context
We need to orchestrate 4 specialized agents (Researcher, Writer, Editor, SEO) in a
sequential pipeline with shared memory and structured output validation.

## Decision
Use **CrewAI** with sequential process and Ollama-based shared memory.

## Alternatives Considered
| Framework | Pros | Cons |
|-----------|------|------|
| **CrewAI** | Role-based agents, sequential/parallel process, Pydantic outputs, built-in memory | Less flexible routing than LangGraph |
| LangGraph (from P08) | Precise control flow, HITL, checkpointing | More boilerplate for sequential pipelines |
| AutoGen | Conversational agents, good for debate | Complex setup, unclear sequential ordering |
| Raw LangChain | Maximum flexibility | No agent role abstraction |

## Rationale
- CrewAI's role-based abstraction (Agent with `role`, `goal`, `backstory`) maps directly
  to our 4 content team roles
- Sequential Process guarantees Researcher → Writer → Editor → SEO ordering
- Pydantic output models enforce structured outputs at each step
- Built-in memory with Ollama embeddings enables context sharing across agents
- `max_rpm=10` rate limiting prevents API throttling

## Consequences
- CrewAI requires OpenAI key in env even when using other LLMs (workaround: dummy value)
- Sequential process means total runtime is sum of all agent runtimes (~5-8 min)
- Parallel process available if independent tasks identified (future optimization)
```

---

## Verification Checklist
- [ ] `python crew.py` runs without ImportError
- [ ] All 4 agents initialized successfully (visible in verbose output)
- [ ] Researcher task produces 5+ sources
- [ ] Writer task produces 1400-1600 word article
- [ ] Editor task reviews and returns updated article
- [ ] SEO task returns valid JSON with all required fields
- [ ] Article saved to `outputs/<topic>.md`
- [ ] SEO metadata saved to `outputs/<topic>-seo.json`
- [ ] `streamlit run app.py` shows all 3 tabs
- [ ] Word count metric shown in Statistics tab
- [ ] Total runtime < 10 minutes
- [ ] `max_rpm=10` prevents rate limit errors
- [ ] ADR-001 written

---

## Troubleshooting

**`ImportError: No module named 'crewai'`**
```bash
pip install crewai crewai-tools
```

**CrewAI raises `OPENAI_API_KEY not set`**
Add to `.env`: `OPENAI_API_KEY=dummy` (only needed if not using OpenAI)

**Ollama memory embedding fails**
Ensure Ollama is running: `ollama serve &` and `nomic-embed-text` model is pulled.

**Article word count outside 1400-1600**
The writer and editor both have `word_count` tool access. Add to task description:
`"You MUST use word_count tool and revise until the count is between 1400 and 1600."`

**Rate limit errors from Anthropic**
Reduce `max_rpm` to 5, or add delays between tasks.

---

## Next Steps → Project 10: MLOps + Docker + CI/CD
```bash
conda activate aiarch
pip install mlflow prometheus-client kafka-python
mkdir -p ~/Documents/ai-journey/projects/10-mlops-docker-cicd
```
