# Project 09: Multi-Agent Content Team with CrewAI

**XP:** 450 | **Cost:** ~$5–10 | **Duration:** Weeks 17–18

## Objective
Build a multi-agent content production system using CrewAI where specialized agents collaborate to research a topic, write a structured article, edit for quality, generate SEO metadata, and produce a final polished report. Implement parallel task execution, inter-agent delegation, and a custom tool shared across the crew.

## Tools
- **CrewAI** — multi-agent orchestration framework
- **LangChain** — LLM backbone
- **Claude 3 Haiku or GPT-4o-mini** — LLM (all agents share it or use different models)
- **Tavily** — web search tool
- **Pydantic** — structured output models

## Project Structure
```
09-crewai-multiagent/
├── crew/
│   ├── agents.py          ← Agent definitions with roles/goals/backstory
│   ├── tasks.py           ← Task definitions with context dependencies
│   ├── tools.py           ← Custom tools shared across agents
│   └── crew.py            ← Crew assembly and kickoff
├── app/
│   └── main.py            ← Streamlit UI
├── outputs/               ← Generated articles (git-ignored)
│   └── .gitkeep
├── notebooks/
│   └── 01_crew_demo.ipynb
├── docs/adr/
│   └── 001-agent-orchestration.md
└── requirements.txt
```

## Setup

```bash
pip install crewai crewai-tools langchain-anthropic tavily-python
pip install streamlit pydantic

export ANTHROPIC_API_KEY="your-key"
export TAVILY_API_KEY="your-key"
```

## Phase 1: Define Custom Tools (Day 1)

`crew/tools.py`:
```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
import re

class SearchInput(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(5, description="Number of results to return")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information. Returns titles, URLs, and summaries."
    args_schema: type[BaseModel] = SearchInput
    
    def _run(self, query: str, max_results: int = 5) -> str:
        client = TavilyClient()
        results = client.search(query=query, max_results=max_results)
        
        formatted = []
        for r in results.get("results", []):
            formatted.append(f"**{r['title']}**\nURL: {r['url']}\n{r['content'][:300]}\n")
        
        return "\n---\n".join(formatted) if formatted else "No results found."

class ReadURLInput(BaseModel):
    url: str = Field(..., description="URL to read and extract content from")

class ReadURLTool(BaseTool):
    name: str = "read_url"
    description: str = "Read and extract the main text content from a given URL."
    args_schema: type[BaseModel] = ReadURLInput
    
    def _run(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return text[:3000]
        except Exception as e:
            return f"Failed to read URL: {e}"

class WordCountInput(BaseModel):
    text: str = Field(..., description="Text to analyze")

class WordCountTool(BaseTool):
    name: str = "word_count"
    description: str = "Count words, sentences, and estimate reading time for a text."
    args_schema: type[BaseModel] = WordCountInput
    
    def _run(self, text: str) -> str:
        words = len(text.split())
        sentences = len(re.split(r"[.!?]+", text))
        reading_time = words // 200  # avg 200 wpm
        return f"Words: {words} | Sentences: {sentences} | Reading time: ~{reading_time} min"

search_tool = WebSearchTool()
read_tool = ReadURLTool()
word_count_tool = WordCountTool()
```

## Phase 2: Define Agents (Days 2–3)

`crew/agents.py`:
```python
from crewai import Agent
from langchain_anthropic import ChatAnthropic
from tools import search_tool, read_tool, word_count_tool

llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)

researcher = Agent(
    role="Senior Research Analyst",
    goal="Discover comprehensive, accurate, and up-to-date information on the given topic from credible sources.",
    backstory="""You are a meticulous researcher with 10 years of experience in technical writing.
    You excel at finding authoritative sources, cross-referencing facts, and identifying the most
    important aspects of complex topics. You always cite your sources.""",
    tools=[search_tool, read_tool],
    llm=llm,
    verbose=True,
    max_iter=5,
    memory=True,
)

writer = Agent(
    role="Technical Content Writer",
    goal="Transform research findings into a well-structured, engaging 1500-word article.",
    backstory="""You are a skilled technical writer who specializes in making complex topics
    accessible. You write clear, engaging prose with strong structure: compelling intro, 
    logical sections, concrete examples, and actionable takeaways. You maintain a professional
    but approachable tone.""",
    tools=[word_count_tool],
    llm=llm,
    verbose=True,
    max_iter=3,
    memory=True,
)

editor = Agent(
    role="Senior Content Editor",
    goal="Ensure the article is factually accurate, well-structured, free of errors, and meets quality standards.",
    backstory="""You are a demanding editor with an eye for clarity, logical flow, and factual precision.
    You check for: grammar/spelling errors, logical inconsistencies, unsupported claims,
    unclear explanations, and weak transitions. You provide specific, actionable feedback.""",
    tools=[word_count_tool],
    llm=llm,
    verbose=True,
    max_iter=3,
)

seo_specialist = Agent(
    role="SEO and Content Strategist",
    goal="Optimize the article for search engines while maintaining readability and value.",
    backstory="""You are an SEO expert who understands both technical optimization and reader psychology.
    You identify target keywords, write compelling meta descriptions, suggest internal linking
    opportunities, and ensure content satisfies search intent without keyword stuffing.""",
    tools=[search_tool],
    llm=llm,
    verbose=True,
    max_iter=3,
)
```

## Phase 3: Define Tasks (Days 4–5)

`crew/tasks.py`:
```python
from crewai import Task
from agents import researcher, writer, editor, seo_specialist
from pydantic import BaseModel

class ResearchFindings(BaseModel):
    summary: str
    key_points: list[str]
    sources: list[str]
    statistics: list[str]

class ArticleDraft(BaseModel):
    title: str
    sections: dict[str, str]
    word_count: int

class SEOMetadata(BaseModel):
    meta_title: str
    meta_description: str
    target_keywords: list[str]
    h1_tag: str
    slug: str

def create_tasks(topic: str):
    research_task = Task(
        description=f"""Research '{topic}' thoroughly. 
        Conduct at least 3 web searches with different angles on the topic.
        Read at least 2 source URLs in detail.
        Extract: key concepts, recent developments (2024-2025), statistics,
        expert opinions, and practical applications.
        
        Organize your findings into:
        - A 200-word executive summary
        - 8-10 key bullet points
        - List of sources with URLs
        - 3-5 relevant statistics""",
        expected_output="Structured research findings with summary, key points, sources, and statistics",
        agent=researcher,
        output_pydantic=ResearchFindings,
    )
    
    writing_task = Task(
        description=f"""Write a comprehensive 1500-word article about '{topic}'.
        
        Use the research findings provided in your context.
        
        Structure:
        - Compelling headline (not clickbait)
        - Introduction: hook + why this matters (150 words)
        - Section 1: Core concepts explained (300 words)
        - Section 2: Current state and trends (300 words)
        - Section 3: Practical applications (300 words)
        - Section 4: Challenges and considerations (200 words)
        - Conclusion: Key takeaways and next steps (150 words)
        
        Style: professional but accessible, active voice, concrete examples.""",
        expected_output="Complete article with title and named sections totaling ~1500 words",
        agent=writer,
        context=[research_task],
    )
    
    editing_task = Task(
        description="""Edit the article draft for quality. Check and fix:
        1. Grammar and spelling errors
        2. Logical flow between sections
        3. Any unsupported claims (flag with [NEEDS SOURCE])
        4. Clarity of technical explanations
        5. Consistency in terminology
        6. Word count (should be 1400-1600 words)
        
        Return the polished, publication-ready article.""",
        expected_output="Edited, publication-ready article with all issues resolved",
        agent=editor,
        context=[writing_task],
    )
    
    seo_task = Task(
        description=f"""Create SEO metadata for the article about '{topic}'.
        Search for the current SEO landscape for this topic.
        
        Provide:
        - meta_title: 50-60 chars, includes primary keyword
        - meta_description: 150-160 chars, compelling, includes keyword
        - target_keywords: 5 keywords (primary + 4 secondary/LSI)
        - h1_tag: Optimized H1 (slightly different from meta title)
        - slug: URL-friendly slug (lowercase, hyphens, max 5 words)""",
        expected_output="SEO metadata as structured JSON",
        agent=seo_specialist,
        context=[editing_task],
        output_pydantic=SEOMetadata,
    )
    
    return [research_task, writing_task, editing_task, seo_task]
```

## Phase 4: Assemble Crew and Run (Days 6–7)

`crew/crew.py`:
```python
from crewai import Crew, Process
from agents import researcher, writer, editor, seo_specialist
from tasks import create_tasks
import time
import json

def run_content_crew(topic: str, output_dir: str = "outputs") -> dict:
    tasks = create_tasks(topic)
    
    crew = Crew(
        agents=[researcher, writer, editor, seo_specialist],
        tasks=tasks,
        process=Process.sequential,          # each agent waits for previous
        verbose=True,
        memory=True,                          # shared memory across agents
        embedder={
            "provider": "ollama",
            "config": {"model": "nomic-embed-text"},
        },
        max_rpm=10,                           # rate limit API calls
    )
    
    start = time.time()
    result = crew.kickoff(inputs={"topic": topic})
    elapsed = time.time() - start
    
    # Save outputs
    import os
    os.makedirs(output_dir, exist_ok=True)
    slug = topic.lower().replace(" ", "-")[:30]
    
    output = {
        "topic": topic,
        "elapsed_seconds": round(elapsed, 1),
        "article": str(result.tasks_output[2]),        # edited article
        "seo": result.tasks_output[3].pydantic.dict() if result.tasks_output[3].pydantic else {},
        "token_usage": result.token_usage.__dict__ if result.token_usage else {},
    }
    
    with open(f"{output_dir}/{slug}.json", "w") as f:
        json.dump(output, f, indent=2)
    
    with open(f"{output_dir}/{slug}.md", "w") as f:
        f.write(f"# {topic}\n\n")
        f.write(str(result.tasks_output[2]))
    
    return output

if __name__ == "__main__":
    result = run_content_crew("AI agents in enterprise software development")
    print(f"\nCompleted in {result['elapsed_seconds']}s")
    print(f"Token usage: {result['token_usage']}")
```

## Phase 5: Streamlit UI (Days 8–10)

`app/main.py`:
```python
import streamlit as st
import sys
sys.path.append("../crew")
from crew import run_content_crew
import json

st.set_page_config(page_title="Multi-Agent Content Studio", layout="wide")
st.title("Multi-Agent Content Production Studio")

with st.sidebar:
    st.header("Configuration")
    model = st.selectbox("LLM", ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"])
    st.divider()
    st.caption("Pipeline: Research → Write → Edit → SEO")

topic = st.text_input("Research Topic", placeholder="e.g., 'Multi-agent AI systems in 2025'")
col1, col2 = st.columns([1, 4])
run_btn = col1.button("Run Crew", type="primary", disabled=not topic)

if run_btn and topic:
    with st.status("Running multi-agent pipeline...", expanded=True) as status:
        st.write("Research agent searching...")
        result = run_content_crew(topic)
        status.update(label="Complete!", state="complete")
    
    tabs = st.tabs(["Article", "SEO Metadata", "Stats"])
    
    with tabs[0]:
        st.markdown(result["article"])
    
    with tabs[1]:
        seo = result.get("seo", {})
        if seo:
            st.json(seo)
        else:
            st.info("SEO metadata not available")
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        col1.metric("Time (seconds)", result["elapsed_seconds"])
        usage = result.get("token_usage", {})
        col2.metric("Total Tokens", usage.get("total_tokens", "N/A"))

streamlit run app/main.py
```

## Acceptance Criteria
- [ ] All 4 agents (researcher, writer, editor, SEO) run in sequence without errors
- [ ] Final article is 1400-1600 words with clear sections
- [ ] Research task produces at least 5 source citations
- [ ] SEO metadata output is valid JSON with all 5 fields
- [ ] Outputs saved as .md and .json files
- [ ] Streamlit UI shows all 3 tabs (article, SEO, stats)
- [ ] Total run time under 5 minutes for a typical topic
- [ ] ADR-001 written: CrewAI vs AutoGen vs LangGraph for multi-agent

## ADR Template
Create `docs/adr/001-agent-orchestration.md`:
```markdown
# ADR-001: Multi-Agent Orchestration Framework

## Status: Accepted

## Context
Need framework to orchestrate 4 specialized agents with sequential dependencies,
shared memory, and structured output passing between agents.

## Decision
CrewAI with sequential Process and Pydantic output models.

## Consequences
+ High-level abstractions (Agent/Task/Crew) — faster to build than raw LangGraph
+ Built-in memory and agent delegation
+ Pydantic output validation ensures structured handoffs
- Less control than LangGraph for complex conditional logic
- CrewAI memory requires embeddings (adds latency and dependency)
- Framework evolving rapidly — breaking changes between minor versions
```
