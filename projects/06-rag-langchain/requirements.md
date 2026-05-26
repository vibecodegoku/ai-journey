# Project 06: RAG Q&A System with LangChain + ChromaDB

**XP:** 350 | **Cost:** Free (local models) or ~$3 (OpenAI) | **Duration:** Weeks 11–12

## Objective
Build a production-quality Retrieval-Augmented Generation (RAG) system that ingests PDFs and web pages, stores embeddings in ChromaDB, and answers questions with cited sources. Add LangSmith observability to trace every LLM call, evaluate pipeline quality with RAGAS, and compare retrieval strategies (naive vs HyDE vs re-ranking).

## Tools
- **LangChain** — document loading, splitting, retrieval chain
- **ChromaDB** — local vector store
- **Ollama** — local LLM (no API cost) OR OpenAI API
- **nomic-embed-text** — local embeddings via Ollama
- **LangSmith** — tracing and observability
- **RAGAS** — RAG evaluation (faithfulness, answer relevance, context precision)
- **Gradio** — Q&A web interface

## Project Structure
```
06-rag-langchain/
├── app/
│   ├── main.py                ← Gradio UI
│   ├── rag_chain.py           ← Core RAG pipeline
│   ├── ingest.py              ← Document ingestion
│   └── evaluator.py           ← RAGAS evaluation
├── data/
│   └── .gitkeep               ← Place PDFs here
├── chroma_db/                 ← Persisted vector store (git-ignored)
├── notebooks/
│   ├── 01_rag_basics.ipynb
│   └── 02_evaluation.ipynb
├── docs/adr/
│   └── 001-retrieval-strategy.md
└── requirements.txt
```

## Setup

```bash
pip install langchain langchain-community langchain-core langchain-chroma
pip install chromadb sentence-transformers
pip install langchain-ollama          # local LLM
pip install ragas datasets            # evaluation
pip install gradio                    # UI
pip install langsmith                 # observability

# Pull local models
ollama pull nomic-embed-text          # embeddings
ollama pull llama3.2                  # generation (or phi3 for speed)
```

## Phase 1: Document Ingestion Pipeline (Days 1–2)

`app/ingest.py`:
```python
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "knowledge_base"

def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )

def ingest_pdfs(pdf_dir: str = "data/") -> int:
    loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return _split_and_store(documents)

def ingest_url(url: str) -> int:
    loader = WebBaseLoader(url)
    documents = loader.load()
    return _split_and_store(documents)

def _split_and_store(documents: list) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    
    print(f"Ingested {len(documents)} documents → {len(chunks)} chunks")
    return len(chunks)

if __name__ == "__main__":
    # Ingest LangChain docs as example
    urls = [
        "https://python.langchain.com/docs/introduction/",
        "https://python.langchain.com/docs/concepts/rag/",
    ]
    for url in urls:
        ingest_url(url)
```

## Phase 2: Core RAG Chain (Days 3–5)

`app/rag_chain.py`:
```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from ingest import get_vectorstore, get_embeddings
import langsmith

CHROMA_PATH = "chroma_db"

RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Always cite the source documents.

Context:
{context}

Question: {question}

Answer with citations:""")

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )

def build_rag_chain(strategy: str = "naive", model: str = "llama3.2"):
    llm = ChatOllama(model=model, temperature=0)
    vectorstore = get_vectorstore()
    
    if strategy == "naive":
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    elif strategy == "mmr":
        # Maximum Marginal Relevance — reduces redundancy
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
        )
    
    elif strategy == "rerank":
        # Re-ranking with cross-encoder — highest quality
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=reranker, top_n=4)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
    
    elif strategy == "hyde":
        # HyDE — Hypothetical Document Embeddings
        from langchain.chains import HypotheticalDocumentEmbedder
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
            llm=llm, embeddings=embeddings, chain_type="stuff"
        )
        vectorstore_hyde = Chroma(
            collection_name="knowledge_base",
            embedding_function=hyde_embeddings,
            persist_directory=CHROMA_PATH,
        )
        retriever = vectorstore_hyde.as_retriever(search_kwargs={"k": 4})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def ask(question: str, strategy: str = "naive") -> dict:
    chain, retriever = build_rag_chain(strategy)
    
    # Get source docs separately for citation display
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)
    
    sources = list(set(
        doc.metadata.get("source", "Unknown") for doc in source_docs
    ))
    
    return {"answer": answer, "sources": sources, "num_docs_retrieved": len(source_docs)}
```

## Phase 3: LangSmith Observability (Day 6)

```bash
# Sign up at https://smith.langchain.com (free tier: 5K traces/month)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="lsv2_pt_YOUR_KEY"
export LANGCHAIN_PROJECT="rag-qa-demo"
```

All LangChain chain invocations are now automatically traced. View in the LangSmith dashboard:
- Token counts per run
- Latency breakdown (retrieval vs LLM)
- Full prompt and response
- Error traces

### Add custom metadata to traces
```python
from langsmith import traceable

@traceable(name="RAG Query", metadata={"strategy": "naive"})
def traced_ask(question: str, strategy: str = "naive") -> dict:
    return ask(question, strategy)
```

## Phase 4: RAGAS Evaluation (Days 7–8)

`app/evaluator.py`:
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from rag_chain import ask

# Build evaluation dataset (ground truth Q&A pairs)
eval_questions = [
    "What is RAG and why is it used?",
    "How does LangChain handle document splitting?",
    "What are the main components of a vector store?",
]

eval_ground_truth = [
    "RAG stands for Retrieval-Augmented Generation. It retrieves relevant context from a knowledge base and provides it to an LLM to generate grounded, factual responses.",
    "LangChain uses TextSplitter classes that break documents into chunks based on size and overlap parameters.",
    "A vector store stores document embeddings (numerical representations) and provides similarity search to find semantically related documents.",
]

def run_evaluation(strategy: str = "naive") -> dict:
    results_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    
    for question, gt in zip(eval_questions, eval_ground_truth):
        result = ask(question, strategy)
        results_data["question"].append(question)
        results_data["answer"].append(result["answer"])
        results_data["contexts"].append([result["answer"]])
        results_data["ground_truth"].append(gt)
    
    dataset = Dataset.from_dict(results_data)
    
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    
    return scores.to_pandas().mean().to_dict()

if __name__ == "__main__":
    import pandas as pd
    
    strategies = ["naive", "mmr"]
    comparison = {}
    for strategy in strategies:
        print(f"\nEvaluating strategy: {strategy}")
        scores = run_evaluation(strategy)
        comparison[strategy] = scores
    
    df = pd.DataFrame(comparison).T
    print("\nRAGAS Evaluation Comparison:")
    print(df.round(3))
```

## Phase 5: Gradio UI (Days 9–10)

`app/main.py`:
```python
import gradio as gr
from ingest import ingest_pdfs, ingest_url
from rag_chain import ask

def query_rag(question, strategy):
    if not question.strip():
        return "Please enter a question.", ""
    result = ask(question, strategy)
    sources_text = "\n".join(f"- {s}" for s in result["sources"])
    return result["answer"], sources_text

def ingest_from_url(url):
    if not url.strip():
        return "Please enter a URL."
    count = ingest_url(url)
    return f"Ingested {count} chunks from {url}"

with gr.Blocks(title="RAG Q&A System") as demo:
    gr.Markdown("# RAG Knowledge Base Q&A")
    
    with gr.Tab("Ask Questions"):
        with gr.Row():
            question_input = gr.Textbox(label="Your Question", lines=2, scale=4)
            strategy_select = gr.Dropdown(
                choices=["naive", "mmr", "rerank"],
                value="naive",
                label="Retrieval Strategy",
                scale=1,
            )
        ask_btn = gr.Button("Ask", variant="primary")
        answer_output = gr.Textbox(label="Answer", lines=6)
        sources_output = gr.Textbox(label="Sources", lines=3)
        ask_btn.click(query_rag, [question_input, strategy_select], [answer_output, sources_output])
    
    with gr.Tab("Add Knowledge"):
        url_input = gr.Textbox(label="URL to ingest")
        ingest_btn = gr.Button("Ingest URL")
        ingest_status = gr.Textbox(label="Status")
        ingest_btn.click(ingest_from_url, url_input, ingest_status)

demo.launch(share=False)
```

Run:
```bash
cd app
python main.py
```

## RAGAS Target Metrics

| Metric | Target | Notes |
|---|---|---|
| Faithfulness | > 0.85 | Response grounded in retrieved context |
| Answer Relevancy | > 0.80 | Response addresses the question |
| Context Precision | > 0.75 | Retrieved chunks are relevant |
| Context Recall | > 0.70 | Retrieved chunks cover the answer |

## Acceptance Criteria
- [ ] PDF and URL ingestion pipeline works
- [ ] Q&A returns answers with cited sources
- [ ] LangSmith traces visible in dashboard with token counts
- [ ] RAGAS evaluation notebook shows scores for 2+ retrieval strategies
- [ ] Gradio UI runs locally with both question answering and URL ingestion tabs
- [ ] Re-ranking strategy outperforms naive on context precision metric
- [ ] ADR-001 written: naive vs MMR vs re-ranking trade-offs

## ADR Template
Create `docs/adr/001-retrieval-strategy.md`:
```markdown
# ADR-001: Default Retrieval Strategy

## Status: Accepted

## Context
RAG system needs retrieval strategy balancing quality, latency, and resource usage.
Evaluation dataset shows MMR and re-ranking both outperform naive cosine similarity.

## Decision
Default to MMR retrieval; offer re-ranking as premium option with explicit flag.

## Consequences
+ MMR reduces redundant chunks (important for long documents)
+ No additional model downloads vs re-ranking (which needs BAAI/bge-reranker)
+ ~20% improvement in context precision vs naive
- MMR adds ~50ms latency vs naive
- Re-ranking adds cross-encoder model (~500MB) and ~200ms additional latency
```
