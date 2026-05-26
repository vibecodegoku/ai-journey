window.DATA = {
  projects: [
    {
      id: "p00", num: "00",
      title: "MacBook Pro AI Dev Setup",
      subtitle: "Pre-flight environment setup",
      description: "Set up your complete AI development environment on macOS. Install all cloud CLIs, run local LLMs with Ollama, and verify all 3 cloud accounts work.",
      providers: ["general"], cost: "free", categories: ["setup"],
      month: 0, week: "Week 0", xp: 100, estimatedCost: "$0",
      tools: ["Homebrew", "Python 3.11", "pyenv", "conda", "Ollama", "Docker Desktop", "AWS CLI", "gcloud SDK", "Azure CLI", "VS Code"],
      outcomes: ["Local LLMs running via Ollama (llama3.2, phi3)", "All 3 cloud CLIs authenticated", "Python 3.11 environment with key ML libraries", "Docker Desktop running", "VS Code with Python, Docker, and cloud extensions"],
      reqPath: "projects/00-macos-setup/requirements.md"
    },
    {
      id: "p01", num: "01",
      title: "ML Foundations & API Deployment",
      subtitle: "Train, evaluate, and serve an ML model",
      description: "Build a complete ML pipeline from raw data to deployed REST API. Train a house price predictor, evaluate it properly, and serve predictions via FastAPI.",
      providers: ["general"], cost: "free", categories: ["ml-basics"],
      month: 1, week: "Weeks 1–2", xp: 200, estimatedCost: "$0",
      tools: ["Python", "scikit-learn", "pandas", "FastAPI", "Jupyter", "matplotlib", "pytest"],
      outcomes: ["Trained regression model with proper CV and metrics", "FastAPI endpoint serving predictions locally", "Model serialized with joblib", "Unit tests for data pipeline and API", "ADR documenting model selection decisions"],
      reqPath: "projects/01-ml-foundations/requirements.md"
    },
    {
      id: "p02", num: "02",
      title: "AWS SageMaker Training Pipeline",
      subtitle: "Train and deploy ML models on AWS",
      description: "Move your ML workflow to AWS. Store data in S3, train a model on SageMaker, deploy an endpoint, and set up basic monitoring with Model Monitor.",
      providers: ["aws"], cost: "paid", categories: ["ml-basics", "mlops"],
      month: 1, week: "Weeks 3–4", xp: 250, estimatedCost: "~$10–20",
      tools: ["AWS SageMaker", "S3", "boto3", "AWS CLI", "SageMaker SDK", "Python"],
      outcomes: ["Data pipeline: local → S3 → SageMaker", "Model trained on SageMaker with built-in XGBoost", "Real-time inference endpoint deployed", "SageMaker Model Monitor baseline established", "Cost tracked and under $20"],
      reqPath: "projects/02-cloud-ml-aws/requirements.md"
    },
    {
      id: "p03", num: "03",
      title: "Azure AI Document Intelligence",
      subtitle: "Extract structured data from documents with Azure AI",
      description: "Build a document processing pipeline using Azure AI Document Intelligence and Azure OpenAI. Extract data from PDFs/invoices, then use GPT to analyze and summarize.",
      providers: ["azure"], cost: "free", categories: ["ml-basics"],
      month: 2, week: "Weeks 5–6", xp: 250, estimatedCost: "~$0–5",
      tools: ["Azure AI Document Intelligence", "Azure OpenAI", "Azure CLI", "Python SDK", "FastAPI"],
      outcomes: ["Invoice/receipt data extracted into structured JSON", "GPT summarization layer on extracted data", "REST API wrapping the full pipeline", "Azure AI services authenticated via managed identity", "Cost tracked in Azure Cost Management"],
      reqPath: "projects/03-azure-ai-services/requirements.md"
    },
    {
      id: "p04", num: "04",
      title: "AWS Bedrock Multi-Modal Chat",
      subtitle: "Build a production-grade chat app with Claude on Bedrock",
      description: "Build a multi-modal chat application using AWS Bedrock. Support text and image inputs, implement conversation memory, and track costs per session.",
      providers: ["aws"], cost: "paid", categories: ["llms"],
      month: 2, week: "Weeks 7–8", xp: 300, estimatedCost: "~$5–15",
      tools: ["AWS Bedrock", "Claude via Bedrock", "boto3", "Streamlit", "LangChain", "LangSmith"],
      outcomes: ["Multi-modal chat app (text + image) using Claude Haiku", "Conversation history with token counting", "Cost tracker: $ per message and per session", "Streamlit UI deployable locally", "LangSmith tracing enabled for all calls"],
      reqPath: "projects/04-aws-bedrock/requirements.md"
    },
    {
      id: "p05", num: "05",
      title: "LLM Fine-Tuning with QLoRA",
      subtitle: "Fine-tune a local LLM on your MacBook with Hugging Face",
      description: "Fine-tune a small LLM (Phi-3 Mini or Llama 3.2 3B) using QLoRA on Apple Silicon. Publish your adapter to Hugging Face Hub and test with Ollama.",
      providers: ["huggingface"], cost: "free", categories: ["llms"],
      month: 3, week: "Weeks 9–10", xp: 350, estimatedCost: "$0",
      tools: ["Hugging Face Transformers", "PEFT", "Unsloth", "Ollama", "Python", "Hugging Face Hub", "W&B"],
      outcomes: ["Fine-tuned LLM adapter pushed to Hugging Face Hub", "Custom model runnable via Ollama locally", "Training metrics logged to Weights & Biases", "ROUGE/BLEU evaluation scores", "Training cost comparison: MacBook vs cloud GPU"],
      reqPath: "projects/05-llm-finetuning/requirements.md"
    },
    {
      id: "p06", num: "06",
      title: "RAG System with LangChain & Vector DB",
      subtitle: "Build a production-grade Retrieval-Augmented Generation pipeline",
      description: "Build a document Q&A system using RAG: ingest PDFs, chunk and embed them, store in a vector database, and retrieve relevant context for LLM answers. Evaluate with RAGAS.",
      providers: ["general"], cost: "free", categories: ["llms", "agents"],
      month: 3, week: "Weeks 11–12", xp: 350, estimatedCost: "$0",
      tools: ["LangChain", "ChromaDB", "Ollama", "RAGAS", "LangSmith", "FastAPI", "nomic-embed-text"],
      outcomes: ["PDF ingestion pipeline with smart chunking", "ChromaDB vector store with hybrid search", "RAG chain with source attribution", "RAGAS evaluation: faithfulness > 0.75", "LangSmith traces for all retrieval + generation steps"],
      reqPath: "projects/06-rag-langchain/requirements.md"
    },
    {
      id: "p07", num: "07",
      title: "GCP Vertex AI End-to-End Pipeline",
      subtitle: "Automated ML pipeline with Vertex AI Pipelines",
      description: "Build a production ML pipeline on Google Cloud Vertex AI. Automate data processing, training, evaluation, and deployment using Vertex AI Pipelines and Kubeflow.",
      providers: ["gcp"], cost: "free", categories: ["mlops"],
      month: 4, week: "Weeks 13–14", xp: 300, estimatedCost: "~$10–20",
      tools: ["GCP Vertex AI", "Vertex AI Pipelines", "Kubeflow Pipelines", "Cloud Storage", "BigQuery", "gcloud CLI"],
      outcomes: ["Automated training pipeline triggered by new data", "Model registry with versioning in Vertex AI", "Online prediction endpoint with autoscaling config", "Vertex AI Model Monitoring with alert policy", "C4 architecture diagram for the pipeline"],
      reqPath: "projects/07-vertex-ai-pipeline/requirements.md"
    },
    {
      id: "p08", num: "08",
      title: "Research Agent with LangGraph",
      subtitle: "Build a stateful AI agent with tool use and human-in-the-loop",
      description: "Build a research assistant agent using LangGraph. The agent autonomously searches the web, reads pages, synthesizes information, and asks for human approval before publishing findings.",
      providers: ["general"], cost: "free", categories: ["agents"],
      month: 4, week: "Weeks 15–16", xp: 400, estimatedCost: "$0",
      tools: ["LangGraph", "LangChain", "Ollama", "DuckDuckGo Search", "LangSmith", "Python"],
      outcomes: ["LangGraph agent with 5+ tools (search, read, calc, write, ask-human)", "State machine with conditional routing", "Human-in-the-loop approval before final output", "LangSmith traces showing full agent reasoning chain", "Agent handles errors and retries automatically"],
      reqPath: "projects/08-langgraph-agent/requirements.md"
    },
    {
      id: "p09", num: "09",
      title: "Multi-Agent System with CrewAI",
      subtitle: "Build a coordinated team of specialized AI agents",
      description: "Build a multi-agent content creation team using CrewAI. A researcher, writer, editor, and fact-checker agent work together to produce high-quality technical blog posts.",
      providers: ["general"], cost: "free", categories: ["agents"],
      month: 5, week: "Weeks 17–18", xp: 450, estimatedCost: "$0–5",
      tools: ["CrewAI", "Ollama", "LangSmith", "Gradio", "Python"],
      outcomes: ["4-agent team: researcher, writer, editor, fact-checker", "Agents pass structured outputs between each other", "Gradio UI for topic input and output display", "Each agent's trace visible in LangSmith", "ADR: CrewAI vs LangGraph for multi-agent orchestration"],
      reqPath: "projects/09-crewai-multiagent/requirements.md"
    },
    {
      id: "p10", num: "10",
      title: "MLOps: Docker + CI/CD + Monitoring",
      subtitle: "Containerize and automate ML model deployment",
      description: "Build a complete MLOps pipeline: containerize a model with Docker, automate deployments with GitHub Actions, track experiments with MLflow, and monitor with Prometheus/Grafana.",
      providers: ["aws", "general"], cost: "free", categories: ["mlops"],
      month: 5, week: "Weeks 19–20", xp: 400, estimatedCost: "$0–10",
      tools: ["Docker", "GitHub Actions", "MLflow", "Prometheus", "Grafana", "AWS Lambda", "FastAPI", "Kafka"],
      outcomes: ["Dockerized ML model with multi-stage build", "GitHub Actions CI/CD: test → build → push → deploy", "MLflow tracking server for experiment comparison", "Prometheus + Grafana dashboard for model serving", "Kafka streaming feature engineering mini-demo"],
      reqPath: "projects/10-mlops-docker-cicd/requirements.md"
    },
    {
      id: "p11", num: "11",
      title: "Multi-Cloud GenAI Architecture",
      subtitle: "Design and document an enterprise multi-cloud AI solution",
      description: "Design a production-grade multi-cloud GenAI solution using AWS + GCP + Azure. Create C4 architecture diagrams, ADRs, cost analysis, and implement the core cross-cloud orchestration layer.",
      providers: ["aws", "gcp", "azure"], cost: "paid", categories: ["architecture"],
      month: 6, week: "Weeks 21–22", xp: 500, estimatedCost: "~$20–40",
      tools: ["AWS Bedrock", "GCP Vertex AI", "Azure OpenAI", "LangChain", "Terraform", "draw.io", "Python"],
      outcomes: ["C4 architecture diagrams (all 4 levels)", "5 ADRs covering key architectural decisions", "Cloud provider failover: if AWS Bedrock fails route to GCP Vertex", "Cost comparison: same workload on 3 clouds", "Terraform IaC for infrastructure provisioning"],
      reqPath: "projects/11-multicloud-architecture/requirements.md"
    },
    {
      id: "p12", num: "12",
      title: "Capstone: Production AI Platform",
      subtitle: "Build and deploy a complete AI solution showcasing all skills",
      description: "Build and deploy a production-grade AI platform combining RAG, multi-agent orchestration, fine-tuned models, multi-cloud deployment, monitoring, and AI governance. Your portfolio centerpiece.",
      providers: ["aws", "gcp", "azure", "huggingface"], cost: "paid", categories: ["architecture", "agents", "mlops", "llms"],
      month: 6, week: "Weeks 23–24", xp: 1000, estimatedCost: "~$30–60",
      tools: ["LangGraph", "CrewAI", "AWS Bedrock", "GCP Vertex AI", "Docker", "GitHub Actions", "LangSmith", "Grafana"],
      outcomes: ["Full-stack AI platform deployed and publicly accessible", "Complete architecture documentation with C4 diagrams", "AI governance report (EU AI Act risk assessment)", "LangSmith dashboard with production traces", "GitHub portfolio with clean README and demo video"],
      reqPath: "projects/12-capstone/requirements.md"
    }
  ],

  certifications: [
    {
      id: "cert-ms-agentic",
      title: "Microsoft: Agentic AI Business Solutions Architect",
      provider: "azure", code: "NEW 2026", cost: "paid", price: "$165", difficulty: 4,
      duration: "TBD",
      link: "https://learn.microsoft.com/en-us/credentials/certifications/agentic-ai-business-solutions-architect/",
      description: "Brand new 2026 cert covering agentic AI design patterns, multi-agent orchestration, and enterprise AI solution architecture on Azure.",
      tags: ["azure", "agents", "architecture"], examMonth: "Month 4–5", hotBadge: "🔥 New 2026"
    },
    {
      id: "cert-aws-aif",
      title: "AWS Certified AI Practitioner",
      provider: "aws", code: "AIF-C01", cost: "paid", price: "$100", difficulty: 2,
      duration: "90 min",
      link: "https://aws.amazon.com/certification/certified-ai-practitioner/",
      description: "Foundational AWS AI certification. Covers AI/ML concepts, generative AI, responsible AI, and AWS AI services. Best first cert to target.",
      tags: ["aws", "ml-basics", "llms"], examMonth: "Month 2", hotBadge: "✅ Start Here"
    },
    {
      id: "cert-aws-mla",
      title: "AWS ML Engineer Associate",
      provider: "aws", code: "MLA-C01", cost: "paid", price: "$300", difficulty: 3,
      duration: "130 min",
      link: "https://aws.amazon.com/certification/certified-machine-learning-engineer-associate/",
      description: "Covers SageMaker, data pipelines, model deployment, monitoring, and MLOps on AWS. Recommended after AIF-C01.",
      tags: ["aws", "ml-basics", "mlops"], examMonth: "Month 3"
    },
    {
      id: "cert-aws-aip",
      title: "AWS Generative AI Developer Professional",
      provider: "aws", code: "AIP-C01", cost: "paid", price: "$300", difficulty: 4,
      duration: "TBD",
      link: "https://aws.amazon.com/certification/certified-generative-ai-developer-professional/",
      description: "Advanced AWS cert for GenAI developers. Covers Bedrock, RAG architectures, fine-tuning, responsible AI, and production GenAI patterns.",
      tags: ["aws", "llms", "agents"], examMonth: "Month 6"
    },
    {
      id: "cert-gcp-pmle",
      title: "Google Professional ML Engineer",
      provider: "gcp", code: "PMLE", cost: "paid", price: "$200", difficulty: 4,
      duration: "120 min",
      link: "https://cloud.google.com/learn/certification/machine-learning-engineer",
      description: "Covers Vertex AI, model training, deployment, monitoring, MLOps, and the new Gemini enterprise features. Updated June 2026.",
      tags: ["gcp", "ml-basics", "mlops"], examMonth: "Month 4–5", hotBadge: "Updated 2026"
    },
    {
      id: "cert-gcp-pca",
      title: "Google Professional Cloud Architect",
      provider: "gcp", code: "PCA", cost: "paid", price: "$200", difficulty: 4,
      duration: "120 min",
      link: "https://cloud.google.com/learn/certification/cloud-architect",
      description: "Broad GCP architecture cert. Covers AI/ML workloads, data architecture, security, and cost optimization on GCP.",
      tags: ["gcp", "architecture"], examMonth: "Month 5–6"
    },
    {
      id: "cert-azure-ai300",
      title: "Azure AI Apps & Agents Developer",
      provider: "azure", code: "AI-300", cost: "paid", price: "$165", difficulty: 3,
      duration: "TBD",
      link: "https://learn.microsoft.com/en-us/credentials/certifications/exams/ai-300/",
      description: "New May 2026 cert replacing AI-102 (retiring June 2026). Focus on generative AI, Azure OpenAI, and agentic workflows.",
      tags: ["azure", "llms", "agents"], examMonth: "Month 5", hotBadge: "Replaces AI-102"
    },
    {
      id: "cert-hf-nlp",
      title: "Hugging Face NLP Course Certificate",
      provider: "huggingface", code: "HF-NLP", cost: "free", price: "Free", difficulty: 2,
      duration: "~40 hours",
      link: "https://huggingface.co/learn/nlp-course/chapter1/1",
      description: "Comprehensive free course covering Transformers, fine-tuning, datasets, tokenizers, and deploying models. Certificate upon completion.",
      tags: ["huggingface", "llms", "ml-basics"], examMonth: "Month 1–2"
    },
    {
      id: "cert-gcp-skills",
      title: "Google Cloud Skills Boost",
      provider: "gcp", code: "Free Labs", cost: "free", price: "Free (35 credits/month)", difficulty: 2,
      duration: "Self-paced",
      link: "https://www.cloudskillsboost.google/",
      description: "600+ free courses and hands-on labs for GCP. Monthly free credits for labs. Completion badges for LinkedIn.",
      tags: ["gcp", "ml-basics"], examMonth: "Month 1–4"
    },
    {
      id: "cert-aws-skillbuilder",
      title: "AWS Skill Builder Free Tier",
      provider: "aws", code: "Free", cost: "free", price: "Free", difficulty: 1,
      duration: "Self-paced",
      link: "https://skillbuilder.aws/",
      description: "1,000+ free learning resources. Covers AIF-C01 and MLA-C01 exam prep. Digital badges for LinkedIn.",
      tags: ["aws", "ml-basics"], examMonth: "Month 1–3"
    },
    {
      id: "cert-azure-learn",
      title: "Microsoft Learn AI Learning Hub",
      provider: "azure", code: "Free", cost: "free", price: "Free", difficulty: 1,
      duration: "Self-paced",
      link: "https://learn.microsoft.com/en-us/ai/",
      description: "Free structured learning paths for all Azure AI certifications. New Agentic AI path for AI-300 prep.",
      tags: ["azure", "llms", "agents"], examMonth: "Month 1–5"
    }
  ],

  courses: [
    { id: "course-fastai", title: "Practical Deep Learning for Coders", provider: "general", platform: "fast.ai", cost: "free", duration: "~20 hours", link: "https://course.fast.ai/", description: "Top-down practical ML by Jeremy Howard. Covers computer vision, NLP, tabular data using PyTorch. Highly respected in industry.", tags: ["ml-basics", "general"], xp: 100 },
    { id: "course-deeplearningai", title: "DeepLearning.AI Short Courses", provider: "general", platform: "deeplearning.ai", cost: "free", duration: "1–4 hours each", link: "https://www.deeplearning.ai/courses/", description: "10+ free short courses: LangChain, LLMOps, RAG, Agents, Prompt Engineering. Taught by researchers at top AI labs.", tags: ["llms", "agents", "general"], xp: 100 },
    { id: "course-google-ml", title: "Google ML Crash Course", provider: "gcp", platform: "Google", cost: "free", duration: "~15 hours", link: "https://developers.google.com/machine-learning/crash-course", description: "Google's foundational ML course with exercises in TensorFlow. Covers regression, classification, neural networks, LLMs.", tags: ["ml-basics", "gcp"], xp: 75 },
    { id: "course-hf-agents", title: "Hugging Face Agents Course", provider: "huggingface", platform: "Hugging Face", cost: "free", duration: "~15 hours", link: "https://huggingface.co/learn/agents-course/unit0/introduction", description: "Hands-on course building AI agents with smolagents, LangGraph, and LlamaIndex. Includes multi-agent systems.", tags: ["agents", "llms", "huggingface"], xp: 100 },
    { id: "course-ragas", title: "RAG Evaluation with RAGAS", provider: "general", platform: "RAGAS Docs", cost: "free", duration: "~4 hours", link: "https://docs.ragas.io/en/latest/getstarted/", description: "Learn to evaluate RAG pipelines with faithfulness, relevance, and precision metrics. Essential for production RAG.", tags: ["llms", "general"], xp: 50 },
    { id: "course-langsmith", title: "LangSmith Observability", provider: "general", platform: "LangChain", cost: "free", duration: "~3 hours", link: "https://docs.smith.langchain.com/", description: "Learn to trace, debug, and monitor LLM applications. Free tier includes 5,000 traces/month.", tags: ["mlops", "llms", "general"], xp: 50 },
    { id: "course-eu-ai-act", title: "EU AI Act Compliance for Practitioners", provider: "general", platform: "artificialintelligenceact.eu", cost: "free", duration: "~5 hours", link: "https://artificialintelligenceact.eu/", description: "Understand EU AI Act risk categories, obligations, and compliance requirements. Full enforcement August 2026.", tags: ["architecture", "general"], xp: 75 },
    { id: "course-finops-ai", title: "FinOps for AI Practitioners", provider: "general", platform: "FinOps Foundation", cost: "free", duration: "~4 hours", link: "https://www.finops.org/wg/finops-for-ai-overview/", description: "Cost optimization frameworks for AI workloads. Learn token economics, GPU cost models, and cloud spend management.", tags: ["mlops", "architecture", "general"], xp: 75 }
  ],

  channels: [
    { id: "yt-krish", title: "Krish Naik", platform: "YouTube", focus: "GenAI, LLMs, MLOps, hands-on tutorials", subscribers: "1M+", link: "https://www.youtube.com/@krishnaik06", tags: ["llms", "ml-basics", "agents"] },
    { id: "yt-deeplearningai", title: "DeepLearning.AI", platform: "YouTube", focus: "ML fundamentals, LLMs, AI research by Andrew Ng", subscribers: "500K+", link: "https://www.youtube.com/@Deeplearningai", tags: ["ml-basics", "llms", "general"] },
    { id: "yt-gcp", title: "Google Cloud Tech", platform: "YouTube", focus: "GCP services, Vertex AI, Gemini tutorials", subscribers: "1.1M+", link: "https://www.youtube.com/@googlecloudtech", tags: ["gcp", "mlops", "architecture"] },
    { id: "yt-2min", title: "Two Minute Papers", platform: "YouTube", focus: "Latest AI research explained simply and quickly", subscribers: "1.5M+", link: "https://www.youtube.com/@TwoMinutePapers", tags: ["ml-basics", "llms", "general"] },
    { id: "yt-mattwolfe", title: "Matt Wolfe", platform: "YouTube", focus: "Weekly AI news, new tools, Future Tools roundups", subscribers: "900K+", link: "https://www.youtube.com/@mreflow", tags: ["general", "llms"] },
    { id: "yt-aws", title: "Amazon Web Services", platform: "YouTube", focus: "AWS tutorials, SageMaker, Bedrock deep dives", subscribers: "1.2M+", link: "https://www.youtube.com/@amazonwebservices", tags: ["aws", "mlops"] },
    { id: "yt-sam", title: "Sam Witteveen (Red Dragon AI)", platform: "YouTube", focus: "LangChain, LangGraph, AI agents deep dives", subscribers: "200K+", link: "https://www.youtube.com/@samwitteveenai", tags: ["agents", "llms", "general"] },
    { id: "yt-nicholas", title: "Nicholas Renotte", platform: "YouTube", focus: "MLOps, Docker, Kubernetes, production ML", subscribers: "300K+", link: "https://www.youtube.com/@NicholasRenotte", tags: ["mlops", "general"] }
  ],

  systemDesign: [
    { id: "sd-rag", title: "Design a Production RAG System", difficulty: "Medium", context: "1 million users, 10TB of documents, P99 latency < 2 seconds", keyComponents: ["API Gateway", "Document Processor", "Embedding Service", "Vector DB (pgvector)", "LLM (Bedrock)", "Cache (Redis)", "LangSmith Monitoring"], tradeoffs: ["Semantic vs BM25 retrieval", "Managed vs self-hosted vector DB", "Chunking strategy impact on recall", "Cache invalidation on document updates"], tags: ["llms", "architecture"] },
    { id: "sd-fraud", title: "Design a Real-Time Fraud Detection System", difficulty: "Hard", context: "10,000 transactions/second, < 100ms decision latency, explainability required for regulators", keyComponents: ["Kafka Streams", "Feature Store (Feast)", "Online ML Serving", "Explainability (SHAP)", "Alert Service", "MLflow Tracking"], tradeoffs: ["Precision vs recall (false positives cost money)", "Online vs batch feature computation", "Model explainability vs accuracy trade-off"], tags: ["mlops", "architecture"] },
    { id: "sd-multiagent", title: "Design a Multi-Agent Orchestration Platform", difficulty: "Hard", context: "Enterprise with 50+ specialized AI agents, needs central coordination and observability", keyComponents: ["Agent Registry", "Task Queue (Celery/RQ)", "LangGraph Orchestrator", "Shared Memory (Redis)", "LangSmith", "Human Approval Layer"], tradeoffs: ["CrewAI vs LangGraph for orchestration", "Shared state vs message passing", "Sync vs async agent communication"], tags: ["agents", "architecture"] },
    { id: "sd-mlplatform", title: "Design an Enterprise ML Platform", difficulty: "Hard", context: "500 data scientists, 1000 models in production, multi-cloud deployment", keyComponents: ["Feature Store", "Experiment Tracking (MLflow)", "Model Registry", "Pipeline Orchestrator (Airflow)", "Serving Layer (KServe)", "Monitoring (Arize)"], tradeoffs: ["Build vs buy (Databricks vs open-source)", "Centralized vs federated feature store", "Real-time vs batch serving costs"], tags: ["mlops", "architecture"] },
    { id: "sd-multimodal", title: "Design a Multimodal AI Pipeline", difficulty: "Medium", context: "Process documents with text + images + tables, extract structured data at scale", keyComponents: ["Document Ingestion", "Image Analysis (GPT-4V/LLaVA)", "OCR Layer", "Table Extractor", "RAG Storage", "Output Formatter"], tradeoffs: ["Cost: cloud vision APIs vs self-hosted", "Latency: sync vs async processing", "Quality: specialized vs general models"], tags: ["llms", "architecture"] }
  ],

  badges: [
    { id: "first-steps", name: "First Steps", emoji: "👣", tier: "bronze", description: "Complete your first resource" },
    { id: "builder", name: "Builder", emoji: "🔨", tier: "bronze", description: "Complete your first project" },
    { id: "cloud-curious", name: "Cloud Curious", emoji: "☁️", tier: "bronze", description: "Mark your first certification as in-progress" },
    { id: "aws-pioneer", name: "AWS Pioneer", emoji: "🟠", tier: "silver", description: "Complete an AWS project" },
    { id: "gcp-explorer", name: "GCP Explorer", emoji: "🔵", tier: "silver", description: "Complete a GCP project" },
    { id: "azure-navigator", name: "Azure Navigator", emoji: "🔷", tier: "silver", description: "Complete an Azure project" },
    { id: "prompt-wizard", name: "Prompt Wizard", emoji: "🧙", tier: "silver", description: "Complete LLM fine-tuning project (P05)" },
    { id: "rag-master", name: "RAG Master", emoji: "📚", tier: "silver", description: "Complete RAG project (P06)" },
    { id: "agent-whisperer", name: "Agent Whisperer", emoji: "🤖", tier: "gold", description: "Complete LangGraph agent project (P08)" },
    { id: "orchestrator", name: "Orchestrator", emoji: "🎭", tier: "gold", description: "Complete CrewAI multi-agent project (P09)" },
    { id: "mlops-hero", name: "MLOps Hero", emoji: "🚀", tier: "gold", description: "Complete MLOps pipeline project (P10)" },
    { id: "multi-cloud", name: "Multi-Cloud Architect", emoji: "🌐", tier: "gold", description: "Complete multi-cloud architecture project (P11)" },
    { id: "project-champion", name: "Project Champion", emoji: "🏆", tier: "gold", description: "Complete 5 projects" },
    { id: "halfway", name: "Halfway There", emoji: "🎯", tier: "silver", description: "Complete 6 projects" },
    { id: "completionist", name: "Completionist", emoji: "💎", tier: "platinum", description: "Complete all 12 projects" },
    { id: "certified", name: "Certified", emoji: "📜", tier: "gold", description: "Pass your first paid certification" },
    { id: "tri-cloud", name: "Tri-Cloud", emoji: "🌍", tier: "platinum", description: "Earn certifications on AWS, GCP, and Azure" },
    { id: "streak-7", name: "Streak Starter", emoji: "🔥", tier: "bronze", description: "Maintain a 7-day streak" },
    { id: "streak-30", name: "On Fire", emoji: "🔥🔥", tier: "gold", description: "Maintain a 30-day streak" },
    { id: "open-source", name: "Open Source Hero", emoji: "🤗", tier: "silver", description: "Push a model to Hugging Face Hub (complete P05)" },
    { id: "deep-diver", name: "Deep Diver", emoji: "🤿", tier: "silver", description: "Complete all resources in any single category" },
    { id: "architect-elite", name: "Architect Elite", emoji: "👑", tier: "platinum", description: "Reach Architect level (20,000 XP)" },
    { id: "governance", name: "Responsible AI", emoji: "⚖️", tier: "gold", description: "Complete AI governance section and EU AI Act course" }
  ]
};
