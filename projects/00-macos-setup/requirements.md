# Project 00: MacBook Pro AI Dev Setup

**XP:** 100 | **Cost:** Free | **Duration:** Week 0 (2–4 hours)

## Objective
Bootstrap your complete AI/ML development environment on macOS before any cloud work begins. This is the non-negotiable foundation.

## Prerequisites
- MacBook Pro (Apple Silicon M1/M2/M3 or Intel)
- macOS 13+ (Ventura or later)
- 16GB RAM recommended (8GB minimum)
- 50GB free disk space
- Credit card for cloud free tiers (no charges expected)

## Tasks

### 1. Core Package Manager
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Verify
brew --version
```

### 2. Python Version Management
```bash
brew install pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

pyenv install 3.11.9
pyenv global 3.11.9
python --version   # should show 3.11.9
```

### 3. Local LLM Runtime (Ollama)
```bash
brew install ollama
ollama serve &           # start server
ollama pull llama3.2     # 3B parameter, fast
ollama pull phi3         # Microsoft, efficient
ollama pull nomic-embed-text  # for embeddings
ollama run llama3.2      # test: ask "What is RAG?"
```

### 4. Cloud CLIs
```bash
# AWS CLI v2
brew install awscli
aws configure           # enter access key, secret, region=us-east-1, format=json
aws sts get-caller-identity   # verify

# Google Cloud SDK
brew install --cask google-cloud-sdk
gcloud init
gcloud auth application-default login
gcloud config list      # verify

# Azure CLI
brew install azure-cli
az login
az account show         # verify
```

### 5. Python AI/ML Environment
```bash
pip install miniconda
conda create -n aiarch python=3.11 -y
conda activate aiarch

# Core ML
pip install numpy pandas scikit-learn matplotlib seaborn jupyter jupyterlab

# PyTorch (Apple Silicon optimized)
pip install torch torchvision torchaudio

# LLM & Agents
pip install transformers datasets tokenizers peft accelerate
pip install langchain langchain-community langchain-core langgraph langsmith
pip install crewai

# RAG & Vector DBs
pip install chromadb sentence-transformers ragas

# APIs & Deployment
pip install fastapi uvicorn httpx streamlit gradio

# Cloud SDKs
pip install boto3 sagemaker google-cloud-aiplatform azure-ai-ml azure-ai-formrecognizer
```

### 6. Docker Desktop
```bash
brew install --cask docker
# Open Docker Desktop from Applications, complete setup
docker --version
docker run hello-world    # verify
```

### 7. VS Code + Extensions
Install VS Code: https://code.visualstudio.com/

Then install these extensions (Cmd+Shift+X):
- **Python** (Microsoft)
- **Pylance**
- **Jupyter**
- **Docker**
- **AWS Toolkit**
- **Azure Tools**
- **Google Cloud Code**
- **GitHub Copilot** (optional)

## Acceptance Criteria
- [ ] `python --version` shows 3.11.x
- [ ] `ollama run llama3.2` answers a question
- [ ] `aws sts get-caller-identity` returns your account ID
- [ ] `gcloud config list` shows your project
- [ ] `az account show` shows your subscription
- [ ] `docker run hello-world` succeeds
- [ ] `jupyter lab` opens in browser

## Apple Silicon Tips
- Use `mps` device for PyTorch GPU acceleration:
  ```python
  import torch
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  ```
- Ollama auto-detects Apple Silicon and uses Metal acceleration
- Expect 15–25 tokens/second for 7B models on M2 with 16GB RAM

## Cloud Free Tier Links
- AWS: https://aws.amazon.com/free/ (12 months free)
- GCP: https://cloud.google.com/free ($300 credit, 90 days)
- Azure: https://azure.microsoft.com/free ($200 credit, 30 days)
