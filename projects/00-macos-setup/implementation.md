# Project 00: macOS AI Dev Setup — Implementation Guide

## Prerequisites
- MacBook Pro (M1/M2/M3 or Intel)
- macOS Ventura 13+ (Sonoma recommended)
- 16 GB RAM minimum, 50 GB free disk space
- Admin rights on your machine
- A terminal app (Terminal.app or iTerm2)

---

## Project Structure
```
~ (home directory)
├── .pyenv/                    # Python version manager
├── .zshrc / .bash_profile     # Shell config (modified by steps below)
├── miniconda3/                # Conda package manager
│   └── envs/aiarch/           # Your AI dev environment
└── Documents/ai-journey/      # Your project workspace
    └── projects/
        └── 00-macos-setup/
            └── verify.sh      # Verification script (created in Step 7)
```

---

## Phase 1: Homebrew

### Step 1.1 — Install Homebrew
Homebrew is the macOS package manager. It installs everything else.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen prompts (you'll need your admin password).

**Apple Silicon only** — after install, add Homebrew to your PATH:
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### Step 1.2 — Verify Homebrew
```bash
brew --version
# Expected: Homebrew 4.x.x
brew doctor
# Expected: Your system is ready to brew.
```

---

## Phase 2: Python via pyenv

### Step 2.1 — Install pyenv
pyenv lets you manage multiple Python versions without conflicts.

```bash
brew install pyenv
```

### Step 2.2 — Add pyenv to shell config
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2.3 — Install Python 3.11.9
```bash
pyenv install 3.11.9
pyenv global 3.11.9
```

### Step 2.4 — Verify Python
```bash
python --version
# Expected: Python 3.11.9
which python
# Expected: /Users/<you>/.pyenv/shims/python
```

---

## Phase 3: Ollama (Local LLMs)

### Step 3.1 — Install Ollama
```bash
brew install ollama
```

Or download from https://ollama.ai (drag to Applications).

### Step 3.2 — Start the Ollama service
```bash
ollama serve &
# Or open the Ollama.app from Applications
```

### Step 3.3 — Pull required models
These three models will be used across the journey:
```bash
ollama pull llama3.2        # General reasoning (2.0 GB)
ollama pull phi3            # Lightweight fast model (2.3 GB)
ollama pull nomic-embed-text # Embeddings for RAG (274 MB)
```
This will take several minutes depending on your connection.

### Step 3.4 — Verify Ollama
```bash
ollama list
# Expected: llama3.2, phi3, nomic-embed-text listed

ollama run phi3 "Say hello in one sentence"
# Expected: A short greeting from the model
```

---

## Phase 4: Cloud CLIs

### Step 4.1 — AWS CLI v2
```bash
brew install awscli
aws --version
# Expected: aws-cli/2.x.x
```

Configure with your AWS account:
```bash
aws configure
# AWS Access Key ID: <your key>
# AWS Secret Access Key: <your secret>
# Default region name: us-east-1
# Default output format: json
```

> Get your keys from AWS Console → IAM → Users → Security credentials → Create access key

### Step 4.2 — Google Cloud SDK
```bash
brew install --cask google-cloud-sdk
```

Add to shell (if not auto-added):
```bash
source "$(brew --prefix)/share/google-cloud-sdk/path.zsh.inc"
source "$(brew --prefix)/share/google-cloud-sdk/completion.zsh.inc"
```

Initialize:
```bash
gcloud init
# Follow prompts: sign in, select/create project
gcloud --version
# Expected: Google Cloud SDK 4xx.x.x
```

### Step 4.3 — Azure CLI
```bash
brew install azure-cli
az --version
# Expected: azure-cli 2.x.x
az login
# Browser opens — sign in with your Azure account
```

---

## Phase 5: Conda + aiarch Environment

### Step 5.1 — Install Miniconda
```bash
brew install --cask miniconda
conda init zsh
source ~/.zshrc
conda --version
# Expected: conda 24.x.x
```

### Step 5.2 — Create the aiarch environment
```bash
conda create -n aiarch python=3.11.9 -y
conda activate aiarch
```

### Step 5.3 — Install core ML libraries
Run this in the activated `aiarch` environment:

```bash
# Core scientific stack
pip install numpy pandas scikit-learn matplotlib seaborn jupyter jupyterlab

# Deep learning (Apple Silicon optimized)
pip install torch torchvision torchaudio

# Hugging Face ecosystem
pip install transformers datasets tokenizers peft accelerate

# LLM frameworks
pip install langchain langchain-community langgraph langsmith
pip install crewai

# RAG & Embeddings
pip install chromadb sentence-transformers ragas

# APIs & serving
pip install fastapi uvicorn[standard] streamlit gradio httpx

# Cloud SDKs
pip install boto3 sagemaker google-cloud-aiplatform
pip install azure-ai-ml azure-ai-documentintelligence azure-cosmos

# MLOps
pip install mlflow pytest joblib python-dotenv pydantic
```

> This takes 5–10 minutes. Go grab a coffee.

### Step 5.4 — Verify key packages
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import langchain; print('LangChain:', langchain.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
```

---

## Phase 6: Docker Desktop

### Step 6.1 — Install Docker Desktop
```bash
brew install --cask docker
```

Launch Docker Desktop from Applications and complete the setup wizard. Accept the license.

### Step 6.2 — Verify Docker
```bash
docker --version
# Expected: Docker version 26.x.x
docker ps
# Expected: empty table (no containers running yet)
docker run hello-world
# Expected: "Hello from Docker!" message
```

---

## Phase 7: VS Code + Extensions

### Step 7.1 — Install VS Code
```bash
brew install --cask visual-studio-code
```

### Step 7.2 — Install the `code` CLI
Open VS Code → ⌘⇧P → "Shell Command: Install 'code' command in PATH"

### Step 7.3 — Install extensions
```bash
code --install-extension ms-python.python
code --install-extension ms-python.pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-azuretools.vscode-docker
code --install-extension amazonwebservices.aws-toolkit-vscode
code --install-extension ms-azuretools.vscode-azurefunctions
code --install-extension googlecloudtools.cloudcode
code --install-extension github.copilot
code --install-extension github.vscode-github-actions
```

### Step 7.4 — Configure Python interpreter
Open VS Code in your project folder:
```bash
code ~/Documents/ai-journey
```
Press ⌘⇧P → "Python: Select Interpreter" → choose `aiarch` conda environment.

---

## Phase 8: Verification Script

### Step 8.1 — Create verify.sh
```bash
cat > ~/Documents/ai-journey/projects/00-macos-setup/verify.sh << 'EOF'
#!/bin/bash
echo "=== AI Dev Environment Verification ==="

check() {
  if eval "$2" &>/dev/null; then
    echo "  [PASS] $1"
  else
    echo "  [FAIL] $1 — run: $2"
  fi
}

echo ""
echo "--- Core Tools ---"
check "Homebrew"       "brew --version"
check "Python 3.11"   "python --version | grep -q '3.11'"
check "pyenv"         "pyenv --version"
check "conda"         "conda --version"
check "Docker"        "docker --version"
check "VS Code"       "code --version"

echo ""
echo "--- Cloud CLIs ---"
check "AWS CLI"       "aws --version"
check "gcloud"        "gcloud --version"
check "Azure CLI"     "az --version"

echo ""
echo "--- Ollama Models ---"
check "ollama"                "ollama --version"
check "llama3.2 model"        "ollama list | grep -q llama3.2"
check "phi3 model"            "ollama list | grep -q phi3"
check "nomic-embed-text"      "ollama list | grep -q nomic-embed-text"

echo ""
echo "--- Python Packages (aiarch env) ---"
conda run -n aiarch python -c "import torch; print('  [PASS] PyTorch', torch.__version__)" 2>/dev/null || echo "  [FAIL] torch"
conda run -n aiarch python -c "import transformers; print('  [PASS] transformers', transformers.__version__)" 2>/dev/null || echo "  [FAIL] transformers"
conda run -n aiarch python -c "import langchain; print('  [PASS] langchain', langchain.__version__)" 2>/dev/null || echo "  [FAIL] langchain"
conda run -n aiarch python -c "import fastapi; print('  [PASS] fastapi', fastapi.__version__)" 2>/dev/null || echo "  [FAIL] fastapi"
conda run -n aiarch python -c "import boto3; print('  [PASS] boto3', boto3.__version__)" 2>/dev/null || echo "  [FAIL] boto3"
conda run -n aiarch python -c "import chromadb; print('  [PASS] chromadb', chromadb.__version__)" 2>/dev/null || echo "  [FAIL] chromadb"

echo ""
echo "=== Done ==="
EOF
chmod +x ~/Documents/ai-journey/projects/00-macos-setup/verify.sh
```

### Step 8.2 — Run the verification
```bash
~/Documents/ai-journey/projects/00-macos-setup/verify.sh
```

All items should show `[PASS]`.

---

## Verification Checklist
- [ ] `brew --version` returns 4.x.x
- [ ] `python --version` returns Python 3.11.9
- [ ] `ollama list` shows llama3.2, phi3, nomic-embed-text
- [ ] `aws --version` works
- [ ] `gcloud --version` works
- [ ] `az --version` works
- [ ] `conda activate aiarch` works
- [ ] PyTorch, transformers, langchain, fastapi importable in aiarch
- [ ] `docker ps` works (Docker Desktop running)
- [ ] VS Code opens with `code .`
- [ ] verify.sh shows all [PASS]

---

## Troubleshooting

**`brew: command not found` after install (Apple Silicon)**
```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

**`pyenv: command not found`**
```bash
source ~/.zshrc
```

**Ollama models fail to download**
Check your disk space: `df -h ~`. You need ~5 GB free for all three models.

**PyTorch install takes forever or fails**
On Apple Silicon, PyTorch has native MPS support. If pip hangs, try:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**`conda activate aiarch` fails**
```bash
conda init zsh && source ~/.zshrc
```

**AWS configure: where do I get keys?**
AWS Console → top-right username → Security credentials → Access keys → Create access key → select "CLI"

---

## Next Steps → Project 01: ML Foundations
You now have a complete AI development environment. In Project 01, you'll:
- Load the California Housing dataset
- Build an EDA notebook with visualizations
- Train 3 ML models and compare them
- Deploy the best model as a FastAPI REST API
- Write tests with pytest

Open your terminal, activate your environment, and start Project 01:
```bash
conda activate aiarch
mkdir -p ~/Documents/ai-journey/projects/01-ml-foundations/{src,notebooks,tests,models,docs/adr}
cd ~/Documents/ai-journey/projects/01-ml-foundations
```
