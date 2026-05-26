# AI Architect Journey — 6-Month Gamified Learning Path

An interactive, self-contained web app for becoming a certified AI/ML Solution Architect. Track your progress through 12 hands-on projects, 6 cloud certifications, and 25 badge achievements — all stored locally with no backend required.

## Live Demo

Deploy to GitHub Pages: push to `main` branch → GitHub Actions deploys automatically.

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/ai-architect-journey.git
cd ai-architect-journey
python3 -m http.server 8181
```

Open: http://localhost:8181

## What's Inside

| Section | Content |
|---|---|
| 12 Projects | ML basics → cloud → agents → MLOps → capstone |
| 6 Certifications | AWS AI Practitioner, ML Engineer, Google ML Eng, Azure AI-300, more |
| 8 Free Courses | Hugging Face, fast.ai, Google ML Crash Course, DeepLearning.AI |
| 8 YouTube Channels | Krish Naik, DeepLearningAI, Google Cloud Tech, and more |
| 7 System Design Challenges | RAG, fraud detection, feature store, multi-agent, responsible AI |
| 25 Badges | Bronze → Platinum progression |
| XP + Levels | Apprentice → Architect (7 levels) |

## 6-Month Path

| Month | Focus | Projects |
|---|---|---|
| 1 | ML Foundations + AWS | P00 Setup, P01 ML Pipeline, P02 SageMaker |
| 2 | Azure + Bedrock | P03 Azure Document AI, P04 Bedrock Chat |
| 3 | LLMs | P05 QLoRA Fine-tuning, P06 RAG + LangChain |
| 4 | GCP + Agents | P07 Vertex AI Pipeline, P08 LangGraph Agent |
| 5 | Multi-Agent + MLOps | P09 CrewAI, P10 Docker + CI/CD + ECS |
| 6 | Architecture + Capstone | P11 Multi-Cloud GenAI, P12 Full-Stack AI Platform |

## Features

- **No backend** — all progress stored in `localStorage`
- **Filter** by cloud provider (AWS/GCP/Azure/HuggingFace), cost (Free/Paid), category
- **Dark/light theme** toggle
- **Gamification** — XP, level-ups, badges, confetti, daily streaks
- **Export/import** progress as JSON
- **GitHub Pages** deployment via GitHub Actions

## Project Requirements

Each project folder (`projects/XX-name/`) contains:
- `requirements.md` — detailed spec with code examples and acceptance criteria

## Deploy to GitHub Pages

1. Fork or push this repo to GitHub
2. Go to **Settings → Pages → Source: GitHub Actions**
3. Push to `main` — the workflow in `.github/workflows/deploy.yml` handles the rest
4. Your site will be at `https://YOUR_USERNAME.github.io/ai-architect-journey/`

## Tech Stack

Pure HTML/CSS/JavaScript — no build tools, no npm, no framework required.
