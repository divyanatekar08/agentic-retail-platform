# 🛍️ Agentic Retail Platform

An asynchronous, AI-powered multi-agent retail engine and API built with Python 3.13, LangGraph, FastAPI, Qdrant, and FastEmbed. Designed for intelligent product recommendation synthesis, spatial location filtering, vector semantic catalog search, and automated agentic reasoning workflows.

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-orange?style=flat)
![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red?style=flat)
![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed%20ONNX-blue?style=flat)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat&logo=githubactions&logoColor=white)

---

## 🛠️ System Architecture & Key Highlights

- **Stateful Multi-Agent Orchestration**: Powered by **LangGraph**, executing cyclic preference parsing, vector RAG catalog retrieval, spatial filtering, and quality critique feedback loops.
- **Local High-Performance RAG Engine**: Integrated **FastEmbed** (`BAAI/bge-small-en-v1.5`) ONNX embeddings with **Qdrant** in-memory vector database for sub-millisecond semantic product catalog search.
- **Hybrid Spatial Analytics**: Combines vector indexing with **Polars** fast columnar data processing for geographic POI filtering (cuisine, walking time, user constraints).
- **Asynchronous API Core**: Built on **FastAPI** and **Uvicorn** with strict Pydantic v2 schemas and centralized structured error handling.
- **Production Containerization**: Fully dockerized with multi-stage builds (`Dockerfile`, `docker-compose.yml`) and isolated environment variable handling.
- **Automated CI/CD**: End-to-end GitHub Actions pipeline (`.github/workflows/ci.yml`) automatically testing python dependencies and agent logic on push.

---

## 🧰 Tech Stack

- **Languages & Core**: Python 3.13, Asyncio, Pydantic v2, Polars
- **Agentic Frameworks & LLMs**: LangGraph, DSPy, Google Gemini API (`gemini-2.5-flash`)
- **Vector Search & RAG**: Qdrant Client (`v1.11+`), FastEmbed (ONNX BGE Embeddings)
- **API Engine**: FastAPI, Uvicorn
- **DevOps & Testing**: Docker, Docker Compose, GitHub Actions, Pytest

---

## 📂 Project Structure

```text
agentic-retail-platform/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow
├── src/
│   ├── agents/
│   │   └── retail_agent.py    # LangGraph state graph & multi-node orchestration
│   ├── rag_engine.py          # FastEmbed + Qdrant Vector Semantic Engine
│   ├── spatial_engine.py      # Polars + Qdrant hybrid spatial retrieval engine
│   └── api.py                 # FastAPI service endpoints & DSPy query analyzer
├── .env                       # Environment variable definitions (API Keys)
├── .dockerignore              # Excludes local artifacts from Docker build
├── Dockerfile                 # Python 3.13 container configuration
├── docker-compose.yml         # Container runtime orchestration
├── requirements.txt           # Frozen production dependencies
└── README.md

---

## 🧰 Tech Stack & Skills

- **Languages & Core Frameworks**: Python 3.13, Asyncio, Pydantic v2, FastAPI, Uvicorn
- **Agentic Orchestration & LLMs**: LangGraph, DSPy, Google GenAI SDK (`gemini-2.5-flash`)
- **Vector Search & Analytics**: FastEmbed, Qdrant Client, Polars, NumPy
- **DevOps, Containerization & Testing**: Docker, Docker Compose, GitHub Actions, Pytest, Pytest-Asyncio
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
