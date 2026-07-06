# arXiv RAG System

**Production-grade Agentic RAG for academic paper research.**
Built with FastAPI · OpenSearch · LangGraph · Groq · Deployed on Hugging Face Spaces.

---

## What This Is

An end-to-end Retrieval-Augmented Generation (RAG) system that:
- **Ingests** arXiv papers automatically via Airflow DAGs
- **Indexes** them with hybrid search (BM25 keyword + vector semantic)
- **Answers** research questions using a LangGraph agent that grades, rewrites, and retrieves intelligently
- **Serves** multiple users via FastAPI REST API, Gradio UI, and Telegram bot

Built as a learning project following production engineering practices — the way it's done at FAANG companies.

---

## Architecture

```
Users → FastAPI (Cloud Run / HF Spaces)
           ├── Hybrid Search  → OpenSearch (BM25 + vector)
           ├── Agentic RAG    → LangGraph → Groq LLM
           ├── Cache          → Upstash Redis
           └── Monitoring     → Langfuse Cloud

Data Pipeline → Astronomer Astro (Airflow)
                    └── arXiv API → PostgreSQL (Neon) → OpenSearch
```

---

## Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| API | FastAPI + uvicorn | Free |
| Database | Neon.tech (serverless Postgres) | Free |
| Search | OpenSearch (Docker, single node) | Free |
| Orchestration | Astronomer Astro (managed Airflow) | Free tier |
| Embeddings | Jina AI | Free (1M tokens) |
| LLM | Groq (Llama 3 70B) | Free tier |
| Cache | Upstash Redis | Free tier |
| Monitoring | Langfuse Cloud | Free tier |
| Deployment | Hugging Face Spaces | Free |
| CI/CD | GitHub Actions | Free |

**Total cost: $0**

---

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker Desktop

### Setup

```bash
# 1. Clone
git clone https://github.com/your-username/arxiv-rag-system
cd arxiv-rag-system

# 2. Configure environment
cp .env.example .env
# Edit .env — most defaults work for local dev

# 3. Install dependencies
make setup

# 4. Start OpenSearch (only Docker service)
make start

# 5. Verify everything
make health

# 6. Run the API
make serve
# Visit: http://localhost:8000/docs
```

---

## Development

```bash
make help          # all available commands

make format        # auto-format code
make lint          # lint check
make type-check    # mypy type checking
make check         # all of the above

make test          # all tests
make test-unit     # fast tests (no services needed)
make test-cov      # tests + coverage report
```

---

## Build Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project scaffold & config | ✅ Complete |
| 2 | FastAPI skeleton + health | 🔄 Next |
| 3 | PostgreSQL models + Neon | ⏳ Planned |
| 4 | arXiv data pipeline (Airflow) | ⏳ Planned |
| 5 | OpenSearch + BM25 search | ⏳ Planned |
| 6 | Embeddings + hybrid search | ⏳ Planned |
| 7 | LangGraph agentic RAG | ⏳ Planned |
| 8 | Redis caching + Langfuse | ⏳ Planned |
| 9 | Telegram bot | ⏳ Planned |
| 10 | HuggingFace Spaces deploy | ⏳ Planned |

---

## Project Structure

```
arxiv-rag-system/
├── src/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # All settings (Pydantic)
│   ├── logger.py            # Structured logging
│   ├── routers/             # API endpoints
│   ├── services/            # Business logic
│   │   ├── agents/          # LangGraph nodes + workflow
│   │   ├── search/          # BM25 + hybrid search
│   │   ├── embeddings/      # Jina AI client
│   │   ├── cache/           # Redis cache
│   │   └── pipeline/        # arXiv ingestion
│   ├── models/              # SQLAlchemy ORM models
│   ├── schemas/             # Pydantic request/response schemas
│   └── db/                  # Database session + migrations
├── tests/
│   ├── unit/                # Fast, no external services
│   └── integration/         # Requires running services
├── airflow/
│   └── dags/                # Airflow DAG definitions
├── notebooks/               # Weekly learning notebooks (week1-7)
├── scripts/                 # Utility scripts
├── .github/workflows/       # CI/CD pipelines
├── docker-compose.yml       # OpenSearch only (lightweight)
├── pyproject.toml           # Dependencies + tool config
├── Makefile                 # Developer commands
└── .env.example             # Environment variable template
```

---

## License

MIT
