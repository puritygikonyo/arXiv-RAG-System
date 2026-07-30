# arXiv RAG System

**Production-grade Agentic RAG for academic paper research.**
Built with FastAPI · OpenSearch · LangGraph · Groq · Deployed on Hugging Face Spaces.

The **academic paper research assistant** automatically fetches academic papers, understands their content, answers the research questions using advanced RAG techniques.

The project is to keep the cost as $0 as possible.

Ask a research question in plain English, get an answer grounded in real arXIv papers - with sources cited, not hallucinated. 

**Live demo · GIF/video walkthrough (links will be added once ready)**
---

## What This Is

This is an agentic RAG(Retrieval-Augmented Generation) system: instead of just searching for documents and stuffing them into a prompt, it makes real decisions along the way.i.e checking whether a question is even answerable from the paper database, judging whether what is found is actually good enough, and automatically rewriting its own search and trying again if not. 
\\
\\


An end-to-end Retrieval-Augmented Generation (RAG) system that:
- **Ingests** arXiv papers automatically via Airflow DAGs
- **Indexes** them with hybrid search (BM25 keyword + vector semantic)
- **Answers** research questions using a LangGraph agent that grades, rewrites, and retrieves intelligently
- **Serves** multiple users via FastAPI REST API, Gradio UI, and Telegram bot

Built as a learning project following production engineering practices — the way it's done at FAANG companies.
---

## Why this isn't "just a chatbot"
Most RAG demos follow one fixed path: retrieve ----> generate. This ssystem can loop and autocorrect:


Question → Is this answerable from our papers?  → No → polite rejection
                    │ Yes
                    ▼
              Search papers  ──────────┐
                    │                  │
                    ▼                  │
          Grade result quality         │
                    │                  │
        Good enough?  ── No ──> Rewrite the search query
                    │ Yes              │  (up to 3 attempts)
                    ▼                  │
             Write final answer <──────┘
             (with citations)



This is implemented as a LangGraph state machine: guardrail -> retriever -> grader -> rewriter -> generator, with a bounded retry loop so it never spins forever, and a graceful fallback (best-effort answer) if it hits the retry limit without a perfect match.

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
| Deployment | Render | Free |
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
| 2 | FastAPI skeleton + health | ✅ Complete |
| 3 | PostgreSQL models + Neon | ✅ Complete |
| 4 | arXiv data pipeline (Airflow) | ✅ Complete |
| 5 | OpenSearch + BM25 search | ✅ Complete |
| 6 | Embeddings + hybrid search | ✅ Complete |
| 7 | LangGraph agentic RAG | ✅ Complete |
| 8 | Redis caching + Langfuse | ✅ Complete |
| 9 | Telegram bot | ✅ Complete |
| 10 | HuggingFace Spaces deploy | ✅ Complete |

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
