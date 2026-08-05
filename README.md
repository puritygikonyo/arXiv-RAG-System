# arXiv RAG System

**Production-grade Agentic RAG for academic paper research.**
Built with FastAPI · OpenSearch · LangGraph · Groq · Deployed on Render.

The **academic paper research assistant** automatically fetches academic papers, understands their content, answers the research questions using advanced RAG techniques.

The project keeps the cost as $0.

Ask a research question in plain English, get an answer grounded in real arXIv papers - with sources cited, not hallucinated. 
![Agentic RAG System Architecture](https://github.com/puritygikonyo/arXiv-RAG-System/blob/main/Agentic%20RAG%20System%20final%20architecture%20image.png)

---
## Table of Contents
- [What this is](#what-this-is)
- [Architecture](#architecture)
- [Build Phases](#build-phases)
  - [Phase 1 — Project scaffold & config](#Phase-1--Project-scaffold-&-config-(Project-Setup)-)
  - [Phase 2 — FastAPI skeleton + health](#phase-2--fastapi-skeleton--health)
  - [Phase 3 — PostgreSQL models + Neon](#phase-3--postgresql-models--neon)
  - [Phase 4 — arXiv data pipeline](#phase-4--arxiv-data-pipeline)
  - [Phase 5 — OpenSearch + BM25 search](#phase-5--opensearch--bm25-search)
  - [Phase 6 — Embeddings + hybrid search](#phase-6--embeddings--hybrid-search)
  - [Phase 7 — LangGraph agentic RAG](#phase-7--langgraph-agentic-rag)
  - [Phase 8 — Redis caching + Langfuse](#phase-8--redis-caching--langfuse)
  - [Phase 9 — Telegram bot](#phase-9--telegram-bot)
  - [Phase 10 — Deployment](#phase-10--deployment)
- [Post-launch hardening](#post-launch-hardening-(beyond-Phase-10))
- [Real engineering challenges](#Real-engineering-challenges-encountered-during-the-project)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Running locally](#running-locally)
---

## What This Is

This is an agentic RAG(Retrieval-Augmented Generation) system: instead of just searching for documents and stuffing them into a prompt, it makes real decisions along the way.i.e checking whether a question is even answerable from the paper database, judging whether what is found is actually good enough, and automatically rewriting its own search and trying again if not. 


It was built and hardened against a real constraint: run on free-tier infrastructure without cutting corners on functionality, which forced genuine architecture decisions(see Real Engineering Challenges), not just "call an API and ship it"


---

## Why this isn't "just a chatbot"
Most RAG demos follow one fixed path: retrieve ----> generate. This system can loop and autocorrect:

```
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
```


This is implemented as a LangGraph state machine: **guardrail -> retriever -> grader -> rewriter -> generator**, with a bounded retry loop so it never spins forever, and a graceful fallback (best-effort answer) if it hits the retry limit without a perfect match.

## Architecture

![Agentic RAG System Architecture](https://github.com/puritygikonyo/arXiv-RAG-System/blob/main/architecture.svg)



Four layers: 
- an offline ingestion script pulls papers from arXiv and embeds them via Jina;
- Postgres (Neon) and OpenSearch (Aiven, managed) store metadata and power hybrid search;
- a FastAPI backend runs a LangGraph agentic loop with Groq as the LLM, Upstash for semantic caching, and Langfuse for tracing;
- an invite-gated access layer serves the same backend to three clients — the raw API, a Gradio web UI, and a Telegram bot.

Version control and deployment triggers run through GitHub — Render's three services (API, Gradio, Telegram worker) each auto-deploy from the same repo on push, using one shared Dockerfile with different Start Commands per service.

## The technical aspects

- **Hybrid search (BM25 + vector), not embeddings-only:** Keyword search and semantic vector searchrun together and get combined - better recall than either alone.

- **Semantic caching:** Near-duplicate questions get answered instantly from cache instead of re-running the full pipeline, cutting both latency and LLM cost. (Real measured hit rate: ~33% in testing.)

- **Full observability:** Every run is traced node-by-node in Langfuse, plus a custom Postgres metrics table for cache hit rate, latency, and top queries — visibility Langfuse's own dashboard doesn't aggregate natively.

- **Cost-aware access control, not just an API key:** Invite-based login with per-user daily question limits and revocation, enforced at question-time (not just login) — because "share it, but keep it controlled" was a real product requirement, not an afterthought.

- **Multi-channel, one backend:** The same FastAPI agent powers a web UI (Gradio), a Telegram bot, and the raw API — no logic duplicated between them.


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
| 10 | Render deploy | ✅ Complete |

### Phase 1 : Project scaffold & config (Project Setup)

Established the project's configuration surface early, before any feature code: a single `Settings` class(Pydantic `BaseSettings`) reads every environment variable the app needs, with `case_sensitive=False` and typed defaults, so a missing or malformed variable fails loudly at startup rather than causing a silent bug three layers deep later. List-typed settings (`cors_origins, arxiv_categories, telegram_allowed_chat_ids`) use `field_validators` to accept comma-separated strings from `.env` — though this later collided with `pydantic-settings`' own JSON-decoding behavior for env vars (see Real engineering challenges). Version control was set up on GitHub from day one, with `.gitignore` scoped for secrets, virtual environments, and build artifacts.


### Phase 2 : FastAPI skeleton + health (Basic API + health check)

Base FastAPI app with a `/health` endpoint that independently checks OpenSearch, Postgres, and Redis connectivity - so a broken dependency shows up as a specific, named failure rather than a generic 500. Structured logging (`structlog`) configured from the start: JSON output in production, pretty console output locally, so log behaviour never had to be retrofitted later.

### Phase 3 : PostgreSQL models + Neon (Database setup)

Async SQLAlchemy setup against Neon (serverless Postgres) via `asyncpg`. Core models: `Paper` (arXiv metadata, keyed by arXiv ID, with an `ingestion_status` enum tracking each paper through `pending -> fetched -> chunked -> embedded -> indexed`) and `Chunk` (one row per text chunk, cascade-deleted with its parent paper).
`pool_pre_ping=True` on the engine specifically to handle Neon's serverless auto-suspend - without it, the first query after Neons's connection went idle would fail with a stable connection

### Phase 4: arXiv data pipeline

Originally scoped and built as **Airflow/Astronomer project** - a separate deployable within its own `dags/arxiv_ingestion_dag.py`, using `feedparser` to parse arXiv's Atom API and `psycopg2` (sync) to upsert into Postgres, orchestrated via Airflow's TaskFlow API (`@dag/@task` decorators) on a daily schedule. This DAG was fully built and is real, working code — but was deliberately **never deployed**, since running Airflow continuously is real infrastructure overhead not justified for occasional/manual ingestion at this project's current scale. Instead, the proven fetch/parse logic was ported into a standalone script (`ingest_arxiv.py`) that reuses the same `feedparser`-based parsing but writes via the main app's own async SQLAlchemy session — no new dependency, no second deployable to maintain. The Airflow project remains in the repo (`airflow/`) as tested, ready-to-deploy infrastructure for scheduled ingestion, if that becomes worth the operational overhead later.

### Phase 5: OpenSearch + BM25 search

Initial local setup: a single-node OpenSearch container (Docker Compose), no security plugin, `index.knn: true` in settings from the start so the later vector field wouldn't require a reindex. 

Custom analyzer (lowercase, stopword removal, ASCII folding) on `title/abstract/authors`, with a boosted `title` field (×3) so exact-title matches rank highly. `/api/v1/search` implemented as pure BM25 keyword search with category/date filtering and pagination.

### Phase 6: Embeddings + hybrid search

Jina AI (`jina-embeddings-v3`, 1024 dimensions) for embeddings. Papers are chunked via a sliding window (512 tokens, 50-token overlap) over **title + abstract only** — full-text PDF chunking was scoped out as a future enhancement, which later turned out to matter for answer quality (see Real engineering challenges). A second OpenSearch index (`arxiv_chunks`) stores one document per chunk with its embedding vector; `/api/v1/hybrid-search` combines BM25 and k-NN vector search. The paper-level document also gets an embedding (the first chunk's vector) so paper-level vector search works without a separate lookup.

### Phase 7: LangGraph agentic RAG

The core of the system: a 5-node LangGraph state machine — `guardrail → retriever → grader → rewriter → generator`, with a separate `reject` exit path. The guardrail checks whether a question is even answerable from the corpus before doing any retrieval. The retriever/grader/rewriter form a bounded loop (`agent_max_retrieval_attempts`, default 3): if the grader scores retrieved chunks below `agent_relevance_threshold` (0.7), the rewriter reformulates the search query and tries again; after the retry limit, the graph proceeds to generation anyway with whatever it has, rather than looping forever. The generator only answers from chunks that individually clear a relevance floor, with an explicit instruction not to fabricate findings or fill gaps with outside knowledge — tightened further after real user testing (see Post-launch hardening). `/api/v1/ask` streams one SSE event per node completion, so a client can show live progress ("searching → grading → rewriting → generating") instead of waiting silently through a run that can take 15–90+ seconds.

### Phase 8: Redis caching + Langfuse

Semantic caching via Upstash (serverless Redis): before running the graph, `/ask` embeds the incoming question and checks for a near-duplicate in cache (cosine similarity ≥ `cache_similarity_threshold`, 0.80) — a hit skips guardrail/retrieval/grading/generation entirely and returns instantly. Measured hit rate in testing: ~33%. Langfuse traces every node individually (including a manually-logged trace for cache hits, so hit rate stays visible in the dashboard even though no graph nodes ran). A custom Postgres `QueryLog` table backs a `/admin/metrics` endpoint (cache hit rate, avg latency by hit/miss, top 10 queries) — data Langfuse's own dashboard doesn't aggregate natively.

Load testing with Locust surfaced a real production bug here: under concurrent load, `"Connection pool is full, discarding connection"` — the OpenSearch client's underlying `urllib3` pool was implicitly capped at 1, serializing every concurrent request through a single connection. Fixed by explicitly setting `pool_maxsize=25` on the client.

### Phase 9: Telegram bot

A thin HTTP client of the app's own `/api/v1/ask` — not a second implementation of the RAG pipeline, so every Telegram message gets the same caching, tracing, and logging as a normal API call for free. Handles Telegram's 4096-character message limit by splitting long answers on paragraph boundaries (falling back to word- and character-level splitting for pathological cases), and strips LLM markdown formatting to plain text rather than risking `MarkdownV2`'s strict escaping rules silently rejecting an entire message over one unescaped character.

### Phase 10: Deployment

The most involved phase, and the one with the most real debugging. Original plan (Hugging Face Spaces) was ruled out when Spaces' Docker SDK moved behind a paid plan. Render's free tier was the next candidate, but self-hosted OpenSearch's JVM heap requirement alone exceeded the entire free-tier memory budget = the actual fix wasn't a bigger free tier, it was recognizing that bundling a search engine's JVM into the same container as the app was the wrong architecture regardless of hosting provider. 

**Aiven's free managed OpenSearch tier** solved this by decoupling search from app compute - the architecturally correct pattern, not a workaround. The Dockerfile was rewritten from `FROM opensearchproject/opensearch` (bundling OpenSearch and the app together) to a plain `python:3.14-slim` image running only the FastAPI app, with the old OpenSearch-startup-wait entrypoint script no longer needed at all. 

The system now runs as **three separate Render services from one shared Dockerfile** (different Start Commands, all triggered by pushes to the same GitHub repo): the FastAPI API, a Gradio web UI, and a Telegram Background Worker.

---
## Post-launch hardening (beyond Phase 10)

Real user testing surfaced work that wasn't part of the original 10-phase plan but was necessary before genuinely sharing the system:

- **Invite-based access control:** a Postgres `invites` table (`username, token, label, revoked, daily_question_limit, telegram_chat_id`) gates all three client channels through one shared enforcement path (`check_invite_allowed()`), checked at question-time, not just login — so revoking someone or hitting their daily limit takes effect on their very next question, and a blocked request never even reaches the semantic cache or the LLM.

- **Admin endpoint security:** `/admin/metrics` was found to have no authentication at all — anyone with the API URL could see total query volume, cache hit rate, and the 10 most-asked questions. Fixed by reusing the same admin-key check already protecting `/admin/cache/invalidate`.

- **Generator hallucination guardrails tightened:** real testing surfaced confidently-wrong answers on questions the corpus couldn't actually support in depth (a symptom of abstract-only chunking against a niche ~500-paper corpus). Fixed by raising the per-chunk relevance floor (0.5 → 0.65) and strengthening the system prompt to explicitly refuse filling gaps with outside knowledge, preferring an honest "I couldn't find enough relevant material" over a fabricated specific.

- **Domain pivot:** the corpus was narrowed from general AI/ML categories to Power Systems (`eess.SY`) to match a specific set of test users, including building small operational scripts (`check_ingestion_status.py`, `check_category.py`, `remove_category.py`, `clear_pending.py`) to inspect and safely reshape the corpus without ad hoc SQL.

---


## Real engineering challenges encountered during the project

A few things that came up building and deploying this for real, worth naming because they're the parts that actually taught something:

- **Found and fixed a real connection-pool bug under load testing:** Locust load testing surfaced "Connection pool is full" errors under concurrent requests — the OpenSearch client's connection pool was implicitly capped at 1, serializing every request through a single connection. Fixed by explicitly raising pool_maxsize.


- **Rearchitected search infrastructure to fit free-tier constraints:** The original design (self-hosted OpenSearch bundled with the app) needed far more RAM than any free hosting tier provides. Migrated to a managed OpenSearch service, decoupling the search engine from the app's compute — the architecturally correct pattern anyway, not just a workaround.

- **Airflow built but consciously not deployed** — a real, working DAG exists and was tested, but shipping it would have meant running a scheduler continuously for infrequent ingestion. Recognizing "built and correct" doesn't automatically mean "should be deployed" was itself a deliberate engineering call, not an oversight.

- **Chased down several silent .gitignore/.dockerignore bugs:** Broad exclusion patterns (models/, *.pem) written for one purpose (blocking ML weight files, private keys) ended up silently excluding legitimate application code and public certificates from deploys — a good reminder that "it works locally" and "it's actually in the deployed image" are different claims.


---

## Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| API | FastAPI + uvicorn | Free |
| Agent orchestartion | LangGraph | Free |
| Search | OpenSearch (Aiven, managed) — hybrid BM25 + k-NN vector search | Free
| Database | Neon.tech (serverless Postgres) | Free |
| Orchestration (built, not deployed) | Astronomer Astro (managed Airflow) | Free tier |
| Embeddings | Jina AI (`jina-embeddings-v3`) | Free (1M tokens) |
| LLM | Groq (Llama 3.3 70B) | Free tier |
| Cache | Upstash Redis (semantic caching)| Free tier |
| Observability & Monitoring | Langfuse Cloud | Free tier |
| Web UI | Gradio | Free |
| Bot Channel | Telegram( `python-telegram-bot`) | Free |
| Deployment | Render(3 services from 1 Dockerfile) | Free |
| CI/CD | GitHub (Render auto-deploys on push) | Free |
| Migrations | Alembic | Free |

**Total cost: $0**

---

## Quick Start

### Prerequisites
- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker Desktop (optional — only needed if you want local OpenSearch instead of a remote Aiven instance)

### Setup

```bash
# 1. Clone
git clone https://github.com/puritygikonyo/arxiv-rag-system
cd arxiv-rag-system

# 2. Configure environment
cp .env.example .env
# Fill in your own OpenSearch (Aiven), Postgres (Neon), Jina, Groq,
# Upstash, and Langfuse credentials

# 3. Install dependencies
make setup

# 4. (Optional) Start local OpenSearch instead of pointing at Aiven
make start

# 5. Apply database migrations
uv run alembic upgrade head

# 6. Generate an invite token so you can actually log in later
uv run python generate_invite.py demo "Local testing" --limit 20

# 7. Verify everything
make health

# 8. Run the API
make serve
# Visit: http://localhost:8000/docs

# 9. (Optional, separate terminals) Run the other clients
uv run python src/ui/gradio_app.py    # web UI — http://localhost:7860
uv run python run_telegram_bot.py     # Telegram bot
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


## Running locally
```
uv sync
cp .env.example .env   # fill in your own keys
uv run uvicorn src.main:app --reload
uv run python src/ui/gradio_app.py   # in a second terminal, for the web UI
uv run python run_telegram_bot.py    # in a third terminal, for the bot
```
---


## Project Structure

```
arxiv-rag-system/
├── src/
│   ├── routers/          # ask, search, hybrid_search, admin, health
│   ├── services/
│   │   ├── agents/       # LangGraph nodes + workflow
│   │   ├── search/       # OpenSearch client, BM25, hybrid search
│   │   ├── embeddings/   # chunking, Jina client, vector indexing
│   │   ├── cache/        # semantic cache (Upstash)
│   │   ├── telegram/     # bot handlers
│   │   ├── monitoring/   # Langfuse client
│   │   └── invite_check.py
│   ├── ui/               # Gradio web interface
│   ├── models/           # Paper, Chunk, QueryLog, Invite
│   └── main.py
├── tests/
│   ├── unit/
│   └── integration/
├── alembic/versions/      # database migrations
├── airflow/               # Airflow project (built, not deployed)
├── scripts/               # operational + admin tooling
│   ├── ingest_arxiv.py
│   ├── generate_invite.py
│   ├── list_invites.py
│   ├── check_ingestion_status.py
│   ├── remove_category.py
│   └── clear_pending.py
├── run_telegram_bot.py    # entry point for the deployed Telegram worker
├── locustfile.py          # load testing
└── Dockerfile              # shared build for all 3 deployed services

```

---

## License

MIT
