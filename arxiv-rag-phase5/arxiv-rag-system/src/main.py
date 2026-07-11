"""
arXiv RAG System — FastAPI application entry point.

This file wires everything together:
  - Logging setup
  - Lifespan (startup / shutdown hooks)
  - Middleware (CORS, rate limiting)
  - Router registration

Run with:
    uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import get_settings
from src.logger import get_logger, setup_logging

# ── Initialise logger before anything else ────────────────────────────────────
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan — startup and shutdown logic ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: runs before the app accepts any requests.
    Shutdown: runs after the app stops accepting requests.

    ORDER MATTERS:
      Start up in dependency order:
        1. Database first (other services may need it)
        2. OpenSearch second
        3. Redis last (optional cache)

      Shut down in reverse order.
    """
    logger.info("application_starting", env=settings.app_env, version="0.1.0")

    # ── Phase 3 will add ──────────────────────────────────────────────────────
    # from src.db.session import init_db, close_db
    # await init_db()

    # ── Phase 5: OpenSearch ───────────────────────────────────────────────────
    from src.services.search.client import close_opensearch, init_opensearch
    await init_opensearch()

    # ── Phase 8 will add ──────────────────────────────────────────────────────
    # from src.services.cache.redis import init_redis, close_redis
    # await init_redis()

    logger.info("application_ready", host=settings.api_host, port=settings.api_port)

    yield  # App is running and accepting requests

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("application_shutting_down")
    await close_opensearch()
    # Phase 8 will add: await close_redis()
    # Phase 3 will add: await close_db()
    logger.info("application_shutdown")


# ── Create FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="arXiv RAG System",
    description=(
        "Production-grade Agentic RAG system for academic paper research. "
        "Hybrid search (BM25 + vector) powered by LangGraph agents."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── Rate limiter ──────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from src.routers.health import router as health_router    # noqa: E402
from src.routers.search import router as search_router    # noqa: E402

app.include_router(health_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")

# Phase 3 will add:  from src.routers.papers import router as papers_router
# Phase 6 will add:  from src.routers.hybrid_search import router as hybrid_router
# Phase 7 will add:  from src.routers.ask import router as ask_router


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": "arXiv RAG System — visit /docs for API documentation"}
