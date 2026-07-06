"""
arXiv RAG System — FastAPI application entry point.

This file wires everything together:
  - Logging setup
  - Lifespan (startup / shutdown hooks)
  - Middleware (CORS, rate limiting)
  - Router registration

Run with:
    make serve
    # or directly:
    uv run uvicorn src.main:app --reload
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


# ── Rate limiter (shared across all routes) ───────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan — startup and shutdown logic ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Code before `yield` runs on startup.
    Code after `yield` runs on shutdown.

    We'll add DB pool init, OpenSearch client init, etc. in later phases.
    """
    logger.info(
        "application_starting",
        env=settings.app_env,
        version="0.1.0",
    )

    # ── Startup ──────────────────────────────────────────────────────────────
    # Phase 3 will add: await init_db()
    # Phase 5 will add: await init_opensearch()
    # Phase 8 will add: await init_redis()

    logger.info("application_ready", host=settings.api_host, port=settings.api_port)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    # Phase 3 will add: await close_db()
    # Phase 5 will add: await close_opensearch()
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

# ── Attach rate limiter ───────────────────────────────────────────────────────
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

# ── Routers (will be added as we build each phase) ────────────────────────────
# Phase 2: health router (already included below)
# Phase 3: from src.routers.papers import router as papers_router
# Phase 5: from src.routers.search import router as search_router
# Phase 6: from src.routers.hybrid_search import router as hybrid_router
# Phase 7: from src.routers.ask import router as ask_router
# Phase 7: from src.routers.agentic_ask import router as agentic_router

from src.routers.health import router as health_router  # noqa: E402

app.include_router(health_router, prefix="/api/v1")


# ── Root redirect ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Redirect root to API docs."""
    return {"message": "arXiv RAG System — visit /docs for API documentation"}
