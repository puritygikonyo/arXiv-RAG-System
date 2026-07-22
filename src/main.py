"""
arXiv RAG System — FastAPI application entry point.
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
from src.routers import health, search, hybrid_search, ask, admin

setup_logging()
logger = get_logger(__name__)
settings = get_settings()

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("application_starting", env=settings.app_env, version="0.1.0")

    from langfuse.decorators import langfuse_context

    langfuse_context.configure(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

    from src.services.search.client import close_opensearch, init_opensearch
    await init_opensearch()

    from src.services.embeddings.vector_indexer import ensure_chunks_index_exists
    ensure_chunks_index_exists()

    # Phase 3 will add: await init_db()
    # Phase 8 will add: await init_redis()

    logger.info("application_ready", host=settings.api_host, port=settings.api_port)
    yield

    logger.info("application_shutting_down")
    from src.services.monitoring.langfuse_client import flush_langfuse
    flush_langfuse()
    await close_opensearch()
    logger.info("application_shutdown")


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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(hybrid_search.router, prefix="/api/v1")
app.include_router(ask.router, prefix="/api/v1")
app.include_router(admin.router)  # admin.py already sets prefix="/api/v1/admin" internally — don't double it


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": "arXiv RAG System — visit /docs for API documentation"}

