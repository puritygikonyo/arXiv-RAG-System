"""
arXiv RAG System — FastAPI application entry point.
Phase 6 update: adds hybrid search router and chunks index initialisation.
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

setup_logging()
logger = get_logger(__name__)
settings = get_settings()

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("application_starting", env=settings.app_env, version="0.1.0")

    # Phase 5: OpenSearch
    from src.services.search.client import close_opensearch, init_opensearch
    await init_opensearch()

    # Phase 6: ensure chunks index exists
    from src.services.embeddings.vector_indexer import ensure_chunks_index_exists
    ensure_chunks_index_exists()

    # Phase 3 will add: await init_db()
    # Phase 8 will add: await init_redis()

    logger.info("application_ready", host=settings.api_host, port=settings.api_port)
    yield

    logger.info("application_shutting_down")
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

from src.routers.health import router as health_router          # noqa: E402
from src.routers.search import router as search_router          # noqa: E402
from src.routers.hybrid_search import router as hybrid_router   # noqa: E402
from src.routers.ask import router as ask_router                # noqa: E402

app.include_router(health_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(hybrid_router, prefix="/api/v1")
app.include_router(ask_router, prefix="/api/v1")




@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": "arXiv RAG System — visit /docs for API documentation"}
