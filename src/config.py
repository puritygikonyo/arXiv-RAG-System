"""
Central application configuration.

All settings are loaded from environment variables (or .env file).
Never import os.environ directly anywhere else — always use `get_settings()`.

Usage:
    from src.config import get_settings
    settings = get_settings()
    print(settings.database_url)
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic validates types automatically — you get an error on startup
    if a required variable is missing or has the wrong type.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars
    )

    # ── Phase 1: App Core ────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    secret_key: str = Field(default="changeme", min_length=8)
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:7860"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Allow comma-separated string from env var."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # ── Phase 2: FastAPI ─────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_per_minute: int = 60

    # ── Phase 3: PostgreSQL ───────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/arxiv_rag"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30

    # ── Phase 4: arXiv Pipeline ───────────────────────────────────────────────
    arxiv_api_base_url: str = "http://export.arxiv.org/api/query"
    arxiv_categories: list[str] = ["cs.AI", "cs.LG", "cs.CL", "cs.IR"]
    arxiv_max_results_per_run: int = 100

    @field_validator("arxiv_categories", mode="before")
    @classmethod
    def parse_categories(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [cat.strip() for cat in v.split(",")]
        return v

    # ── Phase 5 & 6: OpenSearch ───────────────────────────────────────────────
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_user: str = "admin"
    opensearch_password: str = "admin"
    opensearch_use_ssl: bool = False
    opensearch_papers_index: str = "arxiv_papers"
    opensearch_chunks_index: str = "arxiv_chunks"
    search_max_results: int = 20
    search_min_score: float = 0.1
    hybrid_search_vector_weight: float = 0.5

    # ── Phase 6: Embeddings ───────────────────────────────────────────────────
    jina_api_key: str = ""
    jina_embedding_model: str = "jina-embeddings-v3"
    jina_embedding_dimensions: int = 1024
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ── Phase 7: LLM ─────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_streaming: bool = True
    agent_max_retrieval_attempts: int = 3
    agent_relevance_threshold: float = 0.7

    # ── Phase 8: Redis Cache ──────────────────────────────────────────────────
   # ── Phase 8: Redis Cache (Upstash REST API) ────────────────────────────────
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    cache_ttl_seconds: int = 60 * 60 * 24 * 7  # 7 days
    cache_similarity_threshold: float = 0.80

    # admin api key
    admin_api_key: str = ""

    # ── Phase 8: Langfuse Monitoring ──────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Phase 9: Telegram ─────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_allowed_chat_ids: list[int] = []

    @field_validator("telegram_allowed_chat_ids", mode="before")
    @classmethod
    def parse_chat_ids(cls, v: str | list[int] | int) -> list[int]:
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            if not v.strip():
                return []
            return [int(cid.strip()) for cid in v.split(",")]
        return v

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def opensearch_url(self) -> str:
        scheme = "https" if self.opensearch_use_ssl else "http"
        return f"{scheme}://{self.opensearch_host}:{self.opensearch_port}"

from functools import lru_cache


@lru_cache
def get_settings() -> Settings:
    """
    Return cached Settings instance.

    Using lru_cache means settings are only parsed once per process.
    In tests, call get_settings.cache_clear() to reset between tests.
    """
    return Settings()

