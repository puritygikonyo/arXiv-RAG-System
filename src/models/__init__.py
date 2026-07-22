"""
SQLAlchemy ORM models.

Paper  — one row per arXiv paper. Source of truth for metadata and
         pipeline ingestion state.
Chunk  — one row per text chunk of a paper. Holds the chunk text itself;
         the actual embedding vector lives in OpenSearch (opensearch_doc_id
         links the two together).
"""

import enum
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class IngestionStatus(str, enum.Enum):
    pending = "pending"
    fetched = "fetched"
    chunked = "chunked"
    embedded = "embedded"
    indexed = "indexed"
    failed = "failed"


class Paper(Base):
    __tablename__ = "papers"

    # arXiv's own ID as primary key (e.g. "2301.12345") — natural key,
    # avoids needing a separate surrogate key + unique constraint.
    arxiv_id: Mapped[str] = mapped_column(String(32), primary_key=True)

    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    categories: Mapped[list[str]] = mapped_column(JSON, default=list)
    primary_category: Mapped[str] = mapped_column(String(32), nullable=False)

    published_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pdf_url: Mapped[str] = mapped_column(String(512), nullable=False)
    doi: Mapped[str | None] = mapped_column(String(128), nullable=True)

    ingestion_status: Mapped[IngestionStatus] = mapped_column(
        Enum(IngestionStatus), default=IngestionStatus.pending, nullable=False
    )
    ingestion_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Paper {self.arxiv_id!r} status={self.ingestion_status.value}>"


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("papers.arxiv_id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Links to the corresponding document in OpenSearch once indexed.
    # Null until the Phase 5/6 pipeline embeds + indexes this chunk.
    opensearch_doc_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    paper: Mapped["Paper"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk paper={self.paper_id!r} index={self.chunk_index}>"
    
class QueryLog(Base):
    """
    One row per /api/v1/ask request. Powers the Phase 8 metrics endpoint
    (cache hit rate, avg latency, top queries) — data Langfuse's own
    dashboard doesn't aggregate natively. Deliberately minimal: this is
    a lightweight analytics log, not a replacement for Langfuse's
    detailed per-node tracing.
    """
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # answered | off_topic | no_relevant_docs | error

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<QueryLog id={self.id} cache_hit={self.cache_hit} latency_ms={self.latency_ms}>"