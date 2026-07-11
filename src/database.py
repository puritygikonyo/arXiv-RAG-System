"""
Async SQLAlchemy engine and session management.

Usage (in a route or service):
    from src.database import get_db

    async def some_endpoint(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Paper).where(...))
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.config import get_settings

settings = get_settings()

# ── Base class for all ORM models ───────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Engine ───────────────────────────────────────────────────────────────────
# echo=True logs all SQL — useful in dev, noisy in prod.
#
# connect_args={"ssl": "require"} — Neon requires SSL. Note: we deliberately
# connect via the hostname (not a raw IP) because Neon's proxy relies on SNI
# (the hostname sent during the TLS handshake) to route the connection to
# the correct project endpoint — connecting via IP breaks that routing.
engine = create_async_engine(
    settings.database_url,
    echo=settings.is_development,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_pre_ping=True,  # detects stale connections (important for Neon's
                         # serverless auto-suspend — avoids "connection closed" errors)
    connect_args={"ssl": "require"},
)

# ── Session factory ──────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a DB session, closes it after the request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def ping_database() -> tuple[bool, str]:
    """
    Lightweight connectivity check for the health endpoint.
    Returns (is_ok, detail_message).
    """
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, "connected"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}" if str(e) else type(e).__name__