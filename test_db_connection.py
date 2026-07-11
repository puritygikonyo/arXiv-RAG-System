"""
Minimal standalone connectivity test — talks to asyncpg directly,
with no SQLAlchemy or Alembic in the picture at all.

Fill in your real password below, then run:
    uv run python test_db_connection.py
"""

import asyncio

import asyncpg


async def main():
    conn = await asyncpg.connect(
        host="ep-red-morning-at886i26.c-9.us-east-1.aws.neon.tech",
        port=5432,
        user="neondb_owner",
        password="npg_T3XA7slriRvW",
        database="neondb",
        ssl="require",
    )
    result = await conn.fetchval("SELECT 1")
    print("SUCCESS:", result)
    await conn.close()


asyncio.run(main())