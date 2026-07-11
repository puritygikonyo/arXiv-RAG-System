"""
Database helper — upserts fetched arXiv papers into the `papers` table.

Uses plain psycopg2 (sync) rather than the app's async SQLAlchemy setup,
since Airflow tasks run synchronously and this DAG runs in its own
container, decoupled from the FastAPI app's src/ code.
"""

import logging
import os

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


def get_connection():
    """
    Build a psycopg2 connection from the DATABASE_URL_SYNC env var.

    Set this in your Airflow project's .env as the SYNC variant of your
    Neon connection string, e.g.:
        DATABASE_URL_SYNC=postgresql://user:pass@host/dbname?sslmode=require

    (Note: psycopg2 uses plain "postgresql://" with sslmode as a query
    param — unlike the app's asyncpg URL, which uses "postgresql+asyncpg://"
    and passes ssl via connect_args instead of a URL query param.)
    """
    database_url = os.environ["DATABASE_URL_SYNC"]
    return psycopg2.connect(database_url)


def upsert_papers(papers: list[dict]) -> dict:
    """
    Insert papers into the `papers` table, skipping ones that already exist
    (matched by arxiv_id). Newly inserted papers get ingestion_status='fetched'.

    Returns a summary dict: {"inserted": N, "skipped": N}.
    """
    if not papers:
        return {"inserted": 0, "skipped": 0}

    insert_sql = """
        INSERT INTO papers (
            arxiv_id, title, abstract, authors, categories,
            primary_category, published_date, updated_date,
            pdf_url, doi, ingestion_status
        )
        VALUES (
            %(arxiv_id)s, %(title)s, %(abstract)s,
            %(authors)s, %(categories)s,
            %(primary_category)s, %(published_date)s, %(updated_date)s,
            %(pdf_url)s, %(doi)s, 'fetched'
        )
        ON CONFLICT (arxiv_id) DO NOTHING
        RETURNING arxiv_id;
    """

    inserted = 0
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for paper in papers:
                    params = dict(paper)
                    # authors/categories are JSON columns — wrap Python
                    # lists so psycopg2 adapts them correctly.
                    params["authors"] = psycopg2.extras.Json(params["authors"])
                    params["categories"] = psycopg2.extras.Json(params["categories"])
                    cur.execute(insert_sql, params)
                    if cur.fetchone() is not None:
                        inserted += 1
    finally:
        conn.close()

    skipped = len(papers) - inserted
    logger.info("Upsert complete", extra={"inserted": inserted, "skipped": skipped})
    return {"inserted": inserted, "skipped": skipped}