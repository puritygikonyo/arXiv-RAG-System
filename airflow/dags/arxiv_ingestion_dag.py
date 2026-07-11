"""
arXiv paper ingestion DAG.

Fetches recent papers for each configured category from arXiv's API
and upserts them into the `papers` table, tagging new rows with
ingestion_status='fetched' — ready for Phase 5/6 to pick up for
chunking, embedding, and indexing.

Runs daily by default. Adjust the schedule as needed.
"""

import os
from datetime import timedelta

import pendulum
from airflow.decorators import dag, task

from include.arxiv_client import fetch_papers_for_category
from include.db_upsert import upsert_papers

# ── Config (from Airflow project's .env — see setup notes) ─────────────────
ARXIV_API_BASE_URL = os.environ.get(
    "ARXIV_API_BASE_URL", "http://export.arxiv.org/api/query"
)
# Comma-separated in this .env, kept simple — not JSON, unlike the FastAPI
# app's .env, since this is a completely separate config surface.
ARXIV_CATEGORIES = os.environ.get(
    "ARXIV_CATEGORIES", "cs.AI,cs.LG,cs.CL,cs.IR"
).split(",")
ARXIV_MAX_RESULTS_PER_RUN = int(os.environ.get("ARXIV_MAX_RESULTS_PER_RUN", "100"))


@dag(
    dag_id="arxiv_paper_ingestion",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["arxiv", "ingestion", "phase-4"],
)
def arxiv_paper_ingestion():

    @task
    def fetch_papers() -> list[dict]:
        """Fetch recent papers for every configured category."""
        all_papers: dict[str, dict] = {}
        for category in ARXIV_CATEGORIES:
            category = category.strip()
            papers = fetch_papers_for_category(
                base_url=ARXIV_API_BASE_URL,
                category=category,
                max_results=ARXIV_MAX_RESULTS_PER_RUN,
            )
            # De-duplicate across categories (a paper can be cross-listed
            # in multiple categories and would otherwise show up twice).
            for paper in papers:
                all_papers[paper["arxiv_id"]] = paper

        return list(all_papers.values())

    @task
    def store_papers(papers: list[dict]) -> dict:
        """Upsert fetched papers into the papers table."""
        return upsert_papers(papers)

    papers = fetch_papers()
    store_papers(papers)


arxiv_paper_ingestion()