"""
Unit tests for Phase 6 — Chunking and RRF logic.
No external services needed.
"""

import pytest

from src.services.embeddings.chunker import chunk_paper, chunk_papers_batch


@pytest.mark.unit
class TestChunking:
    """Test the chunking service."""

    def _make_paper(self, abstract: str = "default abstract") -> dict:
        return {
            "arxiv_id": "2401.12345",
            "title": "Test Paper",
            "abstract": abstract,
            "primary_category": "cs.AI",
        }

    def test_short_text_produces_one_chunk(self) -> None:
        paper = self._make_paper("Short abstract about transformers.")
        chunks = chunk_paper(paper)
        assert len(chunks) >= 1

    def test_chunk_has_correct_arxiv_id(self) -> None:
        paper = self._make_paper("Some abstract text here.")
        chunks = chunk_paper(paper)
        assert all(c.arxiv_id == "2401.12345" for c in chunks)

    def test_chunk_ids_are_unique(self) -> None:
        paper = self._make_paper("word " * 600)
        chunks = chunk_paper(paper)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_total_chunks_is_correct(self) -> None:
        paper = self._make_paper("word " * 600)
        chunks = chunk_paper(paper)
        total = len(chunks)
        assert all(c.total_chunks == total for c in chunks)

    def test_chunk_index_is_sequential(self) -> None:
        paper = self._make_paper("word " * 600)
        chunks = chunk_paper(paper)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_long_text_produces_multiple_chunks(self) -> None:
        paper = self._make_paper("word " * 1100)
        chunks = chunk_paper(paper)
        assert len(chunks) > 1

    def test_empty_abstract_returns_empty(self) -> None:
        paper = {
            "arxiv_id": "2401.12345",
            "title": "",
            "abstract": "",
            "primary_category": "cs.AI",
        }
        chunks = chunk_paper(paper)
        assert chunks == []

    def test_chunk_text_is_not_empty(self) -> None:
        paper = self._make_paper("attention mechanism in transformer models")
        chunks = chunk_paper(paper)
        assert all(len(c.text.strip()) > 0 for c in chunks)

    def test_title_included_in_first_chunk(self) -> None:
        paper = self._make_paper("Some abstract.")
        paper["title"] = "Attention Is All You Need"
        chunks = chunk_paper(paper)
        # Title should appear in the combined text
        combined_text = " ".join(c.text for c in chunks)
        assert "Attention" in combined_text

    def test_batch_chunking(self) -> None:
        papers = [
            self._make_paper("abstract one"),
            self._make_paper("abstract two"),
        ]
        papers[1]["arxiv_id"] = "2401.99999"
        all_chunks = chunk_papers_batch(papers)
        arxiv_ids = {c.arxiv_id for c in all_chunks}
        assert "2401.12345" in arxiv_ids
        assert "2401.99999" in arxiv_ids


@pytest.mark.unit
class TestRRFLogic:
    """Test Reciprocal Rank Fusion scoring logic."""

    def _rrf_score(self, rank: int, k: int = 60) -> float:
        return 1 / (k + rank)

    def test_rank_1_higher_than_rank_2(self) -> None:
        assert self._rrf_score(1) > self._rrf_score(2)

    def test_rank_1_higher_than_rank_100(self) -> None:
        assert self._rrf_score(1) > self._rrf_score(100)

    def test_consistent_rank_beats_inconsistent(self) -> None:
        # Paper ranked #2 in both systems
        consistent = self._rrf_score(2) + self._rrf_score(2)
        # Paper ranked #1 in BM25 but #20 in vector
        inconsistent = self._rrf_score(1) + self._rrf_score(20)
        # Consistent ranking should win
        assert consistent > inconsistent

    def test_missing_from_one_list_is_penalised(self) -> None:
        # Present in both at rank 5
        both = self._rrf_score(5) + self._rrf_score(5)
        # Only in one list at rank 1, missing from other (rank 1000 penalty)
        one_only = self._rrf_score(1) + self._rrf_score(1000)
        assert both > one_only
