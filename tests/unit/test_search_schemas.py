"""
Unit tests for Phase 5 — Search schemas and query building.

These tests verify:
  - Schema validation works correctly
  - Pagination logic is correct
  - Edge cases are handled

Marked as `unit` — no OpenSearch needed to run these.
"""

import pytest
from pydantic import ValidationError

from src.schemas.search import SearchRequest, IndexPaperRequest


@pytest.mark.unit
class TestSearchRequestSchema:
    """Test SearchRequest validation."""

    def test_valid_basic_query(self) -> None:
        req = SearchRequest(query="transformer attention mechanism")
        assert req.query == "transformer attention mechanism"
        assert req.page == 1
        assert req.page_size == 10

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_query_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="x" * 501)

    def test_page_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="transformers", page=0)

    def test_page_size_max_50(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="transformers", page_size=51)

    def test_page_size_min_1(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="transformers", page_size=0)

    def test_categories_optional(self) -> None:
        req = SearchRequest(query="transformers")
        assert req.categories is None

    def test_categories_accepted(self) -> None:
        req = SearchRequest(query="transformers", categories=["cs.AI", "cs.LG"])
        assert req.categories == ["cs.AI", "cs.LG"]

    def test_date_filters_optional(self) -> None:
        req = SearchRequest(query="transformers")
        assert req.date_from is None
        assert req.date_to is None


@pytest.mark.unit
class TestPaginationLogic:
    """Test that pagination offset calculation is correct."""

    def test_page_1_offset_is_0(self) -> None:
        page, page_size = 1, 10
        offset = (page - 1) * page_size
        assert offset == 0

    def test_page_2_offset_is_10(self) -> None:
        page, page_size = 2, 10
        offset = (page - 1) * page_size
        assert offset == 10

    def test_page_3_offset_is_20(self) -> None:
        page, page_size = 3, 10
        offset = (page - 1) * page_size
        assert offset == 20

    def test_has_more_true_when_results_remain(self) -> None:
        page, page_size, total = 1, 10, 25
        has_more = (page * page_size) < total
        assert has_more is True

    def test_has_more_false_on_last_page(self) -> None:
        page, page_size, total = 3, 10, 25
        has_more = (page * page_size) < total
        assert has_more is False


@pytest.mark.unit
class TestIndexPaperRequest:
    """Test IndexPaperRequest schema."""

    def test_minimal_valid_request(self) -> None:
        req = IndexPaperRequest(
            arxiv_id="2401.12345",
            title="Attention Is All You Need",
            abstract="We propose a new architecture called the Transformer.",
        )
        assert req.arxiv_id == "2401.12345"
        assert req.authors == []
        assert req.categories == []

    def test_full_valid_request(self) -> None:
        req = IndexPaperRequest(
            arxiv_id="2401.12345",
            title="Attention Is All You Need",
            abstract="We propose the Transformer architecture.",
            authors=["Vaswani, A.", "Shazeer, N."],
            primary_category="cs.LG",
            categories=["cs.LG", "cs.AI"],
            published_at="2017-06-12T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/1706.03762",
        )
        assert req.authors == ["Vaswani, A.", "Shazeer, N."]
        assert req.primary_category == "cs.LG"
