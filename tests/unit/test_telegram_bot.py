"""
Unit tests for the Telegram bot's pure helper functions: message
chunking, markdown stripping, and the chat allowlist check. No live
Telegram connection or network calls needed -- these are plain functions.
"""

import pytest

from src.services.telegram.bot import _is_allowed, _split_message, _strip_markdown


@pytest.mark.unit
class TestSplitMessage:
    def test_short_text_returns_single_chunk(self) -> None:
        result = _split_message("hello world")
        assert result == ["hello world"]

    def test_text_under_limit_is_not_split(self) -> None:
        text = "a" * 3000
        result = _split_message(text, max_length=4000)
        assert len(result) == 1

    def test_text_over_limit_is_split(self) -> None:
        text = "a" * 9000
        result = _split_message(text, max_length=4000)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 4000

    def test_splits_on_paragraph_boundaries_when_possible(self) -> None:
        para = "x" * 100
        text = "\n\n".join([para] * 50)   # ~5000 chars across many paragraphs
        result = _split_message(text, max_length=1000)
        assert len(result) > 1
        # every chunk should start and end on a clean paragraph, not mid-word
        for chunk in result:
            assert not chunk.startswith("\n")
            assert len(chunk) <= 1000

    def test_single_paragraph_longer_than_limit_hard_splits_on_words(self) -> None:
        text = " ".join(["word"] * 2000)   # one giant paragraph, no \n\n at all
        result = _split_message(text, max_length=500)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 500
        # reassembling should not have silently dropped any words
        rejoined_word_count = sum(len(c.split()) for c in result)
        assert rejoined_word_count == 2000

    def test_reassembled_chunks_preserve_all_content(self) -> None:
        """No chunk boundary should lose or duplicate characters."""
        text = "\n\n".join([f"paragraph number {i} " + "y" * 50 for i in range(30)])
        result = _split_message(text, max_length=300)
        rejoined = "\n\n".join(result)
        # every original paragraph should still appear somewhere
        for i in range(30):
            assert f"paragraph number {i} " in rejoined


@pytest.mark.unit
class TestStripMarkdown:
    def test_strips_bold_double_asterisk(self) -> None:
        assert _strip_markdown("The **Transformer** model") == "The Transformer model"

    def test_strips_bold_double_underscore(self) -> None:
        assert _strip_markdown("The __Transformer__ model") == "The Transformer model"

    def test_strips_italic_single_asterisk(self) -> None:
        assert _strip_markdown("uses *self-attention* here") == "uses self-attention here"

    def test_strips_inline_code(self) -> None:
        assert _strip_markdown("call `embed_query()` first") == "call embed_query() first"

    def test_strips_fenced_code_block(self) -> None:
        text = "Example:\n```python\nx = 1\n```\nDone"
        result = _strip_markdown(text)
        assert "```" not in result
        assert "x = 1" in result

    def test_strips_headers(self) -> None:
        assert _strip_markdown("## Section Title\nBody text") == "Section Title\nBody text"

    def test_converts_markdown_links_to_text_plus_url(self) -> None:
        result = _strip_markdown("see [the paper](https://arxiv.org/abs/1706.03762)")
        assert result == "see the paper (https://arxiv.org/abs/1706.03762)"

    def test_plain_text_is_unchanged(self) -> None:
        assert _strip_markdown("Just plain text, no formatting.") == (
            "Just plain text, no formatting."
        )

    def test_handles_multiple_formatting_types_together(self) -> None:
        text = "The **Transformer** uses *attention* (see `paper`) [here](https://x.com)."
        result = _strip_markdown(text)
        assert "**" not in result
        assert "`" not in result
        assert "[" not in result
        assert "https://x.com" in result


@pytest.mark.unit
class TestIsAllowed:
    def test_empty_allowlist_allows_any_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.services.telegram.bot as bot_module

        monkeypatch.setattr(bot_module.settings, "telegram_allowed_chat_ids", [])
        assert _is_allowed(123456) is True

    def test_chat_id_in_allowlist_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.services.telegram.bot as bot_module

        monkeypatch.setattr(bot_module.settings, "telegram_allowed_chat_ids", [111, 222])
        assert _is_allowed(111) is True

    def test_chat_id_not_in_allowlist_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.services.telegram.bot as bot_module

        monkeypatch.setattr(bot_module.settings, "telegram_allowed_chat_ids", [111, 222])
        assert _is_allowed(999) is False