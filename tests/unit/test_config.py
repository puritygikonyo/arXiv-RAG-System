"""
Unit tests for application configuration.

These tests verify that:
  - Settings load correctly from environment variables
  - Validators correctly parse comma-separated lists
  - Computed properties return expected values
  - Settings are cached properly

Marked as `unit` — no external services needed.
"""

import pytest

from src.config import Settings, get_settings


@pytest.mark.unit
class TestSettings:
    """Test Settings model validation and defaults."""

    def test_default_app_env_is_development(self) -> None:
        settings = Settings()
        assert settings.app_env == "development"

    def test_is_development_property(self) -> None:
        settings = Settings(app_env="development")
        assert settings.is_development is True
        assert settings.is_production is False

    def test_is_production_property(self) -> None:
        settings = Settings(app_env="production")
        assert settings.is_production is True
        assert settings.is_development is False

    def test_cors_origins_parsed_from_string(self) -> None:
        settings = Settings(cors_origins="http://localhost:3000,http://localhost:7860")
        assert settings.cors_origins == ["http://localhost:3000", "http://localhost:7860"]

    def test_cors_origins_from_list(self) -> None:
        settings = Settings(cors_origins=["http://localhost:3000"])
        assert settings.cors_origins == ["http://localhost:3000"]

    def test_arxiv_categories_parsed_from_string(self) -> None:
        settings = Settings(arxiv_categories="cs.AI,cs.LG,cs.CL")
        assert settings.arxiv_categories == ["cs.AI", "cs.LG", "cs.CL"]

    def test_telegram_allowed_chat_ids_empty_string(self) -> None:
        settings = Settings(telegram_allowed_chat_ids="")
        assert settings.telegram_allowed_chat_ids == []

    def test_telegram_allowed_chat_ids_parsed(self) -> None:
        settings = Settings(telegram_allowed_chat_ids="123456,789012")
        assert settings.telegram_allowed_chat_ids == [123456, 789012]

    def test_opensearch_url_no_ssl(self) -> None:
        settings = Settings(
            opensearch_host="localhost",
            opensearch_port=9200,
            opensearch_use_ssl=False,
        )
        assert settings.opensearch_url == "http://localhost:9200"

    def test_opensearch_url_with_ssl(self) -> None:
        settings = Settings(
            opensearch_host="my-cluster.aws.com",
            opensearch_port=443,
            opensearch_use_ssl=True,
        )
        assert settings.opensearch_url == "https://my-cluster.aws.com:443"

    def test_invalid_app_env_raises(self) -> None:
        with pytest.raises(Exception):
            Settings(app_env="invalid_env")  # type: ignore[arg-type]

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(Exception):
            Settings(log_level="VERBOSE")  # type: ignore[arg-type]


@pytest.mark.unit
class TestGetSettings:
    """Test the cached get_settings() factory."""

    def setup_method(self) -> None:
        """Clear the settings cache before each test."""
        get_settings.cache_clear()

    def test_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_cached_same_instance(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # same object due to lru_cache
