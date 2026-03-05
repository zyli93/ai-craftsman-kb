"""Shared pytest fixtures for AI Craftsman KB tests."""
import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    EmbeddingConfig,
    FiltersConfig,
    HackerNewsConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    SettingsConfig,
    SourcesConfig,
)


@pytest.fixture
def minimal_config() -> AppConfig:
    """Minimal valid AppConfig for testing — no real API keys needed.

    All LLM task configs point to a fake 'test' provider/model so that
    tests which instantiate LLMRouter do not attempt real network calls.
    The data_dir is set to a temporary path that tests can override via
    their own tmp_path fixtures.

    Returns:
        A fully constructed AppConfig with sensible test defaults.
    """
    return AppConfig(
        sources=SourcesConfig(
            hackernews=HackerNewsConfig(mode="top", limit=10),
        ),
        settings=SettingsConfig(
            data_dir="/tmp/test-craftsman-kb",
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test-model"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test-model"),
                briefing=LLMTaskConfig(provider="anthropic", model="test-model"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test-model"),
                keyword_extraction=LLMTaskConfig(provider="openrouter", model="test-model"),
            ),
        ),
        filters=FiltersConfig(),
    )
