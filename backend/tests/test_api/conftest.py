"""Shared fixtures for API tests.

Provides a TestClient with a fully initialised in-memory database and
mocked VectorStore / Embedder so tests do not hit real services.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from ai_craftsman_kb.llm import CompletionResult

import aiosqlite
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from ai_craftsman_kb.api.deps import get_conn
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
from ai_craftsman_kb.db.sqlite import SCHEMA_SQL, get_db, init_db
from ai_craftsman_kb.server import create_app


@pytest.fixture
def minimal_config() -> AppConfig:
    """Minimal valid AppConfig for API tests.

    Returns:
        A fully constructed AppConfig with test defaults.
    """
    return AppConfig(
        sources=SourcesConfig(
            hackernews=HackerNewsConfig(mode="top", limit=10),
        ),
        settings=SettingsConfig(
            data_dir="/tmp/test-craftsman-kb-api",
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


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock VectorStore that returns empty search results.

    Returns:
        A MagicMock with async search method and get_collection_info.
    """
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    store.get_collection_info = MagicMock(
        return_value={"vectors_count": 0, "disk_size_bytes": 0}
    )
    return store


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock Embedder that returns zero vectors.

    Returns:
        A MagicMock with async embed_single method.
    """
    embedder = MagicMock()
    embedder.embed_single = AsyncMock(return_value=[0.0] * 1536)
    return embedder


@pytest.fixture
def mock_llm_router() -> MagicMock:
    """Mock LLMRouter for tests.

    Returns:
        A MagicMock with async complete method.
    """
    router = MagicMock()
    router.complete = AsyncMock(return_value=CompletionResult(text="Test LLM response"))
    return router


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Temporary SQLite database path for testing.

    Args:
        tmp_path: pytest's built-in temporary directory.

    Returns:
        Path to craftsman.db in the temp directory.
    """
    return tmp_path / "craftsman.db"


@pytest.fixture
def api_client(
    minimal_config: AppConfig,
    mock_vector_store: MagicMock,
    mock_embedder: MagicMock,
    mock_llm_router: MagicMock,
    test_db_path: Path,
) -> TestClient:
    """Create a FastAPI TestClient with an in-memory SQLite database.

    Bypasses the lifespan startup by directly setting app state, then
    overrides the ``get_conn`` dependency to use the temporary test DB.

    Args:
        minimal_config: Minimal AppConfig for testing.
        mock_vector_store: Mocked VectorStore.
        mock_embedder: Mocked Embedder.
        mock_llm_router: Mocked LLMRouter.
        test_db_path: Path to the test DB file.

    Returns:
        A configured TestClient.
    """
    app = create_app()

    # Set app state directly (bypassing lifespan startup)
    app.state.config = minimal_config
    app.state.db_path = test_db_path
    app.state.vector_store = mock_vector_store
    app.state.embedder = mock_embedder
    app.state.llm_router = mock_llm_router

    # Initialise the test database schema synchronously
    asyncio.get_event_loop().run_until_complete(init_db(test_db_path.parent))

    # Override the DB dependency to use the test DB
    async def override_get_conn() -> AsyncGenerator[aiosqlite.Connection, None]:
        async with get_db(test_db_path.parent) as conn:
            yield conn

    app.dependency_overrides[get_conn] = override_get_conn

    return TestClient(app, raise_server_exceptions=True)
