"""Tests for the MCP server (task_31).

Verifies that:
- The MCP server can be created without errors.
- All 10 tools are registered on the server instance.
- Each tool has a correct docstring (used as tool description by Claude).
- Core tool logic works with a mocked service layer.

Notes on the MCP SDK call semantics tested here:
- ``FastMCP.call_tool(name, args)`` returns a ``list[TextContent]``.
  The text field of each TextContent item contains JSON-serialised tool output.
- When a tool raises ``ValueError`` or ``RuntimeError``, the SDK re-raises as
  ``mcp.server.fastmcp.exceptions.ToolError``.
- For direct tool invocation without the high-level wrapper, use
  ``mcp._tool_manager._tools[name].run(args, convert_result=False)``,
  which returns the raw return value or raises ToolError.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from ai_craftsman_kb.llm import CompletionResult

import pytest
import pytest_asyncio

from mcp.server.fastmcp.exceptions import ToolError

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
from ai_craftsman_kb.db.models import DocumentRow, EntityRow, SourceRow
from ai_craftsman_kb.db.sqlite import get_db, init_db
from ai_craftsman_kb.mcp_server import create_mcp_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(result: Any) -> Any:
    """Parse the return value from a call_tool() call to a Python object.

    FastMCP.call_tool() returns different shapes depending on the return type
    of the underlying tool function:

    - If the tool returns a **non-list** value (dict, str, etc.), call_tool()
      returns a plain ``list[TextContent]``. The text of the first item is JSON.
    - If the tool returns a **list** value, call_tool() returns a 2-tuple
      ``(list[TextContent], {'result': [...]})`` where the second element holds
      the complete list as-is (avoiding per-item TextContent parsing issues).

    This helper transparently handles both shapes and returns the parsed Python
    object (dict, list, str, etc.).

    Args:
        result: The raw return value from mcp.call_tool().

    Returns:
        The deserialized Python object from the tool's return value.
    """
    if isinstance(result, tuple):
        # Tool returned a list — use the 'result' key from the raw dict
        _content_list, raw = result
        return raw.get("result", [])
    # Tool returned a non-list (dict, str, etc.) — parse the first TextContent
    if not result:
        return None
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config(tmp_path: Path) -> AppConfig:
    """Minimal valid AppConfig for MCP server tests.

    Args:
        tmp_path: pytest built-in temporary directory.

    Returns:
        AppConfig instance with test defaults pointing to tmp_path.
    """
    return AppConfig(
        sources=SourcesConfig(
            hackernews=HackerNewsConfig(mode="top", limit=5),
        ),
        settings=SettingsConfig(
            data_dir=str(tmp_path),
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
    """Mock VectorStore returning empty results.

    Returns:
        MagicMock with async search and sync get_collection_info.
    """
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    store.get_collection_info = MagicMock(
        return_value={"vectors_count": 0, "disk_size_bytes": 0}
    )
    return store


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock Embedder returning zero vectors.

    Returns:
        MagicMock with async embed_single.
    """
    embedder = MagicMock()
    embedder.embed_single = AsyncMock(return_value=[0.0] * 1536)
    return embedder


@pytest.fixture
def mock_llm_router() -> MagicMock:
    """Mock LLMRouter returning a generic response.

    Returns:
        MagicMock with async complete.
    """
    router = MagicMock()
    router.complete = AsyncMock(return_value=CompletionResult(text="Test briefing content"))
    return router


# ---------------------------------------------------------------------------
# Test: server creation and tool registration
# ---------------------------------------------------------------------------


class TestCreateMCPServer:
    """Tests that create_mcp_server() returns a configured FastMCP instance."""

    def test_create_server_returns_fastmcp(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
    ) -> None:
        """create_mcp_server() should return a FastMCP instance without raising."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            from mcp.server.fastmcp import FastMCP
            mcp = create_mcp_server(minimal_config)
            assert isinstance(mcp, FastMCP)

    def test_server_name(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
    ) -> None:
        """The MCP server should be named 'ai-craftsman-kb'."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            assert mcp.name == "ai-craftsman-kb"

    async def test_all_tools_registered(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
    ) -> None:
        """All 10 expected MCP tools should be registered."""
        expected_tools = {
            "search",
            "radar",
            "ingest",
            "ingest_url",
            "briefing",
            "get_entities",
            "get_stats",
            "manage_source",
            "tag_document",
            "discover_sources",
        }
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            tools = await mcp.list_tools()
            registered_names = {t.name for t in tools}
            assert expected_tools == registered_names

    async def test_tools_have_descriptions(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
    ) -> None:
        """Each tool should have a non-empty description (from docstring)."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            tools = await mcp.list_tools()
            for tool in tools:
                assert tool.description, f"Tool '{tool.name}' has no description"
                assert len(tool.description) > 10, (
                    f"Tool '{tool.name}' description is too short: {tool.description!r}"
                )


# ---------------------------------------------------------------------------
# Test: search tool
# ---------------------------------------------------------------------------


class TestSearchTool:
    """Tests for the search() MCP tool."""

    async def test_search_returns_list(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """search() with a valid query should return a list (possibly empty)."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            result = await mcp.call_tool("search", {"query": "machine learning"})
            parsed = _parse_result(result)
            assert isinstance(parsed, list)

    async def test_search_empty_query_raises_tool_error(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """search() with an empty query should raise ToolError."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            with pytest.raises(ToolError, match="query must be non-empty"):
                await mcp.call_tool("search", {"query": ""})

    async def test_search_invalid_mode_raises_tool_error(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """search() with an invalid mode should raise ToolError."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            with pytest.raises(ToolError):
                await mcp.call_tool("search", {"query": "test", "mode": "bogus"})

    async def test_search_result_has_expected_fields(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """search() results (if any) should have the expected field set."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            # Insert a document so FTS has something to match
            from ai_craftsman_kb.db.queries import upsert_document
            doc = DocumentRow(
                id="mcp-search-doc-1",
                source_type="hn",
                origin="pro",
                url="https://example.com/mcp-article",
                title="Neural Networks Guide",
                raw_content="This guide covers neural networks and deep learning.",
            )
            async with get_db(tmp_path) as conn:
                await upsert_document(conn, doc)

            result = await mcp.call_tool(
                "search", {"query": "neural networks", "mode": "keyword"}
            )
            parsed = _parse_result(result)
            assert isinstance(parsed, list)

            # If results are returned, validate their shape
            if parsed:
                item = parsed[0]
                for field in ("id", "title", "url", "source_type", "score"):
                    assert field in item, f"Missing field '{field}' in search result"

    async def test_search_keyword_mode(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """search() with mode='keyword' should succeed."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            result = await mcp.call_tool("search", {"query": "test", "mode": "keyword"})
            parsed = _parse_result(result)
            assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# Test: get_stats tool
# ---------------------------------------------------------------------------


class TestGetStatsTool:
    """Tests for the get_stats() MCP tool."""

    async def test_get_stats_returns_expected_keys(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """get_stats() should return a dict with all expected keys."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            result = await mcp.call_tool("get_stats", {})
            parsed = _parse_result(result)

            assert "total_documents" in parsed
            assert "embedded_documents" in parsed
            assert "total_entities" in parsed
            assert "active_sources" in parsed
            assert "vector_count" in parsed

    async def test_get_stats_empty_db_returns_zeros(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """get_stats() on an empty DB should return zero counts."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            result = await mcp.call_tool("get_stats", {})
            parsed = _parse_result(result)

            assert parsed["total_documents"] == 0
            assert parsed["embedded_documents"] == 0
            assert parsed["total_entities"] == 0
            assert parsed["active_sources"] == 0


# ---------------------------------------------------------------------------
# Test: manage_source tool
# ---------------------------------------------------------------------------


class TestManageSourceTool:
    """Tests for the manage_source() MCP tool."""

    async def test_add_source_returns_info(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """manage_source(action='add') should create a new source and return its info."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            result = await mcp.call_tool(
                "manage_source",
                {
                    "action": "add",
                    "source_type": "hn",
                    "identifier": "top",
                    "display_name": "HN Top Stories",
                },
            )
            parsed = _parse_result(result)
            assert parsed["source_type"] == "hn"
            assert parsed["identifier"] == "top"
            assert parsed["enabled"] is True

    async def test_add_source_invalid_action_raises(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """manage_source() with an invalid action should raise ToolError."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            with pytest.raises(ToolError):
                await mcp.call_tool(
                    "manage_source",
                    {"action": "clone", "source_type": "hn", "identifier": "top"},
                )

    async def test_disable_and_enable_source(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """manage_source() disable then enable should toggle the enabled flag."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            feed_url = "https://example.com/feed"

            # Add a source first
            await mcp.call_tool(
                "manage_source",
                {"action": "add", "source_type": "rss", "identifier": feed_url},
            )

            # Disable it
            result = await mcp.call_tool(
                "manage_source",
                {"action": "disable", "source_type": "rss", "identifier": feed_url},
            )
            parsed = _parse_result(result)
            assert parsed["enabled"] is False

            # Re-enable it
            result2 = await mcp.call_tool(
                "manage_source",
                {"action": "enable", "source_type": "rss", "identifier": feed_url},
            )
            parsed2 = _parse_result(result2)
            assert parsed2["enabled"] is True

    async def test_remove_source(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """manage_source(action='remove') should delete the source."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            # Add then remove
            await mcp.call_tool(
                "manage_source",
                {"action": "add", "source_type": "devto", "identifier": "python"},
            )
            result = await mcp.call_tool(
                "manage_source",
                {"action": "remove", "source_type": "devto", "identifier": "python"},
            )
            parsed = _parse_result(result)
            assert parsed["source_type"] == "devto"
            # After remove, enabled is returned as False
            assert parsed["enabled"] is False


# ---------------------------------------------------------------------------
# Test: tag_document tool
# ---------------------------------------------------------------------------


class TestTagDocumentTool:
    """Tests for the tag_document() MCP tool."""

    async def test_add_tags_to_document(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """tag_document(action='add') should append tags to the document."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            from ai_craftsman_kb.db.queries import upsert_document
            doc = DocumentRow(
                id="tag-test-doc-1",
                source_type="hn",
                origin="pro",
                url="https://example.com/tag-test",
                title="Tag Test Article",
            )
            async with get_db(tmp_path) as conn:
                await upsert_document(conn, doc)

            result = await mcp.call_tool(
                "tag_document",
                {
                    "document_id": "tag-test-doc-1",
                    "tags": ["must-read", "rag"],
                    "action": "add",
                },
            )
            parsed = _parse_result(result)
            assert "must-read" in parsed["user_tags"]
            assert "rag" in parsed["user_tags"]

    async def test_set_tags_replaces_all(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """tag_document(action='set') should replace all existing tags."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            from ai_craftsman_kb.db.queries import upsert_document
            doc = DocumentRow(
                id="tag-test-doc-2",
                source_type="arxiv",
                origin="pro",
                url="https://arxiv.org/abs/9999.00002",
                title="Set Tag Test",
            )
            async with get_db(tmp_path) as conn:
                await upsert_document(conn, doc)

            # Add initial tags
            await mcp.call_tool(
                "tag_document",
                {"document_id": "tag-test-doc-2", "tags": ["old-tag"], "action": "add"},
            )

            # Replace with new set
            result = await mcp.call_tool(
                "tag_document",
                {"document_id": "tag-test-doc-2", "tags": ["new-tag"], "action": "set"},
            )
            parsed = _parse_result(result)
            assert parsed["user_tags"] == ["new-tag"]
            assert "old-tag" not in parsed["user_tags"]

    async def test_remove_tags(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """tag_document(action='remove') should remove only the specified tags."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            from ai_craftsman_kb.db.queries import upsert_document
            doc = DocumentRow(
                id="tag-test-doc-3",
                source_type="rss",
                origin="pro",
                url="https://example.com/remove-tag-test",
                title="Remove Tag Test",
            )
            async with get_db(tmp_path) as conn:
                await upsert_document(conn, doc)

            # Add two tags
            await mcp.call_tool(
                "tag_document",
                {
                    "document_id": "tag-test-doc-3",
                    "tags": ["keep-me", "remove-me"],
                    "action": "add",
                },
            )

            # Remove one tag
            result = await mcp.call_tool(
                "tag_document",
                {
                    "document_id": "tag-test-doc-3",
                    "tags": ["remove-me"],
                    "action": "remove",
                },
            )
            parsed = _parse_result(result)
            assert "keep-me" in parsed["user_tags"]
            assert "remove-me" not in parsed["user_tags"]

    async def test_tag_document_not_found_raises(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """tag_document() with a non-existent document_id should raise ToolError."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            with pytest.raises(ToolError):
                await mcp.call_tool(
                    "tag_document",
                    {"document_id": "nonexistent-uuid", "tags": ["test"], "action": "add"},
                )


# ---------------------------------------------------------------------------
# Test: get_entities tool
# ---------------------------------------------------------------------------


class TestGetEntitiesTool:
    """Tests for the get_entities() MCP tool."""

    async def test_get_entities_empty_db(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """get_entities() on an empty DB should return an empty list."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            result = await mcp.call_tool("get_entities", {})
            parsed = _parse_result(result)
            assert isinstance(parsed, list)
            assert len(parsed) == 0

    async def test_get_entities_invalid_type_raises(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """get_entities() with an invalid entity_type should raise ToolError."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            with pytest.raises(ToolError):
                await mcp.call_tool("get_entities", {"entity_type": "spaceship"})

    async def test_get_entities_returns_inserted_entities(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """get_entities() should return entities that have been stored in DB."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            from ai_craftsman_kb.db.queries import upsert_entity
            entity = EntityRow(
                id="entity-test-1",
                name="Geoffrey Hinton",
                entity_type="person",
                normalized_name="geoffrey hinton",
                mention_count=5,
            )
            async with get_db(tmp_path) as conn:
                await upsert_entity(conn, entity)

            result = await mcp.call_tool("get_entities", {"entity_type": "person"})
            parsed = _parse_result(result)
            assert len(parsed) >= 1
            assert parsed[0]["name"] == "Geoffrey Hinton"
            assert parsed[0]["entity_type"] == "person"
            assert "mention_count" in parsed[0]

    async def test_get_entities_field_shape(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """get_entities() results should have id, name, entity_type, mention_count."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            from ai_craftsman_kb.db.queries import upsert_entity
            entity = EntityRow(
                id="entity-test-2",
                name="OpenAI",
                entity_type="company",
                normalized_name="openai",
                mention_count=10,
            )
            async with get_db(tmp_path) as conn:
                await upsert_entity(conn, entity)

            result = await mcp.call_tool("get_entities", {})
            parsed = _parse_result(result)
            assert len(parsed) >= 1
            item = parsed[0]
            for field in ("id", "name", "entity_type", "mention_count"):
                assert field in item, f"Missing field '{field}' in entity result"


# ---------------------------------------------------------------------------
# Test: ingest_url tool (mocked runner)
# ---------------------------------------------------------------------------


class TestIngestUrlTool:
    """Tests for the ingest_url() MCP tool."""

    async def test_ingest_url_empty_raises_tool_error(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """ingest_url() with an empty URL should raise ToolError."""
        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            with pytest.raises(ToolError, match="url must be non-empty"):
                await mcp.call_tool("ingest_url", {"url": ""})

    async def test_ingest_url_returns_document_metadata(
        self,
        minimal_config: AppConfig,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        mock_llm_router: MagicMock,
        tmp_path: Path,
    ) -> None:
        """ingest_url() should return the ingested document's id, title, url, source_type."""
        from ai_craftsman_kb.ingestors.runner import IngestReport

        mock_report = IngestReport(
            source_type="adhoc",
            fetched=1,
            stored=1,
            embedded=0,
        )

        with (
            patch("ai_craftsman_kb.mcp_server.VectorStore", return_value=mock_vector_store),
            patch("ai_craftsman_kb.mcp_server.Embedder", return_value=mock_embedder),
            patch("ai_craftsman_kb.mcp_server.LLMRouter", return_value=mock_llm_router),
        ):
            mcp = create_mcp_server(minimal_config)
            await init_db(tmp_path)

            # Pre-insert the document so get_document_by_url finds it after ingest
            from ai_craftsman_kb.db.queries import upsert_document
            test_url = "https://example.com/ingest-me"
            doc = DocumentRow(
                id="adhoc-ingest-1",
                source_type="adhoc",
                origin="adhoc",
                url=test_url,
                title="Test Ingest Article",
            )
            async with get_db(tmp_path) as conn:
                await upsert_document(conn, doc)

            # Mock IngestRunner to avoid real network calls
            with patch("ai_craftsman_kb.mcp_server.IngestRunner") as mock_runner_cls:
                mock_runner = MagicMock()
                mock_runner.ingest_url = AsyncMock(return_value=mock_report)
                mock_runner_cls.return_value = mock_runner

                result = await mcp.call_tool("ingest_url", {"url": test_url})

            parsed = _parse_result(result)
            assert parsed["url"] == test_url
            assert "id" in parsed
            assert "source_type" in parsed
            assert "title" in parsed


# ---------------------------------------------------------------------------
# Test: run_mcp_server function
# ---------------------------------------------------------------------------


class TestRunMCPServer:
    """Tests that run_mcp_server is importable and callable."""

    def test_run_mcp_server_importable(self) -> None:
        """run_mcp_server should be importable from the mcp_server module."""
        from ai_craftsman_kb.mcp_server import run_mcp_server
        assert callable(run_mcp_server)

    def test_create_mcp_server_importable(self) -> None:
        """create_mcp_server should be importable from the mcp_server module."""
        from ai_craftsman_kb.mcp_server import create_mcp_server
        assert callable(create_mcp_server)
