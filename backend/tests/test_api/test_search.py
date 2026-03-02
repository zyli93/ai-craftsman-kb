"""Tests for the search endpoint."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from ai_craftsman_kb.db.models import DocumentRow
from ai_craftsman_kb.db.queries import upsert_document
from ai_craftsman_kb.db.sqlite import get_db


async def _insert_doc(test_db_path, doc: DocumentRow) -> None:
    async with get_db(test_db_path.parent) as conn:
        await upsert_document(conn, doc)


class TestSearchEndpoint:
    """Tests for GET /api/search."""

    def test_search_requires_query(self, api_client: TestClient) -> None:
        """GET /api/search without q parameter should return 422."""
        response = api_client.get("/api/search")
        assert response.status_code == 422

    def test_search_empty_query_returns_422(self, api_client: TestClient) -> None:
        """GET /api/search?q= with empty string should return 422."""
        response = api_client.get("/api/search?q=")
        assert response.status_code == 422

    def test_search_returns_list(self, api_client: TestClient) -> None:
        """GET /api/search?q=test should return a list (possibly empty)."""
        response = api_client.get("/api/search?q=test")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_search_keyword_mode(self, api_client: TestClient) -> None:
        """Keyword mode search should succeed and return list."""
        response = api_client.get("/api/search?q=test&mode=keyword")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_search_semantic_mode(self, api_client: TestClient) -> None:
        """Semantic mode search should succeed (mock embedder returns zero vector)."""
        response = api_client.get("/api/search?q=machine+learning&mode=semantic")
        assert response.status_code == 200

    def test_search_hybrid_mode(self, api_client: TestClient) -> None:
        """Hybrid mode is the default and should succeed."""
        response = api_client.get("/api/search?q=neural+networks&mode=hybrid")
        assert response.status_code == 200

    def test_search_invalid_mode_returns_422(self, api_client: TestClient) -> None:
        """An invalid mode value should return 422."""
        response = api_client.get("/api/search?q=test&mode=invalid")
        assert response.status_code == 422

    def test_search_limit_validation(self, api_client: TestClient) -> None:
        """limit > 100 should return 422."""
        response = api_client.get("/api/search?q=test&limit=200")
        assert response.status_code == 422

    def test_search_result_structure(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Search results should have document, score, and mode_used fields."""
        # Insert a document so FTS has something to match
        doc = DocumentRow(
            id="search-doc-1",
            source_type="hn",
            origin="pro",
            url="https://example.com/search-article",
            title="Machine Learning Article",
            raw_content="This article discusses machine learning and neural networks extensively.",
            fetched_at="2025-01-15T10:00:00",
        )
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, doc))

        response = api_client.get("/api/search?q=machine+learning&mode=keyword")
        assert response.status_code == 200
        results = response.json()

        if results:
            result = results[0]
            assert "document" in result
            assert "score" in result
            assert "mode_used" in result
            assert result["mode_used"] == "keyword"
            assert "id" in result["document"]
            assert "url" in result["document"]

    def test_search_source_type_filter(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Search with source_type filter should pass correctly."""
        response = api_client.get("/api/search?q=test&source_type=hn")
        assert response.status_code == 200
