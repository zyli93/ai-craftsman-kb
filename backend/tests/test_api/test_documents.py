"""Tests for document CRUD endpoints."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from ai_craftsman_kb.db.models import DocumentRow
from ai_craftsman_kb.db.queries import upsert_document
from ai_craftsman_kb.db.sqlite import get_db


def _create_test_doc(
    doc_id: str = "test-doc-1",
    url: str = "https://example.com/article",
    title: str = "Test Article",
    source_type: str = "hn",
    origin: str = "pro",
) -> DocumentRow:
    """Create a test DocumentRow with default values.

    Args:
        doc_id: Document UUID.
        url: Document URL.
        title: Document title.
        source_type: Source type.
        origin: Ingest origin.

    Returns:
        A DocumentRow suitable for testing.
    """
    return DocumentRow(
        id=doc_id,
        source_type=source_type,
        origin=origin,
        url=url,
        title=title,
        raw_content="This is the article content for testing purposes.",
        fetched_at="2025-01-15T10:00:00",
    )


async def _insert_doc(test_db_path, doc: DocumentRow) -> None:
    """Insert a document into the test database.

    Args:
        test_db_path: Path to test DB file.
        doc: The DocumentRow to insert.
    """
    async with get_db(test_db_path.parent) as conn:
        await upsert_document(conn, doc)


class TestListDocuments:
    """Tests for GET /api/documents."""

    def test_list_empty(self, api_client: TestClient) -> None:
        """Listing documents on empty DB should return empty list."""
        response = api_client.get("/api/documents")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_returns_documents(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """List should return inserted documents."""
        doc = _create_test_doc()
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, doc))

        response = api_client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "test-doc-1"
        assert data[0]["title"] == "Test Article"
        assert data[0]["url"] == "https://example.com/article"

    def test_list_has_excerpt(self, api_client: TestClient, test_db_path) -> None:
        """Document in list should include excerpt from raw_content."""
        doc = _create_test_doc()
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, doc))

        response = api_client.get("/api/documents")
        data = response.json()
        assert data[0]["excerpt"] is not None
        assert "content" in data[0]["excerpt"].lower()

    def test_list_filter_by_origin(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """List should filter by origin when specified."""
        pro_doc = _create_test_doc(doc_id="pro-1", url="https://example.com/pro", origin="pro")
        radar_doc = _create_test_doc(
            doc_id="radar-1", url="https://example.com/radar", origin="radar"
        )
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, pro_doc))
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, radar_doc))

        response = api_client.get("/api/documents?origin=pro")
        assert response.status_code == 200
        data = response.json()
        assert all(d["origin"] == "pro" for d in data)

    def test_list_filter_by_source_type(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """List should filter by source_type when specified."""
        hn_doc = _create_test_doc(
            doc_id="hn-1", url="https://hn.com/item", source_type="hn"
        )
        arxiv_doc = _create_test_doc(
            doc_id="arxiv-1", url="https://arxiv.org/abs/1234", source_type="arxiv"
        )
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, hn_doc))
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, arxiv_doc))

        response = api_client.get("/api/documents?source_type=hn")
        data = response.json()
        assert all(d["source_type"] == "hn" for d in data)

    def test_list_pagination(self, api_client: TestClient, test_db_path) -> None:
        """List should respect limit and offset parameters."""
        for i in range(5):
            doc = _create_test_doc(
                doc_id=f"doc-{i}",
                url=f"https://example.com/article-{i}",
            )
            asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, doc))

        response = api_client.get("/api/documents?limit=2&offset=0")
        assert len(response.json()) == 2

        response2 = api_client.get("/api/documents?limit=2&offset=2")
        assert len(response2.json()) == 2

    def test_list_limit_validation(self, api_client: TestClient) -> None:
        """limit > 100 should return 422 (FastAPI validation)."""
        response = api_client.get("/api/documents?limit=200")
        assert response.status_code == 422

    def test_list_required_fields(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Document response should include all required fields."""
        doc = _create_test_doc()
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, doc))

        response = api_client.get("/api/documents")
        data = response.json()
        required = {
            "id", "title", "url", "source_type", "origin", "author",
            "published_at", "fetched_at", "word_count", "is_embedded",
            "is_favorited", "is_archived", "user_tags", "excerpt",
        }
        for field in required:
            assert field in data[0], f"Missing field: {field}"


class TestGetDocument:
    """Tests for GET /api/documents/{id}."""

    def test_get_existing(self, api_client: TestClient, test_db_path) -> None:
        """GET by ID should return the document."""
        doc = _create_test_doc()
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, doc))

        response = api_client.get("/api/documents/test-doc-1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-doc-1"

    def test_get_not_found(self, api_client: TestClient) -> None:
        """GET with non-existent ID should return 404."""
        response = api_client.get("/api/documents/nonexistent-id")
        assert response.status_code == 404


class TestDeleteDocument:
    """Tests for DELETE /api/documents/{id}."""

    def test_delete_existing(self, api_client: TestClient, test_db_path) -> None:
        """DELETE should soft-delete the document and return ok=True."""
        doc = _create_test_doc()
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, doc))

        response = api_client.delete("/api/documents/test-doc-1")
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_delete_not_found(self, api_client: TestClient) -> None:
        """DELETE with non-existent ID should return 404."""
        response = api_client.delete("/api/documents/nonexistent-id")
        assert response.status_code == 404

    def test_delete_hides_from_list(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Soft-deleted document should not appear in list."""
        doc = _create_test_doc()
        asyncio.get_event_loop().run_until_complete(_insert_doc(test_db_path, doc))

        api_client.delete("/api/documents/test-doc-1")

        response = api_client.get("/api/documents")
        assert response.json() == []
