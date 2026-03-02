"""Tests for entity endpoints."""
from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from ai_craftsman_kb.db.models import DocumentRow, EntityRow
from ai_craftsman_kb.db.queries import link_document_entity, upsert_document, upsert_entity
from ai_craftsman_kb.db.sqlite import get_db


async def _setup(test_db_path, doc: DocumentRow, entity: EntityRow) -> None:
    async with get_db(test_db_path.parent) as conn:
        await upsert_document(conn, doc)
        await upsert_entity(conn, entity)
        await link_document_entity(conn, doc.id, entity.id, context="test context")


def _make_entity(
    entity_id: str = "ent-1",
    name: str = "OpenAI",
    entity_type: str = "company",
) -> EntityRow:
    return EntityRow(
        id=entity_id,
        name=name,
        entity_type=entity_type,
        normalized_name=name.lower(),
        mention_count=1,
    )


def _make_doc(
    doc_id: str = "doc-1",
    url: str = "https://example.com/doc",
) -> DocumentRow:
    return DocumentRow(
        id=doc_id,
        source_type="hn",
        origin="pro",
        url=url,
        title="Test Document",
        fetched_at="2025-01-15T10:00:00",
    )


class TestListEntities:
    """Tests for GET /api/entities."""

    def test_list_empty(self, api_client: TestClient) -> None:
        """List should return empty list when no entities exist."""
        response = api_client.get("/api/entities")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_returns_entities(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """List should return inserted entities."""
        entity = _make_entity()
        doc = _make_doc()
        asyncio.get_event_loop().run_until_complete(
            _setup(test_db_path, doc, entity)
        )

        response = api_client.get("/api/entities")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(e["id"] == "ent-1" for e in data)

    def test_list_has_required_fields(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Entity response should include required fields."""
        entity = _make_entity()
        doc = _make_doc()
        asyncio.get_event_loop().run_until_complete(_setup(test_db_path, doc, entity))

        response = api_client.get("/api/entities")
        data = response.json()
        required = {
            "id", "name", "entity_type", "normalized_name",
            "description", "mention_count", "first_seen_at",
        }
        for field in required:
            assert field in data[0], f"Missing field: {field}"

    def test_filter_by_type(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """List should filter by entity_type when specified."""
        company = _make_entity(entity_id="comp-1", name="OpenAI", entity_type="company")
        person = _make_entity(entity_id="pers-1", name="Sam Altman", entity_type="person")
        doc1 = _make_doc(doc_id="d1", url="https://e.com/1")
        doc2 = _make_doc(doc_id="d2", url="https://e.com/2")
        asyncio.get_event_loop().run_until_complete(_setup(test_db_path, doc1, company))
        asyncio.get_event_loop().run_until_complete(_setup(test_db_path, doc2, person))

        response = api_client.get("/api/entities?entity_type=company")
        data = response.json()
        assert all(e["entity_type"] == "company" for e in data)


class TestGetEntity:
    """Tests for GET /api/entities/{id}."""

    def test_get_existing(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """GET by ID should return entity with linked documents."""
        entity = _make_entity()
        doc = _make_doc()
        asyncio.get_event_loop().run_until_complete(_setup(test_db_path, doc, entity))

        response = api_client.get("/api/entities/ent-1")
        assert response.status_code == 200
        data = response.json()
        assert "entity" in data
        assert "documents" in data
        assert data["entity"]["id"] == "ent-1"
        assert isinstance(data["documents"], list)

    def test_get_not_found(self, api_client: TestClient) -> None:
        """GET with non-existent ID should return 404."""
        response = api_client.get("/api/entities/nonexistent")
        assert response.status_code == 404

    def test_get_includes_linked_docs(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Entity response should include documents linked to it."""
        entity = _make_entity()
        doc = _make_doc()
        asyncio.get_event_loop().run_until_complete(_setup(test_db_path, doc, entity))

        response = api_client.get("/api/entities/ent-1")
        data = response.json()
        assert len(data["documents"]) == 1
        assert data["documents"][0]["id"] == "doc-1"


class TestEntityDocuments:
    """Tests for GET /api/entities/{id}/documents."""

    def test_get_entity_docs(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Should return documents linked to the entity."""
        entity = _make_entity()
        doc = _make_doc()
        asyncio.get_event_loop().run_until_complete(_setup(test_db_path, doc, entity))

        response = api_client.get("/api/entities/ent-1/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "doc-1"

    def test_entity_not_found(self, api_client: TestClient) -> None:
        """Should return 404 when entity does not exist."""
        response = api_client.get("/api/entities/nonexistent/documents")
        assert response.status_code == 404
