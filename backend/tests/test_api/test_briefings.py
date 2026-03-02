"""Tests for briefing endpoints."""
from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from ai_craftsman_kb.db.models import BriefingRow
from ai_craftsman_kb.db.queries import insert_briefing
from ai_craftsman_kb.db.sqlite import get_db


async def _insert_briefing(test_db_path, briefing: BriefingRow) -> None:
    async with get_db(test_db_path.parent) as conn:
        await insert_briefing(conn, briefing)


def _make_briefing(
    briefing_id: str = "brief-1",
    title: str = "Test Briefing",
    query: str = "machine learning",
) -> BriefingRow:
    return BriefingRow(
        id=briefing_id,
        title=title,
        query=query,
        content="# Test Briefing\n\nContent here.",
        source_document_ids=["doc-1"],
    )


class TestListBriefings:
    """Tests for GET /api/briefings."""

    def test_list_empty(self, api_client: TestClient) -> None:
        """List should return empty list when no briefings exist."""
        response = api_client.get("/api/briefings")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_returns_briefings(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """List should return inserted briefings."""
        briefing = _make_briefing()
        asyncio.get_event_loop().run_until_complete(
            _insert_briefing(test_db_path, briefing)
        )

        response = api_client.get("/api/briefings")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "brief-1"

    def test_list_has_required_fields(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Briefing response should include all required fields."""
        briefing = _make_briefing()
        asyncio.get_event_loop().run_until_complete(
            _insert_briefing(test_db_path, briefing)
        )

        response = api_client.get("/api/briefings")
        data = response.json()
        required = {
            "id", "title", "query", "content",
            "source_document_ids", "created_at", "format",
        }
        for field in required:
            assert field in data[0], f"Missing field: {field}"


class TestGetBriefing:
    """Tests for GET /api/briefings/{id}."""

    def test_get_existing(self, api_client: TestClient, test_db_path) -> None:
        """GET by ID should return the briefing."""
        briefing = _make_briefing()
        asyncio.get_event_loop().run_until_complete(
            _insert_briefing(test_db_path, briefing)
        )

        response = api_client.get("/api/briefings/brief-1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "brief-1"
        assert data["title"] == "Test Briefing"

    def test_get_not_found(self, api_client: TestClient) -> None:
        """GET with non-existent ID should return 404."""
        response = api_client.get("/api/briefings/nonexistent")
        assert response.status_code == 404


class TestCreateBriefing:
    """Tests for POST /api/briefings."""

    def test_create_briefing(self, api_client: TestClient) -> None:
        """POST should create a briefing and return 201."""
        response = api_client.post(
            "/api/briefings",
            json={"query": "test query", "limit": 5},
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert "title" in data
        assert "content" in data
        assert "query" in data
        assert data["query"] == "test query"

    def test_create_briefing_stored(self, api_client: TestClient) -> None:
        """Created briefing should be retrievable via GET."""
        post_response = api_client.post(
            "/api/briefings",
            json={"query": "test storage", "limit": 5},
        )
        assert post_response.status_code == 201
        briefing_id = post_response.json()["id"]

        get_response = api_client.get(f"/api/briefings/{briefing_id}")
        assert get_response.status_code == 200
        assert get_response.json()["id"] == briefing_id


class TestDeleteBriefing:
    """Tests for DELETE /api/briefings/{id}."""

    def test_delete_briefing(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """DELETE should remove the briefing and return ok=True."""
        briefing = _make_briefing()
        asyncio.get_event_loop().run_until_complete(
            _insert_briefing(test_db_path, briefing)
        )

        response = api_client.delete("/api/briefings/brief-1")
        assert response.status_code == 200
        assert response.json() == {"ok": True}

        # Verify it's gone
        response2 = api_client.get("/api/briefings/brief-1")
        assert response2.status_code == 404

    def test_delete_not_found(self, api_client: TestClient) -> None:
        """DELETE on non-existent briefing should return 404."""
        response = api_client.delete("/api/briefings/nonexistent")
        assert response.status_code == 404
