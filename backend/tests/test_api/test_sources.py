"""Tests for source management endpoints."""
from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from ai_craftsman_kb.db.models import SourceRow
from ai_craftsman_kb.db.queries import upsert_source
from ai_craftsman_kb.db.sqlite import get_db


async def _insert_source(test_db_path, source: SourceRow) -> None:
    async with get_db(test_db_path.parent) as conn:
        await upsert_source(conn, source)


def _make_source(
    source_id: str = "src-1",
    source_type: str = "hn",
    identifier: str = "hn",
    display_name: str = "Hacker News",
    enabled: bool = True,
) -> SourceRow:
    return SourceRow(
        id=source_id,
        source_type=source_type,
        identifier=identifier,
        display_name=display_name,
        enabled=enabled,
    )


class TestListSources:
    """Tests for GET /api/sources."""

    def test_list_empty(self, api_client: TestClient) -> None:
        """List should return empty list when no sources exist."""
        response = api_client.get("/api/sources")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_returns_sources(self, api_client: TestClient, test_db_path) -> None:
        """List should return inserted sources."""
        src = _make_source()
        asyncio.get_event_loop().run_until_complete(_insert_source(test_db_path, src))

        response = api_client.get("/api/sources")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "src-1"
        assert data[0]["source_type"] == "hn"

    def test_list_has_required_fields(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Source response should include required fields."""
        src = _make_source()
        asyncio.get_event_loop().run_until_complete(_insert_source(test_db_path, src))

        response = api_client.get("/api/sources")
        data = response.json()
        required = {
            "id", "source_type", "identifier", "display_name",
            "enabled", "last_fetched_at", "fetch_error", "created_at",
        }
        for field in required:
            assert field in data[0], f"Missing field: {field}"


class TestCreateSource:
    """Tests for POST /api/sources."""

    def test_create_source(self, api_client: TestClient) -> None:
        """POST should create a new source and return 201."""
        response = api_client.post(
            "/api/sources",
            json={
                "source_type": "substack",
                "identifier": "test-newsletter",
                "display_name": "Test Newsletter",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["source_type"] == "substack"
        assert data["identifier"] == "test-newsletter"
        assert data["display_name"] == "Test Newsletter"
        assert data["enabled"] is True
        assert "id" in data

    def test_create_duplicate_returns_409(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Creating a source with duplicate type+identifier should return 409."""
        src = _make_source()
        asyncio.get_event_loop().run_until_complete(_insert_source(test_db_path, src))

        response = api_client.post(
            "/api/sources",
            json={"source_type": "hn", "identifier": "hn"},
        )
        assert response.status_code == 409


class TestUpdateSource:
    """Tests for PUT /api/sources/{id}."""

    def test_update_enabled_flag(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """PUT should update the enabled flag."""
        src = _make_source()
        asyncio.get_event_loop().run_until_complete(_insert_source(test_db_path, src))

        response = api_client.put(
            "/api/sources/src-1",
            json={"enabled": False},
        )
        assert response.status_code == 200
        assert response.json()["enabled"] is False

    def test_update_display_name(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """PUT should update the display name."""
        src = _make_source()
        asyncio.get_event_loop().run_until_complete(_insert_source(test_db_path, src))

        response = api_client.put(
            "/api/sources/src-1",
            json={"display_name": "Updated HN"},
        )
        assert response.status_code == 200
        assert response.json()["display_name"] == "Updated HN"

    def test_update_not_found(self, api_client: TestClient) -> None:
        """PUT on non-existent source should return 404."""
        response = api_client.put(
            "/api/sources/nonexistent",
            json={"enabled": False},
        )
        assert response.status_code == 404


class TestDeleteSource:
    """Tests for DELETE /api/sources/{id}."""

    def test_delete_source(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """DELETE should remove the source and return ok=True."""
        src = _make_source()
        asyncio.get_event_loop().run_until_complete(_insert_source(test_db_path, src))

        response = api_client.delete("/api/sources/src-1")
        assert response.status_code == 200
        assert response.json() == {"ok": True}

        # Verify it's gone
        response2 = api_client.get("/api/sources")
        assert response2.json() == []

    def test_delete_not_found(self, api_client: TestClient) -> None:
        """DELETE on non-existent source should return 404."""
        response = api_client.delete("/api/sources/nonexistent")
        assert response.status_code == 404
