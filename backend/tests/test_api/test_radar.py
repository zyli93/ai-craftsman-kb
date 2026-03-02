"""Tests for radar endpoints."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from ai_craftsman_kb.db.models import DocumentRow
from ai_craftsman_kb.db.queries import upsert_document
from ai_craftsman_kb.db.sqlite import get_db


async def _insert_radar_doc(test_db_path, doc: DocumentRow) -> None:
    async with get_db(test_db_path.parent) as conn:
        await upsert_document(conn, doc)


def _make_radar_doc(
    doc_id: str = "radar-1",
    url: str = "https://example.com/radar-article",
    is_archived: bool = False,
    promoted_at: str | None = None,
) -> DocumentRow:
    doc = DocumentRow(
        id=doc_id,
        source_type="hn",
        origin="radar",
        url=url,
        title="Radar Article",
        fetched_at="2025-01-15T10:00:00",
    )
    if promoted_at:
        doc = doc.model_copy(update={"promoted_at": promoted_at})
    if is_archived:
        doc = doc.model_copy(update={"is_archived": True})
    return doc


class TestRadarResults:
    """Tests for GET /api/radar/results."""

    def test_list_pending_empty(self, api_client: TestClient) -> None:
        """Pending radar list should be empty when DB is empty."""
        response = api_client.get("/api/radar/results")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_pending(self, api_client: TestClient, test_db_path) -> None:
        """Should return pending radar documents."""
        doc = _make_radar_doc()
        asyncio.get_event_loop().run_until_complete(
            _insert_radar_doc(test_db_path, doc)
        )

        response = api_client.get("/api/radar/results?status=pending")
        assert response.status_code == 200
        data = response.json()
        assert any(d["id"] == "radar-1" for d in data)

    def test_list_archived(self, api_client: TestClient, test_db_path) -> None:
        """Should return archived radar documents when status=archived."""
        doc = _make_radar_doc(is_archived=True)
        asyncio.get_event_loop().run_until_complete(
            _insert_radar_doc(test_db_path, doc)
        )

        response = api_client.get("/api/radar/results?status=archived")
        assert response.status_code == 200
        data = response.json()
        assert any(d["id"] == "radar-1" for d in data)

    def test_invalid_status_returns_422(self, api_client: TestClient) -> None:
        """Invalid status value should return 422."""
        response = api_client.get("/api/radar/results?status=invalid")
        assert response.status_code == 422


class TestRadarSearch:
    """Tests for POST /api/radar/search."""

    def test_radar_search_with_mock(self, api_client: TestClient) -> None:
        """POST /api/radar/search should run radar engine and return a report."""
        # Mock the RadarEngine.search to avoid real network calls
        from ai_craftsman_kb.radar.engine import RadarReport

        mock_report = RadarReport(
            query="machine learning",
            total_found=2,
            new_documents=1,
            sources_searched=["hn"],
            errors={},
        )

        with patch(
            "ai_craftsman_kb.radar.engine.RadarEngine.search",
            new=AsyncMock(return_value=mock_report),
        ):
            response = api_client.post(
                "/api/radar/search",
                json={"query": "machine learning", "limit_per_source": 5},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "machine learning"
        assert "total_found" in data
        assert "new_documents" in data
        assert "sources_searched" in data
        assert "errors" in data


class TestRadarActions:
    """Tests for POST /api/radar/results/{id}/promote and /archive."""

    def test_promote_radar_doc(self, api_client: TestClient, test_db_path) -> None:
        """Promote should set promoted_at and return the document."""
        doc = _make_radar_doc()
        asyncio.get_event_loop().run_until_complete(
            _insert_radar_doc(test_db_path, doc)
        )

        response = api_client.post("/api/radar/results/radar-1/promote")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "radar-1"

    def test_archive_radar_doc(self, api_client: TestClient, test_db_path) -> None:
        """Archive should set is_archived=True and return the document."""
        doc = _make_radar_doc()
        asyncio.get_event_loop().run_until_complete(
            _insert_radar_doc(test_db_path, doc)
        )

        response = api_client.post("/api/radar/results/radar-1/archive")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "radar-1"

    def test_promote_not_found(self, api_client: TestClient) -> None:
        """Promote on non-existent doc should return 404."""
        response = api_client.post("/api/radar/results/nonexistent/promote")
        assert response.status_code == 404

    def test_promote_non_radar_returns_422(
        self, api_client: TestClient, test_db_path
    ) -> None:
        """Promoting a non-radar document should return 422."""
        pro_doc = DocumentRow(
            id="pro-doc-1",
            source_type="hn",
            origin="pro",
            url="https://example.com/pro",
            title="Pro Article",
            fetched_at="2025-01-15T10:00:00",
        )
        asyncio.get_event_loop().run_until_complete(
            _insert_radar_doc(test_db_path, pro_doc)
        )

        response = api_client.post("/api/radar/results/pro-doc-1/promote")
        assert response.status_code == 422
