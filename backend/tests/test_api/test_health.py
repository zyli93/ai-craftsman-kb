"""Tests for the health and stats endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_ok(self, api_client: TestClient) -> None:
        """GET /api/health should return 200 with status='ok'."""
        response = api_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "db" in data
        assert "qdrant" in data

    def test_health_db_reachable(self, api_client: TestClient) -> None:
        """Health check should report DB as reachable when using test DB."""
        response = api_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["db"] is True

    def test_health_qdrant_reachable(self, api_client: TestClient) -> None:
        """Health check should report Qdrant as reachable when mock is working."""
        response = api_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["qdrant"] is True


class TestStatsEndpoint:
    """Tests for GET /api/stats."""

    def test_stats_returns_200(self, api_client: TestClient) -> None:
        """GET /api/stats should return 200 with system statistics."""
        response = api_client.get("/api/stats")
        assert response.status_code == 200

    def test_stats_has_required_fields(self, api_client: TestClient) -> None:
        """Stats response should include all required fields."""
        response = api_client.get("/api/stats")
        data = response.json()
        required_fields = {
            "total_documents",
            "embedded_documents",
            "total_entities",
            "active_sources",
            "total_briefings",
            "vector_count",
            "db_size_bytes",
        }
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_stats_empty_db_counts(self, api_client: TestClient) -> None:
        """Stats should return zero counts for an empty test database."""
        response = api_client.get("/api/stats")
        data = response.json()
        assert data["total_documents"] == 0
        assert data["total_entities"] == 0
        assert data["total_briefings"] == 0
