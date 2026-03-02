"""Tests for server.py — dashboard mounting and create_app factory."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_craftsman_kb.server import DASHBOARD_DIST, create_app, mount_dashboard


class TestMountDashboard:
    """Tests for the mount_dashboard helper function."""

    def test_mount_dashboard_adds_fallback_route_when_dist_missing(self) -> None:
        """When dashboard/dist does not exist, a GET / route is added."""
        app = create_app(serve_dashboard=False)

        # Manually call mount_dashboard with a non-existent dist path
        with patch("ai_craftsman_kb.server.DASHBOARD_DIST", Path("/nonexistent/dist")):
            mount_dashboard(app)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/")
        # The fallback route returns a JSON with the build instructions
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "pnpm build" in data["message"]

    def test_mount_dashboard_mounts_staticfiles_when_dist_exists(self, tmp_path: Path) -> None:
        """When dashboard/dist exists, StaticFiles is mounted at /."""
        # Create a minimal dist directory with an index.html
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html><body>Dashboard</body></html>")

        app = create_app(serve_dashboard=False)

        with patch("ai_craftsman_kb.server.DASHBOARD_DIST", dist):
            mount_dashboard(app)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/")
        assert response.status_code == 200
        assert b"Dashboard" in response.content


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_create_app_no_dashboard_skips_mounting(self) -> None:
        """create_app(serve_dashboard=False) skips dashboard mounting entirely."""
        with patch("ai_craftsman_kb.server.mount_dashboard") as mock_mount:
            create_app(serve_dashboard=False)
        mock_mount.assert_not_called()

    def test_create_app_with_dashboard_calls_mount(self) -> None:
        """create_app(serve_dashboard=True) calls mount_dashboard."""
        with patch("ai_craftsman_kb.server.mount_dashboard") as mock_mount:
            create_app(serve_dashboard=True)
        mock_mount.assert_called_once()

    def test_create_app_registers_api_health_route(self) -> None:
        """create_app should expose /api/health regardless of dashboard mode."""
        app = create_app(serve_dashboard=False)
        route_paths = [str(r.path) for r in app.routes]  # type: ignore[attr-defined]
        assert "/api/health" in route_paths

    def test_create_app_has_docs_url(self) -> None:
        """API docs should be accessible at /docs."""
        app = create_app(serve_dashboard=False)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/docs")
        # Docs always return 200 even without a live lifespan
        assert response.status_code == 200

    def test_dashboard_dist_path_is_correct(self) -> None:
        """DASHBOARD_DIST should point to dashboard/dist relative to project root."""
        # The path should end with dashboard/dist
        assert DASHBOARD_DIST.parts[-2:] == ("dashboard", "dist")
