"""Tests for the usage API endpoints (GET /api/usage, GET /api/usage/recent)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import MagicMock

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from ai_craftsman_kb.api.deps import get_conn
from ai_craftsman_kb.db.sqlite import get_db
from ai_craftsman_kb.llm.usage_tracker import UsageTracker
from ai_craftsman_kb.server import create_app

# Minimal DDL for the llm_usage table -- avoids calling init_db which runs
# migrations that assume the full schema already exists.
_LLM_USAGE_DDL = """
CREATE TABLE IF NOT EXISTS llm_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    task            TEXT NOT NULL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    duration_ms     INTEGER,
    success         BOOLEAN DEFAULT TRUE
);
"""


async def _create_usage_table(data_dir: Path) -> None:
    """Create only the llm_usage table in a fresh SQLite database."""
    async with get_db(data_dir) as conn:
        await conn.executescript(_LLM_USAGE_DDL)
        await conn.commit()


@pytest.fixture
def usage_client(tmp_path: Path) -> TestClient:
    """TestClient with a real UsageTracker backed by a temporary SQLite DB.

    Returns:
        A configured TestClient with usage_tracker on app.state.
    """
    app = create_app(serve_dashboard=False)

    app.state.config = MagicMock()
    app.state.db_path = tmp_path / "craftsman.db"
    app.state.vector_store = MagicMock()
    app.state.embedder = MagicMock()
    app.state.llm_router = MagicMock()

    # Create only the llm_usage table (not the full schema)
    asyncio.get_event_loop().run_until_complete(_create_usage_table(tmp_path))

    # Create a real UsageTracker
    app.state.usage_tracker = UsageTracker(tmp_path)

    # Override the DB dependency
    async def override_get_conn() -> AsyncGenerator[aiosqlite.Connection, None]:
        async with get_db(tmp_path) as conn:
            yield conn

    app.dependency_overrides[get_conn] = override_get_conn

    return TestClient(app, raise_server_exceptions=True)


def _record_usage(tmp_path: Path, **kwargs) -> None:
    """Helper to synchronously record a usage entry."""
    tracker = UsageTracker(tmp_path)
    asyncio.get_event_loop().run_until_complete(tracker.record(**kwargs))


class TestGetUsageSummary:
    """Tests for GET /api/usage."""

    def test_empty_summary(self, usage_client: TestClient) -> None:
        """Returns an empty summary list when no usage data exists."""
        resp = usage_client.get("/api/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == []
        assert "period_start" in data
        assert "period_end" in data

    def test_summary_with_data(self, usage_client: TestClient, tmp_path: Path) -> None:
        """Returns aggregated summary after recording usage."""
        _record_usage(
            tmp_path,
            provider="openai",
            model="gpt-4o",
            task="filtering",
            input_tokens=100,
            output_tokens=50,
        )
        _record_usage(
            tmp_path,
            provider="openai",
            model="gpt-4o",
            task="filtering",
            input_tokens=200,
            output_tokens=80,
        )

        resp = usage_client.get("/api/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["summary"]) == 1

        item = data["summary"][0]
        assert item["provider"] == "openai"
        assert item["model"] == "gpt-4o"
        assert item["task"] == "filtering"
        assert item["total_input_tokens"] == 300
        assert item["total_output_tokens"] == 130
        assert item["request_count"] == 2

    def test_summary_with_since_param(self, usage_client: TestClient, tmp_path: Path) -> None:
        """The since parameter filters results by timestamp."""
        _record_usage(
            tmp_path,
            provider="anthropic",
            model="claude-3",
            task="briefing",
            input_tokens=500,
            output_tokens=200,
        )

        # Query with a future timestamp should return nothing
        resp = usage_client.get("/api/usage?since=2099-01-01T00:00:00")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == []

        # Query with a past timestamp should include the record
        resp = usage_client.get("/api/usage?since=2020-01-01T00:00:00")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["summary"]) == 1


class TestGetRecentUsage:
    """Tests for GET /api/usage/recent."""

    def test_empty_recent(self, usage_client: TestClient) -> None:
        """Returns an empty list when no usage data exists."""
        resp = usage_client.get("/api/usage/recent")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_recent_with_data(self, usage_client: TestClient, tmp_path: Path) -> None:
        """Returns raw records ordered by most recent first."""
        _record_usage(
            tmp_path,
            provider="openai",
            model="gpt-4o",
            task="filtering",
            input_tokens=100,
            output_tokens=50,
            duration_ms=250,
            success=True,
        )
        _record_usage(
            tmp_path,
            provider="anthropic",
            model="claude-3",
            task="briefing",
            input_tokens=500,
            output_tokens=200,
            duration_ms=1200,
            success=True,
        )

        resp = usage_client.get("/api/usage/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        # Most recent first
        assert data[0]["provider"] == "anthropic"
        assert data[1]["provider"] == "openai"

        # Check all fields are present
        record = data[0]
        assert "id" in record
        assert "timestamp" in record
        assert record["model"] == "claude-3"
        assert record["task"] == "briefing"
        assert record["input_tokens"] == 500
        assert record["output_tokens"] == 200
        assert record["duration_ms"] == 1200
        assert record["success"] is True

    def test_recent_limit_param(self, usage_client: TestClient, tmp_path: Path) -> None:
        """The limit parameter restricts the number of returned records."""
        for i in range(5):
            _record_usage(
                tmp_path,
                provider="openai",
                model="gpt-4o",
                task=f"task_{i}",
                input_tokens=100,
                output_tokens=50,
            )

        resp = usage_client.get("/api/usage/recent?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_recent_null_tokens(self, usage_client: TestClient, tmp_path: Path) -> None:
        """Records with None token counts are serialised as null."""
        _record_usage(
            tmp_path,
            provider="ollama",
            model="llama3",
            task="filtering",
            # No token counts provided
        )

        resp = usage_client.get("/api/usage/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["input_tokens"] is None
        assert data[0]["output_tokens"] is None
        assert data[0]["duration_ms"] is None
