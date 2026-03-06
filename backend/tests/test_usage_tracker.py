"""Tests for UsageTracker -- LLM usage recording and querying."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest

from ai_craftsman_kb.db.sqlite import SCHEMA_SQL, get_db
from ai_craftsman_kb.llm.usage_tracker import UsageTracker


@pytest.fixture
async def tracker(tmp_path: Path) -> UsageTracker:
    """Return a UsageTracker backed by a fresh SQLite DB with full schema."""
    # Use executescript directly to avoid migration ordering issues on fresh DBs.
    db_path = tmp_path / "craftsman.db"
    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript(SCHEMA_SQL)
        await conn.commit()
    return UsageTracker(data_dir=tmp_path)


@pytest.mark.asyncio
async def test_record_and_get_recent(tracker: UsageTracker) -> None:
    """record() inserts a row that get_recent() retrieves."""
    await tracker.record(
        provider="openai",
        model="gpt-4o-mini",
        task="filtering",
        input_tokens=100,
        output_tokens=50,
        duration_ms=320,
        success=True,
    )

    rows = await tracker.get_recent(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "openai"
    assert row["model"] == "gpt-4o-mini"
    assert row["task"] == "filtering"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50
    assert row["duration_ms"] == 320
    assert row["success"] == 1  # SQLite stores booleans as 0/1


@pytest.mark.asyncio
async def test_record_with_none_tokens(tracker: UsageTracker) -> None:
    """record() accepts None for optional numeric fields."""
    await tracker.record(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        task="briefing",
    )

    rows = await tracker.get_recent(limit=10)
    assert len(rows) == 1
    assert rows[0]["input_tokens"] is None
    assert rows[0]["output_tokens"] is None
    assert rows[0]["duration_ms"] is None


@pytest.mark.asyncio
async def test_get_recent_ordering(tracker: UsageTracker) -> None:
    """get_recent() returns rows ordered by timestamp descending."""
    for i in range(5):
        await tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            task=f"task_{i}",
            input_tokens=i * 10,
            output_tokens=i * 5,
        )

    rows = await tracker.get_recent(limit=3)
    assert len(rows) == 3
    # Most recent first -- task_4 was inserted last
    assert rows[0]["task"] == "task_4"
    assert rows[1]["task"] == "task_3"
    assert rows[2]["task"] == "task_2"


@pytest.mark.asyncio
async def test_get_recent_respects_limit(tracker: UsageTracker) -> None:
    """get_recent() returns at most `limit` rows."""
    for i in range(10):
        await tracker.record(
            provider="openai", model="m", task="t", input_tokens=i,
        )

    rows = await tracker.get_recent(limit=5)
    assert len(rows) == 5


@pytest.mark.asyncio
async def test_get_summary_groups_correctly(tracker: UsageTracker) -> None:
    """get_summary() aggregates tokens and counts by provider+model+task."""
    # Two calls for the same provider/model/task
    await tracker.record(
        provider="openai", model="gpt-4o-mini", task="filtering",
        input_tokens=100, output_tokens=50,
    )
    await tracker.record(
        provider="openai", model="gpt-4o-mini", task="filtering",
        input_tokens=200, output_tokens=80,
    )
    # One call for a different task
    await tracker.record(
        provider="anthropic", model="claude-haiku-4-5-20251001", task="briefing",
        input_tokens=500, output_tokens=300,
    )

    since = datetime.now(timezone.utc) - timedelta(hours=1)
    summary = await tracker.get_summary(since)

    assert len(summary) == 2

    # Sorted by request_count DESC, so filtering (2) comes first
    filtering = summary[0]
    assert filtering["provider"] == "openai"
    assert filtering["model"] == "gpt-4o-mini"
    assert filtering["task"] == "filtering"
    assert filtering["total_input_tokens"] == 300
    assert filtering["total_output_tokens"] == 130
    assert filtering["request_count"] == 2

    briefing = summary[1]
    assert briefing["provider"] == "anthropic"
    assert briefing["total_input_tokens"] == 500
    assert briefing["request_count"] == 1


@pytest.mark.asyncio
async def test_get_summary_filters_by_since(tracker: UsageTracker) -> None:
    """get_summary() excludes records older than `since`."""
    await tracker.record(
        provider="openai", model="m", task="t", input_tokens=10,
    )

    # Query with a future timestamp -- should return nothing
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    summary = await tracker.get_summary(future)
    assert summary == []


@pytest.mark.asyncio
async def test_record_failure(tracker: UsageTracker) -> None:
    """record() stores success=False correctly."""
    await tracker.record(
        provider="openai",
        model="gpt-4o-mini",
        task="filtering",
        success=False,
    )

    rows = await tracker.get_recent(limit=1)
    assert rows[0]["success"] == 0  # SQLite boolean


@pytest.mark.asyncio
async def test_usage_tracker_exported_from_llm_package() -> None:
    """UsageTracker is accessible via the top-level llm package import."""
    from ai_craftsman_kb.llm import UsageTracker as Imported
    assert Imported is UsageTracker
