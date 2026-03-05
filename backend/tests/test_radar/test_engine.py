"""Unit tests for the RadarEngine async fan-out orchestrator.

Tests cover:
- search() fans out to all ingestors concurrently via asyncio.gather
- search() filters by sources parameter when provided
- search() deduplicates results across sources by URL
- search() stores new documents with origin='radar'
- search() skips documents already in DB (duplicates counted correctly)
- search() records per-source exceptions in RadarReport.errors
- search() continues with other sources when one fails
- search() returns correct counts (total_found, new_documents)
- search() returns empty report when no active ingestors match
- _search_one_source() ensures origin='radar' on all returned docs
- _search_one_source() re-raises exceptions (for asyncio.gather capture)
- _store_results() returns (new_count, duplicate_count) tuple
- _deduplicate_by_url() keeps first occurrence across sources
- asyncio.gather is used (not sequential awaits) — verified via timing
"""
import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    EmbeddingConfig,
    FiltersConfig,
    HackerNewsConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    SettingsConfig,
    SourcesConfig,
)
from ai_craftsman_kb.db.models import DocumentRow
from ai_craftsman_kb.ingestors.base import BaseIngestor, RawDocument
from ai_craftsman_kb.radar.engine import (
    RadarEngine,
    RadarReport,
    RadarResult,
    _deduplicate_by_url,
)


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _make_config() -> AppConfig:
    """Build a minimal AppConfig for testing."""
    task_cfg = LLMTaskConfig(provider="openai", model="gpt-4o-mini")
    return AppConfig(
        sources=SourcesConfig(hackernews=HackerNewsConfig(limit=10)),
        settings=SettingsConfig(
            data_dir="/tmp/test-radar-engine",
            llm=LLMRoutingConfig(
                filtering=task_cfg,
                entity_extraction=task_cfg,
                briefing=task_cfg,
                source_discovery=task_cfg,
                keyword_extraction=task_cfg,
            ),
        ),
        filters=FiltersConfig(),
    )


def _make_raw_doc(
    url: str = "https://example.com/article",
    source_type: str = "hn",
    title: str = "Test Article",
    origin: str = "radar",
) -> RawDocument:
    """Create a RawDocument for testing.

    Args:
        url: The document URL (used as dedup key).
        source_type: The source type string.
        title: The document title.
        origin: The origin label.

    Returns:
        A RawDocument instance.
    """
    return RawDocument(
        url=url,
        title=title,
        source_type=source_type,
        origin=origin,  # type: ignore[arg-type]
        published_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        content_type="article",
    )


def _make_ingestor(
    source_type: str,
    search_result: list[RawDocument] | Exception | None = None,
    delay: float = 0.0,
) -> MagicMock:
    """Create a mock BaseIngestor with a configurable search_radar() behavior.

    Args:
        source_type: The source_type property value.
        search_result: List of docs to return, an Exception to raise, or None for [].
        delay: Optional asyncio.sleep delay to simulate slow network calls.

    Returns:
        A MagicMock with source_type and search_radar() configured.
    """
    mock = MagicMock(spec=BaseIngestor)
    mock.source_type = source_type

    if search_result is None:
        search_result = []

    async def _search_radar(query: str, limit: int = 20) -> list[RawDocument]:
        if delay > 0:
            await asyncio.sleep(delay)
        if isinstance(search_result, Exception):
            raise search_result
        return list(search_result)

    mock.search_radar = _search_radar
    return mock


@pytest.fixture
def config() -> AppConfig:
    """Provide a minimal AppConfig for radar engine tests."""
    return _make_config()


@pytest.fixture
def five_ingestors(config: AppConfig) -> dict[str, MagicMock]:
    """Provide one mock ingestor per major radar source type.

    Each ingestor returns two documents from its respective source.
    """
    return {
        "hn": _make_ingestor("hn", [
            _make_raw_doc("https://news.ycombinator.com/1", "hn", "HN Article 1"),
            _make_raw_doc("https://news.ycombinator.com/2", "hn", "HN Article 2"),
        ]),
        "reddit": _make_ingestor("reddit", [
            _make_raw_doc("https://reddit.com/r/test/1", "reddit", "Reddit Post 1"),
            _make_raw_doc("https://reddit.com/r/test/2", "reddit", "Reddit Post 2"),
        ]),
        "arxiv": _make_ingestor("arxiv", [
            _make_raw_doc("https://arxiv.org/abs/2501.00001", "arxiv", "ArXiv Paper 1"),
        ]),
        "devto": _make_ingestor("devto", [
            _make_raw_doc("https://dev.to/article-1", "devto", "DEV.to Article 1"),
        ]),
        "youtube": _make_ingestor("youtube", [
            _make_raw_doc("https://youtube.com/watch?v=abc", "youtube", "YouTube Video 1"),
        ]),
    }


# ---------------------------------------------------------------------------
# Model and structure tests
# ---------------------------------------------------------------------------


def test_radar_report_has_required_fields() -> None:
    """RadarReport initializes with expected default field values."""
    report = RadarReport(query="test query")
    assert report.query == "test query"
    assert report.total_found == 0
    assert report.new_documents == 0
    assert report.sources_searched == []
    assert report.errors == {}


def test_radar_result_fields() -> None:
    """RadarResult stores document, source_type, and is_new flag."""
    doc_row = DocumentRow(
        id="test-id",
        origin="radar",
        source_type="hn",
        url="https://example.com",
    )
    result = RadarResult(document=doc_row, source_type="hn", is_new=True)
    assert result.source_type == "hn"
    assert result.is_new is True
    assert result.document.url == "https://example.com"


# ---------------------------------------------------------------------------
# _deduplicate_by_url() unit tests
# ---------------------------------------------------------------------------


def test_deduplicate_empty_list() -> None:
    """_deduplicate_by_url returns empty list for empty input."""
    assert _deduplicate_by_url([]) == []


def test_deduplicate_no_duplicates() -> None:
    """_deduplicate_by_url returns all docs when URLs are unique."""
    docs = [
        _make_raw_doc("https://a.com"),
        _make_raw_doc("https://b.com"),
        _make_raw_doc("https://c.com"),
    ]
    result = _deduplicate_by_url(docs)
    assert len(result) == 3


def test_deduplicate_removes_exact_url_duplicates() -> None:
    """_deduplicate_by_url removes docs with duplicate URLs."""
    docs = [
        _make_raw_doc("https://a.com", "hn", "First"),
        _make_raw_doc("https://a.com", "reddit", "Second"),  # duplicate URL
        _make_raw_doc("https://b.com", "hn", "Third"),
    ]
    result = _deduplicate_by_url(docs)
    assert len(result) == 2


def test_deduplicate_keeps_first_occurrence() -> None:
    """_deduplicate_by_url preserves the first occurrence of a URL."""
    docs = [
        _make_raw_doc("https://a.com", "hn", "First Title"),
        _make_raw_doc("https://a.com", "reddit", "Second Title"),
    ]
    result = _deduplicate_by_url(docs)
    assert len(result) == 1
    assert result[0].title == "First Title"
    assert result[0].source_type == "hn"


def test_deduplicate_handles_all_same_url() -> None:
    """_deduplicate_by_url returns exactly one doc when all URLs are identical."""
    docs = [_make_raw_doc("https://same.com") for _ in range(5)]
    result = _deduplicate_by_url(docs)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# RadarEngine.__init__ tests
# ---------------------------------------------------------------------------


def test_engine_stores_config_and_ingestors(config: AppConfig) -> None:
    """RadarEngine stores config and ingestors passed in __init__."""
    mock_ingestor = _make_ingestor("hn")
    engine = RadarEngine(config=config, ingestors={"hn": mock_ingestor})
    assert engine.config is config
    assert engine.ingestors == {"hn": mock_ingestor}


# ---------------------------------------------------------------------------
# search() — concurrency tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_uses_asyncio_gather_not_sequential() -> None:
    """search() fans out concurrently — total time should be ~max(delays), not sum.

    Each of 3 ingestors has a 0.1s delay. Sequential would take ~0.3s;
    concurrent gather should finish in ~0.1s (allowing generous 0.25s margin).
    """
    config = _make_config()
    delay = 0.1  # 100ms per source

    ingestors = {
        "hn": _make_ingestor("hn", [], delay=delay),
        "reddit": _make_ingestor("reddit", [], delay=delay),
        "arxiv": _make_ingestor("arxiv", [], delay=delay),
    }
    engine = RadarEngine(config=config, ingestors=ingestors)

    # Use an in-memory SQLite for the DB connection
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)

        start = time.monotonic()
        await engine.search(conn, "test query", limit_per_source=5)
        elapsed = time.monotonic() - start

    # If sequential: elapsed >= 3 * delay (~0.3s)
    # If concurrent: elapsed < 2 * delay (well under 0.25s)
    sequential_time = 3 * delay
    assert elapsed < sequential_time * 0.8, (
        f"Expected concurrent execution (< {sequential_time * 0.8:.2f}s), "
        f"but took {elapsed:.3f}s (suggesting sequential calls)"
    )


# ---------------------------------------------------------------------------
# search() — basic functionality
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_radar_report(
    config: AppConfig, five_ingestors: dict[str, MagicMock]
) -> None:
    """search() returns a RadarReport with the correct query string."""
    engine = RadarEngine(config=config, ingestors=five_ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "machine learning")

    assert isinstance(report, RadarReport)
    assert report.query == "machine learning"


@pytest.mark.asyncio
async def test_search_all_sources_searched_by_default(
    config: AppConfig, five_ingestors: dict[str, MagicMock]
) -> None:
    """search() searches all ingestors when sources=None."""
    engine = RadarEngine(config=config, ingestors=five_ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "rust programming")

    assert set(report.sources_searched) == set(five_ingestors.keys())


@pytest.mark.asyncio
async def test_search_sources_filter_limits_to_specified(
    config: AppConfig, five_ingestors: dict[str, MagicMock]
) -> None:
    """search() only searches the specified sources when sources list is given."""
    engine = RadarEngine(config=config, ingestors=five_ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "test", sources=["hn", "arxiv"])

    assert set(report.sources_searched) == {"hn", "arxiv"}


@pytest.mark.asyncio
async def test_search_total_found_counts_deduplicated_results(
    config: AppConfig,
) -> None:
    """search() total_found reflects the deduplicated count, not raw sum."""
    # HN and Reddit both return the same URL — dedup should keep 1
    shared_url = "https://example.com/shared-article"
    ingestors = {
        "hn": _make_ingestor("hn", [
            _make_raw_doc(shared_url, "hn", "Shared from HN"),
            _make_raw_doc("https://hn-only.com", "hn", "HN Only"),
        ]),
        "reddit": _make_ingestor("reddit", [
            _make_raw_doc(shared_url, "reddit", "Shared from Reddit"),  # duplicate
            _make_raw_doc("https://reddit-only.com", "reddit", "Reddit Only"),
        ]),
    }
    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "test")

    # 4 raw results but 3 unique URLs after dedup
    assert report.total_found == 3


@pytest.mark.asyncio
async def test_search_new_documents_counts_only_newly_stored(
    config: AppConfig,
) -> None:
    """search() new_documents counts only docs that were not already in the DB."""
    ingestors = {
        "hn": _make_ingestor("hn", [
            _make_raw_doc("https://new-doc.com", "hn", "New Article"),
            _make_raw_doc("https://existing-doc.com", "hn", "Existing Article"),
        ]),
    }
    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)

        # Pre-insert a document with the URL that the ingestor will return
        existing_row = DocumentRow(
            id="pre-existing-id",
            origin="pro",
            source_type="hn",
            url="https://existing-doc.com",
            title="Already in DB",
        )
        await _insert_document_row(conn, existing_row)

        report = await engine.search(conn, "test")

    # Only 1 new document (the other URL was already in DB)
    assert report.new_documents == 1
    assert report.total_found == 2  # Both were found, only 1 was new


@pytest.mark.asyncio
async def test_search_stores_documents_with_radar_origin(
    config: AppConfig,
) -> None:
    """search() stores all new documents with origin='radar' regardless of ingestor default."""
    # Ingestor returns docs with origin='pro' (not radar)
    ingestors = {
        "hn": _make_ingestor("hn", [
            _make_raw_doc("https://example.com/1", "hn", "Article 1", origin="pro"),
        ]),
    }
    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        await engine.search(conn, "test")

        # Verify the stored document has origin='radar'
        async with conn.execute(
            "SELECT origin FROM documents WHERE url = ?",
            ("https://example.com/1",),
        ) as cursor:
            row = await cursor.fetchone()

    assert row is not None
    assert row["origin"] == "radar"


# ---------------------------------------------------------------------------
# search() — error handling tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_records_failing_source_in_errors(config: AppConfig) -> None:
    """search() records exception message in report.errors for failing source."""
    error = RuntimeError("API rate limit exceeded")
    ingestors = {
        "hn": _make_ingestor("hn", error),
        "arxiv": _make_ingestor("arxiv", [
            _make_raw_doc("https://arxiv.org/abs/1", "arxiv", "Paper"),
        ]),
    }
    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "test")

    assert "hn" in report.errors
    assert "rate limit" in report.errors["hn"].lower() or "api" in report.errors["hn"].lower()


@pytest.mark.asyncio
async def test_search_continues_other_sources_when_one_fails(config: AppConfig) -> None:
    """search() still returns results from working sources when one source fails."""
    error = ConnectionError("Network error")
    ingestors = {
        "hn": _make_ingestor("hn", error),  # fails
        "arxiv": _make_ingestor("arxiv", [
            _make_raw_doc("https://arxiv.org/abs/2501.00001", "arxiv", "ArXiv Paper"),
        ]),
        "devto": _make_ingestor("devto", [
            _make_raw_doc("https://dev.to/article", "devto", "DEV.to Article"),
        ]),
    }
    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "test")

        # Verify documents are actually in the DB
        async with conn.execute("SELECT COUNT(*) as cnt FROM documents") as cursor:
            row = await cursor.fetchone()
        stored_count = row["cnt"]

    # Despite HN failing, we still get results from arxiv and devto
    assert report.total_found == 2
    assert report.new_documents == 2
    assert stored_count == 2
    assert "hn" in report.errors
    assert "hn" in report.sources_searched  # was attempted


@pytest.mark.asyncio
async def test_search_no_errors_when_all_succeed(
    config: AppConfig, five_ingestors: dict[str, MagicMock]
) -> None:
    """search() has empty errors dict when all sources succeed."""
    engine = RadarEngine(config=config, ingestors=five_ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "test")

    assert report.errors == {}


@pytest.mark.asyncio
async def test_search_multiple_sources_fail(config: AppConfig) -> None:
    """search() records all failing sources in errors dict."""
    ingestors = {
        "hn": _make_ingestor("hn", ValueError("HN error")),
        "reddit": _make_ingestor("reddit", TimeoutError("Reddit timeout")),
        "arxiv": _make_ingestor("arxiv", []),
    }
    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "test")

    assert "hn" in report.errors
    assert "reddit" in report.errors
    assert "arxiv" not in report.errors


@pytest.mark.asyncio
async def test_search_empty_ingestors_returns_empty_report(config: AppConfig) -> None:
    """search() with no ingestors returns an empty RadarReport."""
    engine = RadarEngine(config=config, ingestors={})

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "test")

    assert report.total_found == 0
    assert report.new_documents == 0
    assert report.sources_searched == []
    assert report.errors == {}


@pytest.mark.asyncio
async def test_search_sources_filter_with_no_match_returns_empty(
    config: AppConfig, five_ingestors: dict[str, MagicMock]
) -> None:
    """search() with sources list that matches no ingestor returns empty report."""
    engine = RadarEngine(config=config, ingestors=five_ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "test", sources=["nonexistent_source"])

    assert report.total_found == 0
    assert report.sources_searched == []


# ---------------------------------------------------------------------------
# _search_one_source() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_one_source_returns_radar_origin_docs(config: AppConfig) -> None:
    """_search_one_source() ensures all returned docs have origin='radar'."""
    # Ingestor returns docs with origin='pro'
    mock_ingestor = _make_ingestor("hn", [
        _make_raw_doc("https://example.com/1", "hn", "Article", origin="pro"),
        _make_raw_doc("https://example.com/2", "hn", "Article 2", origin="pro"),
    ])
    engine = RadarEngine(config=config, ingestors={"hn": mock_ingestor})

    docs = await engine._search_one_source(mock_ingestor, "test query", limit=10)

    assert all(doc.origin == "radar" for doc in docs)


@pytest.mark.asyncio
async def test_search_one_source_returns_empty_list_when_no_results(
    config: AppConfig,
) -> None:
    """_search_one_source() returns [] when ingestor finds nothing."""
    mock_ingestor = _make_ingestor("hn", [])
    engine = RadarEngine(config=config, ingestors={"hn": mock_ingestor})

    docs = await engine._search_one_source(mock_ingestor, "no results query", limit=5)

    assert docs == []


@pytest.mark.asyncio
async def test_search_one_source_reraises_exception(config: AppConfig) -> None:
    """_search_one_source() re-raises exceptions (not swallowing them)."""
    error = RuntimeError("Connection failed")
    mock_ingestor = _make_ingestor("hn", error)
    engine = RadarEngine(config=config, ingestors={"hn": mock_ingestor})

    with pytest.raises(RuntimeError, match="Connection failed"):
        await engine._search_one_source(mock_ingestor, "test", limit=10)


@pytest.mark.asyncio
async def test_search_one_source_preserves_radar_origin(config: AppConfig) -> None:
    """_search_one_source() does not modify docs that already have origin='radar'."""
    mock_ingestor = _make_ingestor("hn", [
        _make_raw_doc("https://example.com", "hn", "Article", origin="radar"),
    ])
    engine = RadarEngine(config=config, ingestors={"hn": mock_ingestor})

    docs = await engine._search_one_source(mock_ingestor, "test", limit=10)

    assert len(docs) == 1
    assert docs[0].origin == "radar"


# ---------------------------------------------------------------------------
# _store_results() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_results_returns_new_and_duplicate_counts(config: AppConfig) -> None:
    """_store_results() returns (new_count, duplicate_count) tuple."""
    ingestors: dict[str, Any] = {}
    engine = RadarEngine(config=config, ingestors=ingestors)

    docs = [
        _make_raw_doc("https://new1.com", "hn"),
        _make_raw_doc("https://new2.com", "reddit"),
    ]

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        new_count, dup_count = await engine._store_results(conn, docs)

    assert new_count == 2
    assert dup_count == 0


@pytest.mark.asyncio
async def test_store_results_skips_existing_urls(config: AppConfig) -> None:
    """_store_results() skips docs whose URLs already exist in the DB."""
    ingestors: dict[str, Any] = {}
    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)

        # Pre-insert one document
        existing = DocumentRow(
            id="existing-id",
            origin="pro",
            source_type="hn",
            url="https://already-in-db.com",
        )
        await _insert_document_row(conn, existing)

        docs = [
            _make_raw_doc("https://already-in-db.com", "hn"),  # duplicate
            _make_raw_doc("https://brand-new.com", "hn"),  # new
        ]
        new_count, dup_count = await engine._store_results(conn, docs)

    assert new_count == 1
    assert dup_count == 1


@pytest.mark.asyncio
async def test_store_results_empty_list(config: AppConfig) -> None:
    """_store_results() handles empty input and returns (0, 0)."""
    ingestors: dict[str, Any] = {}
    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        new_count, dup_count = await engine._store_results(conn, [])

    assert new_count == 0
    assert dup_count == 0


@pytest.mark.asyncio
async def test_store_results_sets_radar_origin(config: AppConfig) -> None:
    """_store_results() sets origin='radar' on stored documents."""
    ingestors: dict[str, Any] = {}
    engine = RadarEngine(config=config, ingestors=ingestors)

    # Pass a doc with origin='pro' — should be stored as 'radar'
    docs = [_make_raw_doc("https://example.com/article", "hn", origin="pro")]

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        await engine._store_results(conn, docs)

        async with conn.execute(
            "SELECT origin FROM documents WHERE url = ?",
            ("https://example.com/article",),
        ) as cursor:
            row = await cursor.fetchone()

    assert row is not None
    assert row["origin"] == "radar"


# ---------------------------------------------------------------------------
# Integration-style: full search flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_search_flow_five_sources(
    config: AppConfig, five_ingestors: dict[str, MagicMock]
) -> None:
    """End-to-end search with 5 mocked ingestors stores all unique docs correctly."""
    engine = RadarEngine(config=config, ingestors=five_ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "AI engineering", limit_per_source=10)

        # Count stored documents
        async with conn.execute("SELECT COUNT(*) as cnt FROM documents") as cursor:
            row = await cursor.fetchone()
        stored_count = row["cnt"]

    # 5 sources: 2 + 2 + 1 + 1 + 1 = 7 unique docs (no cross-source URL collisions)
    assert report.total_found == 7
    assert report.new_documents == 7
    assert stored_count == 7
    assert report.errors == {}


@pytest.mark.asyncio
async def test_full_search_flow_second_run_shows_zero_new(
    config: AppConfig, five_ingestors: dict[str, MagicMock]
) -> None:
    """Running search twice: second run finds 0 new documents (all are already in DB)."""
    engine = RadarEngine(config=config, ingestors=five_ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)

        # First run: stores everything
        report1 = await engine.search(conn, "test query", limit_per_source=10)
        # Second run: all docs already in DB
        report2 = await engine.search(conn, "test query", limit_per_source=10)

    assert report1.new_documents > 0
    assert report2.new_documents == 0
    assert report2.total_found == report1.total_found  # same total found


# ---------------------------------------------------------------------------
# Helpers for in-memory DB setup
# ---------------------------------------------------------------------------


async def _create_documents_table(conn: aiosqlite.Connection) -> None:
    """Create a minimal documents table in an in-memory SQLite DB for testing.

    This mirrors the schema from the real migration (task_03) but only
    includes the columns needed for radar engine tests.

    Args:
        conn: An open aiosqlite connection.
    """
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            origin TEXT NOT NULL CHECK (origin IN ('pro', 'radar', 'adhoc')),
            source_type TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            title TEXT,
            author TEXT,
            published_at TEXT,
            fetched_at TEXT DEFAULT (datetime('now')),
            content_type TEXT,
            raw_content TEXT,
            word_count INTEGER,
            metadata TEXT DEFAULT '{}',
            is_embedded INTEGER DEFAULT 0,
            is_entities_extracted INTEGER DEFAULT 0,
            filter_score REAL,
            filter_passed INTEGER,
            is_favorited INTEGER DEFAULT 0,
            is_archived INTEGER DEFAULT 0,
            user_tags TEXT DEFAULT '[]',
            user_notes TEXT,
            promoted_at TEXT,
            deleted_at TEXT
        )
    """)
    await conn.commit()


async def _insert_document_row(
    conn: aiosqlite.Connection, doc: DocumentRow
) -> None:
    """Insert a DocumentRow directly for test setup.

    Args:
        conn: An open aiosqlite connection.
        doc: The DocumentRow to insert.
    """
    import json
    await conn.execute(
        """
        INSERT OR IGNORE INTO documents (
            id, source_id, origin, source_type, url, title, author,
            published_at, content_type, raw_content, word_count,
            metadata, is_embedded, is_entities_extracted, filter_score,
            filter_passed, is_favorited, is_archived, user_tags, user_notes,
            promoted_at, deleted_at
        ) VALUES (
            :id, :source_id, :origin, :source_type, :url, :title, :author,
            :published_at, :content_type, :raw_content, :word_count,
            :metadata, :is_embedded, :is_entities_extracted, :filter_score,
            :filter_passed, :is_favorited, :is_archived, :user_tags, :user_notes,
            :promoted_at, :deleted_at
        )
        """,
        {
            "id": doc.id,
            "source_id": doc.source_id,
            "origin": doc.origin,
            "source_type": doc.source_type,
            "url": doc.url,
            "title": doc.title,
            "author": doc.author,
            "published_at": doc.published_at,
            "content_type": doc.content_type,
            "raw_content": doc.raw_content,
            "word_count": doc.word_count,
            "metadata": json.dumps(doc.metadata),
            "is_embedded": doc.is_embedded,
            "is_entities_extracted": doc.is_entities_extracted,
            "filter_score": doc.filter_score,
            "filter_passed": doc.filter_passed,
            "is_favorited": doc.is_favorited,
            "is_archived": doc.is_archived,
            "user_tags": json.dumps(doc.user_tags),
            "user_notes": doc.user_notes,
            "promoted_at": doc.promoted_at,
            "deleted_at": doc.deleted_at,
        },
    )
    await conn.commit()
