"""Phase 2 integration tests: all 7 ingestors + incremental fetch + ingest_url."""
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    ArxivConfig,
    DevtoConfig,
    EmbeddingConfig,
    FiltersConfig,
    HackerNewsConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    RSSSource,
    SettingsConfig,
    SourcesConfig,
    SubstackSource,
    SubredditSource,
    YoutubeChannelSource,
)
from ai_craftsman_kb.db.queries import get_stats, list_documents
from ai_craftsman_kb.db.sqlite import get_db, init_db
from ai_craftsman_kb.ingestors.arxiv import ArxivIngestor
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.ingestors.devto import DevtoIngestor
from ai_craftsman_kb.ingestors.hackernews import HackerNewsIngestor
from ai_craftsman_kb.ingestors.reddit import RedditIngestor
from ai_craftsman_kb.ingestors.rss import RSSIngestor
from ai_craftsman_kb.ingestors.runner import INGESTORS, IngestReport, IngestRunner
from ai_craftsman_kb.ingestors.substack import SubstackIngestor
from ai_craftsman_kb.ingestors.youtube import YouTubeIngestor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_config(tmp_path: Path) -> AppConfig:
    """AppConfig with all 7 source types configured for testing.

    Each source type has at least minimal configuration so that the ingestor
    can attempt to run (even if it immediately returns an empty list due to
    missing credentials or empty feed).

    Args:
        tmp_path: pytest-provided temporary directory, unique per test.

    Returns:
        A fully constructed AppConfig for phase 2 integration testing.
    """
    return AppConfig(
        sources=SourcesConfig(
            hackernews=HackerNewsConfig(mode="top", limit=5),
            substack=[SubstackSource(slug="testpub", name="Test Publication")],
            rss=[RSSSource(url="https://example.com/feed.xml", name="Test Feed")],
            youtube_channels=[YoutubeChannelSource(handle="@TestChannel", name="Test Channel")],
            subreddits=[SubredditSource(name="python", sort="hot", limit=5)],
            arxiv=ArxivConfig(queries=["cat:cs.AI"], max_results=5),
            devto=DevtoConfig(tags=["python"], limit=5),
        ),
        settings=SettingsConfig(
            data_dir=str(tmp_path),
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test"),
                briefing=LLMTaskConfig(provider="anthropic", model="test"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test"),
            ),
        ),
        filters=FiltersConfig(),
    )


@pytest.fixture
def minimal_config(tmp_path: Path) -> AppConfig:
    """Minimal AppConfig with HN only.

    Args:
        tmp_path: pytest-provided temporary directory, unique per test.

    Returns:
        A minimal AppConfig for tests that only need HN.
    """
    return AppConfig(
        sources=SourcesConfig(hackernews=HackerNewsConfig(mode="top", limit=10)),
        settings=SettingsConfig(
            data_dir=str(tmp_path),
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test"),
                briefing=LLMTaskConfig(provider="anthropic", model="test"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test"),
            ),
        ),
        filters=FiltersConfig(),
    )


@pytest.fixture
def mock_llm_router() -> MagicMock:
    """LLMRouter mock that always returns score '8' from complete().

    Score '8' exceeds the default min_score of 5, so all documents pass the
    LLM filter strategy. Using a MagicMock avoids needing real API keys.

    Returns:
        A MagicMock with complete patched to return '8'.
    """
    router = MagicMock()
    router.complete = AsyncMock(return_value="8")
    return router


def _make_doc(
    url: str,
    source_type: str,
    title: str = "Test Doc",
    published_at: datetime | None = None,
    **metadata,
) -> RawDocument:
    """Helper to create a RawDocument for testing.

    Args:
        url: The document URL.
        source_type: The source type string (e.g. 'hn').
        title: Document title.
        published_at: Optional publication timestamp.
        **metadata: Additional metadata key-value pairs.

    Returns:
        A RawDocument configured for testing.
    """
    return RawDocument(
        url=url,
        title=title,
        raw_content=f"Content for {title}",
        source_type=source_type,
        origin="pro",
        published_at=published_at,
        metadata=dict(metadata),
    )


# ---------------------------------------------------------------------------
# Test 1: Full ingest pipeline — mock all 7 APIs, run run_all(), verify docs stored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_ingest_pipeline(
    full_config: AppConfig,
    tmp_path: Path,
    mock_llm_router: MagicMock,
) -> None:
    """Mock all 7 APIs, run run_all(), verify at least one doc stored per source.

    Each ingestor is patched to return exactly one RawDocument. After run_all()
    completes, the DB should contain one document per source type (7 total).
    """
    db_path = tmp_path / "craftsman.db"
    await init_db(tmp_path)

    # One sample doc per source type
    hn_doc = _make_doc("https://hn.com/1", "hn", "HN Post", hn_id="1", points=100, comment_count=5, hn_url="https://news.ycombinator.com/item?id=1")
    substack_doc = _make_doc("https://substack.com/p/1", "substack", "Substack Post")
    rss_doc = _make_doc("https://rss.example.com/1", "rss", "RSS Article")
    youtube_doc = _make_doc("https://youtube.com/watch?v=abc123", "youtube", "YT Video", video_id="abc123")
    reddit_doc = _make_doc("https://reddit.com/r/python/comments/1", "reddit", "Reddit Post")
    arxiv_doc = _make_doc("https://arxiv.org/abs/2501.12345", "arxiv", "ArXiv Paper")
    devto_doc = _make_doc("https://dev.to/user/article-1", "devto", "DEV.to Article")

    with (
        patch.object(HackerNewsIngestor, "fetch_pro", new_callable=AsyncMock, return_value=[hn_doc]),
        patch.object(SubstackIngestor, "fetch_pro", new_callable=AsyncMock, return_value=[substack_doc]),
        patch.object(RSSIngestor, "fetch_pro", new_callable=AsyncMock, return_value=[rss_doc]),
        patch.object(YouTubeIngestor, "fetch_pro", new_callable=AsyncMock, return_value=[youtube_doc]),
        patch.object(RedditIngestor, "fetch_pro", new_callable=AsyncMock, return_value=[reddit_doc]),
        patch.object(ArxivIngestor, "fetch_pro", new_callable=AsyncMock, return_value=[arxiv_doc]),
        patch.object(DevtoIngestor, "fetch_pro", new_callable=AsyncMock, return_value=[devto_doc]),
    ):
        runner = IngestRunner(full_config, mock_llm_router, db_path)
        reports = await runner.run_all()

    # Verify one report per source type
    assert len(reports) == 7
    source_types_in_reports = {r.source_type for r in reports}
    assert source_types_in_reports == set(INGESTORS.keys())

    # Verify each source stored its doc
    for report in reports:
        assert report.stored == 1, (
            f"Expected 1 stored for {report.source_type}, "
            f"got {report.stored}. Errors: {report.errors}"
        )

    # Verify DB stats
    async with get_db(tmp_path) as conn:
        stats = await get_stats(conn)
    assert stats["total_documents"] == 7
    assert stats["total_sources"] == 7


# ---------------------------------------------------------------------------
# Test 2: Incremental fetch — ingest twice, verify second run skips old items
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_fetch(
    minimal_config: AppConfig,
    tmp_path: Path,
    mock_llm_router: MagicMock,
) -> None:
    """Ingest twice: second run should skip docs published before first run.

    Scenario:
    - First run: fetch 2 docs (old + new), both stored.
    - Update the source's last_fetched_at to simulate a completed first run.
    - Second run: fetch 2 docs again (old one predates last_fetched_at, new one is after).
    - Verify: skipped_old == 1, stored == 1 (only the genuinely new doc).
    """
    db_path = tmp_path / "craftsman.db"
    await init_db(tmp_path)

    now = datetime.now(timezone.utc)
    old_published = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new_published = datetime(2026, 3, 1, tzinfo=timezone.utc)
    last_fetch_time = datetime(2026, 2, 1, tzinfo=timezone.utc)

    old_doc = RawDocument(
        url="https://example.com/old-article",
        title="Old Article",
        raw_content="Old content",
        source_type="hn",
        origin="pro",
        published_at=old_published,
        metadata={"hn_id": "100", "points": 50, "comment_count": 5,
                   "hn_url": "https://news.ycombinator.com/item?id=100"},
    )
    new_doc = RawDocument(
        url="https://example.com/new-article",
        title="New Article",
        raw_content="New content",
        source_type="hn",
        origin="pro",
        published_at=new_published,
        metadata={"hn_id": "200", "points": 100, "comment_count": 10,
                   "hn_url": "https://news.ycombinator.com/item?id=200"},
    )

    runner = IngestRunner(minimal_config, mock_llm_router, db_path)
    ingestor = HackerNewsIngestor(minimal_config)

    # --- First run: both docs are new ---
    with patch.object(HackerNewsIngestor, "fetch_pro", new_callable=AsyncMock, return_value=[old_doc, new_doc]):
        report1 = await runner.run_source(ingestor)

    assert report1.stored == 2
    assert report1.skipped_old == 0

    # Manually set last_fetched_at to simulate that the first run happened at last_fetch_time.
    # This simulates a scenario where the second run should skip old_doc.
    async with get_db(tmp_path) as conn:
        await conn.execute(
            "UPDATE sources SET last_fetched_at = ? WHERE source_type = ?",
            (last_fetch_time.isoformat(), "hn"),
        )
        await conn.commit()

    # --- Second run: old_doc is before last_fetched_at, new_doc is after ---
    # old_doc (Jan 1) is before last_fetch_time (Feb 1) → should be skipped
    # new_doc (Mar 1) is after last_fetch_time (Feb 1) → should also be dedup (already in DB)
    with patch.object(HackerNewsIngestor, "fetch_pro", new_callable=AsyncMock, return_value=[old_doc, new_doc]):
        report2 = await runner.run_source(ingestor)

    # old_doc should be skipped by incremental filter
    assert report2.skipped_old == 1
    # new_doc is after the cutoff but already in DB → duplicate
    assert report2.skipped_duplicate == 1
    assert report2.stored == 0


# ---------------------------------------------------------------------------
# Test 3: ingest_url for a generic article
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_url_article(
    minimal_config: AppConfig,
    tmp_path: Path,
) -> None:
    """Ingest an article URL via IngestRunner.ingest_url(), verify stored with origin='adhoc'.

    The AdhocIngestor is mocked to return a predetermined RawDocument so that
    no real HTTP request is made.
    """
    from ai_craftsman_kb.ingestors.adhoc import AdhocIngestor

    db_path = tmp_path / "craftsman.db"
    await init_db(tmp_path)

    article_url = "https://example.com/interesting-article"
    mock_doc = RawDocument(
        url=article_url,
        title="Interesting Article",
        raw_content="This is a very interesting article about technology.",
        source_type="adhoc",
        origin="adhoc",
        content_type="article",
        word_count=10,
        metadata={"adhoc_tags": ["tech"], "url_type": "article"},
    )

    with patch.object(AdhocIngestor, "ingest_url", new_callable=AsyncMock, return_value=mock_doc):
        runner = IngestRunner(minimal_config, llm_router=None, db_path=db_path)
        report = await runner.ingest_url(article_url, tags=["tech"])

    assert report.stored == 1
    assert report.skipped_duplicate == 0
    assert report.errors == []

    # Verify DB: document exists with origin='adhoc'
    async with get_db(tmp_path) as conn:
        docs = await list_documents(conn, origin="adhoc")
    assert len(docs) == 1
    assert docs[0].url == article_url
    assert docs[0].origin == "adhoc"
    assert docs[0].source_type == "adhoc"


# ---------------------------------------------------------------------------
# Test 4: ingest_url for a YouTube video
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_url_youtube(
    minimal_config: AppConfig,
    tmp_path: Path,
) -> None:
    """Ingest a YouTube URL, verify transcript stored with origin='adhoc'.

    The AdhocIngestor is mocked to return a RawDocument with transcript content.
    """
    from ai_craftsman_kb.ingestors.adhoc import AdhocIngestor

    db_path = tmp_path / "craftsman.db"
    await init_db(tmp_path)

    yt_url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
    mock_transcript = "This is the transcript of a very informative video about AI and machine learning."
    mock_doc = RawDocument(
        url=yt_url,
        title="Informative AI Video",
        raw_content=mock_transcript,
        source_type="adhoc",
        origin="adhoc",
        content_type="video",
        word_count=len(mock_transcript.split()),
        metadata={
            "video_id": "dQw4w9WgXcQ",
            "adhoc_tags": ["ai", "video"],
            "url_type": "youtube",
        },
    )

    with patch.object(AdhocIngestor, "ingest_url", new_callable=AsyncMock, return_value=mock_doc):
        runner = IngestRunner(minimal_config, llm_router=None, db_path=db_path)
        report = await runner.ingest_url(yt_url, tags=["ai", "video"])

    assert report.stored == 1
    assert report.errors == []

    # Verify the stored document has transcript content
    async with get_db(tmp_path) as conn:
        docs = await list_documents(conn, origin="adhoc")
    assert len(docs) == 1
    assert docs[0].content_type == "video"
    assert docs[0].raw_content == mock_transcript


# ---------------------------------------------------------------------------
# Test 5: Duplicate skip — ingest same URL twice, verify stored only once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_skip(
    minimal_config: AppConfig,
    tmp_path: Path,
) -> None:
    """Attempt to ingest the same URL twice via ingest_url; verify stored only once.

    The first call should store the document. The second call should detect
    it already exists and return skipped_duplicate=1.
    """
    from ai_craftsman_kb.ingestors.adhoc import AdhocIngestor

    db_path = tmp_path / "craftsman.db"
    await init_db(tmp_path)

    article_url = "https://example.com/duplicate-article"
    mock_doc = RawDocument(
        url=article_url,
        title="Duplicate Article",
        raw_content="Content that will be stored once.",
        source_type="adhoc",
        origin="adhoc",
        content_type="article",
        metadata={"adhoc_tags": [], "url_type": "article"},
    )

    runner = IngestRunner(minimal_config, llm_router=None, db_path=db_path)

    # First ingest — should succeed
    with patch.object(AdhocIngestor, "ingest_url", new_callable=AsyncMock, return_value=mock_doc):
        report1 = await runner.ingest_url(article_url)

    assert report1.stored == 1
    assert report1.skipped_duplicate == 0

    # Second ingest — same URL, should be deduplicated
    with patch.object(AdhocIngestor, "ingest_url", new_callable=AsyncMock, return_value=mock_doc):
        report2 = await runner.ingest_url(article_url)

    assert report2.stored == 0
    assert report2.skipped_duplicate == 1

    # Only one document should exist in the DB
    async with get_db(tmp_path) as conn:
        stats = await get_stats(conn)
    assert stats["total_documents"] == 1


# ---------------------------------------------------------------------------
# Test 6: Failed ingestor logs error and continues (run_all resilience)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_ingestor_continues_run_all(
    full_config: AppConfig,
    tmp_path: Path,
    mock_llm_router: MagicMock,
) -> None:
    """If one ingestor's fetch_pro raises, run_all continues with the remaining sources.

    HN is patched to raise an exception; all other ingestors return one doc each.
    The HN report should have errors, while other reports should have stored=1.
    """
    db_path = tmp_path / "craftsman.db"
    await init_db(tmp_path)

    working_doc = lambda src, url: _make_doc(url, src, f"{src.upper()} Doc")  # noqa: E731

    with (
        patch.object(
            HackerNewsIngestor, "fetch_pro",
            new_callable=AsyncMock,
            side_effect=Exception("HN API is down"),
        ),
        patch.object(
            SubstackIngestor, "fetch_pro",
            new_callable=AsyncMock,
            return_value=[working_doc("substack", "https://substack.com/p/1")],
        ),
        patch.object(
            RSSIngestor, "fetch_pro",
            new_callable=AsyncMock,
            return_value=[working_doc("rss", "https://rss.example.com/1")],
        ),
        patch.object(
            YouTubeIngestor, "fetch_pro",
            new_callable=AsyncMock,
            return_value=[working_doc("youtube", "https://youtube.com/watch?v=abc")],
        ),
        patch.object(
            RedditIngestor, "fetch_pro",
            new_callable=AsyncMock,
            return_value=[working_doc("reddit", "https://reddit.com/r/py/1")],
        ),
        patch.object(
            ArxivIngestor, "fetch_pro",
            new_callable=AsyncMock,
            return_value=[working_doc("arxiv", "https://arxiv.org/abs/2501.00001")],
        ),
        patch.object(
            DevtoIngestor, "fetch_pro",
            new_callable=AsyncMock,
            return_value=[working_doc("devto", "https://dev.to/user/article-1")],
        ),
    ):
        runner = IngestRunner(full_config, mock_llm_router, db_path)
        reports = await runner.run_all()

    assert len(reports) == 7

    hn_report = next(r for r in reports if r.source_type == "hn")
    assert len(hn_report.errors) > 0
    assert hn_report.stored == 0

    # All other sources should have succeeded
    for report in reports:
        if report.source_type != "hn":
            assert report.stored == 1, (
                f"{report.source_type}: expected stored=1, got {report.stored}. "
                f"Errors: {report.errors}"
            )


# ---------------------------------------------------------------------------
# Test 7: sources table updated with last_fetched_at after successful run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sources_table_updated_after_ingest(
    minimal_config: AppConfig,
    tmp_path: Path,
    mock_llm_router: MagicMock,
) -> None:
    """After a successful ingestor run, sources.last_fetched_at is set.

    This is the cornerstone of incremental fetch: the runner must persist
    last_fetched_at so subsequent runs can skip old content.
    """
    db_path = tmp_path / "craftsman.db"
    await init_db(tmp_path)

    doc = _make_doc(
        "https://example.com/article",
        "hn",
        "HN Article",
        hn_id="1",
        points=100,
        comment_count=5,
        hn_url="https://news.ycombinator.com/item?id=1",
    )

    with patch.object(HackerNewsIngestor, "fetch_pro", new_callable=AsyncMock, return_value=[doc]):
        runner = IngestRunner(minimal_config, mock_llm_router, db_path)
        ingestor = HackerNewsIngestor(minimal_config)
        report = await runner.run_source(ingestor)

    assert report.stored == 1

    # Check the sources table has last_fetched_at set
    async with get_db(tmp_path) as conn:
        async with conn.execute(
            "SELECT last_fetched_at, fetch_error FROM sources WHERE source_type = ?",
            ("hn",),
        ) as cursor:
            row = await cursor.fetchone()

    assert row is not None
    assert row["last_fetched_at"] is not None
    assert row["fetch_error"] is None  # no errors on success


# ---------------------------------------------------------------------------
# Test 8: stats command shows accurate counts after ingestion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_after_phase2_ingest(
    full_config: AppConfig,
    tmp_path: Path,
    mock_llm_router: MagicMock,
) -> None:
    """get_stats() returns accurate counts after ingesting from all 7 sources.

    After run_all() with one doc per source, total_documents should be 7
    and total_sources should be 7.
    """
    db_path = tmp_path / "craftsman.db"
    await init_db(tmp_path)

    docs_by_source = {
        "hn": _make_doc("https://hn.com/1", "hn", hn_id="1", points=100, comment_count=5, hn_url="https://news.ycombinator.com/item?id=1"),
        "substack": _make_doc("https://substack.com/p/1", "substack"),
        "rss": _make_doc("https://rss.example.com/1", "rss"),
        "youtube": _make_doc("https://youtube.com/watch?v=abc", "youtube", video_id="abc"),
        "reddit": _make_doc("https://reddit.com/r/python/1", "reddit"),
        "arxiv": _make_doc("https://arxiv.org/abs/2501.12345", "arxiv"),
        "devto": _make_doc("https://dev.to/user/1", "devto"),
    }

    ingestor_classes = {
        "hn": HackerNewsIngestor,
        "substack": SubstackIngestor,
        "rss": RSSIngestor,
        "youtube": YouTubeIngestor,
        "reddit": RedditIngestor,
        "arxiv": ArxivIngestor,
        "devto": DevtoIngestor,
    }

    patches = {
        src: patch.object(cls, "fetch_pro", new_callable=AsyncMock, return_value=[docs_by_source[src]])
        for src, cls in ingestor_classes.items()
    }

    with patches["hn"], patches["substack"], patches["rss"], patches["youtube"], \
         patches["reddit"], patches["arxiv"], patches["devto"]:
        runner = IngestRunner(full_config, mock_llm_router, db_path)
        reports = await runner.run_all()

    # All 7 should succeed
    total_stored = sum(r.stored for r in reports)
    assert total_stored == 7

    async with get_db(tmp_path) as conn:
        stats = await get_stats(conn)

    assert stats["total_documents"] == 7
    assert stats["total_sources"] == 7
