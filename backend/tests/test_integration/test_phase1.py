"""Phase 1 integration tests: HN ingest -> filter -> DB storage."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
from ai_craftsman_kb.db.queries import get_stats, list_documents
from ai_craftsman_kb.db.sqlite import get_db, init_db
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.ingestors.hackernews import HackerNewsIngestor
from ai_craftsman_kb.ingestors.runner import IngestReport, IngestRunner


# ---------------------------------------------------------------------------
# Sample data that mirrors what HackerNewsIngestor.fetch_pro() returns
# ---------------------------------------------------------------------------

_SAMPLE_DOCS = [
    RawDocument(
        url="https://example.com/llm-research",
        title="LLM Research Breakthrough",
        author="researcher",
        raw_content="Full article content about LLMs",
        content_type="post",
        source_type="hn",
        origin="pro",
        metadata={
            "hn_id": "111",
            "points": 200,
            "comment_count": 50,
            "hn_url": "https://news.ycombinator.com/item?id=111",
        },
    ),
    RawDocument(
        url="https://example.com/python-framework",
        title="New Python Framework Released",
        author="developer",
        raw_content="Article about new Python framework",
        content_type="post",
        source_type="hn",
        origin="pro",
        metadata={
            "hn_id": "222",
            "points": 150,
            "comment_count": 30,
            "hn_url": "https://news.ycombinator.com/item?id=222",
        },
    ),
    RawDocument(
        url="https://news.ycombinator.com/item?id=333",
        title="Ask HN: Best AI tools for 2025?",
        author="curious_user",
        raw_content="What are the best AI tools developers should know about in 2025?",
        content_type="post",
        source_type="hn",
        origin="pro",
        metadata={
            "hn_id": "333",
            "points": 75,
            "comment_count": 20,
            "hn_url": "https://news.ycombinator.com/item?id=333",
        },
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config(tmp_path: Path) -> AppConfig:
    """Minimal AppConfig pointing data_dir at tmp_path.

    Each test gets a fresh isolated data directory via pytest's tmp_path.
    No real API keys are needed — LLM calls are mocked in tests.

    Args:
        tmp_path: pytest-provided temporary directory, unique per test.

    Returns:
        A fully constructed AppConfig for testing.
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
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for storing test DB files.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Path to the temporary data directory.
    """
    return tmp_path


@pytest.fixture
def mock_llm_router() -> MagicMock:
    """LLMRouter mock that always returns score '8' from complete().

    Score '8' exceeds the default min_score of 5, so all documents pass
    the LLM filter strategy. Using a MagicMock avoids needing real API keys.

    Returns:
        A MagicMock with complete patched to return '8'.
    """
    router = MagicMock()
    router.complete = AsyncMock(return_value="8")
    return router


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hn_ingest_stores_docs(
    minimal_config: AppConfig,
    tmp_data_dir: Path,
    mock_llm_router: MagicMock,
) -> None:
    """Full pipeline: mock HN fetch_pro -> IngestRunner -> 3 docs stored in DB.

    Verifies that all three sample documents are fetched, pass the filter,
    are deduplicated (none exist yet), and are stored with the correct URLs.
    """
    db_path = tmp_data_dir / "craftsman.db"
    await init_db(tmp_data_dir)

    with patch.object(
        HackerNewsIngestor, "fetch_pro", new_callable=AsyncMock, return_value=_SAMPLE_DOCS
    ):
        runner = IngestRunner(minimal_config, mock_llm_router, db_path)
        ingestor = HackerNewsIngestor(minimal_config)
        report = await runner.run_source(ingestor)

    assert report.fetched == 3
    assert report.stored == 3
    assert report.skipped_duplicate == 0
    assert report.errors == []

    async with get_db(tmp_data_dir) as conn:
        docs = await list_documents(conn)
        assert len(docs) == 3
        urls = {d.url for d in docs}
        assert "https://example.com/llm-research" in urls
        assert "https://example.com/python-framework" in urls
        assert "https://news.ycombinator.com/item?id=333" in urls


@pytest.mark.asyncio
async def test_duplicate_urls_skipped(
    minimal_config: AppConfig,
    tmp_data_dir: Path,
    mock_llm_router: MagicMock,
) -> None:
    """Running ingest twice stores docs only once (URL dedup).

    The second run should see 0 stored and 1 skipped_duplicate because
    the URL already exists in the DB from the first run.
    """
    db_path = tmp_data_dir / "craftsman.db"
    await init_db(tmp_data_dir)

    single_doc = [RawDocument(
        url="https://example.com/article",
        title="Article",
        raw_content="Content",
        source_type="hn",
        origin="pro",
        metadata={
            "hn_id": "999",
            "points": 100,
            "comment_count": 5,
            "hn_url": "https://news.ycombinator.com/item?id=999",
        },
    )]

    with patch.object(
        HackerNewsIngestor, "fetch_pro", new_callable=AsyncMock, return_value=single_doc
    ):
        runner = IngestRunner(minimal_config, mock_llm_router, db_path)
        ingestor = HackerNewsIngestor(minimal_config)
        report1 = await runner.run_source(ingestor)
        report2 = await runner.run_source(ingestor)

    assert report1.stored == 1
    assert report1.skipped_duplicate == 0

    assert report2.stored == 0
    assert report2.skipped_duplicate == 1


@pytest.mark.asyncio
async def test_get_stats_after_ingest(
    minimal_config: AppConfig,
    tmp_data_dir: Path,
    mock_llm_router: MagicMock,
) -> None:
    """Stats reflect stored document count after ingest.

    After ingesting 5 documents, get_stats() should report total_documents=5
    and at least one source in total_sources.
    """
    db_path = tmp_data_dir / "craftsman.db"
    await init_db(tmp_data_dir)

    docs = [
        RawDocument(
            url=f"https://example.com/{i}",
            title=f"Article {i}",
            raw_content="content",
            source_type="hn",
            origin="pro",
            metadata={
                "hn_id": str(i),
                "points": 100,
                "comment_count": 5,
                "hn_url": f"https://news.ycombinator.com/item?id={i}",
            },
        )
        for i in range(5)
    ]

    with patch.object(
        HackerNewsIngestor, "fetch_pro", new_callable=AsyncMock, return_value=docs
    ):
        runner = IngestRunner(minimal_config, mock_llm_router, db_path)
        ingestor = HackerNewsIngestor(minimal_config)
        await runner.run_source(ingestor)

    async with get_db(tmp_data_dir) as conn:
        stats = await get_stats(conn)

    assert stats["total_documents"] == 5
    assert stats["total_sources"] >= 1


@pytest.mark.asyncio
async def test_ingest_runner_handles_fetch_error(
    minimal_config: AppConfig,
    tmp_data_dir: Path,
    mock_llm_router: MagicMock,
) -> None:
    """If fetch_pro raises, report has one error entry and zero stored docs.

    Network failures during fetch should not crash the runner — they should
    be captured as errors in the IngestReport.
    """
    db_path = tmp_data_dir / "craftsman.db"
    await init_db(tmp_data_dir)

    with patch.object(
        HackerNewsIngestor,
        "fetch_pro",
        new_callable=AsyncMock,
        side_effect=Exception("Network timeout"),
    ):
        runner = IngestRunner(minimal_config, mock_llm_router, db_path)
        ingestor = HackerNewsIngestor(minimal_config)
        report = await runner.run_source(ingestor)

    assert report.stored == 0
    assert len(report.errors) == 1
    assert "Network timeout" in report.errors[0]


@pytest.mark.asyncio
async def test_run_all_runs_hn(
    minimal_config: AppConfig,
    tmp_data_dir: Path,
    mock_llm_router: MagicMock,
) -> None:
    """run_all() runs all registered ingestors and returns one report per source.

    With only 'hn' registered in INGESTORS, run_all() should return exactly
    one IngestReport with source_type='hn' and stored=1.
    """
    db_path = tmp_data_dir / "craftsman.db"
    await init_db(tmp_data_dir)

    docs = [RawDocument(
        url="https://example.com/a",
        title="A",
        raw_content="x",
        source_type="hn",
        origin="pro",
        metadata={
            "hn_id": "1",
            "points": 100,
            "comment_count": 5,
            "hn_url": "https://news.ycombinator.com/item?id=1",
        },
    )]

    with patch.object(
        HackerNewsIngestor, "fetch_pro", new_callable=AsyncMock, return_value=docs
    ):
        runner = IngestRunner(minimal_config, mock_llm_router, db_path)
        reports = await runner.run_all()

    assert len(reports) == 1
    assert reports[0].source_type == "hn"
    assert reports[0].stored == 1
    assert reports[0].errors == []


@pytest.mark.asyncio
async def test_source_row_not_duplicated_on_second_run(
    minimal_config: AppConfig,
    tmp_data_dir: Path,
    mock_llm_router: MagicMock,
) -> None:
    """Running ingest twice does not create duplicate source rows.

    The runner should reuse the existing source row ID rather than
    inserting a new row for the same (source_type, identifier) pair.
    """
    db_path = tmp_data_dir / "craftsman.db"
    await init_db(tmp_data_dir)

    doc1 = [RawDocument(
        url="https://example.com/first",
        title="First",
        raw_content="content",
        source_type="hn",
        origin="pro",
        metadata={"hn_id": "10", "points": 100, "comment_count": 5,
                   "hn_url": "https://news.ycombinator.com/item?id=10"},
    )]
    doc2 = [RawDocument(
        url="https://example.com/second",
        title="Second",
        raw_content="content",
        source_type="hn",
        origin="pro",
        metadata={"hn_id": "11", "points": 100, "comment_count": 5,
                   "hn_url": "https://news.ycombinator.com/item?id=11"},
    )]

    runner = IngestRunner(minimal_config, mock_llm_router, db_path)
    ingestor = HackerNewsIngestor(minimal_config)

    with patch.object(HackerNewsIngestor, "fetch_pro", new_callable=AsyncMock, return_value=doc1):
        await runner.run_source(ingestor)

    with patch.object(HackerNewsIngestor, "fetch_pro", new_callable=AsyncMock, return_value=doc2):
        await runner.run_source(ingestor)

    async with get_db(tmp_data_dir) as conn:
        stats = await get_stats(conn)

    # Should still be exactly one source row for 'hn'
    assert stats["total_sources"] == 1
    assert stats["total_documents"] == 2


@pytest.mark.asyncio
async def test_ingest_report_fields(
    minimal_config: AppConfig,
    tmp_data_dir: Path,
    mock_llm_router: MagicMock,
) -> None:
    """IngestReport model validates and serialises correctly.

    Verifies that IngestReport can be constructed from kwargs and that
    its fields have correct default values.
    """
    report = IngestReport(source_type="hn")
    assert report.source_type == "hn"
    assert report.fetched == 0
    assert report.passed_filter == 0
    assert report.stored == 0
    assert report.skipped_duplicate == 0
    assert report.errors == []

    # Verify it serialises cleanly to dict
    as_dict = report.model_dump()
    assert as_dict["source_type"] == "hn"
    assert isinstance(as_dict["errors"], list)
