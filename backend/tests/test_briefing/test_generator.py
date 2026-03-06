"""Unit tests for BriefingGenerator.

Tests cover:
- generate() calls hybrid search and LLM completion
- generate() stores briefing to DB and returns BriefingRow
- generate() triggers radar search when run_radar=True
- generate() triggers ingest when run_ingest=True
- generate() skips radar/ingest when flags are False
- generate() handles empty search results gracefully
- generate() handles LLM failures by re-raising
- generate() handles radar failures gracefully (does not abort)
- generate() handles ingest failures gracefully (does not abort)
- _assemble_context() formats documents correctly
- _assemble_context() truncates per-doc to _MAX_CHARS_PER_DOC
- _assemble_context() respects total context budget _MAX_TOTAL_CHARS
- _assemble_context() returns placeholder string for empty doc list
- _extract_title() extracts H1 headings from LLM output
- _extract_title() falls back to "Briefing: {topic}" when no H1 found
- source_document_ids are stored in the briefing row
- briefing is retrievable via get_briefing() after generation
"""
import re
from unittest.mock import AsyncMock, MagicMock

from ai_craftsman_kb.llm import CompletionResult

import aiosqlite
import pytest

from ai_craftsman_kb.briefing.generator import (
    BriefingGenerator,
    _MAX_CHARS_PER_DOC,
    _MAX_TOTAL_CHARS,
)
from ai_craftsman_kb.db.models import BriefingRow
from ai_craftsman_kb.db.queries import get_briefing
from ai_craftsman_kb.db.sqlite import get_db, init_db
from ai_craftsman_kb.search.hybrid import SearchResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_search_result(
    doc_id: str = "doc-1",
    title: str = "Test Article",
    source_type: str = "hn",
    published_at: str = "2025-01-15T10:00:00+00:00",
    excerpt: str = "This is a test excerpt about LLM inference.",
    origin: str = "pro",
) -> SearchResult:
    """Build a minimal SearchResult for testing."""
    return SearchResult(
        document_id=doc_id,
        score=0.9,
        title=title,
        url=f"https://example.com/{doc_id}",
        source_type=source_type,
        author="Test Author",
        published_at=published_at,
        excerpt=excerpt,
        origin=origin,
    )


def _make_generator(
    config=None,
    llm_response: str = "# Test Briefing\n\n## Key Themes\n- Theme 1\n- Theme 2",
    search_results: list[SearchResult] | None = None,
    radar_raises: bool = False,
    ingest_raises: bool = False,
) -> BriefingGenerator:
    """Build a BriefingGenerator with mocked dependencies.

    Args:
        config: Optional AppConfig. If None, uses MagicMock.
        llm_response: The string the mock LLM router will return.
        search_results: Documents returned by hybrid_search.search(). Defaults to
            one sample SearchResult.
        radar_raises: If True, radar_engine.search() raises an exception.
        ingest_raises: If True, ingest_runner.run_all() raises an exception.

    Returns:
        A BriefingGenerator instance with mocked dependencies.
    """
    if config is None:
        config = MagicMock()

    if search_results is None:
        search_results = [_make_search_result()]

    # Mock LLM router
    llm_router = MagicMock()
    llm_router.complete = AsyncMock(return_value=CompletionResult(text=llm_response))

    # Mock hybrid search
    hybrid_search = MagicMock()
    hybrid_search.search = AsyncMock(return_value=search_results)

    # Mock radar engine
    radar_engine = MagicMock()
    if radar_raises:
        radar_engine.search = AsyncMock(side_effect=RuntimeError("radar failed"))
    else:
        radar_mock = MagicMock(total_found=3, new_documents=2)
        radar_engine.search = AsyncMock(return_value=radar_mock)

    # Mock ingest runner
    ingest_runner = MagicMock()
    if ingest_raises:
        ingest_runner.run_all = AsyncMock(side_effect=RuntimeError("ingest failed"))
    else:
        ingest_runner.run_all = AsyncMock(return_value=([], []))

    return BriefingGenerator(
        config=config,
        llm_router=llm_router,
        hybrid_search=hybrid_search,
        radar_engine=radar_engine,
        ingest_runner=ingest_runner,
    )


@pytest.fixture
async def db_conn(tmp_path):
    """Provide a fresh in-memory-like SQLite connection for testing.

    Uses a temporary file-based DB so that insert_briefing and get_briefing
    can both run on the same initialized schema.
    """
    await init_db(tmp_path)
    async with get_db(tmp_path) as conn:
        yield conn


# ---------------------------------------------------------------------------
# generate() — main pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_briefing_row(db_conn):
    """generate() should return a BriefingRow with non-empty content."""
    generator = _make_generator()
    result = await generator.generate(db_conn, topic="LLM inference")

    assert isinstance(result, BriefingRow)
    assert result.id
    assert result.query == "LLM inference"
    assert len(result.content) > 0
    assert result.format == "markdown"


@pytest.mark.asyncio
async def test_generate_saves_to_db(db_conn):
    """generate() should save the briefing to the DB so it is retrievable."""
    generator = _make_generator()
    result = await generator.generate(db_conn, topic="vector search")

    # Re-fetch from DB to verify persistence
    fetched = await get_briefing(db_conn, result.id)
    assert fetched is not None
    assert fetched.id == result.id
    assert fetched.query == "vector search"
    assert fetched.content == result.content


@pytest.mark.asyncio
async def test_generate_calls_hybrid_search(db_conn):
    """generate() must call hybrid_search.search() with the topic."""
    generator = _make_generator()
    await generator.generate(db_conn, topic="embeddings")

    generator._hybrid_search.search.assert_awaited_once()
    call_args = generator._hybrid_search.search.call_args
    # First positional arg after conn should be the query
    assert "embeddings" in str(call_args)


@pytest.mark.asyncio
async def test_generate_calls_llm_router(db_conn):
    """generate() must call LLMRouter.complete(task='briefing', ...)."""
    generator = _make_generator()
    await generator.generate(db_conn, topic="transformers")

    generator._llm_router.complete.assert_awaited_once()
    call_kwargs = generator._llm_router.complete.call_args
    assert call_kwargs.kwargs.get("task") == "briefing" or call_kwargs.args[0] == "briefing"


@pytest.mark.asyncio
async def test_generate_stores_source_document_ids(db_conn):
    """generate() must store source document IDs in the BriefingRow."""
    docs = [
        _make_search_result(doc_id="doc-aaa"),
        _make_search_result(doc_id="doc-bbb"),
        _make_search_result(doc_id="doc-ccc"),
    ]
    generator = _make_generator(search_results=docs)
    result = await generator.generate(db_conn, topic="AI safety")

    assert set(result.source_document_ids) == {"doc-aaa", "doc-bbb", "doc-ccc"}


@pytest.mark.asyncio
async def test_generate_with_run_radar_true(db_conn):
    """generate() calls radar_engine.search() when run_radar=True."""
    generator = _make_generator()
    await generator.generate(db_conn, topic="RLHF", run_radar=True, run_ingest=False)

    generator._radar_engine.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_with_run_radar_false(db_conn):
    """generate() skips radar_engine.search() when run_radar=False."""
    generator = _make_generator()
    await generator.generate(db_conn, topic="RLHF", run_radar=False, run_ingest=False)

    generator._radar_engine.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_with_run_ingest_true(db_conn):
    """generate() calls ingest_runner.run_all() when run_ingest=True."""
    generator = _make_generator()
    await generator.generate(db_conn, topic="fine-tuning", run_radar=False, run_ingest=True)

    generator._ingest_runner.run_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_with_run_ingest_false(db_conn):
    """generate() skips ingest_runner.run_all() when run_ingest=False."""
    generator = _make_generator()
    await generator.generate(db_conn, topic="fine-tuning", run_radar=False, run_ingest=False)

    generator._ingest_runner.run_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_handles_radar_failure_gracefully(db_conn):
    """generate() should continue (not raise) when radar search fails."""
    generator = _make_generator(radar_raises=True)
    # Should not raise despite radar failure
    result = await generator.generate(db_conn, topic="GPU architecture", run_radar=True)

    assert isinstance(result, BriefingRow)


@pytest.mark.asyncio
async def test_generate_handles_ingest_failure_gracefully(db_conn):
    """generate() should continue (not raise) when pro ingest fails."""
    generator = _make_generator(ingest_raises=True)
    result = await generator.generate(db_conn, topic="attention mechanism", run_ingest=True)

    assert isinstance(result, BriefingRow)


@pytest.mark.asyncio
async def test_generate_with_empty_search_results(db_conn):
    """generate() should handle empty search results without crashing."""
    generator = _make_generator(search_results=[])
    result = await generator.generate(db_conn, topic="obscure topic")

    assert isinstance(result, BriefingRow)
    assert result.source_document_ids == []
    # LLM should still have been called
    generator._llm_router.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_reraises_llm_failure(db_conn):
    """generate() should re-raise LLM completion errors."""
    generator = _make_generator()
    generator._llm_router.complete = AsyncMock(side_effect=RuntimeError("LLM API error"))

    with pytest.raises(RuntimeError, match="LLM API error"):
        await generator.generate(db_conn, topic="test topic")


@pytest.mark.asyncio
async def test_generate_title_extracted_from_llm_output(db_conn):
    """generate() should extract H1 heading from LLM output as the title."""
    generator = _make_generator(
        llm_response="# My Custom Briefing Title\n\n## Key Themes\n- AI\n"
    )
    result = await generator.generate(db_conn, topic="AI trends")

    assert result.title == "My Custom Briefing Title"


@pytest.mark.asyncio
async def test_generate_title_falls_back_when_no_h1(db_conn):
    """generate() should use 'Briefing: {topic}' when no H1 heading is found."""
    generator = _make_generator(
        llm_response="## Key Themes\n- No H1 heading here\n"
    )
    result = await generator.generate(db_conn, topic="my topic")

    assert result.title == "Briefing: my topic"


# ---------------------------------------------------------------------------
# _assemble_context() tests
# ---------------------------------------------------------------------------


def test_assemble_context_formats_documents():
    """_assemble_context() should include title, source_type, published_at, and excerpt."""
    generator = _make_generator()
    doc = _make_search_result(
        doc_id="d1",
        title="LLM Inference Optimization",
        source_type="arxiv",
        published_at="2025-01-15T00:00:00+00:00",
        excerpt="Methods for efficient LLM inference.",
    )
    context = generator._assemble_context([doc], "LLM inference")

    assert "LLM Inference Optimization" in context
    assert "arxiv" in context
    assert "2025-01-15" in context
    assert "Methods for efficient LLM inference." in context


def test_assemble_context_includes_index_number():
    """_assemble_context() should number documents starting from [1]."""
    generator = _make_generator()
    docs = [
        _make_search_result(doc_id=f"doc-{i}", title=f"Article {i}")
        for i in range(1, 4)
    ]
    context = generator._assemble_context(docs, "test")

    assert "[1]" in context
    assert "[2]" in context
    assert "[3]" in context


def test_assemble_context_truncates_per_doc_to_budget():
    """_assemble_context() should truncate individual docs at _MAX_CHARS_PER_DOC chars."""
    generator = _make_generator()
    long_excerpt = "A" * 2000  # Way over per-doc budget
    doc = _make_search_result(excerpt=long_excerpt)
    context = generator._assemble_context([doc], "topic")

    # The block for this doc should be capped and end with '...'
    assert len(context) <= _MAX_CHARS_PER_DOC + 10  # +10 for "..." and small overhead
    assert context.endswith("...")


def test_assemble_context_respects_total_budget():
    """_assemble_context() should stop adding docs when _MAX_TOTAL_CHARS is reached."""
    generator = _make_generator()
    # Each doc excerpt is _MAX_CHARS_PER_DOC - a bit, so after many docs the total exceeds budget
    docs = [
        _make_search_result(
            doc_id=f"doc-{i}",
            title=f"Document {i}",
            excerpt="X" * 700,  # Just under per-doc budget
        )
        for i in range(100)  # Far more than the budget allows
    ]
    context = generator._assemble_context(docs, "topic")

    # Should be within budget
    assert len(context) <= _MAX_TOTAL_CHARS + 100  # small overshoot tolerance for last doc


def test_assemble_context_empty_docs():
    """_assemble_context() should return a placeholder when no docs are provided."""
    generator = _make_generator()
    context = generator._assemble_context([], "my topic")

    assert "No documents found" in context or "my topic" in context


def test_assemble_context_handles_none_fields():
    """_assemble_context() should handle SearchResults with None title/excerpt/published_at."""
    generator = _make_generator()
    doc = SearchResult(
        document_id="doc-null",
        score=0.5,
        title=None,
        url="https://example.com/null",
        source_type="rss",
        author=None,
        published_at=None,
        excerpt=None,
        origin="pro",
    )
    # Should not raise
    context = generator._assemble_context([doc], "topic")
    assert isinstance(context, str)
    assert len(context) > 0


# ---------------------------------------------------------------------------
# _extract_title() tests
# ---------------------------------------------------------------------------


def test_extract_title_from_h1():
    """_extract_title() should extract the first H1 heading."""
    generator = _make_generator()
    content = "# My Great Briefing\n\nSome content here."
    assert generator._extract_title(content, "topic") == "My Great Briefing"


def test_extract_title_from_h1_mid_content():
    """_extract_title() should find H1 heading even when it's not the first line."""
    generator = _make_generator()
    content = "Some preamble\n# The Real Title\n\nMore content."
    assert generator._extract_title(content, "topic") == "The Real Title"


def test_extract_title_fallback_no_h1():
    """_extract_title() should fall back to 'Briefing: {topic}' when no H1."""
    generator = _make_generator()
    content = "## Section Heading\nNo H1 heading.\n### Another Section\n"
    assert generator._extract_title(content, "LLM trends") == "Briefing: LLM trends"


def test_extract_title_fallback_empty_content():
    """_extract_title() should handle empty content gracefully."""
    generator = _make_generator()
    assert generator._extract_title("", "empty topic") == "Briefing: empty topic"


def test_extract_title_with_extra_spaces():
    """_extract_title() should strip whitespace from extracted titles."""
    generator = _make_generator()
    content = "#   Padded Title   \nContent."
    assert generator._extract_title(content, "topic") == "Padded Title"


# ---------------------------------------------------------------------------
# Integration: DB round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_briefing_db_round_trip(tmp_path):
    """Full integration: generate a briefing and verify it is retrievable."""
    await init_db(tmp_path)

    docs = [
        _make_search_result(doc_id="id-1", title="Article about LLMs"),
        _make_search_result(doc_id="id-2", title="Vector DB comparison"),
    ]
    generator = _make_generator(
        llm_response="# LLM Inference Briefing\n\n## Key Themes\n- Theme A\n- Theme B\n",
        search_results=docs,
    )

    async with get_db(tmp_path) as conn:
        result = await generator.generate(conn, topic="LLM inference", run_radar=False, run_ingest=False)

    # Re-open a new connection to verify the row was committed
    async with get_db(tmp_path) as conn2:
        fetched = await get_briefing(conn2, result.id)

    assert fetched is not None
    assert fetched.title == "LLM Inference Briefing"
    assert fetched.query == "LLM inference"
    assert "id-1" in fetched.source_document_ids
    assert "id-2" in fetched.source_document_ids
    assert fetched.format == "markdown"
