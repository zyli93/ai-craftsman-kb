"""Unit tests for SourceDiscoverer — outbound link extraction, YouTube handle
detection, LLM suggestion parsing, and deduplication logic.

All LLM calls are mocked via AsyncMock. DB interactions use an in-memory
SQLite database via the shared async_db fixture.
"""
from __future__ import annotations

import json
import uuid
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
from ai_craftsman_kb.db.models import DiscoveredSourceRow, DocumentRow, SourceRow
from ai_craftsman_kb.db.queries import list_discovered_sources, upsert_discovered_source, upsert_source
from ai_craftsman_kb.db.sqlite import SCHEMA_SQL
from ai_craftsman_kb.llm.router import LLMRouter
from ai_craftsman_kb.processing.discoverer import (
    SourceDiscoverer,
    _CONFIDENCE_DOUBLE,
    _CONFIDENCE_LLM,
    _CONFIDENCE_MULTI,
    _CONFIDENCE_SINGLE,
    _compute_confidence,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_config() -> AppConfig:
    """Build a minimal AppConfig for discoverer tests."""
    return AppConfig(
        sources=SourcesConfig(hackernews=HackerNewsConfig(mode="top", limit=10)),
        settings=SettingsConfig(
            data_dir="/tmp/test-craftsman-kb",
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test-model"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test-model"),
                briefing=LLMTaskConfig(provider="anthropic", model="test-model"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test-model"),
                keyword_extraction=LLMTaskConfig(provider="openrouter", model="test-model"),
            ),
        ),
        filters=FiltersConfig(),
    )


def make_llm_router(complete_return: str = "[]") -> LLMRouter:
    """Build a mock LLMRouter whose complete() returns a fixed string."""
    router = MagicMock(spec=LLMRouter)
    router.complete = AsyncMock(return_value=complete_return)
    return router


def make_doc(
    raw_content: str = "",
    source_type: str = "hn",
    title: str = "Test",
    url: str | None = None,
) -> DocumentRow:
    """Build a minimal DocumentRow for testing."""
    doc_id = str(uuid.uuid4())
    return DocumentRow(
        id=doc_id,
        origin="pro",
        source_type=source_type,
        url=url or f"https://example.com/{doc_id}",
        title=title,
        raw_content=raw_content,
    )


def make_discoverer(complete_return: str = "[]") -> tuple[SourceDiscoverer, LLMRouter]:
    """Create a SourceDiscoverer with a mock LLMRouter."""
    config = make_config()
    router = make_llm_router(complete_return)
    discoverer = SourceDiscoverer(config=config, llm_router=router)
    return discoverer, router


@pytest.fixture
async def mem_db() -> aiosqlite.Connection:
    """Provide an in-memory SQLite connection with the full schema applied."""
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.executescript(SCHEMA_SQL)
        await conn.commit()
        yield conn


# ---------------------------------------------------------------------------
# Tests: _compute_confidence
# ---------------------------------------------------------------------------


def test_compute_confidence_single() -> None:
    """One mention → confidence=0.4."""
    assert _compute_confidence(1) == _CONFIDENCE_SINGLE


def test_compute_confidence_double() -> None:
    """Two mentions → confidence=0.7."""
    assert _compute_confidence(2) == _CONFIDENCE_DOUBLE


def test_compute_confidence_multi() -> None:
    """Three or more mentions → confidence=0.9."""
    assert _compute_confidence(3) == _CONFIDENCE_MULTI
    assert _compute_confidence(5) == _CONFIDENCE_MULTI
    assert _compute_confidence(100) == _CONFIDENCE_MULTI


# ---------------------------------------------------------------------------
# Tests: _extract_outbound_links — Substack
# ---------------------------------------------------------------------------


def test_extract_substack_url() -> None:
    """Substack URLs are classified correctly with slug as identifier."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Check out https://stratechery.substack.com for analysis.")

    results = discoverer._extract_outbound_links(doc)

    substack = [r for r in results if r.source_type == "substack"]
    assert len(substack) == 1
    assert substack[0].identifier == "stratechery"
    assert substack[0].discovery_method == "outbound_link"


def test_extract_multiple_substack_urls() -> None:
    """Multiple Substack URLs in one document produce multiple suggestions."""
    discoverer, _ = make_discoverer()
    doc = make_doc(
        raw_content=(
            "Visit https://stratechery.substack.com and also "
            "https://www.notboring.co and https://benn.substack.com for more."
        )
    )

    results = discoverer._extract_outbound_links(doc)
    substack = [r for r in results if r.source_type == "substack"]
    identifiers = {r.identifier for r in substack}
    assert "stratechery" in identifiers
    assert "benn" in identifiers


def test_extract_no_substack_for_other_domains() -> None:
    """Non-Substack domains are not classified as Substack."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Visit https://medium.com/article-about-ai for more info.")

    results = discoverer._extract_outbound_links(doc)
    substack = [r for r in results if r.source_type == "substack"]
    assert len(substack) == 0


# ---------------------------------------------------------------------------
# Tests: _extract_outbound_links — Reddit
# ---------------------------------------------------------------------------


def test_extract_reddit_url() -> None:
    """Reddit subreddit URLs are classified with subreddit name as identifier."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="See the discussion at https://reddit.com/r/MachineLearning/comments/123")

    results = discoverer._extract_outbound_links(doc)
    reddit = [r for r in results if r.source_type == "reddit"]
    assert len(reddit) >= 1
    assert any(r.identifier == "MachineLearning" for r in reddit)


def test_extract_reddit_display_name_format() -> None:
    """Reddit sources have display_name in 'r/subreddit' format."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Posted to https://reddit.com/r/LocalLLaMA/")

    results = discoverer._extract_outbound_links(doc)
    reddit = [r for r in results if r.source_type == "reddit"]
    assert len(reddit) >= 1
    assert reddit[0].display_name == "r/LocalLLaMA"


# ---------------------------------------------------------------------------
# Tests: _extract_outbound_links — ArXiv
# ---------------------------------------------------------------------------


def test_extract_arxiv_abs_url() -> None:
    """ArXiv abstract URLs are classified with paper ID as identifier."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="See paper at https://arxiv.org/abs/2301.07041 for details.")

    results = discoverer._extract_outbound_links(doc)
    arxiv = [r for r in results if r.source_type == "arxiv"]
    assert len(arxiv) == 1
    assert arxiv[0].identifier == "2301.07041"
    assert arxiv[0].discovery_method == "citation"


def test_extract_arxiv_pdf_url() -> None:
    """ArXiv PDF URLs are also classified."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Download the paper: https://arxiv.org/pdf/2310.06825")

    results = discoverer._extract_outbound_links(doc)
    arxiv = [r for r in results if r.source_type == "arxiv"]
    assert len(arxiv) == 1
    assert arxiv[0].identifier == "2310.06825"


# ---------------------------------------------------------------------------
# Tests: _extract_outbound_links — YouTube
# ---------------------------------------------------------------------------


def test_extract_youtube_channel_handle_url() -> None:
    """YouTube @handle URLs are classified with @handle as identifier."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Check out https://youtube.com/@3blue1brown for math videos.")

    results = discoverer._extract_outbound_links(doc)
    youtube = [r for r in results if r.source_type == "youtube"]
    assert len(youtube) == 1
    assert youtube[0].identifier == "@3blue1brown"


def test_extract_youtube_c_channel_url() -> None:
    """YouTube /c/ channel URLs are classified correctly."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Watch https://youtube.com/c/LexFridman for interviews.")

    results = discoverer._extract_outbound_links(doc)
    youtube = [r for r in results if r.source_type == "youtube"]
    assert len(youtube) == 1
    assert "@LexFridman" in youtube[0].identifier or "LexFridman" in youtube[0].identifier


# ---------------------------------------------------------------------------
# Tests: _extract_youtube_handles
# ---------------------------------------------------------------------------


def test_extract_youtube_handles_from_text() -> None:
    """@handle patterns in text are extracted as YouTube mentions."""
    discoverer, _ = make_discoverer()
    doc = make_doc(
        raw_content="Great video by @AndrejKarpathy and also @lexfridman covered this.",
        source_type="youtube",
    )

    results = discoverer._extract_youtube_handles(doc)
    identifiers = {r.identifier for r in results}
    assert "@AndrejKarpathy" in identifiers
    assert "@lexfridman" in identifiers


def test_extract_youtube_handles_discovery_method() -> None:
    """YouTube handles extracted from text have discovery_method='mention'."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Recommended by @SomeChannel today.")

    results = discoverer._extract_youtube_handles(doc)
    if results:
        assert results[0].discovery_method == "mention"
        assert results[0].source_type == "youtube"


def test_extract_youtube_handles_short_handles_filtered() -> None:
    """Handles shorter than 3 characters are filtered out."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Tagged @ab and @abc and @longer_handle.")

    results = discoverer._extract_youtube_handles(doc)
    identifiers = {r.identifier for r in results}
    # @ab should be filtered (2 chars)
    assert "@ab" not in identifiers


def test_extract_youtube_handles_no_content() -> None:
    """Document with no raw_content returns no YouTube handles."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="")

    results = discoverer._extract_youtube_handles(doc)
    assert results == []


# ---------------------------------------------------------------------------
# Tests: _parse_llm_response
# ---------------------------------------------------------------------------


def test_parse_llm_response_valid_json() -> None:
    """Valid JSON array with all required fields produces DiscoveredSourceRow objects."""
    discoverer, _ = make_discoverer()
    response = json.dumps([
        {"source_type": "substack", "identifier": "stratechery", "display_name": "Stratechery", "reason": "Great tech analysis"},
        {"source_type": "youtube", "identifier": "@3blue1brown", "display_name": "3Blue1Brown", "reason": "Math visualizations"},
    ])

    results = discoverer._parse_llm_response(response)
    assert len(results) == 2
    assert results[0].source_type == "substack"
    assert results[0].identifier == "stratechery"
    assert results[0].discovery_method == "llm_suggestion"
    assert results[0].confidence == _CONFIDENCE_LLM


def test_parse_llm_response_invalid_json() -> None:
    """Invalid JSON in LLM response returns empty list without raising."""
    discoverer, _ = make_discoverer()
    results = discoverer._parse_llm_response("This is not JSON at all!")
    assert results == []


def test_parse_llm_response_with_surrounding_text() -> None:
    """JSON array embedded in surrounding text is still extracted correctly."""
    discoverer, _ = make_discoverer()
    response = (
        "Here are my suggestions:\n"
        '[{"source_type": "reddit", "identifier": "LocalLLaMA", "display_name": "LocalLLaMA", "reason": "Good LLM discussions"}]\n'
        "I hope this helps!"
    )
    results = discoverer._parse_llm_response(response)
    assert len(results) == 1
    assert results[0].source_type == "reddit"
    assert results[0].identifier == "LocalLLaMA"


def test_parse_llm_response_missing_required_fields() -> None:
    """Items missing source_type or identifier are silently skipped."""
    discoverer, _ = make_discoverer()
    response = json.dumps([
        {"source_type": "substack", "display_name": "Missing identifier"},
        {"identifier": "noSourceType", "display_name": "Missing source_type"},
        {"source_type": "youtube", "identifier": "@valid_channel", "display_name": "Valid"},
    ])

    results = discoverer._parse_llm_response(response)
    # Only the third item is valid
    assert len(results) == 1
    assert results[0].identifier == "@valid_channel"


def test_parse_llm_response_respects_limit() -> None:
    """_parse_llm_response truncates results to the specified limit."""
    discoverer, _ = make_discoverer()
    items = [
        {"source_type": "reddit", "identifier": f"sub_{i}", "display_name": f"Sub {i}"}
        for i in range(10)
    ]
    response = json.dumps(items)

    results = discoverer._parse_llm_response(response, limit=3)
    assert len(results) == 3


def test_parse_llm_response_empty_array() -> None:
    """Empty JSON array returns empty list."""
    discoverer, _ = make_discoverer()
    results = discoverer._parse_llm_response("[]")
    assert results == []


def test_parse_llm_response_non_list_json() -> None:
    """Non-list JSON (e.g., a dict) returns empty list."""
    discoverer, _ = make_discoverer()
    results = discoverer._parse_llm_response('{"source_type": "substack"}')
    assert results == []


# ---------------------------------------------------------------------------
# Tests: discover_from_documents — integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_from_documents_basic(mem_db: aiosqlite.Connection) -> None:
    """Documents with known Substack URLs produce discovered source suggestions."""
    from ai_craftsman_kb.db.queries import upsert_document
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Read https://stratechery.substack.com for analysis.")
    await upsert_document(mem_db, doc)

    results = await discoverer.discover_from_documents(mem_db, [doc])

    assert len(results) == 1
    assert results[0].source_type == "substack"
    assert results[0].identifier == "stratechery"
    assert results[0].status == "suggested"


@pytest.mark.asyncio
async def test_discover_from_documents_empty_list(mem_db: aiosqlite.Connection) -> None:
    """Empty document list returns empty results without errors."""
    discoverer, _ = make_discoverer()
    results = await discoverer.discover_from_documents(mem_db, [])
    assert results == []


@pytest.mark.asyncio
async def test_discover_from_documents_dedup_across_docs(mem_db: aiosqlite.Connection) -> None:
    """Same source found in multiple documents yields one suggestion with higher confidence."""
    from ai_craftsman_kb.db.queries import upsert_document
    discoverer, _ = make_discoverer()
    docs = [
        make_doc(raw_content="Visit https://stratechery.substack.com for tech."),
        make_doc(raw_content="Also see https://stratechery.substack.com for business analysis."),
        make_doc(raw_content="Today at https://stratechery.substack.com they wrote about AI."),
    ]
    for doc in docs:
        await upsert_document(mem_db, doc)

    results = await discoverer.discover_from_documents(mem_db, docs)

    substack_results = [r for r in results if r.source_type == "substack" and r.identifier == "stratechery"]
    assert len(substack_results) == 1
    # Found in 3 docs → confidence = 0.9
    assert substack_results[0].confidence == _CONFIDENCE_MULTI


@pytest.mark.asyncio
async def test_discover_from_documents_filters_existing_sources(mem_db: aiosqlite.Connection) -> None:
    """Sources already in the sources table are not suggested again."""
    from ai_craftsman_kb.db.queries import upsert_document
    discoverer, _ = make_discoverer()

    # Pre-load stratechery as an existing configured source
    existing = SourceRow(
        id=str(uuid.uuid4()),
        source_type="substack",
        identifier="stratechery",
        display_name="Stratechery",
    )
    await upsert_source(mem_db, existing)

    doc = make_doc(raw_content="Read https://stratechery.substack.com for analysis.")
    await upsert_document(mem_db, doc)
    results = await discoverer.discover_from_documents(mem_db, [doc])

    # stratechery already in sources — should not be in results
    stratechery = [r for r in results if r.source_type == "substack" and r.identifier == "stratechery"]
    assert len(stratechery) == 0


@pytest.mark.asyncio
async def test_discover_from_documents_filters_existing_discovered(mem_db: aiosqlite.Connection) -> None:
    """Sources already in discovered_sources are not suggested again."""
    from ai_craftsman_kb.db.queries import upsert_document
    discoverer, _ = make_discoverer()

    # Pre-load as already discovered (no document reference needed for this one)
    already = DiscoveredSourceRow(
        id=str(uuid.uuid4()),
        source_type="substack",
        identifier="stratechery",
        discovered_from_document_id=None,
        discovery_method="outbound_link",
        confidence=0.4,
    )
    await upsert_discovered_source(mem_db, already)

    doc = make_doc(raw_content="Read https://stratechery.substack.com for analysis.")
    await upsert_document(mem_db, doc)
    results = await discoverer.discover_from_documents(mem_db, [doc])

    stratechery = [r for r in results if r.source_type == "substack" and r.identifier == "stratechery"]
    assert len(stratechery) == 0


@pytest.mark.asyncio
async def test_discover_from_documents_no_raw_content(mem_db: aiosqlite.Connection) -> None:
    """Documents with no raw_content are safely skipped."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="")
    doc2 = DocumentRow(
        id=str(uuid.uuid4()),
        origin="pro",
        source_type="hn",
        url="https://example.com/no-content",
        raw_content=None,
    )
    # No content → no links → empty candidates → nothing to persist
    results = await discoverer.discover_from_documents(mem_db, [doc, doc2])
    assert results == []


@pytest.mark.asyncio
async def test_discover_from_documents_persists_to_db(mem_db: aiosqlite.Connection) -> None:
    """Discovered sources are persisted to the discovered_sources table."""
    from ai_craftsman_kb.db.queries import upsert_document
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Check out https://newsletter.substack.com for insights.")
    await upsert_document(mem_db, doc)

    await discoverer.discover_from_documents(mem_db, [doc])

    # Verify persistence
    stored = await list_discovered_sources(mem_db, status="suggested")
    assert any(
        s.source_type == "substack" and s.identifier == "newsletter"
        for s in stored
    )


@pytest.mark.asyncio
async def test_discover_from_documents_multiple_source_types(mem_db: aiosqlite.Connection) -> None:
    """Document with multiple source types produces multiple suggestions."""
    from ai_craftsman_kb.db.queries import upsert_document
    discoverer, _ = make_discoverer()
    doc = make_doc(
        raw_content=(
            "References: https://stratechery.substack.com, "
            "https://reddit.com/r/MachineLearning, "
            "https://arxiv.org/abs/2301.07041, "
            "https://youtube.com/@3blue1brown"
        )
    )
    await upsert_document(mem_db, doc)

    results = await discoverer.discover_from_documents(mem_db, [doc])

    source_types = {r.source_type for r in results}
    assert "substack" in source_types
    assert "reddit" in source_types
    assert "arxiv" in source_types
    assert "youtube" in source_types


# ---------------------------------------------------------------------------
# Tests: _llm_suggestions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_suggestions_calls_complete(mem_db: aiosqlite.Connection) -> None:
    """_llm_suggestions calls LLM router with the source_discovery task."""
    llm_response = json.dumps([
        {"source_type": "substack", "identifier": "benn", "display_name": "Benn Stancil", "reason": "Data analysis"},
    ])
    discoverer, router = make_discoverer(complete_return=llm_response)

    # Insert a document so we have something to analyze
    doc = make_doc(title="Test Article", url="https://example.com/test")
    from ai_craftsman_kb.db.queries import upsert_document
    await upsert_document(mem_db, doc)

    results = await discoverer._llm_suggestions(mem_db, limit=5)

    router.complete.assert_called_once()
    call_kwargs = router.complete.call_args.kwargs
    assert call_kwargs.get("task") == "source_discovery"
    assert len(results) == 1
    assert results[0].identifier == "benn"
    assert results[0].confidence == _CONFIDENCE_LLM


@pytest.mark.asyncio
async def test_llm_suggestions_empty_db(mem_db: aiosqlite.Connection) -> None:
    """_llm_suggestions with no documents returns empty list without calling LLM."""
    discoverer, router = make_discoverer()
    results = await discoverer._llm_suggestions(mem_db, limit=5)

    assert results == []
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_llm_suggestions_handles_llm_error(mem_db: aiosqlite.Connection) -> None:
    """_llm_suggestions returns empty list when LLM call fails."""
    discoverer, router = make_discoverer()
    router.complete = AsyncMock(side_effect=RuntimeError("API timeout"))

    # Insert a document
    doc = make_doc(title="Test Article")
    from ai_craftsman_kb.db.queries import upsert_document
    await upsert_document(mem_db, doc)

    results = await discoverer._llm_suggestions(mem_db, limit=5)
    assert results == []


# ---------------------------------------------------------------------------
# Tests: run_periodic_discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_periodic_discovery_returns_count(mem_db: aiosqlite.Connection) -> None:
    """run_periodic_discovery returns the count of newly added suggestions."""
    discoverer, _ = make_discoverer()

    # Insert a recent document with a known link
    doc = make_doc(
        raw_content="Check https://awesome.substack.com for great content.",
    )
    from ai_craftsman_kb.db.queries import upsert_document
    await upsert_document(mem_db, doc)

    count = await discoverer.run_periodic_discovery(mem_db)
    assert count >= 1


@pytest.mark.asyncio
async def test_run_periodic_discovery_empty_db(mem_db: aiosqlite.Connection) -> None:
    """run_periodic_discovery with empty DB returns 0 (no LLM docs to analyze)."""
    discoverer, _ = make_discoverer()
    count = await discoverer.run_periodic_discovery(mem_db)
    assert count == 0


@pytest.mark.asyncio
async def test_run_periodic_discovery_deduplication(mem_db: aiosqlite.Connection) -> None:
    """Running periodic discovery twice does not add duplicates."""
    discoverer, _ = make_discoverer()

    doc = make_doc(raw_content="See https://test-newsletter.substack.com for tips.")
    from ai_craftsman_kb.db.queries import upsert_document
    await upsert_document(mem_db, doc)

    count1 = await discoverer.run_periodic_discovery(mem_db)
    count2 = await discoverer.run_periodic_discovery(mem_db)

    # Second run should add nothing new
    assert count1 >= 1
    assert count2 == 0


# ---------------------------------------------------------------------------
# Tests: Confidence scoring details
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confidence_two_mentions(mem_db: aiosqlite.Connection) -> None:
    """Source mentioned in exactly 2 documents gets confidence=0.7."""
    from ai_craftsman_kb.db.queries import upsert_document
    discoverer, _ = make_discoverer()
    docs = [
        make_doc(raw_content="Visit https://twomention.substack.com for news."),
        make_doc(raw_content="Also https://twomention.substack.com published an update."),
    ]
    for doc in docs:
        await upsert_document(mem_db, doc)

    results = await discoverer.discover_from_documents(mem_db, docs)
    two_mention = [
        r for r in results
        if r.source_type == "substack" and r.identifier == "twomention"
    ]
    assert len(two_mention) == 1
    assert two_mention[0].confidence == _CONFIDENCE_DOUBLE


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


def test_extract_outbound_links_empty_content() -> None:
    """Empty raw_content returns no results from outbound link extraction."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="")
    results = discoverer._extract_outbound_links(doc)
    assert results == []


def test_extract_outbound_links_no_matching_urls() -> None:
    """Content with URLs that don't match any pattern returns empty list."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Visit https://example.com/some/page for info.")
    results = discoverer._extract_outbound_links(doc)
    assert results == []


def test_extract_outbound_links_arxiv_display_name_format() -> None:
    """ArXiv suggestions have display_name in 'arxiv:PAPER_ID' format."""
    discoverer, _ = make_discoverer()
    doc = make_doc(raw_content="Paper at https://arxiv.org/abs/2312.00001")
    results = discoverer._extract_outbound_links(doc)
    arxiv = [r for r in results if r.source_type == "arxiv"]
    assert len(arxiv) == 1
    assert arxiv[0].display_name == "arxiv:2312.00001"
