"""Unit tests for the Substack ingestor.

Tests cover:
- fetch_pro() returns RawDocuments for each configured slug
- fetch_pro() returns [] when no substack sources are configured
- fetch_pro() handles malformed/empty feeds gracefully (log + skip)
- fetch_pro() deduplicates by URL within a single run
- fetch_pro() limits entries to _DEFAULT_MAX_ENTRIES per feed
- search_radar() always returns [] (no public Substack search API)
- _entry_to_raw_doc() maps fields correctly: url, title, author, published_at,
  raw_content, word_count, content_type, source_type, metadata
- _entry_to_raw_doc() prefers content:encoded over summary
- _entry_to_raw_doc() falls back to summary when content:encoded is absent
- _entry_to_raw_doc() converts HTML to plain text via html2text
- _entry_to_raw_doc() handles missing/None fields gracefully
- _entry_to_raw_doc() parses time structs into timezone-aware datetimes
"""
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    FiltersConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    SettingsConfig,
    SourcesConfig,
    SubstackSource,
)
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.ingestors.substack import SubstackIngestor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_routing() -> LLMRoutingConfig:
    """Build a minimal LLMRoutingConfig for testing."""
    task_cfg = LLMTaskConfig(provider="openai", model="gpt-4o-mini")
    return LLMRoutingConfig(
        filtering=task_cfg,
        entity_extraction=task_cfg,
        briefing=task_cfg,
        source_discovery=task_cfg,
        keyword_extraction=task_cfg,
    )


def _make_config(slugs: list[str] | None = None) -> AppConfig:
    """Build a minimal AppConfig with optional Substack slug list.

    Args:
        slugs: List of Substack slug strings to include in config.sources.substack.
               None or empty list simulates no configured publications.

    Returns:
        A minimal AppConfig for testing.
    """
    substack_sources = [SubstackSource(slug=s, name=s.title()) for s in (slugs or [])]
    return AppConfig(
        sources=SourcesConfig(substack=substack_sources),
        settings=SettingsConfig(llm=_make_llm_routing()),
        filters=FiltersConfig(),
    )


def _make_time_struct(year: int = 2025, month: int = 1, day: int = 15) -> time.struct_time:
    """Build a time.struct_time in UTC for the given date.

    Args:
        year: Four-digit year.
        month: Month (1-12).
        day: Day of month (1-31).

    Returns:
        A time.struct_time suitable for feedparser entry.published_parsed.
    """
    return time.struct_time((year, month, day, 10, 0, 0, 0, 15, 0))


# ---------------------------------------------------------------------------
# Fixture feed data
# ---------------------------------------------------------------------------

SAMPLE_ENTRY_FULL_CONTENT = {
    "id": "https://example.substack.com/p/my-post",
    "link": "https://example.substack.com/p/my-post",
    "title": "My Great Post",
    "author": "Jane Doe",
    "published_parsed": _make_time_struct(2025, 1, 15),
    "content": [{"value": "<p>Full post content here.</p>", "type": "text/html"}],
    "summary": "<p>Short summary only.</p>",
}

SAMPLE_ENTRY_SUMMARY_ONLY = {
    "id": "https://example.substack.com/p/another-post",
    "link": "https://example.substack.com/p/another-post",
    "title": "Another Post",
    "author": "Jane Doe",
    "published_parsed": _make_time_struct(2025, 1, 14),
    "content": [],
    "summary": "<p>Only a short excerpt is available for this post.</p>",
}

SAMPLE_ENTRY_MINIMAL = {
    "id": "https://example.substack.com/p/minimal",
    "link": "https://example.substack.com/p/minimal",
    "title": None,
    "author": None,
    "published_parsed": None,
    "content": [],
    "summary": "",
}


def _make_feed(entries: list[dict], bozo: bool = False) -> MagicMock:
    """Create a mock feedparser result with given entries.

    Args:
        entries: List of entry dicts to include in feed.entries.
        bozo: Whether to simulate a malformed feed (bozo=True).

    Returns:
        A MagicMock that behaves like a feedparser.FeedParserDict.
    """
    mock_feed = MagicMock()
    mock_feed.get = lambda key, default=None: {
        "entries": entries,
        "bozo": bozo,
        "bozo_exception": Exception("malformed") if bozo else None,
    }.get(key, default)
    mock_feed.__getitem__ = lambda self, key: mock_feed.get(key)
    return mock_feed


# ---------------------------------------------------------------------------
# fetch_pro() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_no_slugs_configured() -> None:
    """fetch_pro() returns [] when config.sources.substack is empty."""
    config = _make_config(slugs=[])
    ingestor = SubstackIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_returns_raw_documents_for_configured_slug() -> None:
    """fetch_pro() returns RawDocuments for each entry in a configured feed."""
    config = _make_config(slugs=["example"])
    ingestor = SubstackIngestor(config)

    mock_feed = _make_feed([SAMPLE_ENTRY_FULL_CONTENT])

    with patch("ai_craftsman_kb.ingestors.substack.feedparser.parse", return_value=mock_feed):
        docs = await ingestor.fetch_pro()

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)
    assert docs[0].source_type == "substack"
    assert docs[0].origin == "pro"


@pytest.mark.asyncio
async def test_fetch_pro_fetches_all_configured_slugs() -> None:
    """fetch_pro() calls feedparser.parse once per configured slug."""
    config = _make_config(slugs=["pub-one", "pub-two"])
    ingestor = SubstackIngestor(config)

    called_urls: list[str] = []

    def mock_parse(url: str) -> MagicMock:
        called_urls.append(url)
        return _make_feed([SAMPLE_ENTRY_FULL_CONTENT])

    with patch(
        "ai_craftsman_kb.ingestors.substack.feedparser.parse", side_effect=mock_parse
    ):
        with patch("ai_craftsman_kb.ingestors.substack.asyncio.sleep"):
            docs = await ingestor.fetch_pro()

    assert len(called_urls) == 2
    assert any("pub-one" in u for u in called_urls)
    assert any("pub-two" in u for u in called_urls)


@pytest.mark.asyncio
async def test_fetch_pro_deduplicates_by_url() -> None:
    """fetch_pro() deduplicates entries with the same URL across feeds."""
    config = _make_config(slugs=["pub-a", "pub-b"])
    ingestor = SubstackIngestor(config)

    # Both feeds return the exact same entry URL
    duplicate_entry = dict(SAMPLE_ENTRY_FULL_CONTENT)

    def mock_parse(url: str) -> MagicMock:
        return _make_feed([duplicate_entry])

    with patch(
        "ai_craftsman_kb.ingestors.substack.feedparser.parse", side_effect=mock_parse
    ):
        with patch("ai_craftsman_kb.ingestors.substack.asyncio.sleep"):
            docs = await ingestor.fetch_pro()

    # Despite two feeds each returning the same URL, only one doc should appear
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_fetch_pro_handles_malformed_feed_gracefully() -> None:
    """fetch_pro() skips malformed feeds with a log warning, does not crash."""
    config = _make_config(slugs=["bad-feed"])
    ingestor = SubstackIngestor(config)

    mock_feed = _make_feed([], bozo=True)

    with patch("ai_craftsman_kb.ingestors.substack.feedparser.parse", return_value=mock_feed):
        docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_handles_empty_feed_gracefully() -> None:
    """fetch_pro() returns [] and logs info when feed has no entries."""
    config = _make_config(slugs=["empty-pub"])
    ingestor = SubstackIngestor(config)

    mock_feed = _make_feed([])

    with patch("ai_craftsman_kb.ingestors.substack.feedparser.parse", return_value=mock_feed):
        docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_handles_feedparser_exception_gracefully() -> None:
    """fetch_pro() handles exceptions from feedparser.parse and returns []."""
    config = _make_config(slugs=["broken"])
    ingestor = SubstackIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.substack.feedparser.parse",
        side_effect=OSError("network unreachable"),
    ):
        docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_processes_both_slugs_even_if_first_fails() -> None:
    """fetch_pro() continues processing remaining slugs when one feed fails."""
    config = _make_config(slugs=["failing-pub", "working-pub"])
    ingestor = SubstackIngestor(config)

    call_count = 0

    def mock_parse(url: str) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if "failing-pub" in url:
            raise OSError("connection refused")
        return _make_feed([SAMPLE_ENTRY_FULL_CONTENT])

    with patch(
        "ai_craftsman_kb.ingestors.substack.feedparser.parse", side_effect=mock_parse
    ):
        with patch("ai_craftsman_kb.ingestors.substack.asyncio.sleep"):
            docs = await ingestor.fetch_pro()

    # Should still get results from the working feed
    assert len(docs) == 1
    assert docs[0].source_type == "substack"


@pytest.mark.asyncio
async def test_fetch_pro_limits_entries_per_feed() -> None:
    """fetch_pro() processes at most _DEFAULT_MAX_ENTRIES entries per feed."""
    from ai_craftsman_kb.ingestors.substack import _DEFAULT_MAX_ENTRIES

    config = _make_config(slugs=["big-feed"])
    ingestor = SubstackIngestor(config)

    # Create more entries than the limit
    many_entries = [
        {
            "id": f"https://big-feed.substack.com/p/post-{i}",
            "link": f"https://big-feed.substack.com/p/post-{i}",
            "title": f"Post {i}",
            "author": "Author",
            "published_parsed": _make_time_struct(2025, 1, i % 28 + 1),
            "content": [{"value": f"<p>Content {i}</p>", "type": "text/html"}],
            "summary": "",
        }
        for i in range(_DEFAULT_MAX_ENTRIES + 5)
    ]

    mock_feed = _make_feed(many_entries)

    with patch("ai_craftsman_kb.ingestors.substack.feedparser.parse", return_value=mock_feed):
        docs = await ingestor.fetch_pro()

    assert len(docs) <= _DEFAULT_MAX_ENTRIES


# ---------------------------------------------------------------------------
# search_radar() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_returns_empty_list() -> None:
    """search_radar() always returns [] since Substack has no public search API."""
    config = _make_config(slugs=["some-pub"])
    ingestor = SubstackIngestor(config)

    docs = await ingestor.search_radar("machine learning")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_returns_empty_list_regardless_of_query() -> None:
    """search_radar() returns [] for any query string."""
    config = _make_config()
    ingestor = SubstackIngestor(config)

    for query in ["python", "AI", "", "some long complicated query with many words"]:
        docs = await ingestor.search_radar(query)
        assert docs == [], f"Expected [] for query={query!r}, got {docs}"


@pytest.mark.asyncio
async def test_search_radar_returns_empty_list_with_custom_limit() -> None:
    """search_radar() returns [] regardless of the limit parameter."""
    config = _make_config()
    ingestor = SubstackIngestor(config)

    docs = await ingestor.search_radar("test", limit=100)

    assert docs == []


# ---------------------------------------------------------------------------
# _entry_to_raw_doc() tests
# ---------------------------------------------------------------------------


def _make_ingestor(slugs: list[str] | None = None) -> SubstackIngestor:
    """Build a SubstackIngestor for use in unit tests of _entry_to_raw_doc()."""
    return SubstackIngestor(_make_config(slugs=slugs or ["test-pub"]))


def test_entry_to_raw_doc_url_from_link() -> None:
    """_entry_to_raw_doc() uses the entry link as doc.url."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "example")
    assert doc.url == "https://example.substack.com/p/my-post"


def test_entry_to_raw_doc_title() -> None:
    """_entry_to_raw_doc() maps entry title to doc.title."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "example")
    assert doc.title == "My Great Post"


def test_entry_to_raw_doc_author() -> None:
    """_entry_to_raw_doc() maps entry author to doc.author."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "example")
    assert doc.author == "Jane Doe"


def test_entry_to_raw_doc_content_type_is_article() -> None:
    """_entry_to_raw_doc() sets content_type='article' always."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "example")
    assert doc.content_type == "article"


def test_entry_to_raw_doc_source_type_is_substack() -> None:
    """_entry_to_raw_doc() sets source_type='substack' always."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "example")
    assert doc.source_type == "substack"


def test_entry_to_raw_doc_origin_is_pro() -> None:
    """_entry_to_raw_doc() sets origin='pro' always."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "example")
    assert doc.origin == "pro"


def test_entry_to_raw_doc_metadata_includes_substack_slug() -> None:
    """_entry_to_raw_doc() includes substack_slug in metadata."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "my-publication")
    assert doc.metadata["substack_slug"] == "my-publication"


def test_entry_to_raw_doc_metadata_includes_post_id() -> None:
    """_entry_to_raw_doc() includes post_id in metadata from entry.id."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "example")
    assert doc.metadata["post_id"] == "https://example.substack.com/p/my-post"


def test_entry_to_raw_doc_prefers_content_encoded_over_summary() -> None:
    """_entry_to_raw_doc() uses content:encoded (entry.content[]) not summary when both present."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "example")
    # "Full post content here." comes from content:encoded; summary has "Short summary only."
    assert doc.raw_content is not None
    assert "Full post content here" in doc.raw_content
    assert "Short summary only" not in doc.raw_content


def test_entry_to_raw_doc_falls_back_to_summary_when_no_content_encoded() -> None:
    """_entry_to_raw_doc() uses summary when content:encoded is absent/empty."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_SUMMARY_ONLY, "example")
    assert doc.raw_content is not None
    assert "short excerpt" in doc.raw_content.lower()


def test_entry_to_raw_doc_converts_html_to_plain_text() -> None:
    """_entry_to_raw_doc() strips HTML tags from content, returning plain text."""
    ingestor = _make_ingestor()
    entry = {
        **SAMPLE_ENTRY_FULL_CONTENT,
        "content": [{"value": "<h1>Title</h1><p>Plain text content.</p>", "type": "text/html"}],
    }
    doc = ingestor._entry_to_raw_doc(entry, "example")
    assert doc.raw_content is not None
    # HTML tags should not appear in the plain text output
    assert "<h1>" not in doc.raw_content
    assert "<p>" not in doc.raw_content
    assert "Plain text content" in doc.raw_content


def test_entry_to_raw_doc_sets_word_count() -> None:
    """_entry_to_raw_doc() computes word_count from the extracted plain text."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "example")
    assert doc.word_count is not None
    assert doc.word_count > 0


def test_entry_to_raw_doc_word_count_is_none_when_no_content() -> None:
    """_entry_to_raw_doc() sets word_count=None when there is no extractable content."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_MINIMAL, "example")
    assert doc.raw_content is None
    assert doc.word_count is None


def test_entry_to_raw_doc_parses_published_date() -> None:
    """_entry_to_raw_doc() converts published_parsed time struct to UTC datetime."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "example")
    assert doc.published_at is not None
    assert isinstance(doc.published_at, datetime)
    assert doc.published_at.year == 2025
    assert doc.published_at.month == 1
    assert doc.published_at.day == 15
    # Must be timezone-aware (UTC)
    assert doc.published_at.tzinfo is not None


def test_entry_to_raw_doc_published_at_is_none_when_no_time_struct() -> None:
    """_entry_to_raw_doc() sets published_at=None when published_parsed is absent."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_MINIMAL, "example")
    assert doc.published_at is None


def test_entry_to_raw_doc_missing_title_gives_none() -> None:
    """_entry_to_raw_doc() sets title=None when entry has no title."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_MINIMAL, "example")
    assert doc.title is None


def test_entry_to_raw_doc_missing_author_gives_none() -> None:
    """_entry_to_raw_doc() sets author=None when entry has no author field."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_MINIMAL, "example")
    assert doc.author is None


def test_entry_to_raw_doc_falls_back_to_author_detail_name() -> None:
    """_entry_to_raw_doc() extracts author name from author_detail when author is absent."""
    ingestor = _make_ingestor()
    entry = {
        **SAMPLE_ENTRY_MINIMAL,
        "author": None,
        "author_detail": {"name": "Detail Author", "email": "a@b.com"},
    }
    doc = ingestor._entry_to_raw_doc(entry, "example")
    assert doc.author == "Detail Author"


def test_entry_to_raw_doc_slug_stored_in_metadata() -> None:
    """_entry_to_raw_doc() stores the passed slug in metadata.substack_slug."""
    ingestor = _make_ingestor()
    doc = ingestor._entry_to_raw_doc(SAMPLE_ENTRY_FULL_CONTENT, "stratechery")
    assert doc.metadata["substack_slug"] == "stratechery"


def test_entry_to_raw_doc_empty_post_id_in_metadata() -> None:
    """_entry_to_raw_doc() stores empty string for post_id when entry has no id."""
    ingestor = _make_ingestor()
    entry = {**SAMPLE_ENTRY_FULL_CONTENT, "id": None}
    doc = ingestor._entry_to_raw_doc(entry, "example")
    assert doc.metadata["post_id"] == ""


def test_entry_to_raw_doc_with_updated_parsed_fallback() -> None:
    """_entry_to_raw_doc() uses updated_parsed when published_parsed is absent."""
    ingestor = _make_ingestor()
    entry = {
        **SAMPLE_ENTRY_FULL_CONTENT,
        "published_parsed": None,
        "updated_parsed": _make_time_struct(2025, 6, 20),
    }
    doc = ingestor._entry_to_raw_doc(entry, "example")
    assert doc.published_at is not None
    assert doc.published_at.month == 6
    assert doc.published_at.day == 20
