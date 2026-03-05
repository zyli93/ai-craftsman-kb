"""Unit tests for the RSS ingestor.

Tests cover:
- fetch_pro() returns empty list when no feeds configured
- fetch_pro() returns documents from all configured feeds
- fetch_pro() skips entries older than 30 days
- fetch_pro() includes entries with no published_parsed date
- fetch_pro() logs and skips feed on network/parse error (no crash)
- fetch_pro() extracts content from content:encoded when available
- fetch_pro() falls back to summary when content:encoded absent
- fetch_pro() sets raw_content=None when neither available
- search_radar() always returns empty list
- _entry_to_raw_doc() maps url, title, author correctly
- _entry_to_raw_doc() populates metadata with feed_name, feed_url, entry_id
- _entry_to_raw_doc() parses published_parsed struct_time to datetime
- _entry_to_raw_doc() sets published_at=None when published_parsed is None
- _entry_to_raw_doc() sets content_type='article' and source_type='rss'
"""
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import feedparser
import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    FiltersConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    RSSSource,
    SettingsConfig,
    SourcesConfig,
)
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.ingestors.rss import MAX_AGE_DAYS, RSSIngestor

# ---------------------------------------------------------------------------
# Sample XML feed strings for test fixtures
# ---------------------------------------------------------------------------

RSS_FEED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test RSS Feed</title>
    <link>https://example.com</link>
    <description>A test RSS feed</description>
    <item>
      <title>First Article</title>
      <link>https://example.com/first-article</link>
      <author>Jane Doe</author>
      <guid>https://example.com/first-article</guid>
      <pubDate>Mon, 10 Feb 2026 12:00:00 +0000</pubDate>
      <description>Short summary of first article.</description>
    </item>
    <item>
      <title>Second Article</title>
      <link>https://example.com/second-article</link>
      <author>John Smith</author>
      <guid>https://example.com/second-article</guid>
      <pubDate>Tue, 11 Feb 2026 08:30:00 +0000</pubDate>
      <description>Short summary of second article.</description>
    </item>
  </channel>
</rss>
"""

RSS_FEED_WITH_CONTENT_ENCODED = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Full Content Feed</title>
    <link>https://example.com</link>
    <description>Feed with content:encoded</description>
    <item>
      <title>Full Content Article</title>
      <link>https://example.com/full-content</link>
      <guid>https://example.com/full-content</guid>
      <pubDate>Wed, 12 Feb 2026 10:00:00 +0000</pubDate>
      <description>Short summary here.</description>
      <content:encoded><![CDATA[<p>This is the full article content.</p>]]></content:encoded>
    </item>
  </channel>
</rss>
"""

ATOM_FEED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Test Atom Feed</title>
  <link href="https://atom-example.com"/>
  <id>https://atom-example.com</id>
  <entry>
    <title>Atom Article</title>
    <link href="https://atom-example.com/atom-article"/>
    <id>https://atom-example.com/atom-article</id>
    <author><name>Atom Author</name></author>
    <published>2026-02-15T09:00:00Z</published>
    <summary>Atom article summary.</summary>
  </entry>
</feed>
"""

RSS_FEED_OLD_ENTRIES = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Old Entries Feed</title>
    <link>https://example.com</link>
    <description>Feed with old entries</description>
    <item>
      <title>Recent Article</title>
      <link>https://example.com/recent</link>
      <guid>https://example.com/recent</guid>
      <pubDate>Mon, 10 Feb 2026 12:00:00 +0000</pubDate>
      <description>Recent content.</description>
    </item>
    <item>
      <title>Old Article</title>
      <link>https://example.com/old</link>
      <guid>https://example.com/old</guid>
      <pubDate>Mon, 01 Jan 2024 00:00:00 +0000</pubDate>
      <description>Old content from 2024.</description>
    </item>
  </channel>
</rss>
"""

RSS_FEED_NO_DATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>No Date Feed</title>
    <link>https://example.com</link>
    <description>Feed with no dates</description>
    <item>
      <title>Undated Article</title>
      <link>https://example.com/undated</link>
      <guid>https://example.com/undated</guid>
      <description>Article with no publication date.</description>
    </item>
  </channel>
</rss>
"""

RSS_EMPTY_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
    <link>https://example.com</link>
    <description>No entries here</description>
  </channel>
</rss>
"""


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


def _make_config(rss_sources: list[RSSSource] | None = None) -> AppConfig:
    """Build a minimal AppConfig for testing with optional RSS sources.

    Args:
        rss_sources: List of RSSSource configs. Defaults to empty list.

    Returns:
        An AppConfig instance with minimal settings and the given RSS sources.
    """
    return AppConfig(
        sources=SourcesConfig(rss=rss_sources or []),
        settings=SettingsConfig(llm=_make_llm_routing()),
        filters=FiltersConfig(),
    )


def _parse_feed_from_string(xml_string: str) -> object:
    """Parse an RSS/Atom XML string using feedparser.

    Args:
        xml_string: Raw XML content to parse.

    Returns:
        A feedparser FeedParserDict with the parsed feed.
    """
    return feedparser.parse(xml_string)


def _make_mock_feed(xml_string: str) -> MagicMock:
    """Return a MagicMock that wraps a real feedparser result.

    The mock patches feedparser.parse so that _fetch_feed can call it in
    the executor, but here we directly set the return value so the mock
    is called synchronously in tests.

    Args:
        xml_string: Raw RSS/Atom XML to parse.

    Returns:
        The feedparser result (real FeedParserDict, not a MagicMock).
    """
    return feedparser.parse(xml_string)


# ---------------------------------------------------------------------------
# fetch_pro() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_no_feeds_configured() -> None:
    """fetch_pro() returns [] when config.sources.rss is empty."""
    config = _make_config(rss_sources=[])
    ingestor = RSSIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_returns_documents_for_configured_feed() -> None:
    """fetch_pro() returns RawDocuments for each entry in the feed."""
    config = _make_config(
        rss_sources=[RSSSource(url="https://example.com/feed.rss", name="Test Feed")]
    )
    ingestor = RSSIngestor(config)

    feed_result = _parse_feed_from_string(RSS_FEED_XML)

    with patch(
        "ai_craftsman_kb.ingestors.rss.feedparser.parse", return_value=feed_result
    ):
        docs = await ingestor.fetch_pro()

    assert isinstance(docs, list)
    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)
    assert all(d.source_type == "rss" for d in docs)


@pytest.mark.asyncio
async def test_fetch_pro_combines_docs_from_multiple_feeds() -> None:
    """fetch_pro() fetches all feeds and combines results."""
    config = _make_config(
        rss_sources=[
            RSSSource(url="https://feed1.com/rss", name="Feed 1"),
            RSSSource(url="https://feed2.com/rss", name="Feed 2"),
        ]
    )
    ingestor = RSSIngestor(config)

    feed_result = _parse_feed_from_string(RSS_FEED_XML)  # 2 entries

    with patch(
        "ai_craftsman_kb.ingestors.rss.feedparser.parse", return_value=feed_result
    ):
        docs = await ingestor.fetch_pro()

    # 2 feeds * 2 entries each = 4 docs total
    assert len(docs) == 4


@pytest.mark.asyncio
async def test_fetch_pro_skips_entries_older_than_30_days() -> None:
    """fetch_pro() skips entries published more than 30 days ago."""
    config = _make_config(
        rss_sources=[RSSSource(url="https://example.com/feed.rss", name="Old Feed")]
    )
    ingestor = RSSIngestor(config)

    # RSS_FEED_OLD_ENTRIES has 1 recent and 1 old entry
    feed_result = _parse_feed_from_string(RSS_FEED_OLD_ENTRIES)

    with patch(
        "ai_craftsman_kb.ingestors.rss.feedparser.parse", return_value=feed_result
    ):
        docs = await ingestor.fetch_pro()

    # Only the recent entry should be included; "Mon, 10 Feb 2026 12:00:00 +0000"
    # is within 30 days of the test date (2026-03-01)
    assert len(docs) == 1
    assert docs[0].title == "Recent Article"


@pytest.mark.asyncio
async def test_fetch_pro_includes_entries_with_no_date() -> None:
    """fetch_pro() includes entries that have no published date (published_parsed=None)."""
    config = _make_config(
        rss_sources=[RSSSource(url="https://example.com/feed.rss", name="No Date Feed")]
    )
    ingestor = RSSIngestor(config)

    feed_result = _parse_feed_from_string(RSS_FEED_NO_DATE)

    with patch(
        "ai_craftsman_kb.ingestors.rss.feedparser.parse", return_value=feed_result
    ):
        docs = await ingestor.fetch_pro()

    assert len(docs) == 1
    assert docs[0].published_at is None
    assert docs[0].title == "Undated Article"


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_for_empty_feed() -> None:
    """fetch_pro() returns [] when feed has no entries."""
    config = _make_config(
        rss_sources=[RSSSource(url="https://example.com/feed.rss", name="Empty Feed")]
    )
    ingestor = RSSIngestor(config)

    feed_result = _parse_feed_from_string(RSS_EMPTY_FEED)

    with patch(
        "ai_craftsman_kb.ingestors.rss.feedparser.parse", return_value=feed_result
    ):
        docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_skips_erroring_feed_without_crashing() -> None:
    """fetch_pro() logs and skips a feed that fails to parse/fetch, returns others."""
    config = _make_config(
        rss_sources=[
            RSSSource(url="https://bad-feed.com/rss", name="Bad Feed"),
            RSSSource(url="https://good-feed.com/rss", name="Good Feed"),
        ]
    )
    ingestor = RSSIngestor(config)

    good_feed_result = _parse_feed_from_string(RSS_FEED_XML)  # 2 entries

    call_count = 0

    def mock_parse(url: str) -> object:
        nonlocal call_count
        call_count += 1
        if "bad-feed" in url:
            raise ConnectionError("Network unreachable")
        return good_feed_result

    with patch("ai_craftsman_kb.ingestors.rss.feedparser.parse", side_effect=mock_parse):
        docs = await ingestor.fetch_pro()

    # Bad feed should be skipped; good feed should return 2 docs
    assert len(docs) == 2
    assert call_count == 2


@pytest.mark.asyncio
async def test_fetch_pro_extracts_content_encoded_when_available() -> None:
    """fetch_pro() uses content:encoded as raw_content when available."""
    config = _make_config(
        rss_sources=[
            RSSSource(url="https://example.com/feed.rss", name="Content Feed")
        ]
    )
    ingestor = RSSIngestor(config)

    feed_result = _parse_feed_from_string(RSS_FEED_WITH_CONTENT_ENCODED)

    with patch(
        "ai_craftsman_kb.ingestors.rss.feedparser.parse", return_value=feed_result
    ):
        docs = await ingestor.fetch_pro()

    assert len(docs) == 1
    # Should use content:encoded, not the summary
    assert docs[0].raw_content is not None
    assert "full article content" in docs[0].raw_content


@pytest.mark.asyncio
async def test_fetch_pro_falls_back_to_summary_when_no_content_encoded() -> None:
    """fetch_pro() uses summary as raw_content when content:encoded is absent."""
    config = _make_config(
        rss_sources=[RSSSource(url="https://example.com/feed.rss", name="Summary Feed")]
    )
    ingestor = RSSIngestor(config)

    feed_result = _parse_feed_from_string(RSS_FEED_XML)

    with patch(
        "ai_craftsman_kb.ingestors.rss.feedparser.parse", return_value=feed_result
    ):
        docs = await ingestor.fetch_pro()

    assert len(docs) == 2
    assert docs[0].raw_content == "Short summary of first article."


@pytest.mark.asyncio
async def test_fetch_pro_origin_is_pro() -> None:
    """fetch_pro() sets origin='pro' on all returned documents."""
    config = _make_config(
        rss_sources=[RSSSource(url="https://example.com/feed.rss", name="Test Feed")]
    )
    ingestor = RSSIngestor(config)

    feed_result = _parse_feed_from_string(RSS_FEED_XML)

    with patch(
        "ai_craftsman_kb.ingestors.rss.feedparser.parse", return_value=feed_result
    ):
        docs = await ingestor.fetch_pro()

    assert all(d.origin == "pro" for d in docs)


# ---------------------------------------------------------------------------
# search_radar() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_always_returns_empty_list() -> None:
    """search_radar() always returns [] (RSS has no search API)."""
    config = _make_config(
        rss_sources=[RSSSource(url="https://example.com/feed.rss", name="Test Feed")]
    )
    ingestor = RSSIngestor(config)

    docs = await ingestor.search_radar("machine learning")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_returns_empty_regardless_of_limit() -> None:
    """search_radar() returns [] for any limit value."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    docs = await ingestor.search_radar("python", limit=100)

    assert docs == []


# ---------------------------------------------------------------------------
# _entry_to_raw_doc() tests
# ---------------------------------------------------------------------------


def test_entry_to_raw_doc_maps_url_correctly() -> None:
    """_entry_to_raw_doc() uses entry.link as the document URL."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "Test Feed", "https://example.com/feed.rss")

    assert doc.url == "https://example.com/first-article"


def test_entry_to_raw_doc_maps_title_correctly() -> None:
    """_entry_to_raw_doc() maps entry.title to doc.title."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "Test Feed", "https://example.com/feed.rss")

    assert doc.title == "First Article"


def test_entry_to_raw_doc_maps_author_correctly() -> None:
    """_entry_to_raw_doc() maps entry.author to doc.author."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "Test Feed", "https://example.com/feed.rss")

    assert doc.author == "Jane Doe"


def test_entry_to_raw_doc_metadata_includes_feed_name() -> None:
    """_entry_to_raw_doc() includes feed_name in metadata."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "My Test Feed", "https://example.com/feed.rss")

    assert doc.metadata["feed_name"] == "My Test Feed"


def test_entry_to_raw_doc_metadata_includes_feed_url() -> None:
    """_entry_to_raw_doc() includes feed_url in metadata."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(
        entry, "Test Feed", "https://example.com/feed.rss"
    )

    assert doc.metadata["feed_url"] == "https://example.com/feed.rss"


def test_entry_to_raw_doc_metadata_includes_entry_id() -> None:
    """_entry_to_raw_doc() includes entry_id in metadata."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "Test Feed", "https://example.com/feed.rss")

    assert "entry_id" in doc.metadata
    assert doc.metadata["entry_id"] != ""


def test_entry_to_raw_doc_sets_content_type_article() -> None:
    """_entry_to_raw_doc() always sets content_type='article'."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "Test Feed", "https://example.com/feed.rss")

    assert doc.content_type == "article"


def test_entry_to_raw_doc_sets_source_type_rss() -> None:
    """_entry_to_raw_doc() always sets source_type='rss'."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "Test Feed", "https://example.com/feed.rss")

    assert doc.source_type == "rss"


def test_entry_to_raw_doc_parses_published_parsed_to_datetime() -> None:
    """_entry_to_raw_doc() converts published_parsed struct_time to a UTC datetime."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]  # pubDate: Mon, 10 Feb 2026 12:00:00 +0000

    doc = ingestor._entry_to_raw_doc(entry, "Test Feed", "https://example.com/feed.rss")

    assert doc.published_at is not None
    assert isinstance(doc.published_at, datetime)
    assert doc.published_at.year == 2026
    assert doc.published_at.month == 2
    assert doc.published_at.day == 10
    assert doc.published_at.hour == 12
    # Should be timezone-aware UTC
    assert doc.published_at.tzinfo is not None


def test_entry_to_raw_doc_sets_published_at_none_when_no_date() -> None:
    """_entry_to_raw_doc() sets published_at=None when published_parsed is absent."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_NO_DATE)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "No Date Feed", "https://example.com/feed.rss")

    assert doc.published_at is None


def test_entry_to_raw_doc_uses_content_encoded_over_summary() -> None:
    """_entry_to_raw_doc() prefers content:encoded over summary for raw_content."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_WITH_CONTENT_ENCODED)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "Content Feed", "https://example.com/feed.rss")

    assert doc.raw_content is not None
    assert "full article content" in doc.raw_content
    # Should NOT be the short summary
    assert "Short summary" not in doc.raw_content


def test_entry_to_raw_doc_uses_summary_when_no_content_encoded() -> None:
    """_entry_to_raw_doc() uses summary as raw_content when content:encoded absent."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "Test Feed", "https://example.com/feed.rss")

    assert doc.raw_content == "Short summary of first article."


def test_entry_to_raw_doc_sets_none_raw_content_when_no_content() -> None:
    """_entry_to_raw_doc() sets raw_content=None when neither content nor summary present."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    # Parse a minimal feed entry with no description/summary
    minimal_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Minimal Feed</title>
    <link>https://example.com</link>
    <item>
      <title>Minimal Entry</title>
      <link>https://example.com/minimal</link>
      <guid>https://example.com/minimal</guid>
    </item>
  </channel>
</rss>
"""
    feed = feedparser.parse(minimal_xml)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "Minimal Feed", "https://example.com/feed.rss")

    assert doc.raw_content is None


def test_entry_to_raw_doc_handles_atom_entry() -> None:
    """_entry_to_raw_doc() correctly parses an Atom feed entry."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(ATOM_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(
        entry, "Atom Feed", "https://atom-example.com/feed"
    )

    assert doc.url == "https://atom-example.com/atom-article"
    assert doc.title == "Atom Article"
    assert doc.author == "Atom Author"
    assert doc.source_type == "rss"
    assert doc.content_type == "article"
    assert doc.published_at is not None
    assert doc.published_at.year == 2026


def test_entry_to_raw_doc_default_origin_is_pro() -> None:
    """_entry_to_raw_doc() sets origin='pro' by default."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    feed = feedparser.parse(RSS_FEED_XML)
    entry = feed.entries[0]

    doc = ingestor._entry_to_raw_doc(entry, "Test Feed", "https://example.com/feed.rss")

    assert doc.origin == "pro"


# ---------------------------------------------------------------------------
# source_type property test
# ---------------------------------------------------------------------------


def test_rss_ingestor_source_type() -> None:
    """RSSIngestor.source_type returns 'rss'."""
    config = _make_config()
    ingestor = RSSIngestor(config)

    assert ingestor.source_type == "rss"


# ---------------------------------------------------------------------------
# MAX_AGE_DAYS constant test
# ---------------------------------------------------------------------------


def test_max_age_days_constant() -> None:
    """MAX_AGE_DAYS is set to 30 days."""
    assert MAX_AGE_DAYS == 30
