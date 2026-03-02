"""Unit tests for the ArXiv ingestor.

Tests cover:
- _parse_atom_feed() parses fixture Atom XML into RawDocuments
- _entry_to_raw_doc() maps fields correctly: url, title, author, abstract, metadata
- Canonical URL stored without version suffix
- Multiple authors joined as comma-separated string
- PDF URL constructed correctly
- arxiv_id stored in metadata (short form)
- categories stored in metadata as list
- published_at parsed into timezone-aware datetime
- fetch_pro() returns [] when arxiv config is None
- fetch_pro() handles HTTP error gracefully (returns [])
- fetch_pro() filters papers older than lookback window
- fetch_pro() deduplicates across queries
- search_radar() searches with arbitrary query string
- search_radar() handles HTTP error gracefully
- search_radar() respects limit
- Rate-limit sleep is called between consecutive requests
"""
import textwrap
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call
import xml.etree.ElementTree as ET

import httpx
import pytest

from ai_craftsman_kb.ingestors.arxiv import ArxivIngestor, _canonical_url, _pdf_url, _arxiv_id_from_raw
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.config.models import (
    AppConfig,
    ArxivConfig,
    FiltersConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    SettingsConfig,
    SourcesConfig,
)


# ---------------------------------------------------------------------------
# Fixture XML helpers
# ---------------------------------------------------------------------------

ATOM_NS = "http://www.w3.org/2005/Atom"

SAMPLE_ATOM_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <title>ArXiv Query Results</title>
      <entry>
        <id>http://arxiv.org/abs/2501.12345v1</id>
        <title>
          Attention Is All You Need: A Revisit
        </title>
        <author><name>Alice Smith</name></author>
        <author><name>Bob Jones</name></author>
        <summary>
          We revisit the transformer architecture and propose improvements.
        </summary>
        <published>2025-01-15T00:00:00Z</published>
        <updated>2025-01-16T00:00:00Z</updated>
        <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
        <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
        <link href="https://arxiv.org/abs/2501.12345" rel="alternate" type="text/html"/>
        <link href="https://arxiv.org/pdf/2501.12345" rel="related" type="application/pdf"/>
      </entry>
    </feed>
""")

SAMPLE_ATOM_XML_MULTI = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <title>ArXiv Query Results</title>
      <entry>
        <id>http://arxiv.org/abs/2501.11111v2</id>
        <title>Paper One</title>
        <author><name>Alice Smith</name></author>
        <summary>Abstract for paper one.</summary>
        <published>2025-01-10T00:00:00Z</published>
        <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2501.22222v1</id>
        <title>Paper Two</title>
        <author><name>Carol White</name></author>
        <summary>Abstract for paper two.</summary>
        <published>2025-01-12T00:00:00Z</published>
        <category term="cs.NE" scheme="http://arxiv.org/schemas/atom"/>
      </entry>
    </feed>
""")

EMPTY_ATOM_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>ArXiv Query Results</title>
    </feed>
""")

MALFORMED_XML = "this is not valid xml <<<"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_llm_routing() -> LLMRoutingConfig:
    """Build a minimal LLMRoutingConfig for testing."""
    task_cfg = LLMTaskConfig(provider="openai", model="gpt-4o-mini")
    return LLMRoutingConfig(
        filtering=task_cfg,
        entity_extraction=task_cfg,
        briefing=task_cfg,
        source_discovery=task_cfg,
    )


def _make_config(arxiv_cfg: ArxivConfig | None = None) -> AppConfig:
    """Build a minimal AppConfig for testing.

    Args:
        arxiv_cfg: ArxivConfig to include, or None to simulate unconfigured arxiv.

    Returns:
        An AppConfig instance with minimal settings.
    """
    return AppConfig(
        sources=SourcesConfig(arxiv=arxiv_cfg),
        settings=SettingsConfig(llm=_make_llm_routing()),
        filters=FiltersConfig(),
    )


def _make_mock_response(body: str, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response returning `body` as text.

    Args:
        body: The response body text (Atom XML).
        status_code: HTTP status code.

    Returns:
        A MagicMock mimicking an httpx.Response.
    """
    mock_resp = MagicMock()
    mock_resp.text = body
    mock_resp.status_code = status_code
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=MagicMock(),
        )
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# Helper URL function tests
# ---------------------------------------------------------------------------


def test_canonical_url_strips_version_suffix() -> None:
    """_canonical_url() strips the version suffix from the raw ArXiv ID."""
    raw = "http://arxiv.org/abs/2501.12345v1"
    assert _canonical_url(raw) == "https://arxiv.org/abs/2501.12345"


def test_canonical_url_strips_higher_version() -> None:
    """_canonical_url() strips version suffixes beyond v1."""
    raw = "http://arxiv.org/abs/2501.12345v3"
    assert _canonical_url(raw) == "https://arxiv.org/abs/2501.12345"


def test_canonical_url_upgrades_to_https() -> None:
    """_canonical_url() uses HTTPS even when raw ID uses HTTP."""
    raw = "http://arxiv.org/abs/2501.12345v1"
    assert _canonical_url(raw).startswith("https://")


def test_pdf_url_strips_version() -> None:
    """_pdf_url() builds PDF URL without version suffix."""
    raw = "http://arxiv.org/abs/2501.12345v1"
    assert _pdf_url(raw) == "https://arxiv.org/pdf/2501.12345"


def test_arxiv_id_from_raw_returns_short_id() -> None:
    """_arxiv_id_from_raw() returns just the paper ID portion."""
    raw = "http://arxiv.org/abs/2501.12345v1"
    assert _arxiv_id_from_raw(raw) == "2501.12345"


# ---------------------------------------------------------------------------
# _parse_atom_feed() tests
# ---------------------------------------------------------------------------


def test_parse_atom_feed_returns_raw_documents() -> None:
    """_parse_atom_feed() returns a list of RawDocuments from valid Atom XML."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    docs = ingestor._parse_atom_feed(SAMPLE_ATOM_XML)

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)


def test_parse_atom_feed_parses_multiple_entries() -> None:
    """_parse_atom_feed() handles feeds with multiple entries."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    docs = ingestor._parse_atom_feed(SAMPLE_ATOM_XML_MULTI)

    assert len(docs) == 2


def test_parse_atom_feed_returns_empty_on_no_entries() -> None:
    """_parse_atom_feed() returns [] when feed has no <entry> elements."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    docs = ingestor._parse_atom_feed(EMPTY_ATOM_XML)

    assert docs == []


def test_parse_atom_feed_returns_empty_on_malformed_xml() -> None:
    """_parse_atom_feed() returns [] and logs error on malformed XML."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    docs = ingestor._parse_atom_feed(MALFORMED_XML)

    assert docs == []


def test_parse_atom_feed_sets_origin() -> None:
    """_parse_atom_feed() applies the origin parameter to all documents."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    docs = ingestor._parse_atom_feed(SAMPLE_ATOM_XML, origin="radar")

    assert docs[0].origin == "radar"


# ---------------------------------------------------------------------------
# _entry_to_raw_doc() field mapping tests
# ---------------------------------------------------------------------------


def _get_sample_entry() -> ET.Element:
    """Parse SAMPLE_ATOM_XML and return the first <entry> Element."""
    root = ET.fromstring(SAMPLE_ATOM_XML)
    return root.find(f"{{{ATOM_NS}}}entry")  # type: ignore[return-value]


def test_entry_to_raw_doc_canonical_url() -> None:
    """_entry_to_raw_doc() stores canonical URL without version suffix."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert doc.url == "https://arxiv.org/abs/2501.12345"


def test_entry_to_raw_doc_url_is_https() -> None:
    """_entry_to_raw_doc() URL always uses HTTPS."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert doc.url.startswith("https://")


def test_entry_to_raw_doc_title_normalized() -> None:
    """_entry_to_raw_doc() normalizes whitespace in the title."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    # Newlines in title should be collapsed
    assert doc.title == "Attention Is All You Need: A Revisit"
    assert "\n" not in (doc.title or "")


def test_entry_to_raw_doc_multiple_authors_joined() -> None:
    """_entry_to_raw_doc() joins multiple author names with ', '."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert doc.author == "Alice Smith, Bob Jones"


def test_entry_to_raw_doc_raw_content_is_abstract() -> None:
    """_entry_to_raw_doc() uses the summary (abstract) as raw_content."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert doc.raw_content is not None
    assert "transformer architecture" in doc.raw_content


def test_entry_to_raw_doc_raw_content_normalized() -> None:
    """_entry_to_raw_doc() normalizes whitespace in the abstract."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert doc.raw_content is not None
    assert "\n" not in doc.raw_content


def test_entry_to_raw_doc_content_type_is_paper() -> None:
    """_entry_to_raw_doc() sets content_type='paper'."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert doc.content_type == "paper"


def test_entry_to_raw_doc_source_type_is_arxiv() -> None:
    """_entry_to_raw_doc() sets source_type='arxiv'."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert doc.source_type == "arxiv"


def test_entry_to_raw_doc_default_origin_is_pro() -> None:
    """_entry_to_raw_doc() defaults to origin='pro'."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert doc.origin == "pro"


def test_entry_to_raw_doc_radar_origin() -> None:
    """_entry_to_raw_doc() applies origin='radar' when specified."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry, origin="radar")

    assert doc.origin == "radar"


def test_entry_to_raw_doc_published_at_parsed() -> None:
    """_entry_to_raw_doc() parses published date into a timezone-aware datetime."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert doc.published_at is not None
    assert isinstance(doc.published_at, datetime)
    assert doc.published_at.year == 2025
    assert doc.published_at.month == 1
    assert doc.published_at.day == 15
    assert doc.published_at.tzinfo is not None


def test_entry_to_raw_doc_metadata_arxiv_id() -> None:
    """_entry_to_raw_doc() includes short arxiv_id in metadata."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert "arxiv_id" in doc.metadata
    assert doc.metadata["arxiv_id"] == "2501.12345"


def test_entry_to_raw_doc_metadata_categories() -> None:
    """_entry_to_raw_doc() includes categories list in metadata."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert "categories" in doc.metadata
    assert isinstance(doc.metadata["categories"], list)
    assert "cs.CL" in doc.metadata["categories"]
    assert "cs.LG" in doc.metadata["categories"]


def test_entry_to_raw_doc_metadata_pdf_url() -> None:
    """_entry_to_raw_doc() includes pdf_url in metadata."""
    config = _make_config()
    ingestor = ArxivIngestor(config)
    entry = _get_sample_entry()

    doc = ingestor._entry_to_raw_doc(entry)

    assert "pdf_url" in doc.metadata
    assert doc.metadata["pdf_url"] == "https://arxiv.org/pdf/2501.12345"


def test_entry_to_raw_doc_single_author() -> None:
    """_entry_to_raw_doc() correctly handles a single author."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    root = ET.fromstring(SAMPLE_ATOM_XML_MULTI)
    entry = root.find(f"{{{ATOM_NS}}}entry")  # First entry has one author

    doc = ingestor._entry_to_raw_doc(entry)  # type: ignore[arg-type]

    assert doc.author == "Alice Smith"
    assert "," not in (doc.author or "")


# ---------------------------------------------------------------------------
# fetch_pro() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_arxiv_config_is_none() -> None:
    """fetch_pro() returns [] when arxiv config is None."""
    config = _make_config(arxiv_cfg=None)
    ingestor = ArxivIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_returns_documents() -> None:
    """fetch_pro() returns RawDocuments for a configured query."""
    config = _make_config(
        arxiv_cfg=ArxivConfig(queries=["cat:cs.CL"], max_results=5)
    )
    ingestor = ArxivIngestor(config)

    # Paper published recently (within lookback window)
    recent_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    xml_with_recent = SAMPLE_ATOM_XML.replace("2025-01-15T00:00:00Z", recent_date)

    mock_resp = _make_mock_response(xml_with_recent)
    ingestor._rate_limited_get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0].source_type == "arxiv"
    assert docs[0].origin == "pro"


@pytest.mark.asyncio
async def test_fetch_pro_filters_old_papers() -> None:
    """fetch_pro() excludes papers older than the lookback window."""
    config = _make_config(
        arxiv_cfg=ArxivConfig(queries=["cat:cs.CL"], max_results=5)
    )
    ingestor = ArxivIngestor(config)

    # Paper published 30 days ago (outside 7-day lookback)
    old_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    xml_with_old = SAMPLE_ATOM_XML.replace("2025-01-15T00:00:00Z", old_date)

    mock_resp = _make_mock_response(xml_with_old)
    ingestor._rate_limited_get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_deduplicates_across_queries() -> None:
    """fetch_pro() deduplicates papers that appear in multiple query results."""
    config = _make_config(
        arxiv_cfg=ArxivConfig(queries=["cat:cs.CL", "cat:cs.AI"], max_results=5)
    )
    ingestor = ArxivIngestor(config)

    recent_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    xml_recent = SAMPLE_ATOM_XML.replace("2025-01-15T00:00:00Z", recent_date)
    mock_resp = _make_mock_response(xml_recent)
    # Both queries return the same paper
    ingestor._rate_limited_get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    # Should only appear once despite being returned by 2 queries
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_fetch_pro_handles_http_error_gracefully() -> None:
    """fetch_pro() skips query and continues on HTTP error, returns [] if all fail."""
    config = _make_config(
        arxiv_cfg=ArxivConfig(queries=["cat:cs.CL"], max_results=5)
    )
    ingestor = ArxivIngestor(config)

    async def raise_error(params: dict) -> None:
        raise httpx.ConnectError("Connection refused")

    ingestor._rate_limited_get = raise_error  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_handles_http_status_error() -> None:
    """fetch_pro() skips query on HTTP 5xx response."""
    config = _make_config(
        arxiv_cfg=ArxivConfig(queries=["cat:cs.CL"], max_results=5)
    )
    ingestor = ArxivIngestor(config)

    mock_resp = _make_mock_response("", status_code=503)
    ingestor._rate_limited_get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_multiple_queries_all_succeed() -> None:
    """fetch_pro() fetches all queries and combines results."""
    config = _make_config(
        arxiv_cfg=ArxivConfig(
            queries=["cat:cs.CL", "cat:cs.AI"], max_results=5
        )
    )
    ingestor = ArxivIngestor(config)

    recent_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # Two different papers for two different queries
    xml_paper1 = SAMPLE_ATOM_XML.replace("2025-01-15T00:00:00Z", recent_date)
    xml_paper2 = SAMPLE_ATOM_XML.replace(
        "http://arxiv.org/abs/2501.12345v1",
        "http://arxiv.org/abs/2501.99999v1",
    ).replace("2025-01-15T00:00:00Z", recent_date)

    responses = [
        _make_mock_response(xml_paper1),
        _make_mock_response(xml_paper2),
    ]
    call_count = 0

    async def mock_get(params: dict) -> MagicMock:
        nonlocal call_count
        resp = responses[call_count % len(responses)]
        call_count += 1
        return resp

    ingestor._rate_limited_get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    # Should have 2 unique papers from 2 queries
    assert len(docs) == 2
    assert call_count == 2


# ---------------------------------------------------------------------------
# search_radar() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_returns_documents() -> None:
    """search_radar() returns RawDocuments with origin='radar'."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    mock_resp = _make_mock_response(SAMPLE_ATOM_XML)
    ingestor._rate_limited_get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    docs = await ingestor.search_radar("large language models")

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0].origin == "radar"
    assert docs[0].source_type == "arxiv"


@pytest.mark.asyncio
async def test_search_radar_passes_query_to_api() -> None:
    """search_radar() passes the query string as search_query param."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    captured_params: list[dict] = []
    mock_resp = _make_mock_response(EMPTY_ATOM_XML)

    async def mock_get(params: dict) -> MagicMock:
        captured_params.append(dict(params))
        return mock_resp

    ingestor._rate_limited_get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("cat:cs.CL AND abs:attention")

    assert len(captured_params) == 1
    assert captured_params[0]["search_query"] == "cat:cs.CL AND abs:attention"


@pytest.mark.asyncio
async def test_search_radar_passes_limit_as_max_results() -> None:
    """search_radar() passes limit as max_results param."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    captured_params: list[dict] = []
    mock_resp = _make_mock_response(EMPTY_ATOM_XML)

    async def mock_get(params: dict) -> MagicMock:
        captured_params.append(dict(params))
        return mock_resp

    ingestor._rate_limited_get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("test query", limit=15)

    assert captured_params[0]["max_results"] == 15


@pytest.mark.asyncio
async def test_search_radar_respects_limit() -> None:
    """search_radar() returns at most limit documents."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    mock_resp = _make_mock_response(SAMPLE_ATOM_XML_MULTI)
    ingestor._rate_limited_get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    docs = await ingestor.search_radar("llm", limit=1)

    assert len(docs) <= 1


@pytest.mark.asyncio
async def test_search_radar_handles_http_error() -> None:
    """search_radar() returns [] on HTTP connection error."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    async def raise_error(params: dict) -> None:
        raise httpx.ConnectError("Connection refused")

    ingestor._rate_limited_get = raise_error  # type: ignore[method-assign]

    docs = await ingestor.search_radar("test query")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_handles_http_status_error() -> None:
    """search_radar() returns [] on HTTP 5xx response."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    mock_resp = _make_mock_response("", status_code=500)
    ingestor._rate_limited_get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    docs = await ingestor.search_radar("test query")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_no_date_filtering() -> None:
    """search_radar() does not filter by date (old papers can surface)."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    # Paper published 30 days ago — should still be returned in radar mode
    old_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    xml_with_old = SAMPLE_ATOM_XML.replace("2025-01-15T00:00:00Z", old_date)

    mock_resp = _make_mock_response(xml_with_old)
    ingestor._rate_limited_get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    docs = await ingestor.search_radar("test query")

    # Old paper should still be returned in radar mode
    assert len(docs) == 1


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_returns_self() -> None:
    """ArxivIngestor works as async context manager and returns self."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    async with ingestor as ctx:
        assert ctx is ingestor


@pytest.mark.asyncio
async def test_context_manager_closes_client_on_exit() -> None:
    """ArxivIngestor.__aexit__ calls aclose() on the internal httpx client."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    close_called = []
    original_aclose = ingestor._client.aclose

    async def mock_aclose() -> None:
        close_called.append(True)
        await original_aclose()

    ingestor._client.aclose = mock_aclose  # type: ignore[method-assign]

    async with ingestor:
        pass

    assert close_called, "aclose() was not called on __aexit__"


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_sleep_between_requests() -> None:
    """_rate_limited_get() sleeps between consecutive API calls."""
    import time

    config = _make_config(
        arxiv_cfg=ArxivConfig(queries=["q1", "q2"], max_results=1)
    )
    ingestor = ArxivIngestor(config)

    recent_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    # Two different papers to avoid deduplication filtering out the second
    xml_paper1 = SAMPLE_ATOM_XML.replace("2025-01-15T00:00:00Z", recent_date)
    xml_paper2 = SAMPLE_ATOM_XML.replace(
        "http://arxiv.org/abs/2501.12345v1",
        "http://arxiv.org/abs/2501.99999v1",
    ).replace("2025-01-15T00:00:00Z", recent_date)

    responses = [
        _make_mock_response(xml_paper1),
        _make_mock_response(xml_paper2),
    ]
    call_idx = 0

    sleep_durations: list[float] = []

    async def mock_get_client(url: str, params: dict | None = None) -> MagicMock:
        nonlocal call_idx
        resp = responses[call_idx % len(responses)]
        call_idx += 1
        return resp

    ingestor._client.get = mock_get_client  # type: ignore[method-assign]

    with patch("asyncio.sleep") as mock_sleep:
        mock_sleep.return_value = None
        # Simulate first request already having been made recently
        import time as time_module
        ingestor._last_request_time = time_module.monotonic()

        # Force a second call by manually calling _rate_limited_get twice
        mock_resp1 = _make_mock_response(EMPTY_ATOM_XML)
        mock_resp2 = _make_mock_response(EMPTY_ATOM_XML)

        async def _fast_client_get(url: str, **kwargs: object) -> MagicMock:
            return mock_resp1

        ingestor._client.get = _fast_client_get  # type: ignore[method-assign]
        await ingestor._rate_limited_get({"search_query": "q1", "start": 0, "max_results": 1})
        await ingestor._rate_limited_get({"search_query": "q2", "start": 0, "max_results": 1})

        # asyncio.sleep should have been called for the second request
        assert mock_sleep.called, "asyncio.sleep should be called to enforce rate limit"
