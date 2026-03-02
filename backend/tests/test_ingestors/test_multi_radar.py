"""Integration tests for multi-source radar fan-out: Reddit, HN, ArXiv, DEV.to.

Tests cover:
- RadarEngine.search() fans out to all 4 sources concurrently
- All 4 sources return well-formed RawDocuments with origin='radar'
- Results are deduplicated by URL across sources
- New documents stored with origin='radar'
- HN: link story content fetch attempted via ContentExtractor (semaphore=5)
- HN: text-only (Ask HN) stories returned without extra fetch
- Reddit: self-post content from selftext, no extra fetch
- Reddit: link post content fetch attempted via ContentExtractor (semaphore=3)
- ArXiv: abstract used as content (no additional fetch needed)
- DEV.to: full body_markdown fetched for each result (semaphore=5)
- Per-source errors recorded without interrupting other sources
"""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import httpx
import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    ArxivConfig,
    DevtoConfig,
    FiltersConfig,
    HackerNewsConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    RedditAPIConfig,
    SettingsConfig,
    SourcesConfig,
)
from ai_craftsman_kb.ingestors.arxiv import ArxivIngestor
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.ingestors.devto import DevtoIngestor
from ai_craftsman_kb.ingestors.hackernews import HackerNewsIngestor
from ai_craftsman_kb.ingestors.reddit import RedditIngestor
from ai_craftsman_kb.radar.engine import RadarEngine, RadarReport


# ---------------------------------------------------------------------------
# Sample API responses
# ---------------------------------------------------------------------------

SAMPLE_HN_LINK_HIT = {
    "objectID": "42001234",
    "title": "LLM Inference Optimization Techniques",
    "url": "https://example.com/llm-inference",
    "author": "researcher",
    "points": 250,
    "num_comments": 80,
    "created_at": "2025-01-15T10:00:00Z",
    "story_text": None,
}

SAMPLE_HN_TEXT_HIT = {
    "objectID": "42009999",
    "title": "Ask HN: Best LLM inference tools?",
    "url": None,
    "author": "asker",
    "points": 55,
    "num_comments": 20,
    "created_at": "2025-01-15T12:00:00Z",
    "story_text": "Looking for good LLM inference tools for on-premise deployment.",
}

SAMPLE_HN_SEARCH_RESPONSE = {
    "hits": [SAMPLE_HN_LINK_HIT, SAMPLE_HN_TEXT_HIT],
    "nbHits": 2,
}

SAMPLE_REDDIT_TOKEN = {
    "access_token": "mock_reddit_token",
    "token_type": "bearer",
    "expires_in": 3600,
}

SAMPLE_REDDIT_LINK_POST = {
    "id": "abc123",
    "title": "LLM inference guide",
    "url": "https://example.com/llm-guide",
    "permalink": "/r/MachineLearning/comments/abc123/llm_inference_guide/",
    "author": "mlresearcher",
    "score": 450,
    "num_comments": 87,
    "created_utc": 1700000000.0,
    "is_self": False,
    "selftext": "",
    "subreddit": "MachineLearning",
}

SAMPLE_REDDIT_SELF_POST = {
    "id": "def456",
    "title": "My LLM inference benchmarks",
    "permalink": "/r/LocalLLaMA/comments/def456/my_llm_inference_benchmarks/",
    "author": "benchmarker",
    "score": 120,
    "num_comments": 34,
    "created_utc": 1700001000.0,
    "is_self": True,
    "selftext": "I have been benchmarking LLM inference runtimes (vLLM, llama.cpp, TGI) "
                "and wanted to share detailed performance numbers with the community.",
    "subreddit": "LocalLLaMA",
}

SAMPLE_REDDIT_SEARCH_RESPONSE = {
    "data": {
        "children": [
            {"data": SAMPLE_REDDIT_LINK_POST},
            {"data": SAMPLE_REDDIT_SELF_POST},
        ]
    }
}

SAMPLE_ARXIV_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query Results</title>
  <entry>
    <id>http://arxiv.org/abs/2501.12345v1</id>
    <title>Efficient LLM Inference via Speculative Decoding</title>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <summary>We present a novel speculative decoding approach that significantly
    reduces LLM inference latency while maintaining output quality.</summary>
    <published>2025-01-15T00:00:00Z</published>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""

SAMPLE_DEVTO_SEARCH_RESPONSE = [
    {
        "id": 98765,
        "title": "Understanding LLM Inference Performance",
        "canonical_url": "https://dev.to/author/understanding-llm-inference-xyz",
        "url": "https://dev.to/author/understanding-llm-inference-xyz",
        "description": "A deep dive into LLM inference optimization.",
        "published_at": "2025-01-15T10:00:00Z",
        "user": {"name": "Jane Developer", "username": "jdev"},
        "tags": ["llm", "ai", "performance"],
        "positive_reactions_count": 120,
        "comments_count": 15,
        "reading_time_minutes": 8,
    }
]

SAMPLE_DEVTO_FULL_ARTICLE = {
    **SAMPLE_DEVTO_SEARCH_RESPONSE[0],
    "body_markdown": "# Understanding LLM Inference Performance\n\nThis is the full article...",
}


# ---------------------------------------------------------------------------
# Config and DB helpers
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


def _make_config() -> AppConfig:
    """Build a minimal AppConfig with all 4 radar sources configured.

    Returns:
        An AppConfig instance suitable for multi-source radar tests.
    """
    return AppConfig(
        sources=SourcesConfig(
            hackernews=HackerNewsConfig(limit=10),
            arxiv=ArxivConfig(queries=["LLM inference"], max_results=5),
            devto=DevtoConfig(tags=["llm"], limit=10),
        ),
        settings=SettingsConfig(
            data_dir="/tmp/test-multi-radar",
            llm=_make_llm_routing(),
            reddit=RedditAPIConfig(
                client_id="test_client_id",
                client_secret="test_client_secret",
            ),
        ),
        filters=FiltersConfig(),
    )


def _make_mock_http_response(json_data: object, status_code: int = 200) -> MagicMock:
    """Create a mock httpx response.

    Args:
        json_data: The JSON payload to return from .json().
        status_code: HTTP status code for the mock.

    Returns:
        A MagicMock mimicking an httpx.Response.
    """
    mock_resp = MagicMock()
    mock_resp.json.return_value = json_data
    mock_resp.text = ""
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


async def _create_documents_table(conn: aiosqlite.Connection) -> None:
    """Create the documents table for an in-memory test DB.

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


# ---------------------------------------------------------------------------
# HN radar tests — content fetching for link stories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hn_radar_returns_documents_with_radar_origin() -> None:
    """HackerNewsIngestor.search_radar() returns docs with origin='radar'."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_HN_SEARCH_RESPONSE)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    # Patch ContentExtractor to avoid real HTTP in tests
    with patch(
        "ai_craftsman_kb.ingestors.hackernews.ContentExtractor"
    ) as mock_extractor_cls:
        mock_extractor = AsyncMock()
        mock_extractor_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_extractor
        )
        mock_extractor_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        extracted = MagicMock()
        extracted.text = "Full article content about LLM inference."
        extracted.word_count = 6
        extracted.title = "LLM Inference Optimization"
        mock_extractor.fetch_and_extract = AsyncMock(return_value=extracted)

        docs = await ingestor.search_radar("LLM inference", limit=5)

    assert len(docs) == 2
    assert all(doc.origin == "radar" for doc in docs)
    assert all(doc.source_type == "hn" for doc in docs)


@pytest.mark.asyncio
async def test_hn_radar_fetches_content_for_link_stories() -> None:
    """HN radar fetches article content for link stories via ContentExtractor."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    # Only return the link story (has external url)
    link_only_response = {"hits": [SAMPLE_HN_LINK_HIT]}
    mock_resp = _make_mock_http_response(link_only_response)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    fetch_urls: list[str] = []

    with patch(
        "ai_craftsman_kb.ingestors.hackernews.ContentExtractor"
    ) as mock_extractor_cls:
        mock_extractor = AsyncMock()
        mock_extractor_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_extractor
        )
        mock_extractor_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        extracted = MagicMock()
        extracted.text = "Full LLM inference content."
        extracted.word_count = 4
        extracted.title = None

        async def mock_fetch(url: str) -> MagicMock:
            fetch_urls.append(url)
            return extracted

        mock_extractor.fetch_and_extract = mock_fetch

        docs = await ingestor.search_radar("LLM inference", limit=5)

    assert len(docs) == 1
    assert docs[0].raw_content == "Full LLM inference content."
    assert fetch_urls == ["https://example.com/llm-inference"]


@pytest.mark.asyncio
async def test_hn_radar_text_only_stories_skip_content_fetch() -> None:
    """HN radar does NOT call ContentExtractor for Ask HN text-only stories."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    # Only return the text-only Ask HN story (no url)
    text_only_response = {"hits": [SAMPLE_HN_TEXT_HIT]}
    mock_resp = _make_mock_http_response(text_only_response)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    fetch_called: list[bool] = []

    with patch(
        "ai_craftsman_kb.ingestors.hackernews.ContentExtractor"
    ) as mock_extractor_cls:
        mock_extractor = AsyncMock()
        mock_extractor_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_extractor
        )
        mock_extractor_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        async def mock_fetch(url: str) -> MagicMock:
            fetch_called.append(True)
            return MagicMock(text="content", word_count=1, title=None)

        mock_extractor.fetch_and_extract = mock_fetch

        docs = await ingestor.search_radar("LLM inference", limit=5)

    # ContentExtractor should NOT have been called for the text-only story
    assert not fetch_called
    # The text story's content comes from story_text
    assert docs[0].raw_content == SAMPLE_HN_TEXT_HIT["story_text"]


@pytest.mark.asyncio
async def test_hn_radar_content_fetch_error_does_not_fail_other_results() -> None:
    """HN radar: content fetch failure for one story does not drop other results."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    # Two link stories — first fetch fails, second succeeds
    link_hit_2 = {**SAMPLE_HN_LINK_HIT, "objectID": "99999", "url": "https://other.com/article"}
    two_link_response = {"hits": [SAMPLE_HN_LINK_HIT, link_hit_2]}
    mock_resp = _make_mock_http_response(two_link_response)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    call_count = 0

    with patch(
        "ai_craftsman_kb.ingestors.hackernews.ContentExtractor"
    ) as mock_extractor_cls:
        mock_extractor = AsyncMock()
        mock_extractor_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_extractor
        )
        mock_extractor_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        async def mock_fetch(url: str) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            result = MagicMock()
            result.text = "Content from second article."
            result.word_count = 4
            result.title = None
            return result

        mock_extractor.fetch_and_extract = mock_fetch

        docs = await ingestor.search_radar("LLM inference", limit=5)

    # Both docs are returned — error on first doesn't drop second
    assert len(docs) == 2
    # First doc has no content (fetch failed), second has content
    first_doc = next(d for d in docs if d.url == "https://example.com/llm-inference")
    second_doc = next(d for d in docs if d.url == "https://other.com/article")
    assert first_doc.raw_content is None
    assert second_doc.raw_content == "Content from second article."


# ---------------------------------------------------------------------------
# Reddit radar tests — self-post vs link-post content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reddit_radar_self_post_uses_selftext() -> None:
    """Reddit radar: self-post raw_content comes from selftext, no ContentExtractor."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_http_response(SAMPLE_REDDIT_TOKEN)
    self_only_response = {
        "data": {"children": [{"data": SAMPLE_REDDIT_SELF_POST}]}
    }
    search_resp = _make_mock_http_response(self_only_response)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return search_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    fetch_called: list[bool] = []

    with patch(
        "ai_craftsman_kb.ingestors.reddit.ContentExtractor"
    ) as mock_extractor_cls:
        mock_extractor = AsyncMock()
        mock_extractor_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_extractor
        )
        mock_extractor_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        async def mock_fetch(url: str) -> MagicMock:
            fetch_called.append(True)
            return MagicMock(text="content", word_count=1, title=None)

        mock_extractor.fetch_and_extract = mock_fetch

        docs = await ingestor.search_radar("LLM inference", limit=5)

    # ContentExtractor should NOT have been called for the self-post
    assert not fetch_called
    assert len(docs) == 1
    assert docs[0].raw_content == SAMPLE_REDDIT_SELF_POST["selftext"]
    assert docs[0].origin == "radar"
    assert docs[0].source_type == "reddit"


@pytest.mark.asyncio
async def test_reddit_radar_link_post_fetches_content() -> None:
    """Reddit radar: link-post content fetched via ContentExtractor from linked_url."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_http_response(SAMPLE_REDDIT_TOKEN)
    link_only_response = {
        "data": {"children": [{"data": SAMPLE_REDDIT_LINK_POST}]}
    }
    search_resp = _make_mock_http_response(link_only_response)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return search_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    fetch_urls: list[str] = []

    with patch(
        "ai_craftsman_kb.ingestors.reddit.ContentExtractor"
    ) as mock_extractor_cls:
        mock_extractor = AsyncMock()
        mock_extractor_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_extractor
        )
        mock_extractor_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        extracted = MagicMock()
        extracted.text = "Full LLM guide content."
        extracted.word_count = 4
        extracted.title = None

        async def mock_fetch(url: str) -> MagicMock:
            fetch_urls.append(url)
            return extracted

        mock_extractor.fetch_and_extract = mock_fetch

        docs = await ingestor.search_radar("LLM inference", limit=5)

    assert len(docs) == 1
    assert docs[0].raw_content == "Full LLM guide content."
    assert docs[0].origin == "radar"
    # Should have fetched the linked_url (external article), not the Reddit permalink
    assert fetch_urls == ["https://example.com/llm-guide"]


@pytest.mark.asyncio
async def test_reddit_radar_link_post_content_error_returns_doc_without_content() -> None:
    """Reddit radar: content fetch failure returns doc with raw_content=None."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_http_response(SAMPLE_REDDIT_TOKEN)
    link_only_response = {
        "data": {"children": [{"data": SAMPLE_REDDIT_LINK_POST}]}
    }
    search_resp = _make_mock_http_response(link_only_response)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return search_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.reddit.ContentExtractor"
    ) as mock_extractor_cls:
        mock_extractor = AsyncMock()
        mock_extractor_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_extractor
        )
        mock_extractor_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_extractor.fetch_and_extract = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        docs = await ingestor.search_radar("LLM inference", limit=5)

    # Doc is still returned, just without content
    assert len(docs) == 1
    assert docs[0].raw_content is None
    assert docs[0].origin == "radar"


@pytest.mark.asyncio
async def test_reddit_radar_returns_empty_when_no_credentials() -> None:
    """Reddit radar returns [] when no credentials are configured."""
    config = AppConfig(
        sources=SourcesConfig(),
        settings=SettingsConfig(
            llm=_make_llm_routing(),
            reddit=RedditAPIConfig(client_id=None, client_secret=None),
        ),
        filters=FiltersConfig(),
    )
    ingestor = RedditIngestor(config)

    docs = await ingestor.search_radar("LLM inference")

    assert docs == []


# ---------------------------------------------------------------------------
# ArXiv radar tests — abstract as content (no extra fetch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arxiv_radar_uses_abstract_as_content() -> None:
    """ArXiv search_radar() uses the paper abstract as raw_content (no URL fetch)."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    mock_resp = MagicMock()
    mock_resp.text = SAMPLE_ARXIV_ATOM_XML
    mock_resp.raise_for_status = MagicMock()

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("LLM inference", limit=5)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_type == "arxiv"
    assert doc.origin == "radar"
    assert doc.raw_content is not None
    assert "speculative decoding" in doc.raw_content.lower()
    assert doc.url == "https://arxiv.org/abs/2501.12345"


@pytest.mark.asyncio
async def test_arxiv_radar_content_is_abstract_no_extractor_needed() -> None:
    """ArXiv radar: abstract content requires no ContentExtractor call."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    mock_resp = MagicMock()
    mock_resp.text = SAMPLE_ARXIV_ATOM_XML
    mock_resp.raise_for_status = MagicMock()

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    # Verify no ContentExtractor import/usage by checking raw_content is set
    # from the abstract without any additional network calls.
    docs = await ingestor.search_radar("LLM inference", limit=5)

    assert docs[0].raw_content is not None
    # Abstract content is set, no None
    assert len(docs[0].raw_content) > 0


@pytest.mark.asyncio
async def test_arxiv_radar_handles_api_error() -> None:
    """ArXiv search_radar() returns [] on HTTP error."""
    config = _make_config()
    ingestor = ArxivIngestor(config)

    async def mock_get(url: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection refused")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("LLM inference")

    assert docs == []


# ---------------------------------------------------------------------------
# DEV.to radar tests — full body_markdown fetching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_devto_radar_fetches_full_body_markdown() -> None:
    """DEV.to search_radar() fetches full body_markdown from individual article endpoint."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    call_count = 0

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: search endpoint returns summaries
            return _make_mock_http_response(SAMPLE_DEVTO_SEARCH_RESPONSE)
        else:
            # Subsequent calls: individual article endpoints return full data
            return _make_mock_http_response(SAMPLE_DEVTO_FULL_ARTICLE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("LLM inference", limit=5)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_type == "devto"
    assert doc.origin == "radar"
    assert doc.raw_content is not None
    # Full body_markdown should be present (not just description)
    assert "# Understanding LLM Inference Performance" in doc.raw_content


@pytest.mark.asyncio
async def test_devto_radar_returns_documents_with_radar_origin() -> None:
    """DEV.to search_radar() returns docs with origin='radar'."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    call_count = 0

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_mock_http_response(SAMPLE_DEVTO_SEARCH_RESPONSE)
        return _make_mock_http_response(SAMPLE_DEVTO_FULL_ARTICLE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("LLM inference", limit=5)

    assert all(doc.origin == "radar" for doc in docs)
    assert all(doc.source_type == "devto" for doc in docs)


@pytest.mark.asyncio
async def test_devto_radar_handles_api_error() -> None:
    """DEV.to search_radar() returns [] on HTTP error."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    async def mock_get(url: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection refused")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("LLM inference")

    assert docs == []


# ---------------------------------------------------------------------------
# Integration test: RadarEngine fan-out across all 4 sources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_radar_fan_out_all_sources() -> None:
    """RadarEngine.search() fans out to all 4 sources concurrently.

    Given: mocked APIs for HN, Reddit, ArXiv, DEV.to
    When: RadarEngine.search(conn, 'LLM inference') called
    Then:
    - All 4 source search APIs called concurrently
    - Results deduplicated by URL
    - New documents stored with origin='radar'
    """
    config = _make_config()

    # Build one mock ingestor per source with pre-set return values
    hn_docs = [
        RawDocument(
            url="https://news.ycombinator.com/item?id=42001234",
            title="LLM Inference Optimization",
            source_type="hn",
            origin="radar",
            content_type="post",
            raw_content="HN story text about LLM inference.",
            published_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        )
    ]
    reddit_docs = [
        RawDocument(
            url="https://reddit.com/r/MachineLearning/comments/abc123/llm_guide/",
            title="LLM inference guide",
            source_type="reddit",
            origin="radar",
            content_type="post",
            raw_content="Reddit discussion about LLM inference.",
            published_at=datetime(2025, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
        )
    ]
    arxiv_docs = [
        RawDocument(
            url="https://arxiv.org/abs/2501.12345",
            title="Efficient LLM Inference via Speculative Decoding",
            source_type="arxiv",
            origin="radar",
            content_type="paper",
            raw_content="Abstract: We present a novel speculative decoding approach...",
            published_at=datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
        )
    ]
    devto_docs = [
        RawDocument(
            url="https://dev.to/author/understanding-llm-inference-xyz",
            title="Understanding LLM Inference Performance",
            source_type="devto",
            origin="radar",
            content_type="article",
            raw_content="# Understanding LLM Inference Performance\n\nFull article...",
            published_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        )
    ]

    def _make_mock_ingestor(source_type: str, docs: list[RawDocument]) -> MagicMock:
        """Create a mock ingestor with a specific search_radar() return value."""
        mock = MagicMock()
        mock.source_type = source_type

        async def _search_radar(query: str, limit: int = 20) -> list[RawDocument]:
            return list(docs)

        mock.search_radar = _search_radar
        return mock

    ingestors = {
        "hn": _make_mock_ingestor("hn", hn_docs),
        "reddit": _make_mock_ingestor("reddit", reddit_docs),
        "arxiv": _make_mock_ingestor("arxiv", arxiv_docs),
        "devto": _make_mock_ingestor("devto", devto_docs),
    }

    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)

        report = await engine.search(conn, "LLM inference", limit_per_source=10)

        # Check stored count
        async with conn.execute("SELECT COUNT(*) as cnt FROM documents") as cursor:
            row = await cursor.fetchone()
        stored_count = row["cnt"]

    # All 4 sources searched
    assert set(report.sources_searched) == {"hn", "reddit", "arxiv", "devto"}
    # 4 unique documents (one from each source, all different URLs)
    assert report.total_found == 4
    assert report.new_documents == 4
    assert stored_count == 4
    assert report.errors == {}
    assert report.query == "LLM inference"


@pytest.mark.asyncio
async def test_radar_fan_out_deduplicates_cross_source_urls() -> None:
    """RadarEngine deduplicates when HN and Reddit return the same article URL."""
    config = _make_config()

    shared_url = "https://example.com/shared-llm-article"

    def _make_mock_ingestor(source_type: str, docs: list[RawDocument]) -> MagicMock:
        mock = MagicMock()
        mock.source_type = source_type

        async def _search_radar(query: str, limit: int = 20) -> list[RawDocument]:
            return list(docs)

        mock.search_radar = _search_radar
        return mock

    ingestors = {
        "hn": _make_mock_ingestor("hn", [
            RawDocument(
                url=shared_url,
                title="LLM Article from HN",
                source_type="hn",
                origin="radar",
                content_type="post",
            ),
        ]),
        "reddit": _make_mock_ingestor("reddit", [
            RawDocument(
                url=shared_url,  # same URL as HN result
                title="LLM Article from Reddit",
                source_type="reddit",
                origin="radar",
                content_type="post",
            ),
        ]),
        "arxiv": _make_mock_ingestor("arxiv", [
            RawDocument(
                url="https://arxiv.org/abs/2501.99999",
                title="ArXiv Paper",
                source_type="arxiv",
                origin="radar",
                content_type="paper",
            ),
        ]),
    }

    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "LLM inference")

    # 3 raw results but 2 unique URLs after dedup (shared_url counted once)
    assert report.total_found == 2
    assert report.new_documents == 2


@pytest.mark.asyncio
async def test_radar_fan_out_records_errors_without_stopping_other_sources() -> None:
    """RadarEngine fan-out: failing source recorded in errors, others succeed."""
    config = _make_config()

    def _make_mock_ingestor(
        source_type: str,
        docs: list[RawDocument] | Exception,
    ) -> MagicMock:
        mock = MagicMock()
        mock.source_type = source_type

        async def _search_radar(query: str, limit: int = 20) -> list[RawDocument]:
            if isinstance(docs, Exception):
                raise docs
            return list(docs)

        mock.search_radar = _search_radar
        return mock

    ingestors = {
        "hn": _make_mock_ingestor("hn", RuntimeError("HN API rate limit exceeded")),
        "reddit": _make_mock_ingestor("reddit", RuntimeError("Reddit auth failed")),
        "arxiv": _make_mock_ingestor("arxiv", [
            RawDocument(
                url="https://arxiv.org/abs/2501.12345",
                title="ArXiv Paper",
                source_type="arxiv",
                origin="radar",
                content_type="paper",
                raw_content="Abstract content.",
            )
        ]),
        "devto": _make_mock_ingestor("devto", [
            RawDocument(
                url="https://dev.to/author/article",
                title="DEV.to Article",
                source_type="devto",
                origin="radar",
                content_type="article",
                raw_content="Article body.",
            )
        ]),
    }

    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "LLM inference")

    # HN and Reddit errors recorded
    assert "hn" in report.errors
    assert "reddit" in report.errors
    # ArXiv and DEV.to still returned results
    assert report.total_found == 2
    assert report.new_documents == 2
    # All sources were attempted
    assert set(report.sources_searched) == {"hn", "reddit", "arxiv", "devto"}


@pytest.mark.asyncio
async def test_radar_fan_out_all_sources_run_concurrently() -> None:
    """RadarEngine runs all 4 source searches concurrently (not sequentially).

    Each source has a 100ms delay. If sequential, total time >= 400ms.
    If concurrent, total time should be ~100ms (well under 300ms).
    """
    import time

    config = _make_config()
    delay = 0.1  # 100ms per source

    def _make_delayed_ingestor(source_type: str) -> MagicMock:
        mock = MagicMock()
        mock.source_type = source_type

        async def _search_radar(query: str, limit: int = 20) -> list[RawDocument]:
            await asyncio.sleep(delay)
            return []

        mock.search_radar = _search_radar
        return mock

    ingestors = {
        "hn": _make_delayed_ingestor("hn"),
        "reddit": _make_delayed_ingestor("reddit"),
        "arxiv": _make_delayed_ingestor("arxiv"),
        "devto": _make_delayed_ingestor("devto"),
    }

    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)

        start = time.monotonic()
        await engine.search(conn, "LLM inference")
        elapsed = time.monotonic() - start

    sequential_time = 4 * delay  # 0.4s if sequential
    # Concurrent should finish in roughly 1 delay period (~0.1s), not 4x
    assert elapsed < sequential_time * 0.75, (
        f"Expected concurrent execution (< {sequential_time * 0.75:.2f}s), "
        f"but took {elapsed:.3f}s (suggesting sequential calls)"
    )


@pytest.mark.asyncio
async def test_radar_all_sources_documents_have_correct_source_type() -> None:
    """Each source's documents have the correct source_type set."""
    config = _make_config()

    def _make_mock_ingestor(source_type: str) -> MagicMock:
        mock = MagicMock()
        mock.source_type = source_type

        async def _search_radar(query: str, limit: int = 20) -> list[RawDocument]:
            return [
                RawDocument(
                    url=f"https://example.com/{source_type}/article",
                    title=f"{source_type.upper()} Article",
                    source_type=source_type,
                    origin="radar",
                    content_type="article",
                    raw_content=f"Content from {source_type}.",
                )
            ]

        mock.search_radar = _search_radar
        return mock

    ingestors = {
        st: _make_mock_ingestor(st) for st in ["hn", "reddit", "arxiv", "devto"]
    }
    engine = RadarEngine(config=config, ingestors=ingestors)

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_documents_table(conn)
        report = await engine.search(conn, "LLM inference")

        async with conn.execute(
            "SELECT source_type FROM documents ORDER BY source_type"
        ) as cursor:
            rows = await cursor.fetchall()

    stored_source_types = {row["source_type"] for row in rows}
    assert stored_source_types == {"hn", "reddit", "arxiv", "devto"}
    assert report.total_found == 4
