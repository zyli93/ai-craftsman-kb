"""Unit tests for the HackerNews ingestor.

Tests cover:
- fetch_pro() returns up to limit stories
- fetch_pro() returns empty list when hackernews config is None
- fetch_pro() handles HTTP error gracefully (returns [])
- search_radar() passes query param to Algolia
- search_radar() handles HTTP error gracefully
- _hit_to_raw_doc() maps fields correctly: url, title, author, metadata
- _hit_to_raw_doc() uses story_text for Ask HN (no url)
- _hit_to_raw_doc() sets hn_url in metadata always
- _hit_to_raw_doc() parses created_at into datetime
"""
import pytest
import httpx
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from ai_craftsman_kb.ingestors.hackernews import HackerNewsIngestor
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.config.models import (
    AppConfig,
    SourcesConfig,
    SettingsConfig,
    FiltersConfig,
    HackerNewsConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_HIT = {
    "objectID": "42001234",
    "title": "Show HN: My Cool Project",
    "url": "https://example.com/project",
    "author": "pg",
    "points": 142,
    "num_comments": 37,
    "created_at": "2025-01-15T10:00:00Z",
    "story_text": None,
}

SAMPLE_ASK_HIT = {
    "objectID": "42009999",
    "title": "Ask HN: Best resources for learning Rust?",
    "url": None,
    "author": "rustlover",
    "points": 89,
    "num_comments": 52,
    "created_at": "2025-01-15T12:00:00Z",
    "story_text": "Looking for good Rust learning resources for 2025.",
}

SAMPLE_ALGOLIA_RESPONSE = {
    "hits": [SAMPLE_HIT],
    "nbHits": 1,
    "page": 0,
    "hitsPerPage": 1,
}

SAMPLE_ALGOLIA_MULTI_RESPONSE = {
    "hits": [SAMPLE_HIT, SAMPLE_ASK_HIT],
    "nbHits": 2,
    "page": 0,
    "hitsPerPage": 2,
}


def _make_llm_routing() -> LLMRoutingConfig:
    """Build a minimal LLMRoutingConfig for testing."""
    task_cfg = LLMTaskConfig(provider="openai", model="gpt-4o-mini")
    return LLMRoutingConfig(
        filtering=task_cfg,
        entity_extraction=task_cfg,
        briefing=task_cfg,
        source_discovery=task_cfg,
    )


def _make_config(hn_cfg: HackerNewsConfig | None = None) -> AppConfig:
    """Build a minimal AppConfig for testing.

    Args:
        hn_cfg: HackerNewsConfig to include, or None to simulate unconfigured HN.

    Returns:
        An AppConfig instance with minimal settings.
    """
    return AppConfig(
        sources=SourcesConfig(hackernews=hn_cfg),
        settings=SettingsConfig(llm=_make_llm_routing()),
        filters=FiltersConfig(),
    )


def _make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response that returns json_data from .json().

    Args:
        json_data: The dict that .json() should return.
        status_code: HTTP status code for the mock response.

    Returns:
        A MagicMock mimicking an httpx.Response.
    """
    mock_resp = MagicMock()
    mock_resp.json.return_value = json_data
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
# fetch_pro() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_pro_returns_stories() -> None:
    """fetch_pro() returns RawDocuments for each hit from Algolia."""
    config = _make_config(hn_cfg=HackerNewsConfig(limit=10))
    ingestor = HackerNewsIngestor(config)

    mock_resp = _make_mock_response(SAMPLE_ALGOLIA_RESPONSE)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)
    assert docs[0].source_type == "hn"
    assert docs[0].origin == "pro"


@pytest.mark.asyncio
async def test_fetch_pro_respects_limit() -> None:
    """fetch_pro() returns at most config.limit stories."""
    # Config limit of 1, but response has 2 hits
    config = _make_config(hn_cfg=HackerNewsConfig(limit=1))
    ingestor = HackerNewsIngestor(config)

    mock_resp = _make_mock_response(SAMPLE_ALGOLIA_MULTI_RESPONSE)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    # Should be capped at limit=1
    assert len(docs) <= 1


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_hn_config_is_none() -> None:
    """fetch_pro() returns [] when hackernews config is None (source not configured)."""
    config = _make_config(hn_cfg=None)
    ingestor = HackerNewsIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_handles_http_error_gracefully() -> None:
    """fetch_pro() returns [] and logs error when HTTP request fails."""
    config = _make_config(hn_cfg=HackerNewsConfig(limit=10))
    ingestor = HackerNewsIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection refused")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_handles_http_status_error() -> None:
    """fetch_pro() returns [] and logs error on HTTP 5xx response."""
    config = _make_config(hn_cfg=HackerNewsConfig(limit=10))
    ingestor = HackerNewsIngestor(config)

    mock_resp = _make_mock_response({}, status_code=503)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_handles_empty_hits() -> None:
    """fetch_pro() returns [] when Algolia returns no hits."""
    config = _make_config(hn_cfg=HackerNewsConfig(limit=10))
    ingestor = HackerNewsIngestor(config)

    mock_resp = _make_mock_response({"hits": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_calls_search_by_date_endpoint() -> None:
    """fetch_pro() calls the /search_by_date endpoint (not /search)."""
    config = _make_config(hn_cfg=HackerNewsConfig(limit=5))
    ingestor = HackerNewsIngestor(config)

    called_paths: list[str] = []
    mock_resp = _make_mock_response({"hits": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        called_paths.append(path)
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.fetch_pro()

    assert len(called_paths) == 1
    assert "search_by_date" in called_paths[0]


# ---------------------------------------------------------------------------
# search_radar() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_returns_documents() -> None:
    """search_radar() returns RawDocuments with origin='radar'."""
    config = _make_config(hn_cfg=HackerNewsConfig(limit=10))
    ingestor = HackerNewsIngestor(config)

    mock_resp = _make_mock_response(SAMPLE_ALGOLIA_RESPONSE)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    # Patch ContentExtractor to avoid real network calls in unit tests.
    # SAMPLE_ALGOLIA_RESPONSE has a link story with an external URL.
    with patch("ai_craftsman_kb.ingestors.hackernews.ContentExtractor") as mock_cls:
        mock_extractor = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_extractor)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        extracted = MagicMock()
        extracted.text = "Article content."
        extracted.word_count = 2
        extracted.title = None
        mock_extractor.fetch_and_extract = AsyncMock(return_value=extracted)

        docs = await ingestor.search_radar("python machine learning")

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0].origin == "radar"
    assert docs[0].source_type == "hn"


@pytest.mark.asyncio
async def test_search_radar_passes_query_param() -> None:
    """search_radar() passes the query string to Algolia as a param."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    captured_kwargs: list[dict] = []
    mock_resp = _make_mock_response({"hits": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        captured_kwargs.append(dict(kwargs))
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("rust async programming", limit=5)

    assert len(captured_kwargs) == 1
    params = captured_kwargs[0].get("params", {})
    assert params.get("query") == "rust async programming"


@pytest.mark.asyncio
async def test_search_radar_passes_hitsperpage() -> None:
    """search_radar() passes limit as hitsPerPage param to Algolia."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    captured_kwargs: list[dict] = []
    mock_resp = _make_mock_response({"hits": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        captured_kwargs.append(dict(kwargs))
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("test query", limit=15)

    params = captured_kwargs[0].get("params", {})
    assert params.get("hitsPerPage") == 15


@pytest.mark.asyncio
async def test_search_radar_calls_search_endpoint() -> None:
    """search_radar() calls the /search endpoint (relevance-ranked)."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    called_paths: list[str] = []
    mock_resp = _make_mock_response({"hits": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        called_paths.append(path)
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("query")

    assert len(called_paths) == 1
    # Should use /search (relevance), not /search_by_date
    assert called_paths[0].endswith("/search") or called_paths[0] == "/search"


@pytest.mark.asyncio
async def test_search_radar_handles_http_error() -> None:
    """search_radar() returns [] and logs error on HTTP failure."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> None:
        raise httpx.TimeoutException("Request timed out")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("test query")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_handles_http_status_error() -> None:
    """search_radar() returns [] on HTTP 5xx response."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    mock_resp = _make_mock_response({}, status_code=500)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("test query")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_respects_limit() -> None:
    """search_radar() returns at most limit documents."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    # Response has 2 hits but limit is 1
    mock_resp = _make_mock_response(SAMPLE_ALGOLIA_MULTI_RESPONSE)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    # SAMPLE_ALGOLIA_MULTI_RESPONSE has a link story — patch ContentExtractor.
    with patch("ai_craftsman_kb.ingestors.hackernews.ContentExtractor") as mock_cls:
        mock_extractor = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_extractor)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        extracted = MagicMock()
        extracted.text = "Content."
        extracted.word_count = 1
        extracted.title = None
        mock_extractor.fetch_and_extract = AsyncMock(return_value=extracted)

        docs = await ingestor.search_radar("rust", limit=1)

    assert len(docs) <= 1


# ---------------------------------------------------------------------------
# _hit_to_raw_doc() tests
# ---------------------------------------------------------------------------


def test_hit_to_raw_doc_maps_url_from_hit_url() -> None:
    """_hit_to_raw_doc() uses the story's external URL as doc.url."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.url == "https://example.com/project"


def test_hit_to_raw_doc_maps_title() -> None:
    """_hit_to_raw_doc() maps the title field correctly."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.title == "Show HN: My Cool Project"


def test_hit_to_raw_doc_maps_author() -> None:
    """_hit_to_raw_doc() maps the author field correctly."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.author == "pg"


def test_hit_to_raw_doc_metadata_includes_hn_id() -> None:
    """_hit_to_raw_doc() includes hn_id in metadata."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.metadata["hn_id"] == "42001234"


def test_hit_to_raw_doc_metadata_includes_points() -> None:
    """_hit_to_raw_doc() includes points in metadata."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.metadata["points"] == 142


def test_hit_to_raw_doc_metadata_includes_comment_count() -> None:
    """_hit_to_raw_doc() includes comment_count in metadata."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.metadata["comment_count"] == 37


def test_hit_to_raw_doc_metadata_includes_hn_url_always() -> None:
    """_hit_to_raw_doc() always sets hn_url in metadata pointing to HN discussion."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert "hn_url" in doc.metadata
    expected_hn_url = "https://news.ycombinator.com/item?id=42001234"
    assert doc.metadata["hn_url"] == expected_hn_url


def test_hit_to_raw_doc_hn_url_differs_from_story_url_for_external_articles() -> None:
    """For external articles, hn_url in metadata != doc.url (they point to different pages)."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    # doc.url is the external article, metadata["hn_url"] is the HN thread
    assert doc.url == "https://example.com/project"
    assert doc.metadata["hn_url"] == "https://news.ycombinator.com/item?id=42001234"
    assert doc.url != doc.metadata["hn_url"]


def test_hit_to_raw_doc_ask_hn_uses_story_text_as_raw_content() -> None:
    """For Ask HN posts (no url), raw_content is populated from story_text."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_ASK_HIT)

    assert doc.raw_content == "Looking for good Rust learning resources for 2025."


def test_hit_to_raw_doc_ask_hn_url_is_hn_discussion() -> None:
    """For Ask HN posts (no external url), doc.url is the HN discussion URL."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_ASK_HIT)

    expected_url = "https://news.ycombinator.com/item?id=42009999"
    assert doc.url == expected_url


def test_hit_to_raw_doc_ask_hn_hn_url_in_metadata() -> None:
    """For Ask HN posts, hn_url in metadata matches doc.url (both are the HN thread)."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_ASK_HIT)

    expected_url = "https://news.ycombinator.com/item?id=42009999"
    assert doc.metadata["hn_url"] == expected_url
    assert doc.url == doc.metadata["hn_url"]


def test_hit_to_raw_doc_external_story_has_none_raw_content() -> None:
    """For stories with external URL, raw_content is None (must be fetched by caller)."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.raw_content is None


def test_hit_to_raw_doc_parses_created_at_into_datetime() -> None:
    """_hit_to_raw_doc() parses created_at ISO string into a timezone-aware datetime."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.published_at is not None
    assert isinstance(doc.published_at, datetime)
    assert doc.published_at.year == 2025
    assert doc.published_at.month == 1
    assert doc.published_at.day == 15
    assert doc.published_at.hour == 10
    # Should be UTC-aware
    assert doc.published_at.tzinfo is not None


def test_hit_to_raw_doc_missing_created_at_gives_none() -> None:
    """_hit_to_raw_doc() sets published_at=None when created_at is absent."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    hit_no_date = {**SAMPLE_HIT, "created_at": None}
    doc = ingestor._hit_to_raw_doc(hit_no_date)

    assert doc.published_at is None


def test_hit_to_raw_doc_invalid_created_at_gives_none() -> None:
    """_hit_to_raw_doc() sets published_at=None on unparseable timestamp (no crash)."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    hit_bad_date = {**SAMPLE_HIT, "created_at": "not-a-date"}
    doc = ingestor._hit_to_raw_doc(hit_bad_date)

    assert doc.published_at is None


def test_hit_to_raw_doc_content_type_is_post() -> None:
    """_hit_to_raw_doc() always sets content_type='post'."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.content_type == "post"


def test_hit_to_raw_doc_source_type_is_hn() -> None:
    """_hit_to_raw_doc() always sets source_type='hn'."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.source_type == "hn"


def test_hit_to_raw_doc_default_origin_is_pro() -> None:
    """_hit_to_raw_doc() uses origin='pro' by default."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT)

    assert doc.origin == "pro"


def test_hit_to_raw_doc_radar_origin() -> None:
    """_hit_to_raw_doc() uses origin='radar' when explicitly passed."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    doc = ingestor._hit_to_raw_doc(SAMPLE_HIT, origin="radar")

    assert doc.origin == "radar"


def test_hit_to_raw_doc_missing_points_defaults_to_zero() -> None:
    """_hit_to_raw_doc() defaults points to 0 when absent from hit."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    hit_no_points = {**SAMPLE_HIT, "points": None}
    doc = ingestor._hit_to_raw_doc(hit_no_points)

    assert doc.metadata["points"] == 0


def test_hit_to_raw_doc_missing_num_comments_defaults_to_zero() -> None:
    """_hit_to_raw_doc() defaults comment_count to 0 when num_comments is absent."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    hit_no_comments = {**SAMPLE_HIT, "num_comments": None}
    doc = ingestor._hit_to_raw_doc(hit_no_comments)

    assert doc.metadata["comment_count"] == 0


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_returns_self() -> None:
    """HackerNewsIngestor works as async context manager and returns self."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    async with ingestor as ctx:
        assert ctx is ingestor

    # After exit, client should be closed — no exception expected


@pytest.mark.asyncio
async def test_context_manager_closes_client_on_exit() -> None:
    """HackerNewsIngestor.__aexit__ calls aclose() on the internal httpx client."""
    config = _make_config()
    ingestor = HackerNewsIngestor(config)

    close_called = []
    original_aclose = ingestor._client.aclose

    async def mock_aclose() -> None:
        close_called.append(True)
        await original_aclose()

    ingestor._client.aclose = mock_aclose  # type: ignore[method-assign]

    async with ingestor:
        pass

    assert close_called, "aclose() was not called on __aexit__"
