"""Unit tests for the DEV.to ingestor.

Tests cover:
- fetch_pro() returns articles for all configured tags
- fetch_pro() returns [] when devto config is None
- fetch_pro() returns [] when no tags are configured
- fetch_pro() handles HTTP errors per tag gracefully (continues with others)
- fetch_pro() deduplicates articles that appear under multiple tags
- search_radar() uses the ?q= search parameter
- search_radar() handles HTTP errors gracefully (returns [])
- search_radar() returns documents with origin='radar'
- _article_to_raw_doc() maps all required fields: url, title, author, metadata
- _article_to_raw_doc() uses canonical_url as doc.url
- _article_to_raw_doc() populates raw_content from body_markdown when available
- _article_to_raw_doc() falls back to description when body_markdown is absent
- _article_to_raw_doc() metadata includes devto_id, tags, reactions, comments, reading_time_minutes
- _fetch_full_article() merges summary with full article data
- _fetch_full_article() falls back to summary on HTTP error
- Concurrency is capped at 5 simultaneous fetches
"""
import asyncio
import pytest
import httpx
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from ai_craftsman_kb.ingestors.devto import DevtoIngestor
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.config.models import (
    AppConfig,
    SourcesConfig,
    SettingsConfig,
    FiltersConfig,
    DevtoConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_ARTICLE_SUMMARY = {
    "id": 12345,
    "title": "Building a CLI tool in Python",
    "canonical_url": "https://dev.to/author/building-a-cli-tool-in-python-abc123",
    "url": "https://dev.to/author/building-a-cli-tool-in-python-abc123",
    "description": "A short guide to building CLI tools in Python using Click.",
    "published_at": "2025-01-15T10:00:00Z",
    "user": {
        "name": "Jane Developer",
        "username": "janedev",
    },
    "tags": ["python", "cli", "tutorial"],
    "positive_reactions_count": 42,
    "comments_count": 7,
    "reading_time_minutes": 5,
}

SAMPLE_ARTICLE_FULL = {
    **SAMPLE_ARTICLE_SUMMARY,
    "body_markdown": "# Building a CLI tool in Python\n\nThis is the full article content...",
    "body_html": "<h1>Building a CLI tool in Python</h1><p>This is the full article content...</p>",
}

SAMPLE_ARTICLE_SUMMARY_2 = {
    "id": 67890,
    "title": "Async Python Best Practices",
    "canonical_url": "https://dev.to/author/async-python-abc456",
    "url": "https://dev.to/author/async-python-abc456",
    "description": "Best practices for async Python in 2025.",
    "published_at": "2025-01-14T08:00:00Z",
    "user": {
        "name": "Bob Coder",
        "username": "bobcoder",
    },
    "tags": ["python", "async", "performance"],
    "positive_reactions_count": 89,
    "comments_count": 12,
    "reading_time_minutes": 8,
}

SAMPLE_ARTICLE_FULL_2 = {
    **SAMPLE_ARTICLE_SUMMARY_2,
    "body_markdown": "# Async Python Best Practices\n\nFull content of async article...",
}


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
    )


def _make_config(devto_cfg: DevtoConfig | None = None) -> AppConfig:
    """Build a minimal AppConfig for testing.

    Args:
        devto_cfg: DevtoConfig to include, or None to simulate unconfigured DEV.to.

    Returns:
        An AppConfig instance with minimal settings.
    """
    return AppConfig(
        sources=SourcesConfig(devto=devto_cfg),
        settings=SettingsConfig(llm=_make_llm_routing()),
        filters=FiltersConfig(),
    )


def _make_mock_response(json_data: object, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response.

    Args:
        json_data: The object that .json() should return.
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
async def test_fetch_pro_returns_empty_when_devto_config_is_none() -> None:
    """fetch_pro() returns [] when devto config is None (source not configured)."""
    config = _make_config(devto_cfg=None)
    ingestor = DevtoIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_no_tags_configured() -> None:
    """fetch_pro() returns [] when devto config has no tags."""
    config = _make_config(devto_cfg=DevtoConfig(tags=[], limit=20))
    ingestor = DevtoIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_returns_articles_for_configured_tags() -> None:
    """fetch_pro() returns RawDocuments for articles found under configured tags."""
    config = _make_config(devto_cfg=DevtoConfig(tags=["python"], limit=10))
    ingestor = DevtoIngestor(config)

    # Track calls to distinguish list vs detail endpoints
    call_count = [0]

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        call_count[0] += 1
        if path == "/articles" or path.endswith("/articles"):
            # List endpoint returns summary list
            return _make_mock_response([SAMPLE_ARTICLE_SUMMARY])
        else:
            # Individual article endpoint returns full article
            return _make_mock_response(SAMPLE_ARTICLE_FULL)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert len(docs) >= 1
    assert isinstance(docs[0], RawDocument)
    assert docs[0].source_type == "devto"
    assert docs[0].origin == "pro"


@pytest.mark.asyncio
async def test_fetch_pro_deduplicates_across_tags() -> None:
    """fetch_pro() deduplicates articles appearing under multiple tags."""
    config = _make_config(devto_cfg=DevtoConfig(tags=["python", "tutorial"], limit=10))
    ingestor = DevtoIngestor(config)

    # Both tags return the same article ID
    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/articles/" in path:
            return _make_mock_response(SAMPLE_ARTICLE_FULL)
        return _make_mock_response([SAMPLE_ARTICLE_SUMMARY])

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    # Article 12345 should appear only once despite being in both tags
    article_ids = [doc.metadata.get("devto_id") for doc in docs]
    assert article_ids.count(12345) == 1


@pytest.mark.asyncio
async def test_fetch_pro_handles_http_error_for_tag_gracefully() -> None:
    """fetch_pro() skips failed tags and continues with remaining tags."""
    config = _make_config(devto_cfg=DevtoConfig(tags=["python", "rust"], limit=10))
    ingestor = DevtoIngestor(config)

    call_count = [0]

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        params = kwargs.get("params", {})
        # Fail the "python" tag request, succeed for "rust"
        if params.get("tag") == "python":
            raise httpx.ConnectError("Connection refused")
        if "/articles/" in path:
            return _make_mock_response(SAMPLE_ARTICLE_FULL_2)
        return _make_mock_response([SAMPLE_ARTICLE_SUMMARY_2])

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    # Should return articles from "rust" even though "python" failed
    assert isinstance(docs, list)
    assert len(docs) >= 1


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_all_tags_fail() -> None:
    """fetch_pro() returns [] when all tag requests fail."""
    config = _make_config(devto_cfg=DevtoConfig(tags=["python"], limit=10))
    ingestor = DevtoIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        raise httpx.ConnectError("Connection refused")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_handles_http_status_error() -> None:
    """fetch_pro() returns [] on HTTP 5xx response for the tag list endpoint."""
    config = _make_config(devto_cfg=DevtoConfig(tags=["python"], limit=10))
    ingestor = DevtoIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response({}, status_code=503)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_handles_empty_tag_results() -> None:
    """fetch_pro() returns [] gracefully when a tag returns no articles."""
    config = _make_config(devto_cfg=DevtoConfig(tags=["obscuretag"], limit=10))
    ingestor = DevtoIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response([])

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_fetches_full_articles_for_body_markdown() -> None:
    """fetch_pro() calls the individual article endpoint to get body_markdown."""
    config = _make_config(devto_cfg=DevtoConfig(tags=["python"], limit=10))
    ingestor = DevtoIngestor(config)

    individual_endpoints_called: list[str] = []

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/articles/" in path:
            individual_endpoints_called.append(path)
            return _make_mock_response(SAMPLE_ARTICLE_FULL)
        return _make_mock_response([SAMPLE_ARTICLE_SUMMARY])

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    # Should have called the individual article endpoint
    assert len(individual_endpoints_called) >= 1
    # The doc should have body_markdown as raw_content
    assert docs[0].raw_content == SAMPLE_ARTICLE_FULL["body_markdown"]


@pytest.mark.asyncio
async def test_fetch_pro_uses_per_page_param() -> None:
    """fetch_pro() passes per_page param equal to the configured limit."""
    config = _make_config(devto_cfg=DevtoConfig(tags=["python"], limit=15))
    ingestor = DevtoIngestor(config)

    captured_params: list[dict] = []

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        params = kwargs.get("params", {})
        captured_params.append(dict(params))
        if "/articles/" in path:
            return _make_mock_response(SAMPLE_ARTICLE_FULL)
        return _make_mock_response([])

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.fetch_pro()

    # First call should be the list endpoint with per_page=15
    list_params = [p for p in captured_params if "tag" in p]
    assert len(list_params) >= 1
    assert list_params[0]["per_page"] == 15


# ---------------------------------------------------------------------------
# search_radar() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_returns_documents_with_radar_origin() -> None:
    """search_radar() returns RawDocuments with origin='radar'."""
    config = _make_config(devto_cfg=DevtoConfig(tags=["python"], limit=20))
    ingestor = DevtoIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/articles/" in path:
            return _make_mock_response(SAMPLE_ARTICLE_FULL)
        return _make_mock_response([SAMPLE_ARTICLE_SUMMARY])

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("python cli")

    assert len(docs) >= 1
    assert docs[0].origin == "radar"
    assert docs[0].source_type == "devto"


@pytest.mark.asyncio
async def test_search_radar_passes_q_param() -> None:
    """search_radar() passes the query string as the 'q' parameter."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    captured_params: list[dict] = []

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        params = kwargs.get("params", {})
        captured_params.append(dict(params))
        return _make_mock_response([])

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("rust async programming")

    assert len(captured_params) >= 1
    assert captured_params[0].get("q") == "rust async programming"


@pytest.mark.asyncio
async def test_search_radar_passes_per_page_param() -> None:
    """search_radar() passes the limit as per_page parameter."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    captured_params: list[dict] = []

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        params = kwargs.get("params", {})
        captured_params.append(dict(params))
        return _make_mock_response([])

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("machine learning", limit=15)

    assert captured_params[0].get("per_page") == 15


@pytest.mark.asyncio
async def test_search_radar_handles_http_error() -> None:
    """search_radar() returns [] and logs error on HTTP failure."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        raise httpx.TimeoutException("Request timed out")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("test query")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_handles_http_status_error() -> None:
    """search_radar() returns [] on HTTP 5xx response."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response({}, status_code=500)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("test query")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_returns_empty_on_no_results() -> None:
    """search_radar() returns [] when DEV.to returns no matching articles."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response([])

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("very obscure query with no results")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_fetches_full_articles() -> None:
    """search_radar() fetches individual article pages to get body_markdown."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    individual_endpoints_called: list[str] = []

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/articles/" in path and path != "/articles":
            individual_endpoints_called.append(path)
            return _make_mock_response(SAMPLE_ARTICLE_FULL)
        return _make_mock_response([SAMPLE_ARTICLE_SUMMARY])

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("python cli")

    assert len(individual_endpoints_called) >= 1
    assert docs[0].raw_content == SAMPLE_ARTICLE_FULL["body_markdown"]


# ---------------------------------------------------------------------------
# _article_to_raw_doc() tests
# ---------------------------------------------------------------------------


def test_article_to_raw_doc_uses_canonical_url() -> None:
    """_article_to_raw_doc() uses canonical_url as doc.url."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {**SAMPLE_ARTICLE_FULL, "canonical_url": "https://example.com/canonical"}

    doc = ingestor._article_to_raw_doc(article)

    assert doc.url == "https://example.com/canonical"


def test_article_to_raw_doc_falls_back_to_url_when_no_canonical() -> None:
    """_article_to_raw_doc() falls back to url field when canonical_url is absent."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {
        **SAMPLE_ARTICLE_FULL,
        "canonical_url": None,
        "url": "https://dev.to/author/article",
    }

    doc = ingestor._article_to_raw_doc(article)

    assert doc.url == "https://dev.to/author/article"


def test_article_to_raw_doc_maps_title() -> None:
    """_article_to_raw_doc() maps the title field correctly."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.title == "Building a CLI tool in Python"


def test_article_to_raw_doc_maps_author_from_user_name() -> None:
    """_article_to_raw_doc() maps author from user.name field."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.author == "Jane Developer"


def test_article_to_raw_doc_maps_author_from_username_when_name_missing() -> None:
    """_article_to_raw_doc() falls back to username when name is absent."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {**SAMPLE_ARTICLE_FULL, "user": {"username": "janedev"}}
    doc = ingestor._article_to_raw_doc(article)

    assert doc.author == "janedev"


def test_article_to_raw_doc_uses_body_markdown_as_raw_content() -> None:
    """_article_to_raw_doc() uses body_markdown as raw_content when present."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.raw_content == SAMPLE_ARTICLE_FULL["body_markdown"]


def test_article_to_raw_doc_falls_back_to_description_when_no_markdown() -> None:
    """_article_to_raw_doc() uses description as raw_content when body_markdown is absent."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {**SAMPLE_ARTICLE_SUMMARY}  # No body_markdown key

    doc = ingestor._article_to_raw_doc(article)

    assert doc.raw_content == article["description"]


def test_article_to_raw_doc_raw_content_is_none_when_no_markdown_or_description() -> None:
    """_article_to_raw_doc() sets raw_content=None when neither body_markdown nor description."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {
        **SAMPLE_ARTICLE_SUMMARY,
        "description": None,
    }

    doc = ingestor._article_to_raw_doc(article)

    assert doc.raw_content is None


def test_article_to_raw_doc_content_type_is_article() -> None:
    """_article_to_raw_doc() always sets content_type='article'."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.content_type == "article"


def test_article_to_raw_doc_source_type_is_devto() -> None:
    """_article_to_raw_doc() always sets source_type='devto'."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.source_type == "devto"


def test_article_to_raw_doc_default_origin_is_pro() -> None:
    """_article_to_raw_doc() uses origin='pro' by default."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.origin == "pro"


def test_article_to_raw_doc_radar_origin() -> None:
    """_article_to_raw_doc() uses origin='radar' when explicitly passed."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL, origin="radar")

    assert doc.origin == "radar"


def test_article_to_raw_doc_parses_published_at_into_datetime() -> None:
    """_article_to_raw_doc() parses published_at ISO string into timezone-aware datetime."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.published_at is not None
    assert isinstance(doc.published_at, datetime)
    assert doc.published_at.year == 2025
    assert doc.published_at.month == 1
    assert doc.published_at.day == 15
    assert doc.published_at.tzinfo is not None


def test_article_to_raw_doc_missing_published_at_gives_none() -> None:
    """_article_to_raw_doc() sets published_at=None when published_at is absent."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {**SAMPLE_ARTICLE_FULL, "published_at": None}
    doc = ingestor._article_to_raw_doc(article)

    assert doc.published_at is None


def test_article_to_raw_doc_invalid_published_at_gives_none() -> None:
    """_article_to_raw_doc() sets published_at=None on unparseable timestamp."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {**SAMPLE_ARTICLE_FULL, "published_at": "not-a-date"}
    doc = ingestor._article_to_raw_doc(article)

    assert doc.published_at is None


def test_article_to_raw_doc_metadata_includes_devto_id() -> None:
    """_article_to_raw_doc() includes devto_id in metadata."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.metadata["devto_id"] == 12345


def test_article_to_raw_doc_metadata_includes_tags() -> None:
    """_article_to_raw_doc() includes tags list in metadata."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.metadata["tags"] == ["python", "cli", "tutorial"]


def test_article_to_raw_doc_metadata_includes_reactions() -> None:
    """_article_to_raw_doc() includes reactions (positive_reactions_count) in metadata."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.metadata["reactions"] == 42


def test_article_to_raw_doc_metadata_includes_comments() -> None:
    """_article_to_raw_doc() includes comments (comments_count) in metadata."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.metadata["comments"] == 7


def test_article_to_raw_doc_metadata_includes_reading_time_minutes() -> None:
    """_article_to_raw_doc() includes reading_time_minutes in metadata."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    assert doc.metadata["reading_time_minutes"] == 5


def test_article_to_raw_doc_metadata_reactions_defaults_to_zero() -> None:
    """_article_to_raw_doc() defaults reactions to 0 when absent."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {**SAMPLE_ARTICLE_FULL, "positive_reactions_count": None}
    doc = ingestor._article_to_raw_doc(article)

    assert doc.metadata["reactions"] == 0


def test_article_to_raw_doc_metadata_comments_defaults_to_zero() -> None:
    """_article_to_raw_doc() defaults comments to 0 when absent."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {**SAMPLE_ARTICLE_FULL, "comments_count": None}
    doc = ingestor._article_to_raw_doc(article)

    assert doc.metadata["comments"] == 0


def test_article_to_raw_doc_metadata_tags_defaults_to_empty_list() -> None:
    """_article_to_raw_doc() defaults tags to [] when absent."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {**SAMPLE_ARTICLE_FULL, "tags": None}
    doc = ingestor._article_to_raw_doc(article)

    assert doc.metadata["tags"] == []


def test_article_to_raw_doc_computes_word_count_from_markdown() -> None:
    """_article_to_raw_doc() computes word_count from body_markdown content."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    doc = ingestor._article_to_raw_doc(SAMPLE_ARTICLE_FULL)

    # body_markdown: "# Building a CLI tool in Python\n\nThis is the full article content..."
    # Word count should be non-zero
    assert doc.word_count is not None
    assert doc.word_count > 0


def test_article_to_raw_doc_word_count_is_none_when_no_content() -> None:
    """_article_to_raw_doc() sets word_count=None when there is no content."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    article = {**SAMPLE_ARTICLE_FULL, "body_markdown": None, "description": None}
    doc = ingestor._article_to_raw_doc(article)

    assert doc.word_count is None


# ---------------------------------------------------------------------------
# _fetch_full_article() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_full_article_merges_summary_with_full_data() -> None:
    """_fetch_full_article() merges summary with full article data from detail endpoint."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_ARTICLE_FULL)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    semaphore = asyncio.Semaphore(5)
    result = await ingestor._fetch_full_article(SAMPLE_ARTICLE_SUMMARY, semaphore)

    assert result is not None
    # Full article should include body_markdown from the detail endpoint
    assert "body_markdown" in result
    assert result["body_markdown"] == SAMPLE_ARTICLE_FULL["body_markdown"]


@pytest.mark.asyncio
async def test_fetch_full_article_falls_back_to_summary_on_http_error() -> None:
    """_fetch_full_article() falls back to summary data when detail endpoint fails."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        raise httpx.ConnectError("Connection refused")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    semaphore = asyncio.Semaphore(5)
    result = await ingestor._fetch_full_article(SAMPLE_ARTICLE_SUMMARY, semaphore)

    # Falls back to summary — should still return the summary dict
    assert result is not None
    assert result["id"] == SAMPLE_ARTICLE_SUMMARY["id"]
    assert result["title"] == SAMPLE_ARTICLE_SUMMARY["title"]


@pytest.mark.asyncio
async def test_fetch_full_article_returns_summary_when_no_article_id() -> None:
    """_fetch_full_article() returns summary directly when the summary has no 'id'."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    summary_no_id = {k: v for k, v in SAMPLE_ARTICLE_SUMMARY.items() if k != "id"}

    semaphore = asyncio.Semaphore(5)
    result = await ingestor._fetch_full_article(summary_no_id, semaphore)

    # Should return the summary unchanged (no ID to fetch)
    assert result is not None
    assert result.get("title") == SAMPLE_ARTICLE_SUMMARY["title"]


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_pro_limits_concurrent_fetches() -> None:
    """fetch_pro() does not exceed 5 concurrent individual article fetches."""
    config = _make_config(devto_cfg=DevtoConfig(tags=["python"], limit=20))
    ingestor = DevtoIngestor(config)

    # Create 10 articles to trigger concurrency limiting
    summaries = [
        {**SAMPLE_ARTICLE_SUMMARY, "id": i, "title": f"Article {i}"}
        for i in range(1, 11)
    ]
    full_articles = [
        {**SAMPLE_ARTICLE_FULL, "id": i, "title": f"Article {i}"}
        for i in range(1, 11)
    ]
    full_by_id = {a["id"]: a for a in full_articles}

    concurrent_count = [0]
    max_concurrent = [0]

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/articles/" in path and path != "/articles":
            # Extract ID from path like /articles/5
            article_id = int(path.split("/")[-1])
            concurrent_count[0] += 1
            max_concurrent[0] = max(max_concurrent[0], concurrent_count[0])
            await asyncio.sleep(0.01)  # Simulate network delay
            concurrent_count[0] -= 1
            return _make_mock_response(full_by_id.get(article_id, SAMPLE_ARTICLE_FULL))
        return _make_mock_response(summaries)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    # Should have returned docs and respected concurrency limit
    assert len(docs) > 0
    assert max_concurrent[0] <= 5


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_returns_self() -> None:
    """DevtoIngestor works as async context manager and returns self."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    async with ingestor as ctx:
        assert ctx is ingestor

    # After exit, client should be closed — no exception expected


@pytest.mark.asyncio
async def test_context_manager_closes_client_on_exit() -> None:
    """DevtoIngestor.__aexit__ calls aclose() on the internal httpx client."""
    config = _make_config()
    ingestor = DevtoIngestor(config)

    close_called = []
    original_aclose = ingestor._client.aclose

    async def mock_aclose() -> None:
        close_called.append(True)
        await original_aclose()

    ingestor._client.aclose = mock_aclose  # type: ignore[method-assign]

    async with ingestor:
        pass

    assert close_called, "aclose() was not called on __aexit__"
