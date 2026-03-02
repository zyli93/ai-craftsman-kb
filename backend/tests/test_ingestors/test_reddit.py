"""Unit tests for the Reddit ingestor.

Tests cover:
- _authenticate() fetches token and caches it; refreshes when expired
- fetch_pro() returns [] when credentials are missing
- fetch_pro() returns [] when no subreddits configured
- fetch_pro() returns RawDocuments for link posts (raw_content=None)
- fetch_pro() returns RawDocuments for self-posts (raw_content=selftext)
- fetch_pro() skips deleted/short self-posts
- fetch_pro() respects min_upvotes filter
- fetch_pro() handles HTTP errors gracefully (returns [])
- search_radar() returns [] when credentials are missing
- search_radar() returns RawDocuments with origin='radar'
- search_radar() passes query and limit params to the API
- search_radar() handles HTTP errors gracefully (returns [])
- _post_to_raw_doc() maps all required metadata fields
- _post_to_raw_doc() parses created_utc into a timezone-aware datetime
- _post_to_raw_doc() returns None for [deleted] self-posts
- _post_to_raw_doc() returns None for self-posts shorter than 50 chars
- Context manager returns self and closes client on exit
"""
import base64
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    FiltersConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    RedditAPIConfig,
    SettingsConfig,
    SourceFilterConfig,
    SourcesConfig,
    SubredditSource,
)
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.ingestors.reddit import RedditIngestor, _MIN_SELFTEXT_LEN

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_LINK_POST = {
    "id": "abc123",
    "title": "Interesting AI article",
    "url": "https://example.com/ai-article",
    "permalink": "/r/MachineLearning/comments/abc123/interesting_ai_article/",
    "author": "mlresearcher",
    "score": 450,
    "num_comments": 87,
    "created_utc": 1700000000.0,
    "is_self": False,
    "selftext": "",
    "subreddit": "MachineLearning",
}

SAMPLE_SELF_POST = {
    "id": "def456",
    "title": "My experience learning Rust",
    "url": "https://www.reddit.com/r/rust/comments/def456/my_experience_learning_rust/",
    "permalink": "/r/rust/comments/def456/my_experience_learning_rust/",
    "author": "rustnoob",
    "score": 120,
    "num_comments": 34,
    "created_utc": 1700001000.0,
    "is_self": True,
    "selftext": "I have been learning Rust for the past three months and wanted to share my experience. "
                "It has been challenging but very rewarding overall.",
    "subreddit": "rust",
}

SAMPLE_DELETED_POST = {
    "id": "ghi789",
    "title": "Deleted post",
    "permalink": "/r/test/comments/ghi789/deleted_post/",
    "author": "[deleted]",
    "score": 5,
    "num_comments": 2,
    "created_utc": 1700002000.0,
    "is_self": True,
    "selftext": "[deleted]",
    "subreddit": "test",
}

SAMPLE_SHORT_SELF_POST = {
    "id": "jkl012",
    "title": "Short post",
    "permalink": "/r/test/comments/jkl012/short_post/",
    "author": "user",
    "score": 10,
    "num_comments": 1,
    "created_utc": 1700003000.0,
    "is_self": True,
    "selftext": "Too short.",
    "subreddit": "test",
}

SAMPLE_TOKEN_RESPONSE = {
    "access_token": "mock_token_abc",
    "token_type": "bearer",
    "expires_in": 3600,
}

SAMPLE_LISTING_RESPONSE = {
    "data": {
        "children": [
            {"data": SAMPLE_LINK_POST},
            {"data": SAMPLE_SELF_POST},
        ]
    }
}

SAMPLE_SEARCH_RESPONSE = {
    "data": {
        "children": [
            {"data": SAMPLE_LINK_POST},
        ]
    }
}


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


def _make_config(
    client_id: str | None = "test_client_id",
    client_secret: str | None = "test_client_secret",
    subreddits: list[SubredditSource] | None = None,
    min_upvotes: int | None = None,
) -> AppConfig:
    """Build a minimal AppConfig for Reddit testing.

    Args:
        client_id: Reddit API client ID (None to simulate missing credentials).
        client_secret: Reddit API client secret (None to simulate missing credentials).
        subreddits: List of SubredditSource configs to include.
        min_upvotes: Minimum upvotes filter for Reddit posts.

    Returns:
        An AppConfig instance for testing.
    """
    reddit_filter = SourceFilterConfig(min_upvotes=min_upvotes)
    return AppConfig(
        sources=SourcesConfig(
            subreddits=subreddits or [],
        ),
        settings=SettingsConfig(
            llm=_make_llm_routing(),
            reddit=RedditAPIConfig(
                client_id=client_id,
                client_secret=client_secret,
            ),
        ),
        filters=FiltersConfig(reddit=reddit_filter),
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
# _authenticate() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_fetches_token() -> None:
    """_authenticate() returns the access_token from the token endpoint."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]

    token = await ingestor._authenticate()

    assert token == "mock_token_abc"
    assert ingestor._token == "mock_token_abc"


@pytest.mark.asyncio
async def test_authenticate_caches_token() -> None:
    """_authenticate() returns the cached token without a new request when not expired."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    call_count = 0
    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        return token_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]

    # First call should fetch the token
    token1 = await ingestor._authenticate()
    # Second call should return the cached token
    token2 = await ingestor._authenticate()

    assert token1 == token2 == "mock_token_abc"
    assert call_count == 1  # Only one POST request made


@pytest.mark.asyncio
async def test_authenticate_refreshes_expired_token() -> None:
    """_authenticate() fetches a new token when the cached one is near expiry."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    # Pre-populate a token that appears to be expired (expires_at in the past)
    ingestor._token = "old_token"
    ingestor._token_expires_at = time.monotonic() - 1.0  # already expired

    call_count = 0
    token_resp = _make_mock_response({**SAMPLE_TOKEN_RESPONSE, "access_token": "new_token"})

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        return token_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]

    token = await ingestor._authenticate()

    assert token == "new_token"
    assert call_count == 1


@pytest.mark.asyncio
async def test_authenticate_sends_basic_auth_header() -> None:
    """_authenticate() sends correct Basic auth header with base64-encoded credentials."""
    config = _make_config(client_id="myid", client_secret="mysecret")
    ingestor = RedditIngestor(config)

    captured_kwargs: list[dict] = []
    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        captured_kwargs.append(dict(kwargs))
        return token_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]

    await ingestor._authenticate()

    assert len(captured_kwargs) == 1
    headers = captured_kwargs[0].get("headers", {})
    expected_b64 = base64.b64encode(b"myid:mysecret").decode()
    assert headers.get("Authorization") == f"Basic {expected_b64}"


@pytest.mark.asyncio
async def test_authenticate_raises_on_http_error() -> None:
    """_authenticate() raises httpx.HTTPError when the token endpoint fails."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    async def mock_post(url: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection refused")

    ingestor._client.post = mock_post  # type: ignore[method-assign]

    with pytest.raises(httpx.HTTPError):
        await ingestor._authenticate()


# ---------------------------------------------------------------------------
# fetch_pro() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_no_credentials() -> None:
    """fetch_pro() returns [] and logs warning when credentials are missing."""
    config = _make_config(client_id=None, client_secret=None)
    ingestor = RedditIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_client_id_missing() -> None:
    """fetch_pro() returns [] when client_id is None even if client_secret is set."""
    config = _make_config(client_id=None, client_secret="secret")
    ingestor = RedditIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_no_subreddits() -> None:
    """fetch_pro() returns [] when no subreddits are configured."""
    config = _make_config(subreddits=[])
    ingestor = RedditIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_returns_documents_for_subreddit() -> None:
    """fetch_pro() returns RawDocuments for each valid post in the subreddit feed."""
    config = _make_config(
        subreddits=[SubredditSource(name="MachineLearning", sort="hot", limit=25)]
    )
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    listing_resp = _make_mock_response(SAMPLE_LISTING_RESPONSE)

    post_counter = 0

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        nonlocal post_counter
        post_counter += 1
        return listing_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    # SAMPLE_LISTING_RESPONSE has 2 posts: 1 link post + 1 self-post
    # Both should be returned (self-post has enough text)
    assert len(docs) == 2
    assert all(isinstance(doc, RawDocument) for doc in docs)
    assert all(doc.source_type == "reddit" for doc in docs)
    assert all(doc.origin == "pro" for doc in docs)


@pytest.mark.asyncio
async def test_fetch_pro_link_post_has_none_raw_content() -> None:
    """fetch_pro() sets raw_content=None for link posts (ContentExtractor fills it)."""
    config = _make_config(
        subreddits=[SubredditSource(name="MachineLearning", sort="hot", limit=25)]
    )
    ingestor = RedditIngestor(config)

    # Response with only the link post
    link_only_response = {"data": {"children": [{"data": SAMPLE_LINK_POST}]}}
    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    listing_resp = _make_mock_response(link_only_response)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return listing_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert len(docs) == 1
    assert docs[0].raw_content is None
    assert docs[0].content_type == "article"


@pytest.mark.asyncio
async def test_fetch_pro_self_post_uses_selftext_as_raw_content() -> None:
    """fetch_pro() sets raw_content=selftext for self-posts with sufficient text."""
    config = _make_config(
        subreddits=[SubredditSource(name="rust", sort="hot", limit=25)]
    )
    ingestor = RedditIngestor(config)

    self_only_response = {"data": {"children": [{"data": SAMPLE_SELF_POST}]}}
    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    listing_resp = _make_mock_response(self_only_response)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return listing_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert len(docs) == 1
    assert docs[0].raw_content == SAMPLE_SELF_POST["selftext"]
    assert docs[0].content_type == "post"


@pytest.mark.asyncio
async def test_fetch_pro_skips_deleted_self_posts() -> None:
    """fetch_pro() skips self-posts with selftext='[deleted]'."""
    config = _make_config(
        subreddits=[SubredditSource(name="test", sort="new", limit=10)]
    )
    ingestor = RedditIngestor(config)

    deleted_only = {"data": {"children": [{"data": SAMPLE_DELETED_POST}]}}
    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    listing_resp = _make_mock_response(deleted_only)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return listing_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_skips_short_self_posts() -> None:
    """fetch_pro() skips self-posts with selftext shorter than 50 characters."""
    config = _make_config(
        subreddits=[SubredditSource(name="test", sort="new", limit=10)]
    )
    ingestor = RedditIngestor(config)

    short_only = {"data": {"children": [{"data": SAMPLE_SHORT_SELF_POST}]}}
    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    listing_resp = _make_mock_response(short_only)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return listing_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_respects_min_upvotes_filter() -> None:
    """fetch_pro() filters out posts below the min_upvotes threshold."""
    # Set min_upvotes to 200; SAMPLE_LINK_POST has 450 (passes), SAMPLE_SELF_POST has 120 (filtered)
    config = _make_config(
        subreddits=[SubredditSource(name="MachineLearning", sort="hot", limit=25)],
        min_upvotes=200,
    )
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    listing_resp = _make_mock_response(SAMPLE_LISTING_RESPONSE)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return listing_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    # Only the link post with 450 upvotes should pass
    assert len(docs) == 1
    assert docs[0].metadata["upvotes"] == 450


@pytest.mark.asyncio
async def test_fetch_pro_handles_http_error_per_subreddit() -> None:
    """fetch_pro() skips failing subreddits and continues with the next one."""
    config = _make_config(
        subreddits=[
            SubredditSource(name="failing_sub", sort="hot", limit=10),
            SubredditSource(name="ok_sub", sort="hot", limit=10),
        ]
    )
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    success_resp = _make_mock_response({"data": {"children": [{"data": SAMPLE_LINK_POST}]}})
    call_count = 0

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("Connection failed")
        return success_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    # First subreddit failed, second succeeded with 1 doc
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_fetch_pro_top_sort_includes_time_param() -> None:
    """fetch_pro() with sort='top' passes t=day to the Reddit API."""
    config = _make_config(
        subreddits=[SubredditSource(name="programming", sort="top", limit=10)]
    )
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    captured_kwargs: list[dict] = []

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        captured_kwargs.append({"url": url, **kwargs})
        return _make_mock_response({"data": {"children": []}})

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.fetch_pro()

    assert len(captured_kwargs) == 1
    params = captured_kwargs[0].get("params", {})
    assert params.get("t") == "day"


@pytest.mark.asyncio
async def test_fetch_pro_non_top_sort_excludes_time_param() -> None:
    """fetch_pro() with sort='hot' does not pass t= param to the Reddit API."""
    config = _make_config(
        subreddits=[SubredditSource(name="programming", sort="hot", limit=10)]
    )
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    captured_kwargs: list[dict] = []

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        captured_kwargs.append({"url": url, **kwargs})
        return _make_mock_response({"data": {"children": []}})

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.fetch_pro()

    assert len(captured_kwargs) == 1
    params = captured_kwargs[0].get("params", {})
    assert "t" not in params


# ---------------------------------------------------------------------------
# search_radar() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_returns_empty_when_no_credentials() -> None:
    """search_radar() returns [] and logs warning when credentials are missing."""
    config = _make_config(client_id=None, client_secret=None)
    ingestor = RedditIngestor(config)

    docs = await ingestor.search_radar("machine learning")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_returns_documents_with_radar_origin() -> None:
    """search_radar() returns RawDocuments with origin='radar'."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    search_resp = _make_mock_response(SAMPLE_SEARCH_RESPONSE)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return search_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("machine learning transformers")

    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)
    assert docs[0].origin == "radar"
    assert docs[0].source_type == "reddit"


@pytest.mark.asyncio
async def test_search_radar_passes_query_param() -> None:
    """search_radar() passes the query string as 'q' param to the Reddit API."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    captured_kwargs: list[dict] = []

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        captured_kwargs.append({"url": url, **kwargs})
        return _make_mock_response({"data": {"children": []}})

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("rust async programming", limit=10)

    assert len(captured_kwargs) == 1
    params = captured_kwargs[0].get("params", {})
    assert params.get("q") == "rust async programming"


@pytest.mark.asyncio
async def test_search_radar_passes_limit_param() -> None:
    """search_radar() passes the limit as a param to the Reddit API."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    captured_kwargs: list[dict] = []

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        captured_kwargs.append({"url": url, **kwargs})
        return _make_mock_response({"data": {"children": []}})

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("query", limit=15)

    params = captured_kwargs[0].get("params", {})
    assert params.get("limit") == 15


@pytest.mark.asyncio
async def test_search_radar_caps_limit_at_100() -> None:
    """search_radar() caps the limit at 100 (Reddit API maximum)."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    captured_kwargs: list[dict] = []

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        captured_kwargs.append({"url": url, **kwargs})
        return _make_mock_response({"data": {"children": []}})

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("query", limit=500)

    params = captured_kwargs[0].get("params", {})
    assert params.get("limit") == 100


@pytest.mark.asyncio
async def test_search_radar_handles_http_error() -> None:
    """search_radar() returns [] and logs error on HTTP failure."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Network unreachable")

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("test query")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_handles_auth_http_error() -> None:
    """search_radar() returns [] and logs error when authentication fails."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    async def mock_post(url: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Auth endpoint unreachable")

    ingestor._client.post = mock_post  # type: ignore[method-assign]

    docs = await ingestor.search_radar("test query")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_handles_http_status_error() -> None:
    """search_radar() returns [] on HTTP 5xx response."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    token_resp = _make_mock_response(SAMPLE_TOKEN_RESPONSE)
    error_resp = _make_mock_response({}, status_code=503)

    async def mock_post(url: str, **kwargs: object) -> MagicMock:
        return token_resp

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return error_resp

    ingestor._client.post = mock_post  # type: ignore[method-assign]
    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("test query")

    assert docs == []


# ---------------------------------------------------------------------------
# _post_to_raw_doc() unit tests
# ---------------------------------------------------------------------------


def test_post_to_raw_doc_link_post_url_from_permalink() -> None:
    """_post_to_raw_doc() sets url from the Reddit permalink for link posts."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    expected_url = f"https://reddit.com{SAMPLE_LINK_POST['permalink']}"
    assert doc.url == expected_url


def test_post_to_raw_doc_title_mapped() -> None:
    """_post_to_raw_doc() maps the title field correctly."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.title == "Interesting AI article"


def test_post_to_raw_doc_author_mapped() -> None:
    """_post_to_raw_doc() maps the author field correctly."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.author == "mlresearcher"


def test_post_to_raw_doc_metadata_subreddit() -> None:
    """_post_to_raw_doc() includes subreddit in metadata."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.metadata["subreddit"] == "MachineLearning"


def test_post_to_raw_doc_metadata_post_id() -> None:
    """_post_to_raw_doc() includes post_id in metadata."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.metadata["post_id"] == "abc123"


def test_post_to_raw_doc_metadata_upvotes() -> None:
    """_post_to_raw_doc() includes upvotes (score) in metadata."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.metadata["upvotes"] == 450


def test_post_to_raw_doc_metadata_comment_count() -> None:
    """_post_to_raw_doc() includes comment_count in metadata."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.metadata["comment_count"] == 87


def test_post_to_raw_doc_metadata_is_self_false_for_link() -> None:
    """_post_to_raw_doc() sets is_self=False for link posts in metadata."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.metadata["is_self"] is False


def test_post_to_raw_doc_metadata_is_self_true_for_selfpost() -> None:
    """_post_to_raw_doc() sets is_self=True for self-posts in metadata."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_SELF_POST)

    assert doc is not None
    assert doc.metadata["is_self"] is True


def test_post_to_raw_doc_metadata_linked_url_for_link_post() -> None:
    """_post_to_raw_doc() sets linked_url in metadata for link posts."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.metadata["linked_url"] == "https://example.com/ai-article"


def test_post_to_raw_doc_metadata_linked_url_none_for_self_post() -> None:
    """_post_to_raw_doc() sets linked_url=None in metadata for self-posts."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_SELF_POST)

    assert doc is not None
    assert doc.metadata["linked_url"] is None


def test_post_to_raw_doc_parses_created_utc() -> None:
    """_post_to_raw_doc() parses created_utc into a timezone-aware datetime."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.published_at is not None
    assert isinstance(doc.published_at, datetime)
    assert doc.published_at.tzinfo is not None
    # created_utc=1700000000.0 → 2023-11-14 22:13:20 UTC
    assert doc.published_at == datetime.fromtimestamp(1700000000.0, tz=timezone.utc)


def test_post_to_raw_doc_missing_created_utc_gives_none() -> None:
    """_post_to_raw_doc() sets published_at=None when created_utc is absent."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    post_no_ts = {**SAMPLE_LINK_POST, "created_utc": None}
    doc = ingestor._post_to_raw_doc(post_no_ts)

    assert doc is not None
    assert doc.published_at is None


def test_post_to_raw_doc_returns_none_for_deleted_selftext() -> None:
    """_post_to_raw_doc() returns None when selftext='[deleted]'."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    result = ingestor._post_to_raw_doc(SAMPLE_DELETED_POST)

    assert result is None


def test_post_to_raw_doc_returns_none_for_removed_selftext() -> None:
    """_post_to_raw_doc() returns None when selftext='[removed]'."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    removed_post = {**SAMPLE_SELF_POST, "selftext": "[removed]"}
    result = ingestor._post_to_raw_doc(removed_post)

    assert result is None


def test_post_to_raw_doc_returns_none_for_short_selftext() -> None:
    """_post_to_raw_doc() returns None for self-posts shorter than _MIN_SELFTEXT_LEN."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    # selftext exactly one character short of the minimum
    borderline_post = {
        **SAMPLE_SELF_POST,
        "selftext": "x" * (_MIN_SELFTEXT_LEN - 1),
    }
    result = ingestor._post_to_raw_doc(borderline_post)

    assert result is None


def test_post_to_raw_doc_accepts_selftext_at_min_length() -> None:
    """_post_to_raw_doc() accepts self-posts with selftext exactly _MIN_SELFTEXT_LEN chars."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    min_length_post = {
        **SAMPLE_SELF_POST,
        "selftext": "x" * _MIN_SELFTEXT_LEN,
    }
    doc = ingestor._post_to_raw_doc(min_length_post)

    assert doc is not None
    assert doc.raw_content == "x" * _MIN_SELFTEXT_LEN


def test_post_to_raw_doc_default_origin_is_pro() -> None:
    """_post_to_raw_doc() uses origin='pro' by default."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.origin == "pro"


def test_post_to_raw_doc_radar_origin() -> None:
    """_post_to_raw_doc() uses origin='radar' when explicitly passed."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST, origin="radar")

    assert doc is not None
    assert doc.origin == "radar"


def test_post_to_raw_doc_source_type_is_reddit() -> None:
    """_post_to_raw_doc() always sets source_type='reddit'."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    doc = ingestor._post_to_raw_doc(SAMPLE_LINK_POST)

    assert doc is not None
    assert doc.source_type == "reddit"


def test_post_to_raw_doc_missing_score_defaults_to_zero() -> None:
    """_post_to_raw_doc() defaults upvotes to 0 when score is absent."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    post_no_score = {**SAMPLE_LINK_POST, "score": None}
    doc = ingestor._post_to_raw_doc(post_no_score)

    assert doc is not None
    assert doc.metadata["upvotes"] == 0


def test_post_to_raw_doc_missing_num_comments_defaults_to_zero() -> None:
    """_post_to_raw_doc() defaults comment_count to 0 when num_comments is absent."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    post_no_comments = {**SAMPLE_LINK_POST, "num_comments": None}
    doc = ingestor._post_to_raw_doc(post_no_comments)

    assert doc is not None
    assert doc.metadata["comment_count"] == 0


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_returns_self() -> None:
    """RedditIngestor works as async context manager and returns self."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    async with ingestor as ctx:
        assert ctx is ingestor


@pytest.mark.asyncio
async def test_context_manager_closes_client_on_exit() -> None:
    """RedditIngestor.__aexit__ calls aclose() on the internal httpx client."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    close_called: list[bool] = []
    original_aclose = ingestor._client.aclose

    async def mock_aclose() -> None:
        close_called.append(True)
        await original_aclose()

    ingestor._client.aclose = mock_aclose  # type: ignore[method-assign]

    async with ingestor:
        pass

    assert close_called, "aclose() was not called on __aexit__"


# ---------------------------------------------------------------------------
# source_type property test
# ---------------------------------------------------------------------------


def test_source_type_is_reddit() -> None:
    """RedditIngestor.source_type is 'reddit'."""
    config = _make_config()
    ingestor = RedditIngestor(config)

    assert ingestor.source_type == "reddit"
