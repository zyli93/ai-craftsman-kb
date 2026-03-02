"""Unit tests for the YouTubeIngestor radar search implementation.

Tests cover the new/enhanced radar-specific methods:
- search_radar() returns up to `limit` documents with origin='radar'
- search_radar() fetches transcripts concurrently via asyncio.gather
- search_radar() includes videos without transcripts (description fallback)
- search_radar() sets metadata['has_transcript'] accurately
- search_radar() returns [] when API key is missing (with warning log)
- search_radar() returns [] on HTTP errors (connection/status)
- search_radar() handles quotaExceeded (403) gracefully
- _search_videos() returns list of raw API item dicts
- _search_videos() respects max 50 cap on maxResults param
- _search_videos() returns [] on HTTPStatusError (non-403)
- _search_videos() returns [] on HTTPError
- _get_transcript_safe() returns None on exception (never propagates)
- _snippet_to_raw_doc() maps all fields correctly
- _snippet_to_raw_doc() sets has_transcript=True when transcript present
- _snippet_to_raw_doc() uses description as fallback when no transcript
- _snippet_to_raw_doc() sets has_transcript=False when no transcript
- _snippet_to_raw_doc() includes thumbnail_url in metadata
- Concurrent gather: transcript fetches run in parallel (not sequential)
"""
import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    FiltersConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    SettingsConfig,
    SourcesConfig,
    YoutubeAPIConfig,
    YoutubeChannelSource,
)
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.ingestors.youtube import YouTubeIngestor


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SAMPLE_VIDEO_ITEM = {
    "id": {"videoId": "abc123"},
    "snippet": {
        "title": "Introduction to Transformers",
        "description": "A deep dive into attention mechanisms and transformers.",
        "channelTitle": "ML Channel",
        "channelId": "UCtest123",
        "publishedAt": "2025-02-10T12:00:00Z",
        "thumbnails": {
            "default": {"url": "https://img.youtube.com/vi/abc123/default.jpg"},
            "medium": {"url": "https://img.youtube.com/vi/abc123/mqdefault.jpg"},
        },
    },
}

SAMPLE_VIDEO_ITEM_2 = {
    "id": {"videoId": "xyz789"},
    "snippet": {
        "title": "Fine-tuning LLMs in 2025",
        "description": "How to fine-tune large language models efficiently.",
        "channelTitle": "AI Research",
        "channelId": "UCresearch456",
        "publishedAt": "2025-01-20T09:00:00Z",
        "thumbnails": {
            "default": {"url": "https://img.youtube.com/vi/xyz789/default.jpg"},
        },
    },
}

SAMPLE_VIDEO_ITEM_3 = {
    "id": {"videoId": "def456"},
    "snippet": {
        "title": "RAG Explained Simply",
        "description": "Retrieval Augmented Generation explained step by step.",
        "channelTitle": "Coding with AI",
        "channelId": "UCcoding789",
        "publishedAt": "2025-03-01T15:30:00Z",
        "thumbnails": {},
    },
}

SAMPLE_SEARCH_RESPONSE_1 = {"items": [SAMPLE_VIDEO_ITEM]}
SAMPLE_SEARCH_RESPONSE_2 = {"items": [SAMPLE_VIDEO_ITEM, SAMPLE_VIDEO_ITEM_2]}
SAMPLE_SEARCH_RESPONSE_3 = {
    "items": [SAMPLE_VIDEO_ITEM, SAMPLE_VIDEO_ITEM_2, SAMPLE_VIDEO_ITEM_3]
}
SAMPLE_TRANSCRIPT = "Welcome to this video on transformers. Today we will cover attention."
QUOTA_EXCEEDED_BODY = {
    "error": {
        "code": 403,
        "errors": [{"reason": "quotaExceeded", "domain": "youtube.quota"}],
    }
}


# ---------------------------------------------------------------------------
# Fixtures / helpers
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


def _make_config(api_key: str | None = "test-api-key") -> AppConfig:
    """Build a minimal AppConfig with a YouTube API key.

    Args:
        api_key: The YouTube API key, or None to simulate missing key.

    Returns:
        An AppConfig instance with YouTube settings configured.
    """
    yt_config = YoutubeAPIConfig(api_key=api_key, transcript_langs=["en"])
    return AppConfig(
        sources=SourcesConfig(
            youtube_channels=[YoutubeChannelSource(handle="@Test", name="Test")]
        ),
        settings=SettingsConfig(
            llm=_make_llm_routing(),
            youtube=yt_config,
        ),
        filters=FiltersConfig(),
    )


def _make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response that returns json_data from .json().

    Args:
        json_data: The dict that .json() should return.
        status_code: HTTP status code.

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
            response=mock_resp,
        )
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# search_radar() — basic acceptance criteria
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_returns_empty_when_no_api_key() -> None:
    """search_radar() returns [] and logs warning when API key is not configured."""
    config = _make_config(api_key=None)
    ingestor = YouTubeIngestor(config)

    docs = await ingestor.search_radar("machine learning")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_returns_radar_origin() -> None:
    """search_radar() returns RawDocuments with origin='radar'."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_1)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        docs = await ingestor.search_radar("transformers")

    assert len(docs) == 1
    assert docs[0].origin == "radar"


@pytest.mark.asyncio
async def test_search_radar_returns_source_type_youtube() -> None:
    """search_radar() returns RawDocuments with source_type='youtube'."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_1)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        docs = await ingestor.search_radar("transformers")

    assert all(doc.source_type == "youtube" for doc in docs)


@pytest.mark.asyncio
async def test_search_radar_content_type_is_video() -> None:
    """search_radar() returns RawDocuments with content_type='video'."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_1)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        docs = await ingestor.search_radar("transformers")

    assert all(doc.content_type == "video" for doc in docs)


@pytest.mark.asyncio
async def test_search_radar_respects_limit() -> None:
    """search_radar() returns at most `limit` documents."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    # Return 3 videos from search API
    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_3)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        docs = await ingestor.search_radar("LLM", limit=2)

    assert len(docs) <= 2


@pytest.mark.asyncio
async def test_search_radar_returns_empty_when_api_returns_no_items() -> None:
    """search_radar() returns [] when the YouTube API returns no video items."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response({"items": []})

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("very obscure query")

    assert docs == []


# ---------------------------------------------------------------------------
# search_radar() — transcript handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_includes_videos_with_transcripts() -> None:
    """search_radar() includes videos with available transcripts."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_1)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        docs = await ingestor.search_radar("transformers")

    assert len(docs) == 1
    assert docs[0].raw_content == SAMPLE_TRANSCRIPT
    assert docs[0].metadata["has_transcript"] is True


@pytest.mark.asyncio
async def test_search_radar_includes_videos_without_transcripts_as_fallback() -> None:
    """search_radar() includes videos without transcripts using description as fallback."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_1)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=None,
    ):
        # limit=4 means up to limit//2 = 2 fallback videos are accepted
        docs = await ingestor.search_radar("transformers", limit=4)

    assert len(docs) == 1
    # Falls back to description as raw_content
    assert docs[0].raw_content == SAMPLE_VIDEO_ITEM["snippet"]["description"]
    assert docs[0].metadata["has_transcript"] is False


@pytest.mark.asyncio
async def test_search_radar_has_transcript_true_when_transcript_present() -> None:
    """search_radar() sets metadata['has_transcript']=True when transcript is available."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_1)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        docs = await ingestor.search_radar("transformers")

    assert docs[0].metadata["has_transcript"] is True


@pytest.mark.asyncio
async def test_search_radar_has_transcript_false_when_no_transcript() -> None:
    """search_radar() sets metadata['has_transcript']=False when no transcript available."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_1)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=None,
    ):
        docs = await ingestor.search_radar("transformers", limit=4)

    assert docs[0].metadata["has_transcript"] is False


@pytest.mark.asyncio
async def test_search_radar_prefers_transcript_over_description() -> None:
    """search_radar() uses transcript as raw_content when available (not description)."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_1)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        docs = await ingestor.search_radar("transformers")

    # Should use transcript, not description
    assert docs[0].raw_content == SAMPLE_TRANSCRIPT
    assert docs[0].raw_content != SAMPLE_VIDEO_ITEM["snippet"]["description"]


@pytest.mark.asyncio
async def test_search_radar_mixed_transcripts_includes_fallbacks_up_to_half_limit() -> None:
    """search_radar() includes no-transcript videos up to limit//2 as fallback slots."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    # 3 videos: first has transcript, others do not
    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_3)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    def mock_transcript_sync(video_id: str, langs: list) -> str | None:
        """Return transcript only for the first video."""
        if video_id == "abc123":
            return SAMPLE_TRANSCRIPT
        return None

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        side_effect=mock_transcript_sync,
    ):
        # limit=4 → limit//2=2 fallback slots
        docs = await ingestor.search_radar("LLM", limit=4)

    # 1 transcript video + up to 2 fallback videos = 3 total (all 3 returned)
    assert len(docs) == 3
    transcript_docs = [d for d in docs if d.metadata["has_transcript"]]
    fallback_docs = [d for d in docs if not d.metadata["has_transcript"]]
    assert len(transcript_docs) == 1
    assert len(fallback_docs) == 2


# ---------------------------------------------------------------------------
# search_radar() — concurrent transcript fetching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_fetches_transcripts_concurrently() -> None:
    """search_radar() fetches transcripts concurrently (total time ~ max, not sum).

    3 transcript fetches each taking 0.1s should complete in ~0.1s total
    (concurrent), not ~0.3s (sequential).
    """
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_3)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    call_delay = 0.1

    async def slow_get_transcript(video_id: str) -> str | None:
        """Simulate a slow transcript fetch."""
        await asyncio.sleep(call_delay)
        return SAMPLE_TRANSCRIPT

    ingestor._get_transcript_safe = slow_get_transcript  # type: ignore[method-assign]

    start = time.monotonic()
    docs = await ingestor.search_radar("concurrent test", limit=10)
    elapsed = time.monotonic() - start

    # Sequential would take 3 * 0.1 = 0.3s; concurrent should be well under 0.25s
    sequential_time = 3 * call_delay
    assert elapsed < sequential_time * 0.85, (
        f"Expected concurrent execution (< {sequential_time * 0.85:.2f}s), "
        f"but took {elapsed:.3f}s — suggests sequential fetching"
    )
    # Also verify we got docs back
    assert len(docs) == 3


# ---------------------------------------------------------------------------
# search_radar() — error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_returns_empty_on_http_connection_error() -> None:
    """search_radar() returns [] on HTTP connection errors."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection refused")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("python")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_returns_empty_on_http_status_error() -> None:
    """search_radar() returns [] on HTTP 5xx responses."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    mock_resp = _make_mock_response({}, status_code=503)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("python")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_logs_warning_on_quota_exceeded(caplog: pytest.LogCaptureFixture) -> None:
    """search_radar() logs an error message when quotaExceeded (403)."""
    import logging

    config = _make_config()
    ingestor = YouTubeIngestor(config)

    mock_resp = _make_mock_response(QUOTA_EXCEEDED_BODY, status_code=403)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with caplog.at_level(logging.ERROR, logger="ai_craftsman_kb.ingestors.youtube"):
        docs = await ingestor.search_radar("python")

    assert docs == []
    assert any("quota" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# _search_videos() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_videos_returns_list_of_items() -> None:
    """_search_videos() returns the list of item dicts from the API response."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE_2)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    items = await ingestor._search_videos("transformers", limit=10)

    assert isinstance(items, list)
    assert len(items) == 2
    assert items[0]["id"]["videoId"] == "abc123"


@pytest.mark.asyncio
async def test_search_videos_caps_limit_at_50() -> None:
    """_search_videos() never requests more than 50 results (YouTube API max)."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    captured_params: list[dict] = []
    mock_resp = _make_mock_response({"items": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        captured_params.append(dict(kwargs.get("params", {})))
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor._search_videos("test", limit=100)

    assert captured_params[0].get("maxResults") == 50


@pytest.mark.asyncio
async def test_search_videos_passes_query_param() -> None:
    """_search_videos() passes the query string as 'q' parameter to the API."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    captured_params: list[dict] = []
    mock_resp = _make_mock_response({"items": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        captured_params.append(dict(kwargs.get("params", {})))
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor._search_videos("rust async programming", limit=5)

    assert captured_params[0].get("q") == "rust async programming"


@pytest.mark.asyncio
async def test_search_videos_returns_empty_on_connection_error() -> None:
    """_search_videos() returns [] on network connection errors."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection failed")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    items = await ingestor._search_videos("python", limit=10)

    assert items == []


@pytest.mark.asyncio
async def test_search_videos_returns_empty_on_5xx_error() -> None:
    """_search_videos() returns [] on HTTP 5xx status errors."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    mock_resp = _make_mock_response({}, status_code=500)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    items = await ingestor._search_videos("python", limit=10)

    assert items == []


@pytest.mark.asyncio
async def test_search_videos_returns_empty_on_quota_exceeded() -> None:
    """_search_videos() returns [] on 403 quotaExceeded error."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    mock_resp = _make_mock_response(QUOTA_EXCEEDED_BODY, status_code=403)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    items = await ingestor._search_videos("machine learning", limit=10)

    assert items == []


@pytest.mark.asyncio
async def test_search_videos_passes_correct_api_params() -> None:
    """_search_videos() passes correct part, type, and order params."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    captured_params: list[dict] = []
    mock_resp = _make_mock_response({"items": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        captured_params.append(dict(kwargs.get("params", {})))
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor._search_videos("AI", limit=10)

    params = captured_params[0]
    assert params.get("part") == "snippet"
    assert params.get("type") == "video"
    assert params.get("order") == "relevance"


# ---------------------------------------------------------------------------
# _get_transcript_safe() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_transcript_safe_returns_transcript() -> None:
    """_get_transcript_safe() returns transcript text when available."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        result = await ingestor._get_transcript_safe("abc123")

    assert result == SAMPLE_TRANSCRIPT


@pytest.mark.asyncio
async def test_get_transcript_safe_returns_none_on_exception() -> None:
    """_get_transcript_safe() returns None (not raises) when an exception occurs."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def failing_get_transcript(video_id: str) -> str:
        raise RuntimeError("Unexpected transcript error")

    ingestor._get_transcript = failing_get_transcript  # type: ignore[method-assign]

    result = await ingestor._get_transcript_safe("problem_video")

    assert result is None


@pytest.mark.asyncio
async def test_get_transcript_safe_returns_none_when_transcript_unavailable() -> None:
    """_get_transcript_safe() returns None when transcript API returns None."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=None,
    ):
        result = await ingestor._get_transcript_safe("no_transcript_video")

    assert result is None


@pytest.mark.asyncio
async def test_get_transcript_safe_does_not_propagate_exceptions() -> None:
    """_get_transcript_safe() never raises — always returns None on error."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    # Even a very nasty error should be swallowed
    async def raising_get_transcript(video_id: str) -> str:
        raise ValueError(f"Critical failure for {video_id}")

    ingestor._get_transcript = raising_get_transcript  # type: ignore[method-assign]

    # Should not raise
    result = await ingestor._get_transcript_safe("any_video_id")
    assert result is None


# ---------------------------------------------------------------------------
# _snippet_to_raw_doc() tests
# ---------------------------------------------------------------------------


def test_snippet_to_raw_doc_with_transcript() -> None:
    """_snippet_to_raw_doc() uses transcript as raw_content when available."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, SAMPLE_TRANSCRIPT)

    assert doc.raw_content == SAMPLE_TRANSCRIPT
    assert doc.metadata["has_transcript"] is True


def test_snippet_to_raw_doc_without_transcript_uses_description() -> None:
    """_snippet_to_raw_doc() uses description as raw_content fallback when no transcript."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, None)

    assert doc.raw_content == SAMPLE_VIDEO_ITEM["snippet"]["description"]
    assert doc.metadata["has_transcript"] is False


def test_snippet_to_raw_doc_maps_url() -> None:
    """_snippet_to_raw_doc() builds the correct YouTube watch URL."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, SAMPLE_TRANSCRIPT)

    assert doc.url == "https://youtube.com/watch?v=abc123"


def test_snippet_to_raw_doc_maps_title() -> None:
    """_snippet_to_raw_doc() maps the title from the snippet."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, SAMPLE_TRANSCRIPT)

    assert doc.title == "Introduction to Transformers"


def test_snippet_to_raw_doc_maps_author_from_channel_title() -> None:
    """_snippet_to_raw_doc() sets author from channelTitle."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, SAMPLE_TRANSCRIPT)

    assert doc.author == "ML Channel"


def test_snippet_to_raw_doc_origin_is_radar() -> None:
    """_snippet_to_raw_doc() always sets origin='radar'."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, None)

    assert doc.origin == "radar"


def test_snippet_to_raw_doc_content_type_is_video() -> None:
    """_snippet_to_raw_doc() always sets content_type='video'."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, None)

    assert doc.content_type == "video"


def test_snippet_to_raw_doc_source_type_is_youtube() -> None:
    """_snippet_to_raw_doc() always sets source_type='youtube'."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, None)

    assert doc.source_type == "youtube"


def test_snippet_to_raw_doc_metadata_video_id() -> None:
    """_snippet_to_raw_doc() includes video_id in metadata."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, SAMPLE_TRANSCRIPT)

    assert doc.metadata["video_id"] == "abc123"


def test_snippet_to_raw_doc_metadata_channel_id() -> None:
    """_snippet_to_raw_doc() includes channel_id in metadata."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, SAMPLE_TRANSCRIPT)

    assert doc.metadata["channel_id"] == "UCtest123"


def test_snippet_to_raw_doc_metadata_channel_title() -> None:
    """_snippet_to_raw_doc() includes channel_title in metadata."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, SAMPLE_TRANSCRIPT)

    assert doc.metadata["channel_title"] == "ML Channel"


def test_snippet_to_raw_doc_metadata_description() -> None:
    """_snippet_to_raw_doc() includes description in metadata."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, SAMPLE_TRANSCRIPT)

    assert doc.metadata["description"] == "A deep dive into attention mechanisms and transformers."


def test_snippet_to_raw_doc_metadata_thumbnail_url_medium_preferred() -> None:
    """_snippet_to_raw_doc() uses 'medium' thumbnail URL when available."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, None)

    assert doc.metadata["thumbnail_url"] == "https://img.youtube.com/vi/abc123/mqdefault.jpg"


def test_snippet_to_raw_doc_metadata_thumbnail_url_fallback_to_default() -> None:
    """_snippet_to_raw_doc() falls back to 'default' thumbnail URL when 'medium' absent."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    # SAMPLE_VIDEO_ITEM_2 only has 'default' thumbnail
    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM_2, None)

    assert doc.metadata["thumbnail_url"] == "https://img.youtube.com/vi/xyz789/default.jpg"


def test_snippet_to_raw_doc_metadata_thumbnail_url_none_when_no_thumbnails() -> None:
    """_snippet_to_raw_doc() sets thumbnail_url=None when thumbnails dict is empty."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    # SAMPLE_VIDEO_ITEM_3 has empty thumbnails dict
    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM_3, None)

    assert doc.metadata["thumbnail_url"] is None


def test_snippet_to_raw_doc_word_count_from_transcript() -> None:
    """_snippet_to_raw_doc() computes word_count from transcript when available."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, SAMPLE_TRANSCRIPT)

    expected_count = len(SAMPLE_TRANSCRIPT.split())
    assert doc.word_count == expected_count


def test_snippet_to_raw_doc_word_count_from_description_fallback() -> None:
    """_snippet_to_raw_doc() computes word_count from description when no transcript."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    description = SAMPLE_VIDEO_ITEM["snippet"]["description"]
    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, None)

    expected_count = len(description.split())
    assert doc.word_count == expected_count


def test_snippet_to_raw_doc_parses_published_at() -> None:
    """_snippet_to_raw_doc() parses publishedAt into a timezone-aware datetime."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, None)

    assert doc.published_at is not None
    assert isinstance(doc.published_at, datetime)
    assert doc.published_at.year == 2025
    assert doc.published_at.month == 2
    assert doc.published_at.day == 10
    assert doc.published_at.tzinfo is not None


def test_snippet_to_raw_doc_has_transcript_true() -> None:
    """_snippet_to_raw_doc() sets has_transcript=True when transcript arg is a string."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, "some transcript text")

    assert doc.metadata["has_transcript"] is True


def test_snippet_to_raw_doc_has_transcript_false() -> None:
    """_snippet_to_raw_doc() sets has_transcript=False when transcript arg is None."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._snippet_to_raw_doc(SAMPLE_VIDEO_ITEM, None)

    assert doc.metadata["has_transcript"] is False


def test_snippet_to_raw_doc_raw_content_none_when_no_transcript_no_description() -> None:
    """_snippet_to_raw_doc() sets raw_content=None when both transcript and description are absent."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    item_no_description = {
        "id": {"videoId": "empty123"},
        "snippet": {
            "title": "Silent Video",
            "description": "",  # empty description
            "channelTitle": "Channel",
            "channelId": "UCchannel",
            "publishedAt": "2025-01-01T00:00:00Z",
            "thumbnails": {},
        },
    }

    doc = ingestor._snippet_to_raw_doc(item_no_description, None)

    assert doc.raw_content is None
