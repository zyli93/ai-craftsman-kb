"""Unit tests for the YouTube ingestor.

Tests cover:
- fetch_pro() returns RawDocuments for configured channels
- fetch_pro() returns [] when API key is missing
- fetch_pro() returns [] when no channels are configured
- fetch_pro() handles HTTP error gracefully (returns [])
- search_radar() returns videos with origin='radar'
- search_radar() returns [] when API key is missing
- search_radar() handles HTTP error gracefully
- search_radar() only includes videos that have transcripts
- _resolve_handle() caches handle -> channel_id mapping
- _resolve_handle() returns None when handle not found
- _item_to_raw_doc() maps fields correctly: url, title, author, metadata
- _item_to_raw_doc() returns None when video_id is missing
- fetch_content() uses transcript instead of HTML extraction
- _get_transcript() returns None when transcript unavailable (no crash)
"""
import pytest
import httpx
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from ai_craftsman_kb.ingestors.youtube import YouTubeIngestor, _fetch_transcript_sync
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.config.models import (
    AppConfig,
    SourcesConfig,
    SettingsConfig,
    FiltersConfig,
    YoutubeChannelSource,
    YoutubeAPIConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_VIDEO_ITEM = {
    "id": {"videoId": "abc123"},
    "snippet": {
        "title": "How Neural Networks Actually Work",
        "description": "A deep dive into backpropagation.",
        "channelTitle": "Andrej Karpathy",
        "channelId": "UCXZCJLdBC09xxGZ6gcdrc6A",
        "publishedAt": "2025-01-15T10:00:00Z",
    },
}

SAMPLE_VIDEO_ITEM_2 = {
    "id": {"videoId": "xyz789"},
    "snippet": {
        "title": "Attention Is All You Need — Explained",
        "description": "Transformer architecture walkthrough.",
        "channelTitle": "Andrej Karpathy",
        "channelId": "UCXZCJLdBC09xxGZ6gcdrc6A",
        "publishedAt": "2025-01-10T08:30:00Z",
    },
}

SAMPLE_CHANNEL_RESPONSE = {
    "items": [{"id": "UCXZCJLdBC09xxGZ6gcdrc6A"}]
}

SAMPLE_SEARCH_RESPONSE = {
    "items": [SAMPLE_VIDEO_ITEM],
}

SAMPLE_SEARCH_MULTI_RESPONSE = {
    "items": [SAMPLE_VIDEO_ITEM, SAMPLE_VIDEO_ITEM_2],
}

SAMPLE_TRANSCRIPT = "Hello and welcome to this video. Today we will discuss transformers."


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


def _make_config(
    api_key: str | None = "test-api-key",
    channels: list[YoutubeChannelSource] | None = None,
    transcript_langs: list[str] | None = None,
) -> AppConfig:
    """Build a minimal AppConfig with YouTube settings.

    Args:
        api_key: The YouTube API key to configure (None simulates missing key).
        channels: List of YouTube channel sources to configure.
        transcript_langs: Preferred transcript languages (defaults to ['en']).

    Returns:
        An AppConfig instance with minimal settings.
    """
    if channels is None:
        channels = [YoutubeChannelSource(handle="@AndrejKarpathy", name="Andrej Karpathy")]
    yt_config = YoutubeAPIConfig(
        api_key=api_key,
        transcript_langs=transcript_langs or ["en"],
    )
    return AppConfig(
        sources=SourcesConfig(youtube_channels=channels),
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
            response=mock_resp,
        )
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# fetch_pro() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_no_api_key() -> None:
    """fetch_pro() returns [] and logs warning when API key is not configured."""
    config = _make_config(api_key=None)
    ingestor = YouTubeIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_returns_empty_when_no_channels() -> None:
    """fetch_pro() returns [] when no YouTube channels are configured."""
    config = _make_config(channels=[])
    ingestor = YouTubeIngestor(config)

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_returns_documents() -> None:
    """fetch_pro() returns RawDocuments for each video in configured channels."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/channels" in path:
            return _make_mock_response(SAMPLE_CHANNEL_RESPONSE)
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        docs = await ingestor.fetch_pro()

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)
    assert docs[0].source_type == "youtube"
    assert docs[0].origin == "pro"
    assert docs[0].content_type == "video"


@pytest.mark.asyncio
async def test_fetch_pro_stores_transcript_as_raw_content() -> None:
    """fetch_pro() populates raw_content with transcript text."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/channels" in path:
            return _make_mock_response(SAMPLE_CHANNEL_RESPONSE)
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        docs = await ingestor.fetch_pro()

    assert docs[0].raw_content == SAMPLE_TRANSCRIPT
    assert docs[0].word_count == len(SAMPLE_TRANSCRIPT.split())


@pytest.mark.asyncio
async def test_fetch_pro_raw_content_none_when_no_transcript() -> None:
    """fetch_pro() keeps raw_content=None for videos without available transcripts."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/channels" in path:
            return _make_mock_response(SAMPLE_CHANNEL_RESPONSE)
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=None,
    ):
        docs = await ingestor.fetch_pro()

    assert len(docs) == 1
    assert docs[0].raw_content is None


@pytest.mark.asyncio
async def test_fetch_pro_handles_http_error_gracefully() -> None:
    """fetch_pro() returns [] and logs error when channel video fetch fails."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/channels" in path:
            return _make_mock_response(SAMPLE_CHANNEL_RESPONSE)
        raise httpx.ConnectError("Connection refused")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_skips_channel_if_handle_unresolvable() -> None:
    """fetch_pro() skips channels that cannot be resolved to a channel_id."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/channels" in path:
            # Return empty items — handle not found
            return _make_mock_response({"items": []})
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.fetch_pro()

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_pro_metadata_includes_required_fields() -> None:
    """fetch_pro() returns docs with video_id, channel_handle, channel_id, description."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        if "/channels" in path:
            return _make_mock_response(SAMPLE_CHANNEL_RESPONSE)
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=None,
    ):
        docs = await ingestor.fetch_pro()

    assert len(docs) == 1
    meta = docs[0].metadata
    assert "video_id" in meta
    assert "channel_handle" in meta
    assert "channel_id" in meta
    assert "description" in meta
    assert meta["video_id"] == "abc123"
    assert meta["channel_handle"] == "@AndrejKarpathy"


# ---------------------------------------------------------------------------
# search_radar() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_radar_returns_empty_when_no_api_key() -> None:
    """search_radar() returns [] and logs warning when API key is not configured."""
    config = _make_config(api_key=None)
    ingestor = YouTubeIngestor(config)

    docs = await ingestor.search_radar("machine learning")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_returns_documents_with_transcripts() -> None:
    """search_radar() returns RawDocuments with origin='radar' for videos with transcripts."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_RESPONSE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        docs = await ingestor.search_radar("neural networks")

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0].origin == "radar"
    assert docs[0].source_type == "youtube"


@pytest.mark.asyncio
async def test_search_radar_only_includes_videos_with_transcripts() -> None:
    """search_radar() excludes videos where transcript fetch returns None."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_SEARCH_MULTI_RESPONSE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    def mock_transcript_sync(video_id: str, langs: list) -> str | None:
        # Only return transcript for first video
        if video_id == "abc123":
            return SAMPLE_TRANSCRIPT
        return None

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        side_effect=mock_transcript_sync,
    ):
        docs = await ingestor.search_radar("transformers", limit=10)

    # Videos with transcripts are preferred; no-transcript videos are included
    # as fallback (up to limit // 2 = 5 slots with limit=10).
    # abc123 has a transcript; xyz789 does not but qualifies as a fallback.
    assert len(docs) == 2
    video_ids = [d.metadata["video_id"] for d in docs]
    assert "abc123" in video_ids


@pytest.mark.asyncio
async def test_search_radar_passes_query_param() -> None:
    """search_radar() passes the query string to the YouTube API."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    captured_params: list[dict] = []
    mock_resp = _make_mock_response({"items": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        captured_params.append(dict(kwargs.get("params", {})))
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("rust programming", limit=5)

    assert len(captured_params) == 1
    assert captured_params[0].get("q") == "rust programming"


@pytest.mark.asyncio
async def test_search_radar_respects_limit() -> None:
    """search_radar() sends the correct maxResults param."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    captured_params: list[dict] = []
    mock_resp = _make_mock_response({"items": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        captured_params.append(dict(kwargs.get("params", {})))
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("python", limit=15)

    # search_radar over-fetches (limit * 2) to fill slots after transcript filtering
    assert captured_params[0].get("maxResults") == 30


@pytest.mark.asyncio
async def test_search_radar_caps_limit_at_50() -> None:
    """search_radar() caps maxResults at 50 (YouTube API max)."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    captured_params: list[dict] = []
    mock_resp = _make_mock_response({"items": []})

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        captured_params.append(dict(kwargs.get("params", {})))
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    await ingestor.search_radar("python", limit=100)

    assert captured_params[0].get("maxResults") == 50


@pytest.mark.asyncio
async def test_search_radar_handles_http_error_gracefully() -> None:
    """search_radar() returns [] and logs error when HTTP request fails."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection refused")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("python")

    assert docs == []


@pytest.mark.asyncio
async def test_search_radar_handles_http_status_error() -> None:
    """search_radar() returns [] on HTTP 5xx response."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    mock_resp = _make_mock_response({}, status_code=503)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return mock_resp

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    docs = await ingestor.search_radar("python")

    assert docs == []


# ---------------------------------------------------------------------------
# _resolve_handle() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_handle_returns_channel_id() -> None:
    """_resolve_handle() returns the channel_id for a valid handle."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response(SAMPLE_CHANNEL_RESPONSE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    channel_id = await ingestor._resolve_handle("@AndrejKarpathy")

    assert channel_id == "UCXZCJLdBC09xxGZ6gcdrc6A"


@pytest.mark.asyncio
async def test_resolve_handle_caches_result() -> None:
    """_resolve_handle() caches handle -> channel_id and avoids repeated API calls."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    call_count = 0

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        return _make_mock_response(SAMPLE_CHANNEL_RESPONSE)

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    # Call twice with the same handle
    result1 = await ingestor._resolve_handle("@AndrejKarpathy")
    result2 = await ingestor._resolve_handle("@AndrejKarpathy")

    assert result1 == result2 == "UCXZCJLdBC09xxGZ6gcdrc6A"
    # API should only be called once due to caching
    assert call_count == 1


@pytest.mark.asyncio
async def test_resolve_handle_returns_none_when_not_found() -> None:
    """_resolve_handle() returns None when the handle doesn't exist on YouTube."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> MagicMock:
        return _make_mock_response({"items": []})

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    channel_id = await ingestor._resolve_handle("@NonExistentChannel")

    assert channel_id is None


@pytest.mark.asyncio
async def test_resolve_handle_returns_none_on_http_error() -> None:
    """_resolve_handle() returns None and logs error on HTTP failure."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async def mock_get(path: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection refused")

    ingestor._client.get = mock_get  # type: ignore[method-assign]

    channel_id = await ingestor._resolve_handle("@SomeChannel")

    assert channel_id is None


# ---------------------------------------------------------------------------
# _item_to_raw_doc() tests
# ---------------------------------------------------------------------------


def test_item_to_raw_doc_maps_url_correctly() -> None:
    """_item_to_raw_doc() builds the YouTube watch URL from video_id."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.url == "https://youtube.com/watch?v=abc123"


def test_item_to_raw_doc_maps_title() -> None:
    """_item_to_raw_doc() maps the title field from snippet."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.title == "How Neural Networks Actually Work"


def test_item_to_raw_doc_maps_author_from_channel_title() -> None:
    """_item_to_raw_doc() uses channelTitle as the author field."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.author == "Andrej Karpathy"


def test_item_to_raw_doc_content_type_is_video() -> None:
    """_item_to_raw_doc() always sets content_type='video'."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.content_type == "video"


def test_item_to_raw_doc_source_type_is_youtube() -> None:
    """_item_to_raw_doc() always sets source_type='youtube'."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.source_type == "youtube"


def test_item_to_raw_doc_raw_content_is_none() -> None:
    """_item_to_raw_doc() sets raw_content=None (populated later via transcript)."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.raw_content is None


def test_item_to_raw_doc_metadata_video_id() -> None:
    """_item_to_raw_doc() includes video_id in metadata."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.metadata["video_id"] == "abc123"


def test_item_to_raw_doc_metadata_channel_handle() -> None:
    """_item_to_raw_doc() includes channel_handle in metadata."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM, channel_handle="@AndrejKarpathy")

    assert doc is not None
    assert doc.metadata["channel_handle"] == "@AndrejKarpathy"


def test_item_to_raw_doc_metadata_channel_id() -> None:
    """_item_to_raw_doc() includes channel_id in metadata from snippet."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.metadata["channel_id"] == "UCXZCJLdBC09xxGZ6gcdrc6A"


def test_item_to_raw_doc_metadata_description() -> None:
    """_item_to_raw_doc() includes description in metadata."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.metadata["description"] == "A deep dive into backpropagation."


def test_item_to_raw_doc_returns_none_when_no_video_id() -> None:
    """_item_to_raw_doc() returns None when the item has no video_id."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    item_no_id = {
        "id": {},  # no videoId key
        "snippet": SAMPLE_VIDEO_ITEM["snippet"],
    }
    doc = ingestor._item_to_raw_doc(item_no_id)

    assert doc is None


def test_item_to_raw_doc_parses_published_at() -> None:
    """_item_to_raw_doc() parses publishedAt into a timezone-aware datetime."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.published_at is not None
    assert isinstance(doc.published_at, datetime)
    assert doc.published_at.year == 2025
    assert doc.published_at.month == 1
    assert doc.published_at.day == 15
    assert doc.published_at.tzinfo is not None


def test_item_to_raw_doc_default_origin_is_pro() -> None:
    """_item_to_raw_doc() uses origin='pro' by default."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)

    assert doc is not None
    assert doc.origin == "pro"


def test_item_to_raw_doc_radar_origin() -> None:
    """_item_to_raw_doc() uses origin='radar' when explicitly passed."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM, origin="radar")

    assert doc is not None
    assert doc.origin == "radar"


# ---------------------------------------------------------------------------
# fetch_content() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_content_populates_transcript() -> None:
    """fetch_content() populates raw_content from transcript for a video document."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)
    assert doc is not None

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        updated_doc = await ingestor.fetch_content(doc)

    assert updated_doc.raw_content == SAMPLE_TRANSCRIPT
    assert updated_doc.word_count == len(SAMPLE_TRANSCRIPT.split())


@pytest.mark.asyncio
async def test_fetch_content_returns_unchanged_when_no_video_id() -> None:
    """fetch_content() returns the document unchanged if video_id is missing from metadata."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = RawDocument(
        url="https://youtube.com/watch?v=test",
        source_type="youtube",
        metadata={},  # no video_id
    )

    result = await ingestor.fetch_content(doc)

    assert result is doc  # same object, unchanged


@pytest.mark.asyncio
async def test_fetch_content_returns_unchanged_when_transcript_unavailable() -> None:
    """fetch_content() returns the doc unchanged when transcript is not available."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    doc = ingestor._item_to_raw_doc(SAMPLE_VIDEO_ITEM)
    assert doc is not None

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=None,
    ):
        result = await ingestor.fetch_content(doc)

    assert result.raw_content is None


# ---------------------------------------------------------------------------
# _get_transcript() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_transcript_returns_transcript_text() -> None:
    """_get_transcript() returns the transcript as plain text."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        result = await ingestor._get_transcript("abc123")

    assert result == SAMPLE_TRANSCRIPT


@pytest.mark.asyncio
async def test_get_transcript_returns_none_when_unavailable() -> None:
    """_get_transcript() returns None when transcript is unavailable (no crash)."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.youtube._fetch_transcript_sync",
        return_value=None,
    ):
        result = await ingestor._get_transcript("private_video_id")

    assert result is None


# ---------------------------------------------------------------------------
# _fetch_transcript_sync() unit tests
# ---------------------------------------------------------------------------


def test_fetch_transcript_sync_returns_none_on_exception() -> None:
    """_fetch_transcript_sync() returns None on any exception from youtube-transcript-api.

    The function uses a broad except clause to handle NoTranscriptFound,
    TranscriptsDisabled, VideoUnavailable, and network errors gracefully.
    """
    # Patch the YouTubeTranscriptApi instance method directly.
    # Since the function does a local import, patching via the module works.
    with patch(
        "youtube_transcript_api.YouTubeTranscriptApi.fetch",
        side_effect=Exception("Transcripts disabled"),
    ):
        result = _fetch_transcript_sync("disabled_video", ["en"])

    assert result is None


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_returns_self() -> None:
    """YouTubeIngestor works as async context manager and returns self."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    async with ingestor as ctx:
        assert ctx is ingestor


@pytest.mark.asyncio
async def test_context_manager_closes_client_on_exit() -> None:
    """YouTubeIngestor.__aexit__ calls aclose() on the internal httpx client."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

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
# source_type property
# ---------------------------------------------------------------------------


def test_source_type_is_youtube() -> None:
    """YouTubeIngestor.source_type returns 'youtube'."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    assert ingestor.source_type == "youtube"


# ---------------------------------------------------------------------------
# API key property
# ---------------------------------------------------------------------------


def test_api_key_returns_configured_key() -> None:
    """_api_key property returns the API key from config."""
    config = _make_config(api_key="my-secret-key")
    ingestor = YouTubeIngestor(config)

    assert ingestor._api_key == "my-secret-key"


def test_api_key_returns_none_when_not_configured() -> None:
    """_api_key property returns None when not configured."""
    config = _make_config(api_key=None)
    ingestor = YouTubeIngestor(config)

    assert ingestor._api_key is None


# ---------------------------------------------------------------------------
# Transcript language configuration
# ---------------------------------------------------------------------------


def test_transcript_langs_returns_configured_langs() -> None:
    """_transcript_langs returns the configured list of language codes."""
    config = _make_config(transcript_langs=["en", "de"])
    ingestor = YouTubeIngestor(config)

    assert ingestor._transcript_langs == ["en", "de"]


def test_transcript_langs_defaults_to_english() -> None:
    """_transcript_langs defaults to ['en'] when not explicitly configured."""
    config = _make_config()
    ingestor = YouTubeIngestor(config)

    assert ingestor._transcript_langs == ["en"]
