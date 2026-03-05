"""Unit tests for the Adhoc URL ingestor.

Tests cover:
- _detect_url_type() correctly classifies YouTube, ArXiv, and article URLs
- _extract_youtube_video_id() parses all supported YouTube URL formats
- _extract_arxiv_id() parses /abs/ and /pdf/ URL paths and strips versions
- ingest_url() routes to correct handler based on detected URL type
- _handle_youtube(): transcript fetched, metadata includes video_id
- _handle_youtube(): raw_content=None when transcript unavailable
- _handle_youtube(): returns doc with video_id=None for unrecognized YT URL
- _handle_arxiv(): abstract fetched, metadata includes arxiv_id
- _handle_arxiv(): raw_content=None when ArXiv API fails
- _handle_arxiv(): raw_content=None when paper not found in feed
- _handle_article(): ContentExtractor used, article text returned
- origin='adhoc' on all returned documents
- source_type='adhoc' on all returned documents
- metadata includes adhoc_tags (from tags arg) and url_type
- fetch_pro() raises NotImplementedError
- search_radar() raises NotImplementedError
- Context manager: __aexit__ calls aclose() on the httpx client
"""
import textwrap
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
)
from ai_craftsman_kb.ingestors.adhoc import (
    AdhocIngestor,
    _extract_arxiv_id,
    _extract_youtube_video_id,
)
from ai_craftsman_kb.ingestors.base import RawDocument


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


def _make_config() -> AppConfig:
    """Build a minimal AppConfig for testing."""
    return AppConfig(
        sources=SourcesConfig(),
        settings=SettingsConfig(llm=_make_llm_routing()),
        filters=FiltersConfig(),
    )


def _make_mock_http_response(body: str, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response returning `body` as text.

    Args:
        body: The response body text.
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


SAMPLE_TRANSCRIPT = "Hello and welcome. Today we discuss transformers and attention mechanisms."

SAMPLE_ARXIV_ATOM = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <title>ArXiv Query Results</title>
      <entry>
        <id>http://arxiv.org/abs/2501.12345v1</id>
        <title>Attention Is All You Need: A Deep Dive</title>
        <author><name>Alice Smith</name></author>
        <author><name>Bob Jones</name></author>
        <summary>We revisit the transformer architecture and propose key improvements.</summary>
        <published>2025-01-15T00:00:00Z</published>
        <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
      </entry>
    </feed>
""")

EMPTY_ARXIV_ATOM = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>ArXiv Query Results</title>
    </feed>
""")


# ---------------------------------------------------------------------------
# _extract_youtube_video_id() tests
# ---------------------------------------------------------------------------


def test_extract_youtube_video_id_watch_url() -> None:
    """_extract_youtube_video_id() extracts video ID from /watch?v= URL."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert _extract_youtube_video_id(url) == "dQw4w9WgXcQ"


def test_extract_youtube_video_id_youtu_be() -> None:
    """_extract_youtube_video_id() extracts video ID from youtu.be short link."""
    url = "https://youtu.be/dQw4w9WgXcQ"
    assert _extract_youtube_video_id(url) == "dQw4w9WgXcQ"


def test_extract_youtube_video_id_shorts() -> None:
    """_extract_youtube_video_id() extracts video ID from /shorts/ URL."""
    url = "https://www.youtube.com/shorts/abc123xyz"
    assert _extract_youtube_video_id(url) == "abc123xyz"


def test_extract_youtube_video_id_embed() -> None:
    """_extract_youtube_video_id() extracts video ID from /embed/ URL."""
    url = "https://www.youtube.com/embed/abc123xyz"
    assert _extract_youtube_video_id(url) == "abc123xyz"


def test_extract_youtube_video_id_returns_none_for_non_video_url() -> None:
    """_extract_youtube_video_id() returns None for a non-video YouTube URL."""
    url = "https://www.youtube.com/channel/UCXZCJLdBC09xxGZ6gcdrc6A"
    assert _extract_youtube_video_id(url) is None


def test_extract_youtube_video_id_returns_none_for_empty_youtu_be() -> None:
    """_extract_youtube_video_id() returns None for bare youtu.be domain with no path."""
    url = "https://youtu.be/"
    assert _extract_youtube_video_id(url) is None


# ---------------------------------------------------------------------------
# _extract_arxiv_id() tests
# ---------------------------------------------------------------------------


def test_extract_arxiv_id_abs_url() -> None:
    """_extract_arxiv_id() extracts paper ID from /abs/ URL."""
    url = "https://arxiv.org/abs/2501.12345"
    assert _extract_arxiv_id(url) == "2501.12345"


def test_extract_arxiv_id_pdf_url() -> None:
    """_extract_arxiv_id() extracts paper ID from /pdf/ URL."""
    url = "https://arxiv.org/pdf/2501.12345"
    assert _extract_arxiv_id(url) == "2501.12345"


def test_extract_arxiv_id_strips_version_suffix() -> None:
    """_extract_arxiv_id() strips version suffix from paper ID."""
    url = "https://arxiv.org/abs/2501.12345v2"
    assert _extract_arxiv_id(url) == "2501.12345"


def test_extract_arxiv_id_returns_none_for_invalid_url() -> None:
    """_extract_arxiv_id() returns None for non-paper arxiv URLs."""
    url = "https://arxiv.org/search/"
    assert _extract_arxiv_id(url) is None


# ---------------------------------------------------------------------------
# _detect_url_type() tests
# ---------------------------------------------------------------------------


def test_detect_url_type_youtube_watch() -> None:
    """_detect_url_type() returns 'youtube' for youtube.com watch URL."""
    config = _make_config()
    ingestor = AdhocIngestor(config)
    assert ingestor._detect_url_type("https://www.youtube.com/watch?v=abc") == "youtube"


def test_detect_url_type_youtu_be() -> None:
    """_detect_url_type() returns 'youtube' for youtu.be URL."""
    config = _make_config()
    ingestor = AdhocIngestor(config)
    assert ingestor._detect_url_type("https://youtu.be/abc123") == "youtube"


def test_detect_url_type_arxiv() -> None:
    """_detect_url_type() returns 'arxiv' for arxiv.org URL."""
    config = _make_config()
    ingestor = AdhocIngestor(config)
    assert ingestor._detect_url_type("https://arxiv.org/abs/2501.12345") == "arxiv"


def test_detect_url_type_article() -> None:
    """_detect_url_type() returns 'article' for any other URL."""
    config = _make_config()
    ingestor = AdhocIngestor(config)
    assert ingestor._detect_url_type("https://blog.example.com/post/123") == "article"


def test_detect_url_type_article_for_generic_https() -> None:
    """_detect_url_type() returns 'article' for generic HTTPS URL."""
    config = _make_config()
    ingestor = AdhocIngestor(config)
    assert ingestor._detect_url_type("https://substack.com/some-newsletter") == "article"


# ---------------------------------------------------------------------------
# _handle_youtube() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_youtube_returns_raw_document() -> None:
    """_handle_youtube() returns a RawDocument."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.adhoc._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        doc = await ingestor._handle_youtube("https://www.youtube.com/watch?v=abc123")

    assert isinstance(doc, RawDocument)


@pytest.mark.asyncio
async def test_handle_youtube_fetches_transcript() -> None:
    """_handle_youtube() populates raw_content with the transcript."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.adhoc._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        doc = await ingestor._handle_youtube("https://www.youtube.com/watch?v=abc123")

    assert doc.raw_content == SAMPLE_TRANSCRIPT
    assert doc.word_count == len(SAMPLE_TRANSCRIPT.split())


@pytest.mark.asyncio
async def test_handle_youtube_raw_content_none_when_no_transcript() -> None:
    """_handle_youtube() sets raw_content=None when transcript is unavailable."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.adhoc._fetch_transcript_sync",
        return_value=None,
    ):
        doc = await ingestor._handle_youtube("https://www.youtube.com/watch?v=abc123")

    assert doc.raw_content is None


@pytest.mark.asyncio
async def test_handle_youtube_metadata_includes_video_id() -> None:
    """_handle_youtube() includes video_id in metadata."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.adhoc._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        doc = await ingestor._handle_youtube("https://www.youtube.com/watch?v=abc123")

    assert doc.metadata.get("video_id") == "abc123"


@pytest.mark.asyncio
async def test_handle_youtube_content_type_is_video() -> None:
    """_handle_youtube() sets content_type='video'."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.adhoc._fetch_transcript_sync",
        return_value=None,
    ):
        doc = await ingestor._handle_youtube("https://www.youtube.com/watch?v=abc123")

    assert doc.content_type == "video"


@pytest.mark.asyncio
async def test_handle_youtube_unrecognized_url_returns_doc_with_none_video_id() -> None:
    """_handle_youtube() handles unrecognized YouTube URLs gracefully."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    # A YouTube URL with no video ID
    doc = await ingestor._handle_youtube("https://www.youtube.com/channel/UC123")

    assert doc.raw_content is None
    assert doc.metadata.get("video_id") is None
    assert doc.source_type == "adhoc"


# ---------------------------------------------------------------------------
# _handle_arxiv() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_arxiv_returns_raw_document() -> None:
    """_handle_arxiv() returns a RawDocument."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor._handle_arxiv("https://arxiv.org/abs/2501.12345")

    assert isinstance(doc, RawDocument)


@pytest.mark.asyncio
async def test_handle_arxiv_fetches_abstract() -> None:
    """_handle_arxiv() populates raw_content with the abstract."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor._handle_arxiv("https://arxiv.org/abs/2501.12345")

    assert doc.raw_content is not None
    assert "transformer architecture" in doc.raw_content


@pytest.mark.asyncio
async def test_handle_arxiv_metadata_includes_arxiv_id() -> None:
    """_handle_arxiv() includes arxiv_id in metadata."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor._handle_arxiv("https://arxiv.org/abs/2501.12345")

    assert doc.metadata.get("arxiv_id") == "2501.12345"


@pytest.mark.asyncio
async def test_handle_arxiv_metadata_includes_pdf_url() -> None:
    """_handle_arxiv() includes pdf_url in metadata."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor._handle_arxiv("https://arxiv.org/abs/2501.12345")

    assert "pdf_url" in doc.metadata
    assert doc.metadata["pdf_url"] == "https://arxiv.org/pdf/2501.12345"


@pytest.mark.asyncio
async def test_handle_arxiv_content_type_is_paper() -> None:
    """_handle_arxiv() sets content_type='paper'."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor._handle_arxiv("https://arxiv.org/abs/2501.12345")

    assert doc.content_type == "paper"


@pytest.mark.asyncio
async def test_handle_arxiv_raw_content_none_on_http_error() -> None:
    """_handle_arxiv() sets raw_content=None when ArXiv API request fails."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    ingestor._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=httpx.ConnectError("Connection refused")
    )

    doc = await ingestor._handle_arxiv("https://arxiv.org/abs/2501.12345")

    assert doc.raw_content is None
    assert doc.metadata.get("arxiv_id") == "2501.12345"


@pytest.mark.asyncio
async def test_handle_arxiv_raw_content_none_when_entry_not_found() -> None:
    """_handle_arxiv() sets raw_content=None when paper not found in feed."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(EMPTY_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor._handle_arxiv("https://arxiv.org/abs/9999.00000")

    assert doc.raw_content is None
    assert doc.source_type == "adhoc"


@pytest.mark.asyncio
async def test_handle_arxiv_resolves_pdf_url_to_canonical() -> None:
    """_handle_arxiv() resolves a /pdf/ URL to the canonical /abs/ URL."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    # Pass a PDF URL — the returned doc.url should be the abs canonical URL
    doc = await ingestor._handle_arxiv("https://arxiv.org/pdf/2501.12345")

    assert doc.url == "https://arxiv.org/abs/2501.12345"


@pytest.mark.asyncio
async def test_handle_arxiv_unrecognized_url_returns_doc_with_none_arxiv_id() -> None:
    """_handle_arxiv() handles unrecognized ArXiv URL gracefully."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    doc = await ingestor._handle_arxiv("https://arxiv.org/search/?query=llm")

    assert doc.raw_content is None
    assert doc.metadata.get("arxiv_id") is None
    assert doc.source_type == "adhoc"


@pytest.mark.asyncio
async def test_handle_arxiv_title_and_author_populated() -> None:
    """_handle_arxiv() extracts title and author from the Atom feed."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor._handle_arxiv("https://arxiv.org/abs/2501.12345")

    assert doc.title == "Attention Is All You Need: A Deep Dive"
    assert doc.author == "Alice Smith, Bob Jones"


# ---------------------------------------------------------------------------
# _handle_article() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_article_uses_content_extractor() -> None:
    """_handle_article() uses ContentExtractor to fetch and extract article text."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_extracted = MagicMock()
    mock_extracted.text = "This is the extracted article content."
    mock_extracted.word_count = 7
    mock_extracted.title = "Test Article"
    mock_extracted.author = None

    mock_extractor = AsyncMock()
    mock_extractor.fetch_and_extract = AsyncMock(return_value=mock_extracted)
    mock_extractor.__aenter__ = AsyncMock(return_value=mock_extractor)
    mock_extractor.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "ai_craftsman_kb.processing.extractor.ContentExtractor",
        return_value=mock_extractor,
    ):
        doc = await ingestor._handle_article("https://blog.example.com/post/1")

    assert doc.raw_content == "This is the extracted article content."
    assert doc.word_count == 7
    assert doc.title == "Test Article"


@pytest.mark.asyncio
async def test_handle_article_content_type_is_article() -> None:
    """_handle_article() sets content_type='article'."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_extracted = MagicMock()
    mock_extracted.text = "Some article text."
    mock_extracted.word_count = 3
    mock_extracted.title = None
    mock_extracted.author = None

    mock_extractor = AsyncMock()
    mock_extractor.fetch_and_extract = AsyncMock(return_value=mock_extracted)
    mock_extractor.__aenter__ = AsyncMock(return_value=mock_extractor)
    mock_extractor.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "ai_craftsman_kb.processing.extractor.ContentExtractor",
        return_value=mock_extractor,
    ):
        doc = await ingestor._handle_article("https://example.com/article")

    assert doc.content_type == "article"


@pytest.mark.asyncio
async def test_handle_article_raw_content_none_on_empty_extraction() -> None:
    """_handle_article() sets raw_content=None when extracted text is empty."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_extracted = MagicMock()
    mock_extracted.text = ""
    mock_extracted.word_count = 0
    mock_extracted.title = None
    mock_extracted.author = None

    mock_extractor = AsyncMock()
    mock_extractor.fetch_and_extract = AsyncMock(return_value=mock_extracted)
    mock_extractor.__aenter__ = AsyncMock(return_value=mock_extractor)
    mock_extractor.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "ai_craftsman_kb.processing.extractor.ContentExtractor",
        return_value=mock_extractor,
    ):
        doc = await ingestor._handle_article("https://example.com/404")

    assert doc.raw_content is None


# ---------------------------------------------------------------------------
# ingest_url() routing tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_url_routes_youtube_url() -> None:
    """ingest_url() calls _handle_youtube() for YouTube URLs."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with patch.object(
        ingestor,
        "_handle_youtube",
        new_callable=AsyncMock,
        return_value=RawDocument(
            url="https://www.youtube.com/watch?v=abc123",
            source_type="adhoc",
            origin="adhoc",
            content_type="video",
            metadata={"video_id": "abc123"},
        ),
    ) as mock_yt:
        doc = await ingestor.ingest_url("https://www.youtube.com/watch?v=abc123")

    mock_yt.assert_awaited_once_with("https://www.youtube.com/watch?v=abc123")
    assert doc.origin == "adhoc"


@pytest.mark.asyncio
async def test_ingest_url_routes_arxiv_url() -> None:
    """ingest_url() calls _handle_arxiv() for ArXiv URLs."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with patch.object(
        ingestor,
        "_handle_arxiv",
        new_callable=AsyncMock,
        return_value=RawDocument(
            url="https://arxiv.org/abs/2501.12345",
            source_type="adhoc",
            origin="adhoc",
            content_type="paper",
            metadata={"arxiv_id": "2501.12345"},
        ),
    ) as mock_arxiv:
        doc = await ingestor.ingest_url("https://arxiv.org/abs/2501.12345")

    mock_arxiv.assert_awaited_once_with("https://arxiv.org/abs/2501.12345")
    assert doc.origin == "adhoc"


@pytest.mark.asyncio
async def test_ingest_url_routes_article_url() -> None:
    """ingest_url() calls _handle_article() for generic URLs."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with patch.object(
        ingestor,
        "_handle_article",
        new_callable=AsyncMock,
        return_value=RawDocument(
            url="https://blog.example.com/post",
            source_type="adhoc",
            origin="adhoc",
            content_type="article",
            metadata={},
        ),
    ) as mock_article:
        doc = await ingestor.ingest_url("https://blog.example.com/post")

    mock_article.assert_awaited_once_with("https://blog.example.com/post")
    assert doc.origin == "adhoc"


# ---------------------------------------------------------------------------
# origin and source_type guarantee tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_url_origin_is_always_adhoc_for_youtube() -> None:
    """ingest_url() always sets origin='adhoc' for YouTube docs."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.adhoc._fetch_transcript_sync",
        return_value=SAMPLE_TRANSCRIPT,
    ):
        doc = await ingestor.ingest_url("https://www.youtube.com/watch?v=abc123")

    assert doc.origin == "adhoc"
    assert doc.source_type == "adhoc"


@pytest.mark.asyncio
async def test_ingest_url_origin_is_always_adhoc_for_arxiv() -> None:
    """ingest_url() always sets origin='adhoc' for ArXiv docs."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor.ingest_url("https://arxiv.org/abs/2501.12345")

    assert doc.origin == "adhoc"
    assert doc.source_type == "adhoc"


@pytest.mark.asyncio
async def test_ingest_url_origin_is_always_adhoc_for_article() -> None:
    """ingest_url() always sets origin='adhoc' for article docs."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_extracted = MagicMock()
    mock_extracted.text = "Article content."
    mock_extracted.word_count = 2
    mock_extracted.title = "My Article"
    mock_extracted.author = None

    mock_extractor = AsyncMock()
    mock_extractor.fetch_and_extract = AsyncMock(return_value=mock_extracted)
    mock_extractor.__aenter__ = AsyncMock(return_value=mock_extractor)
    mock_extractor.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "ai_craftsman_kb.processing.extractor.ContentExtractor",
        return_value=mock_extractor,
    ):
        doc = await ingestor.ingest_url("https://example.com/article")

    assert doc.origin == "adhoc"
    assert doc.source_type == "adhoc"


# ---------------------------------------------------------------------------
# metadata (adhoc_tags, url_type) tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_url_stores_adhoc_tags_in_metadata() -> None:
    """ingest_url() stores user-supplied tags in metadata['adhoc_tags']."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor.ingest_url(
        "https://arxiv.org/abs/2501.12345",
        tags=["ml", "transformers"],
    )

    assert doc.metadata.get("adhoc_tags") == ["ml", "transformers"]


@pytest.mark.asyncio
async def test_ingest_url_adhoc_tags_empty_list_when_none() -> None:
    """ingest_url() stores [] as adhoc_tags when no tags are passed."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor.ingest_url("https://arxiv.org/abs/2501.12345")

    assert doc.metadata.get("adhoc_tags") == []


@pytest.mark.asyncio
async def test_ingest_url_stores_url_type_in_metadata_youtube() -> None:
    """ingest_url() stores url_type='youtube' in metadata for YouTube URLs."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with patch(
        "ai_craftsman_kb.ingestors.adhoc._fetch_transcript_sync",
        return_value=None,
    ):
        doc = await ingestor.ingest_url("https://www.youtube.com/watch?v=abc123")

    assert doc.metadata.get("url_type") == "youtube"


@pytest.mark.asyncio
async def test_ingest_url_stores_url_type_in_metadata_arxiv() -> None:
    """ingest_url() stores url_type='arxiv' in metadata for ArXiv URLs."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_resp = _make_mock_http_response(SAMPLE_ARXIV_ATOM)
    ingestor._client.get = AsyncMock(return_value=mock_resp)  # type: ignore[method-assign]

    doc = await ingestor.ingest_url("https://arxiv.org/abs/2501.12345")

    assert doc.metadata.get("url_type") == "arxiv"


@pytest.mark.asyncio
async def test_ingest_url_stores_url_type_in_metadata_article() -> None:
    """ingest_url() stores url_type='article' in metadata for generic URLs."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    mock_extracted = MagicMock()
    mock_extracted.text = "Content."
    mock_extracted.word_count = 1
    mock_extracted.title = None
    mock_extracted.author = None

    mock_extractor = AsyncMock()
    mock_extractor.fetch_and_extract = AsyncMock(return_value=mock_extracted)
    mock_extractor.__aenter__ = AsyncMock(return_value=mock_extractor)
    mock_extractor.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "ai_craftsman_kb.processing.extractor.ContentExtractor",
        return_value=mock_extractor,
    ):
        doc = await ingestor.ingest_url("https://example.com/page")

    assert doc.metadata.get("url_type") == "article"


# ---------------------------------------------------------------------------
# fetch_pro() / search_radar() raise NotImplementedError tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_pro_raises_not_implemented() -> None:
    """fetch_pro() raises NotImplementedError for AdhocIngestor."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with pytest.raises(NotImplementedError):
        await ingestor.fetch_pro()


@pytest.mark.asyncio
async def test_search_radar_raises_not_implemented() -> None:
    """search_radar() raises NotImplementedError for AdhocIngestor."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    with pytest.raises(NotImplementedError):
        await ingestor.search_radar("some query")


# ---------------------------------------------------------------------------
# source_type tests
# ---------------------------------------------------------------------------


def test_source_type_is_adhoc() -> None:
    """AdhocIngestor.source_type is 'adhoc'."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    assert ingestor.source_type == "adhoc"


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_returns_self() -> None:
    """AdhocIngestor works as async context manager and returns self."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    async with ingestor as ctx:
        assert ctx is ingestor


@pytest.mark.asyncio
async def test_context_manager_closes_client_on_exit() -> None:
    """AdhocIngestor.__aexit__ calls aclose() on the internal httpx client."""
    config = _make_config()
    ingestor = AdhocIngestor(config)

    close_called = []
    original_aclose = ingestor._client.aclose

    async def mock_aclose() -> None:
        close_called.append(True)
        await original_aclose()

    ingestor._client.aclose = mock_aclose  # type: ignore[method-assign]

    async with ingestor:
        pass

    assert close_called, "aclose() was not called on __aexit__"
