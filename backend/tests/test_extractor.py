"""Unit tests for ContentExtractor and related models.

Tests cover:
- HTML extraction with known content
- Word count accuracy
- Title extraction
- Empty HTML edge case
- RawDocument.to_document_row() field mapping
- BaseIngestor cannot be instantiated directly
- ContentExtractor as async context manager (mocked httpx)
- fetch_and_extract() with HTTP error (mocked httpx)
"""
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from ai_craftsman_kb.processing.extractor import ContentExtractor, ExtractedContent
from ai_craftsman_kb.ingestors.base import BaseIngestor, RawDocument


# ---------------------------------------------------------------------------
# Sample HTML fixtures
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<html>
<head><title>Test Article</title></head>
<body>
<article>
<h1>Hello World</h1>
<p>This is a test article with some content for testing extraction.</p>
<p>It has multiple paragraphs to test word counting and text extraction.</p>
</article>
</body>
</html>
"""

EMPTY_HTML = ""

MINIMAL_HTML = "<html><head><title>Empty Page</title></head><body></body></html>"


# ---------------------------------------------------------------------------
# ContentExtractor.extract_from_html() tests
# ---------------------------------------------------------------------------


def test_extract_from_html_basic() -> None:
    """Parse HTML with known content and verify text is extracted."""
    extractor = ContentExtractor()
    result = extractor.extract_from_html(url="https://example.com", html=SAMPLE_HTML)

    assert isinstance(result, ExtractedContent)
    assert result.url == "https://example.com"
    # Should contain meaningful text from the article
    assert "Hello World" in result.text or "test article" in result.text
    assert result.text != ""


def test_extract_from_html_word_count() -> None:
    """Verify word_count is accurate for extracted text."""
    extractor = ContentExtractor()
    result = extractor.extract_from_html(url="https://example.com", html=SAMPLE_HTML)

    assert result.word_count > 0
    # word_count must equal number of whitespace-separated tokens in text
    assert result.word_count == len(result.text.split())


def test_extract_from_html_title() -> None:
    """Verify title is extracted from <title> tag."""
    extractor = ContentExtractor()
    result = extractor.extract_from_html(url="https://example.com", html=SAMPLE_HTML)

    assert result.title is not None
    assert "Test Article" in result.title


def test_extract_from_html_empty() -> None:
    """Empty HTML string returns empty text and word_count=0."""
    extractor = ContentExtractor()
    result = extractor.extract_from_html(url="https://example.com", html=EMPTY_HTML)

    assert isinstance(result, ExtractedContent)
    assert result.text == "" or result.word_count == 0
    # word_count must be consistent with text
    assert result.word_count == len(result.text.split()) if result.text else result.word_count == 0


def test_extract_from_html_stores_url() -> None:
    """The url field should match what was passed in."""
    extractor = ContentExtractor()
    url = "https://example.com/article"
    result = extractor.extract_from_html(url=url, html=SAMPLE_HTML)

    assert result.url == url


def test_extract_from_html_stores_raw_html() -> None:
    """The html field stores the original HTML for debugging."""
    extractor = ContentExtractor()
    result = extractor.extract_from_html(url="https://example.com", html=SAMPLE_HTML)

    assert result.html == SAMPLE_HTML


# ---------------------------------------------------------------------------
# RawDocument.to_document_row() tests
# ---------------------------------------------------------------------------


def test_raw_document_to_document_row_generates_uuid() -> None:
    """to_document_row() generates a unique UUID for each call."""
    doc = RawDocument(
        url="https://example.com",
        title="Test",
        source_type="hn",
        origin="pro",
    )
    row1 = doc.to_document_row()
    row2 = doc.to_document_row()

    # Each call should produce a different UUID
    assert row1.id != row2.id
    # Should be valid UUID format (36 chars with dashes)
    assert len(row1.id) == 36
    assert row1.id.count("-") == 4


def test_raw_document_to_document_row_maps_fields() -> None:
    """to_document_row() correctly maps all fields to DocumentRow."""
    published = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    doc = RawDocument(
        url="https://news.ycombinator.com/item?id=12345",
        title="Interesting AI Article",
        author="johndoe",
        raw_content="Some article content here.",
        content_type="article",
        published_at=published,
        source_type="hn",
        origin="pro",
        word_count=5,
        filter_score=8.5,
        metadata={"points": 123, "comments": 45},
    )
    source_id = "source-uuid-abc"
    row = doc.to_document_row(source_id=source_id)

    assert row.url == doc.url
    assert row.title == doc.title
    assert row.author == doc.author
    assert row.raw_content == doc.raw_content
    assert row.content_type == doc.content_type
    assert row.published_at == published.isoformat()
    assert row.source_type == doc.source_type
    assert row.origin == doc.origin
    assert row.word_count == doc.word_count
    assert row.filter_score == doc.filter_score
    assert row.metadata == doc.metadata
    assert row.source_id == source_id


def test_raw_document_to_document_row_without_source_id() -> None:
    """to_document_row() works with source_id=None (adhoc docs)."""
    doc = RawDocument(
        url="https://example.com",
        source_type="rss",
        origin="adhoc",
    )
    row = doc.to_document_row()

    assert row.source_id is None
    assert row.origin == "adhoc"


def test_raw_document_to_document_row_none_published_at() -> None:
    """None published_at maps to None in the row (not a crash)."""
    doc = RawDocument(
        url="https://example.com",
        source_type="devto",
        origin="pro",
        published_at=None,
    )
    row = doc.to_document_row()

    assert row.published_at is None


# ---------------------------------------------------------------------------
# BaseIngestor abstract class tests
# ---------------------------------------------------------------------------


def test_base_ingestor_is_abstract() -> None:
    """BaseIngestor cannot be instantiated directly (it's an ABC)."""
    with pytest.raises(TypeError):
        BaseIngestor(config=None)  # type: ignore[arg-type]


def test_base_ingestor_concrete_subclass_requires_abstractmethods() -> None:
    """A partial subclass that skips abstract methods also cannot be instantiated."""

    class IncompleteIngestor(BaseIngestor):
        # Missing: source_type, fetch_pro, search_radar
        pass

    with pytest.raises(TypeError):
        IncompleteIngestor(config=None)  # type: ignore[arg-type]


def test_base_ingestor_concrete_subclass_can_be_instantiated() -> None:
    """A complete concrete subclass can be instantiated."""

    class ConcreteIngestor(BaseIngestor):
        @property
        def source_type(self) -> str:
            return "test"

        async def fetch_pro(self) -> list[RawDocument]:
            return []

        async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
            return []

    ingestor = ConcreteIngestor(config=None)  # type: ignore[arg-type]
    assert ingestor.source_type == "test"


# ---------------------------------------------------------------------------
# ContentExtractor async context manager tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_extractor_as_context_manager() -> None:
    """ContentExtractor works as an async context manager and closes the client."""
    async with ContentExtractor() as extractor:
        assert isinstance(extractor, ContentExtractor)
        # Should be able to extract from HTML inside context
        result = extractor.extract_from_html(
            url="https://example.com", html=SAMPLE_HTML
        )
        assert isinstance(result, ExtractedContent)
    # After exit, client should be closed (no assertions needed beyond no exceptions)


@pytest.mark.asyncio
async def test_content_extractor_context_manager_closes_client() -> None:
    """__aexit__ calls aclose() on the internal httpx client."""
    extractor = ContentExtractor()
    closed = []

    original_aclose = extractor._client.aclose

    async def mock_aclose() -> None:
        closed.append(True)
        await original_aclose()

    extractor._client.aclose = mock_aclose  # type: ignore[method-assign]

    async with extractor:
        pass

    assert closed, "aclose() was not called on __aexit__"


# ---------------------------------------------------------------------------
# fetch_and_extract() with mocked httpx tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_and_extract_http_error() -> None:
    """HTTP error during fetch returns empty ExtractedContent gracefully."""
    extractor = ContentExtractor()

    # Mock the httpx client's get() to raise an HTTPError
    async def mock_get(url: str, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection refused")

    extractor._client.get = mock_get  # type: ignore[method-assign]

    result = await extractor.fetch_and_extract("https://example.com/article")

    assert result.url == "https://example.com/article"
    assert result.text == ""
    assert result.word_count == 0
    assert result.title is None


@pytest.mark.asyncio
async def test_fetch_and_extract_non_html_content_type() -> None:
    """Non-HTML content types (PDF, images) return empty ExtractedContent."""
    extractor = ContentExtractor()

    # Create a mock response with PDF content type
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/pdf"}
    mock_response.raise_for_status = MagicMock()
    mock_response.text = "%PDF-1.4..."

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return mock_response

    extractor._client.get = mock_get  # type: ignore[method-assign]

    result = await extractor.fetch_and_extract("https://example.com/paper.pdf")

    assert result.text == ""
    assert result.word_count == 0


@pytest.mark.asyncio
async def test_fetch_and_extract_success() -> None:
    """Successful fetch returns extracted content."""
    extractor = ContentExtractor()

    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/html; charset=utf-8"}
    mock_response.raise_for_status = MagicMock()
    mock_response.text = SAMPLE_HTML

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return mock_response

    extractor._client.get = mock_get  # type: ignore[method-assign]

    result = await extractor.fetch_and_extract("https://example.com/article")

    assert result.url == "https://example.com/article"
    assert result.text != ""
    assert result.word_count > 0


@pytest.mark.asyncio
async def test_fetch_and_extract_http_status_error() -> None:
    """HTTP 4xx/5xx status raises and is caught, returning empty content."""
    extractor = ContentExtractor()

    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/html"}
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(),
        )
    )

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        return mock_response

    extractor._client.get = mock_get  # type: ignore[method-assign]

    result = await extractor.fetch_and_extract("https://example.com/missing")

    assert result.text == ""
    assert result.word_count == 0
