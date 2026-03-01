"""Content extraction using readability-lxml + html2text."""
import logging
from typing import Self

import html2text
import httpx
from pydantic import BaseModel
from readability import Document

logger = logging.getLogger(__name__)

# Content types to skip extraction for (non-text)
_SKIP_CONTENT_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}


class ExtractedContent(BaseModel):
    """Result of content extraction from a URL or HTML."""

    url: str
    title: str | None
    text: str  # clean plain text / markdown
    word_count: int
    author: str | None = None
    html: str | None = None  # raw HTML (kept for debugging)


class ContentExtractor:
    """Fetch a URL and extract clean text using readability-lxml + html2text.

    Uses readability-lxml to identify the main content area of an HTML page,
    then converts it to plain text (markdown-ish) using html2text.

    The httpx client is reused across calls for connection pooling.

    Can be used as an async context manager to ensure client cleanup:

        async with ContentExtractor() as extractor:
            content = await extractor.fetch_and_extract(url)

    Or without context manager (caller must ensure no resource leaks for
    long-lived instances):

        extractor = ContentExtractor()
        content = await extractor.fetch_and_extract(url)
    """

    def __init__(self, timeout: float = 30.0) -> None:
        """Initialize the extractor with an httpx async client.

        Args:
            timeout: Request timeout in seconds (default 30).
        """
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": "ai-craftsman-kb/1.0 (+https://github.com/ai-craftsman-kb)"
            },
            follow_redirects=True,
        )
        self._h2t = html2text.HTML2Text()
        self._h2t.ignore_links = False
        self._h2t.ignore_images = True
        self._h2t.body_width = 0  # don't wrap lines

    async def fetch_and_extract(self, url: str) -> ExtractedContent:
        """Fetch URL, extract readable HTML, and convert to plain text.

        Handles HTTP errors and non-HTML content types gracefully by
        returning an ExtractedContent with text='' and word_count=0.

        Args:
            url: The URL to fetch and extract content from.

        Returns:
            ExtractedContent with text='' on fetch/parse failure.
        """
        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return ExtractedContent(url=url, title=None, text="", word_count=0)

        # Check content type — skip non-HTML
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type in _SKIP_CONTENT_TYPES:
            logger.debug(
                "Skipping extraction for content-type %s at %s", content_type, url
            )
            return ExtractedContent(url=url, title=None, text="", word_count=0)

        html = response.text
        return self.extract_from_html(url=url, html=html)

    def extract_from_html(self, url: str, html: str) -> ExtractedContent:
        """Extract clean text from already-fetched HTML.

        Uses readability-lxml for main content extraction, then html2text
        for plain text conversion. Useful when the caller already has the
        raw HTML (e.g., some ingestors receive HTML in their API response).

        Args:
            url: The source URL (used for metadata only, no network call).
            html: The raw HTML string to extract content from.

        Returns:
            ExtractedContent with extracted title and clean text.
            Returns text='' and word_count=0 on parse failure.
        """
        try:
            doc = Document(html)
            title = doc.title()
            readable_html = doc.summary()
            text = self._h2t.handle(readable_html)
            text = text.strip()
            word_count = len(text.split()) if text else 0
            return ExtractedContent(
                url=url,
                title=title or None,
                text=text,
                word_count=word_count,
                html=html,
            )
        except Exception as e:
            logger.warning("Extraction failed for %s: %s", url, e)
            return ExtractedContent(url=url, title=None, text="", word_count=0)

    async def __aenter__(self) -> "ContentExtractor":
        """Enter async context manager, returning self."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager, closing the httpx client."""
        await self._client.aclose()
