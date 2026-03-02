"""Adhoc URL ingestor — detects URL type and delegates to the appropriate handler.

Supports three URL types:
- YouTube videos (youtube.com, youtu.be): extracts video ID, fetches transcript
- ArXiv papers (arxiv.org): extracts paper ID, fetches abstract via Atom API
- All other URLs: uses ContentExtractor for HTML extraction

This ingestor is called directly via ``AdhocIngestor.ingest_url(url)``; the
``fetch_pro()`` and ``search_radar()`` methods are not applicable and raise
``NotImplementedError``.
"""
import logging
import re
import xml.etree.ElementTree as ET
from urllib.parse import parse_qs, urlparse

import httpx

from ..config.models import AppConfig
from .base import BaseIngestor, RawDocument
from .youtube import _fetch_transcript_sync

logger = logging.getLogger(__name__)

# ArXiv Atom API base URL (no authentication required)
_ARXIV_BASE_URL = "http://export.arxiv.org/api/query"

# ArXiv Atom XML namespace
_ATOM_NS = "http://www.w3.org/2005/Atom"


def _extract_youtube_video_id(url: str) -> str | None:
    """Extract the YouTube video ID from various YouTube URL formats.

    Supports:
    - ``https://www.youtube.com/watch?v=VIDEO_ID``
    - ``https://youtu.be/VIDEO_ID``
    - ``https://www.youtube.com/shorts/VIDEO_ID``

    Args:
        url: The YouTube URL string.

    Returns:
        The video ID string, or None if the URL format is not recognized.
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()

    # youtu.be short links: path is /<VIDEO_ID>
    if "youtu.be" in netloc:
        path = parsed.path.lstrip("/")
        return path if path else None

    if "youtube.com" in netloc:
        # /watch?v=VIDEO_ID
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

        # /shorts/VIDEO_ID
        match = re.match(r"^/shorts/([^/?&]+)", parsed.path)
        if match:
            return match.group(1)

        # /embed/VIDEO_ID
        match = re.match(r"^/embed/([^/?&]+)", parsed.path)
        if match:
            return match.group(1)

    return None


def _extract_arxiv_id(url: str) -> str | None:
    """Extract the ArXiv paper ID from an arxiv.org URL.

    Supports:
    - ``https://arxiv.org/abs/2501.12345`` → ``2501.12345``
    - ``https://arxiv.org/pdf/2501.12345`` → ``2501.12345``
    - ``https://arxiv.org/abs/2501.12345v2`` → ``2501.12345`` (version stripped)

    Args:
        url: The ArXiv URL string.

    Returns:
        The paper ID without version suffix, or None if not matched.
    """
    parsed = urlparse(url)
    # Match /abs/<id> or /pdf/<id> paths
    match = re.match(r"^/(abs|pdf)/([^/?&]+)", parsed.path)
    if match:
        raw_id = match.group(2)
        # Strip version suffix (e.g. v1, v2)
        paper_id = re.sub(r"v\d+$", "", raw_id)
        return paper_id
    return None


class AdhocIngestor(BaseIngestor):
    """Ingestor for single adhoc URLs provided directly by the user.

    Detects the URL type (YouTube, ArXiv, or generic article) and delegates
    to the appropriate handler. Returns a single ``RawDocument`` with
    ``origin='adhoc'``.

    ``fetch_pro()`` and ``search_radar()`` are not applicable for adhoc
    ingestion and raise ``NotImplementedError``.

    Usage::

        ingestor = AdhocIngestor(config)
        doc = await ingestor.ingest_url("https://arxiv.org/abs/2501.12345", tags=["ml"])
    """

    source_type = "adhoc"

    def __init__(self, config: AppConfig) -> None:
        """Initialize the ingestor with application config and httpx client.

        Args:
            config: The application configuration.
        """
        super().__init__(config)
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "ai-craftsman-kb/1.0"},
        )

    @property
    def _transcript_langs(self) -> list[str]:
        """Return the preferred transcript language codes from settings."""
        return self.config.settings.youtube.transcript_langs

    async def fetch_pro(self) -> list[RawDocument]:
        """Not applicable for adhoc ingestion.

        Raises:
            NotImplementedError: Always raised; use ``ingest_url()`` instead.
        """
        raise NotImplementedError(
            "AdhocIngestor does not support fetch_pro(). Use ingest_url() instead."
        )

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Not applicable for adhoc ingestion.

        Args:
            query: Unused.
            limit: Unused.

        Raises:
            NotImplementedError: Always raised; use ``ingest_url()`` instead.
        """
        raise NotImplementedError(
            "AdhocIngestor does not support search_radar(). Use ingest_url() instead."
        )

    async def ingest_url(
        self,
        url: str,
        tags: list[str] | None = None,
    ) -> RawDocument:
        """Detect URL type and delegate to the appropriate handler.

        The URL type is determined by the domain:
        - ``youtube.com`` / ``youtu.be`` → YouTube video handler
        - ``arxiv.org`` → ArXiv abstract handler
        - All other domains → generic article handler (HTML extraction)

        The ``tags`` argument is stored in ``metadata['adhoc_tags']`` so that
        the caller can later populate ``user_tags`` on the DB row.

        Args:
            url: The URL to ingest.
            tags: Optional list of user-supplied tag strings stored in metadata.

        Returns:
            A ``RawDocument`` with ``origin='adhoc'`` and ``source_type='adhoc'``.
        """
        url_type = self._detect_url_type(url)
        logger.debug("Adhoc ingest: url=%s type=%s tags=%s", url, url_type, tags)

        if url_type == "youtube":
            doc = await self._handle_youtube(url)
        elif url_type == "arxiv":
            doc = await self._handle_arxiv(url)
        else:
            doc = await self._handle_article(url)

        # Merge adhoc_tags and url_type into whatever metadata the handler set
        merged_metadata = {
            **doc.metadata,
            "adhoc_tags": tags or [],
            "url_type": url_type,
        }
        return doc.model_copy(update={"metadata": merged_metadata})

    def _detect_url_type(self, url: str) -> str:
        """Determine whether a URL points to a YouTube video, ArXiv paper, or article.

        Args:
            url: The URL string to classify.

        Returns:
            One of ``'youtube'``, ``'arxiv'``, or ``'article'``.
        """
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()

        if "youtube.com" in netloc or "youtu.be" in netloc:
            return "youtube"
        if "arxiv.org" in netloc:
            return "arxiv"
        return "article"

    async def _handle_youtube(self, url: str) -> RawDocument:
        """Handle a YouTube URL by extracting the video ID and fetching the transcript.

        Falls back to ``raw_content=None`` if the transcript is unavailable
        (disabled, private video, unsupported language, etc.).

        Args:
            url: A YouTube watch, short, or embed URL.

        Returns:
            A RawDocument with ``content_type='video'``, ``source_type='adhoc'``,
            and ``origin='adhoc'``. Transcript is stored in ``raw_content`` when
            available; ``raw_content=None`` otherwise.
        """
        import asyncio

        video_id = _extract_youtube_video_id(url)
        if not video_id:
            logger.warning("Could not extract video_id from YouTube URL: %s", url)
            return RawDocument(
                url=url,
                source_type="adhoc",
                origin="adhoc",
                content_type="video",
                metadata={"video_id": None},
            )

        # Fetch transcript asynchronously using thread executor (youtube-transcript-api is sync)
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(
            None, _fetch_transcript_sync, video_id, self._transcript_langs
        )

        word_count = len(transcript.split()) if transcript else None

        return RawDocument(
            url=url,
            source_type="adhoc",
            origin="adhoc",
            content_type="video",
            raw_content=transcript,
            word_count=word_count,
            metadata={"video_id": video_id},
        )

    async def _handle_arxiv(self, url: str) -> RawDocument:
        """Handle an ArXiv URL by fetching the paper abstract via the Atom API.

        Resolves both ``/abs/`` and ``/pdf/`` URL paths to the abstract.
        Falls back to ``raw_content=None`` on HTTP failure or parse error.

        Args:
            url: An arxiv.org URL (abs or pdf path).

        Returns:
            A RawDocument with ``content_type='paper'``, ``source_type='adhoc'``,
            and ``origin='adhoc'``. Abstract text is stored in ``raw_content`` when
            available; ``raw_content=None`` on failure.
        """
        paper_id = _extract_arxiv_id(url)
        if not paper_id:
            logger.warning("Could not extract paper_id from ArXiv URL: %s", url)
            return RawDocument(
                url=url,
                source_type="adhoc",
                origin="adhoc",
                content_type="paper",
                metadata={"arxiv_id": None},
            )

        canonical_url = f"https://arxiv.org/abs/{paper_id}"

        params = {
            "id_list": paper_id,
            "max_results": 1,
        }

        try:
            resp = await self._client.get(_ARXIV_BASE_URL, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("ArXiv Atom API request failed for paper %s: %s", paper_id, exc)
            return RawDocument(
                url=canonical_url,
                source_type="adhoc",
                origin="adhoc",
                content_type="paper",
                raw_content=None,
                metadata={"arxiv_id": paper_id},
            )

        # Parse the Atom XML response
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            logger.error("Failed to parse ArXiv Atom response for %s: %s", paper_id, exc)
            return RawDocument(
                url=canonical_url,
                source_type="adhoc",
                origin="adhoc",
                content_type="paper",
                raw_content=None,
                metadata={"arxiv_id": paper_id},
            )

        entry = root.find(f"{{{_ATOM_NS}}}entry")
        if entry is None:
            logger.warning("No entry found in ArXiv Atom response for paper: %s", paper_id)
            return RawDocument(
                url=canonical_url,
                source_type="adhoc",
                origin="adhoc",
                content_type="paper",
                raw_content=None,
                metadata={"arxiv_id": paper_id},
            )

        def _text(tag: str) -> str | None:
            """Return stripped text of the first matching child element."""
            el = entry.find(f"{{{_ATOM_NS}}}{tag}")
            return el.text.strip() if el is not None and el.text else None

        # Extract title (normalize whitespace)
        raw_title = _text("title")
        title = " ".join(raw_title.split()) if raw_title else None

        # Extract authors
        author_names: list[str] = []
        for author_el in entry.findall(f"{{{_ATOM_NS}}}author"):
            name_el = author_el.find(f"{{{_ATOM_NS}}}name")
            if name_el is not None and name_el.text:
                author_names.append(name_el.text.strip())
        author = ", ".join(author_names) if author_names else None

        # Extract abstract (normalize whitespace)
        raw_summary = _text("summary")
        abstract = " ".join(raw_summary.split()) if raw_summary else None

        # Extract categories
        categories: list[str] = []
        for cat_el in entry.findall(f"{{{_ATOM_NS}}}category"):
            term = cat_el.get("term")
            if term:
                categories.append(term)

        # Build PDF URL
        pdf_url = f"https://arxiv.org/pdf/{paper_id}"

        word_count = len(abstract.split()) if abstract else None

        return RawDocument(
            url=canonical_url,
            title=title,
            author=author,
            source_type="adhoc",
            origin="adhoc",
            content_type="paper",
            raw_content=abstract,
            word_count=word_count,
            metadata={
                "arxiv_id": paper_id,
                "categories": categories,
                "pdf_url": pdf_url,
            },
        )

    async def _handle_article(self, url: str) -> RawDocument:
        """Handle a generic article URL using ContentExtractor (readability + html2text).

        Extracts clean article text from the HTML page at the given URL.
        Returns ``raw_content=None`` on fetch failure.

        Args:
            url: Any HTTP/HTTPS URL for an article or blog post.

        Returns:
            A RawDocument with ``content_type='article'``, ``source_type='adhoc'``,
            and ``origin='adhoc'``. Extracted text is in ``raw_content``.
        """
        from ..processing.extractor import ContentExtractor

        async with ContentExtractor() as extractor:
            extracted = await extractor.fetch_and_extract(url)

        raw_content = extracted.text if extracted.text else None
        word_count = extracted.word_count if extracted.word_count else None

        return RawDocument(
            url=url,
            title=extracted.title,
            author=extracted.author,
            source_type="adhoc",
            origin="adhoc",
            content_type="article",
            raw_content=raw_content,
            word_count=word_count,
            metadata={},
        )

    async def __aenter__(self) -> "AdhocIngestor":
        """Enter async context manager, returning self."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager, closing the httpx client."""
        await self._client.aclose()
