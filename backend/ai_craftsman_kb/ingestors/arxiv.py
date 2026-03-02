"""ArXiv ingestor using the public ArXiv Atom API.

Fetches papers matching configured search queries (pro mode) or arbitrary
query strings (radar mode). Uses the abstract (summary) as raw_content —
no PDF download is needed.

ArXiv API documentation: https://arxiv.org/help/api/user-manual
Rate limit: 3 seconds between API calls as requested by ArXiv.
"""
import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Literal

import httpx

from ..config.models import AppConfig
from .base import BaseIngestor, RawDocument

logger = logging.getLogger(__name__)

# ArXiv Atom API base URL (no authentication required)
BASE_URL = "http://export.arxiv.org/api/query"

# ArXiv Atom XML namespaces
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

# Rate limit enforced by ArXiv: 3 seconds between requests
_RATE_LIMIT_SECONDS = 3.0

# Default lookback window for pro-mode date filtering
_DEFAULT_LOOKBACK_DAYS = 7


def _strip_version(arxiv_id: str) -> str:
    """Remove version suffix from an ArXiv ID string.

    Example: 'http://arxiv.org/abs/2501.12345v1' -> '2501.12345'
             '2501.12345v2' -> '2501.12345'

    Args:
        arxiv_id: Raw ArXiv ID, possibly with a 'v<N>' version suffix.

    Returns:
        ArXiv ID without version suffix.
    """
    # Strip trailing version suffix like 'v1', 'v2', etc.
    return re.sub(r"v\d+$", "", arxiv_id)


def _canonical_url(raw_id: str) -> str:
    """Build the canonical https abs URL from a raw ArXiv entry ID.

    The raw ID from the Atom feed looks like:
        http://arxiv.org/abs/2501.12345v1

    We extract the paper ID, strip the version suffix, and return:
        https://arxiv.org/abs/2501.12345

    Args:
        raw_id: The <id> element text from the ArXiv Atom entry.

    Returns:
        Canonical HTTPS abs URL without version suffix.
    """
    # Extract the paper ID part after '/abs/'
    match = re.search(r"/abs/(.+)$", raw_id)
    if match:
        paper_id = _strip_version(match.group(1))
        return f"https://arxiv.org/abs/{paper_id}"
    # Fallback: return the raw_id as-is (should not happen in practice)
    return raw_id


def _pdf_url(raw_id: str) -> str:
    """Build the PDF URL from a raw ArXiv entry ID.

    Example: 'http://arxiv.org/abs/2501.12345v1' -> 'https://arxiv.org/pdf/2501.12345'

    Args:
        raw_id: The <id> element text from the ArXiv Atom entry.

    Returns:
        HTTPS PDF URL without version suffix.
    """
    match = re.search(r"/abs/(.+)$", raw_id)
    if match:
        paper_id = _strip_version(match.group(1))
        return f"https://arxiv.org/pdf/{paper_id}"
    return raw_id


def _arxiv_id_from_raw(raw_id: str) -> str:
    """Extract just the ArXiv paper ID (e.g. '2501.12345') from the full entry ID.

    Args:
        raw_id: The <id> element text from the ArXiv Atom entry.

    Returns:
        Short ArXiv paper ID without version suffix.
    """
    match = re.search(r"/abs/(.+)$", raw_id)
    if match:
        return _strip_version(match.group(1))
    return _strip_version(raw_id)


class ArxivIngestor(BaseIngestor):
    """Ingestor for ArXiv papers via the public Atom API.

    Pro mode: fetches papers for each configured query, filtered by
    submission date within the last N days.

    Radar mode: searches ArXiv with an arbitrary query string and returns
    up to `limit` results.

    No authentication is required. ArXiv requests a 3-second delay between
    API calls, which this ingestor enforces.
    """

    source_type = "arxiv"
    BASE_URL = BASE_URL

    def __init__(self, config: AppConfig) -> None:
        """Initialize the ingestor with application config and httpx client.

        Args:
            config: The application configuration (see config.models.AppConfig).
        """
        super().__init__(config)
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "ai-craftsman-kb/1.0"},
        )
        # Track the time of the last API call for rate limiting
        self._last_request_time: float = 0.0

    async def _rate_limited_get(self, params: dict) -> httpx.Response:
        """Perform a GET request to the ArXiv API, enforcing the 3-second rate limit.

        ArXiv's usage policy requests a minimum of 3 seconds between requests.
        This method sleeps as needed before issuing the next request.

        Args:
            params: Query parameters to include in the GET request.

        Returns:
            The httpx.Response from the ArXiv API.

        Raises:
            httpx.HTTPError: On network or HTTP-level errors.
        """
        import time

        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _RATE_LIMIT_SECONDS and self._last_request_time > 0:
            wait = _RATE_LIMIT_SECONDS - elapsed
            logger.debug("ArXiv rate limit: sleeping %.2fs before next request", wait)
            await asyncio.sleep(wait)

        resp = await self._client.get(self.BASE_URL, params=params)
        self._last_request_time = time.monotonic()
        return resp

    async def fetch_pro(self) -> list[RawDocument]:
        """Fetch papers for each query in config.sources.arxiv.queries.

        For each configured query:
        1. Fetch papers via the ArXiv Atom API with max_results limit.
        2. Filter to papers submitted within the last 7 days.
        3. Collect results across all queries, deduplicating by URL.

        Returns:
            Combined deduplicated list of RawDocuments. Returns empty list
            if arxiv config is None or all API calls fail.
        """
        arxiv_cfg = self.config.sources.arxiv
        if arxiv_cfg is None:
            return []

        seen_urls: set[str] = set()
        all_docs: list[RawDocument] = []

        cutoff = datetime.now(timezone.utc) - timedelta(days=_DEFAULT_LOOKBACK_DAYS)

        for query in arxiv_cfg.queries:
            params = {
                "search_query": query,
                "start": 0,
                "max_results": arxiv_cfg.max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            try:
                resp = await self._rate_limited_get(params)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.error("ArXiv pro fetch failed for query %r: %s", query, e)
                continue

            docs = self._parse_atom_feed(resp.text)
            for doc in docs:
                # Filter by submission date
                if doc.published_at is not None and doc.published_at < cutoff:
                    logger.debug(
                        "Skipping old ArXiv paper %s (published %s)",
                        doc.url,
                        doc.published_at,
                    )
                    continue
                # Deduplicate by URL across queries
                if doc.url in seen_urls:
                    continue
                seen_urls.add(doc.url)
                all_docs.append(doc)

        return all_docs

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search ArXiv for papers matching an arbitrary query string (Radar mode).

        Uses the ArXiv Atom API with the provided query. Results are sorted
        by submission date (most recent first). No date filtering is applied
        in radar mode so that relevant older papers can surface.

        Args:
            query: The ArXiv search query string (e.g. 'cat:cs.CL AND abs:LLM').
            limit: Maximum number of results to return.

        Returns:
            Up to `limit` RawDocuments with origin='radar'. Returns empty list
            on error.
        """
        params = {
            "search_query": query,
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            resp = await self._rate_limited_get(params)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("ArXiv radar search failed for query %r: %s", query, e)
            return []

        docs = self._parse_atom_feed(resp.text, origin="radar")
        return docs[:limit]

    def _parse_atom_feed(
        self,
        xml_text: str,
        origin: Literal["pro", "radar", "adhoc"] = "pro",
    ) -> list[RawDocument]:
        """Parse an ArXiv Atom XML response into a list of RawDocuments.

        Uses xml.etree.ElementTree (stdlib). Handles the ArXiv Atom namespace
        (http://www.w3.org/2005/Atom) and ignores the arxiv-specific namespace
        for now (no arxiv:comment extraction required).

        Args:
            xml_text: The raw Atom XML response body from the ArXiv API.
            origin: The ingest origin label to apply to each document.

        Returns:
            List of RawDocuments parsed from the Atom feed entries. Returns
            empty list if the XML is malformed or contains no entries.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error("Failed to parse ArXiv Atom XML: %s", e)
            return []

        docs: list[RawDocument] = []
        # Atom entries are direct children of the feed root
        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            try:
                doc = self._entry_to_raw_doc(entry, origin=origin)
                docs.append(doc)
            except Exception as e:
                logger.warning("Failed to parse ArXiv entry: %s", e)

        return docs

    def _entry_to_raw_doc(
        self,
        entry_elem: ET.Element,
        origin: Literal["pro", "radar", "adhoc"] = "pro",
    ) -> RawDocument:
        """Map an ArXiv Atom XML entry element to a RawDocument.

        Extracts:
        - url: canonical abs URL without version suffix
        - title: paper title (whitespace normalized)
        - author: comma-separated author names
        - raw_content: abstract text (summary element)
        - published_at: submission date as timezone-aware UTC datetime
        - metadata: arxiv_id (short), categories (list), pdf_url

        Args:
            entry_elem: An <entry> Element from the ArXiv Atom feed.
            origin: The ingest origin label.

        Returns:
            A RawDocument populated from the entry fields.
        """
        def _text(tag: str) -> str | None:
            """Return stripped text content of the first matching child, or None."""
            el = entry_elem.find(f"{{{ATOM_NS}}}{tag}")
            return el.text.strip() if el is not None and el.text else None

        # Raw entry ID, e.g. "http://arxiv.org/abs/2501.12345v1"
        raw_id = _text("id") or ""

        canonical_url = _canonical_url(raw_id)
        arxiv_id = _arxiv_id_from_raw(raw_id)
        pdf_url = _pdf_url(raw_id)

        # Title — normalize internal whitespace (ArXiv titles often have newlines)
        raw_title = _text("title")
        title = " ".join(raw_title.split()) if raw_title else None

        # Authors — join multiple <author><name>...</name></author> elements
        author_names: list[str] = []
        for author_el in entry_elem.findall(f"{{{ATOM_NS}}}author"):
            name_el = author_el.find(f"{{{ATOM_NS}}}name")
            if name_el is not None and name_el.text:
                author_names.append(name_el.text.strip())
        author = ", ".join(author_names) if author_names else None

        # Abstract — normalize whitespace
        raw_summary = _text("summary")
        raw_content = " ".join(raw_summary.split()) if raw_summary else None

        # Published date — ISO 8601 format: "2025-01-15T00:00:00Z"
        published_at: datetime | None = None
        published_str = _text("published")
        if published_str:
            try:
                published_at = datetime.fromisoformat(
                    published_str.replace("Z", "+00:00")
                )
            except ValueError:
                logger.debug("Could not parse ArXiv published date: %s", published_str)

        # Categories — multiple <category term="cs.CL" scheme="..."/>
        categories: list[str] = []
        for cat_el in entry_elem.findall(f"{{{ATOM_NS}}}category"):
            term = cat_el.get("term")
            if term:
                categories.append(term)

        return RawDocument(
            url=canonical_url,
            title=title,
            author=author,
            raw_content=raw_content,
            content_type="paper",
            published_at=published_at,
            source_type="arxiv",
            origin=origin,
            metadata={
                "arxiv_id": arxiv_id,
                "categories": categories,
                "pdf_url": pdf_url,
            },
        )

    async def __aenter__(self) -> "ArxivIngestor":
        """Enter async context manager, returning self."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager, closing the httpx client."""
        await self._client.aclose()
