"""Hacker News ingestor using the Algolia HN Search API."""
import logging
from datetime import datetime

import httpx

from ..config.models import AppConfig
from .base import BaseIngestor, RawDocument

logger = logging.getLogger(__name__)

ALGOLIA_BASE = "https://hn.algolia.com/api/v1"


class HackerNewsIngestor(BaseIngestor):
    """Ingestor for Hacker News via Algolia Search API.

    Pro mode: fetches recent top/new stories above a point threshold.
    Radar mode: searches HN by keyword query.

    Both modes use the Algolia HN Search API which requires no authentication.
    The hn_url in metadata always points to the HN discussion page, even when
    the story itself links to an external article.
    """

    source_type = "hn"

    def __init__(self, config: AppConfig) -> None:
        """Initialize the ingestor with application config and httpx client.

        Args:
            config: The application configuration (see config.models.AppConfig).
        """
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=ALGOLIA_BASE,
            timeout=30.0,
            headers={"User-Agent": "ai-craftsman-kb/1.0"},
        )

    async def fetch_pro(self) -> list[RawDocument]:
        """Fetch recent HN stories above min_points threshold.

        Uses hackernews config from sources.yaml: {mode, limit}.
        Stories with a URL have raw_content set to None (caller should
        use fetch_content() or ContentExtractor to populate it).
        Text-only stories (Ask HN) use story_text directly as raw_content.

        Returns:
            List of RawDocuments, up to configured limit. Returns empty list
            if hackernews config is None or if the API call fails.
        """
        hn_cfg = self.config.sources.hackernews
        if hn_cfg is None:
            return []

        limit = hn_cfg.limit
        # Use Algolia search_by_date to fetch recent stories with point filter.
        # Points > 10 is a reasonable threshold to filter noise.
        params = {
            "tags": "story",
            "numericFilters": "points>10",
            "hitsPerPage": min(limit, 50),
        }
        try:
            resp = await self._client.get("/search_by_date", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error("HN pro fetch failed: %s", e)
            return []

        docs: list[RawDocument] = []
        for hit in data.get("hits", [])[:limit]:
            doc = self._hit_to_raw_doc(hit)
            docs.append(doc)

        return docs

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search HN Algolia for query. Returns stories sorted by relevance.

        Uses the /search endpoint (relevance-ranked) with a minimum points filter
        to avoid surfacing low-quality results. Does not require authentication.

        Args:
            query: The keyword or phrase to search for on HN.
            limit: Maximum number of results to return (capped at 50).

        Returns:
            List of RawDocuments with origin='radar'. Returns empty list on error.
        """
        params = {
            "query": query,
            "tags": "story",
            "numericFilters": "points>5",
            "hitsPerPage": min(limit, 50),
        }
        try:
            resp = await self._client.get("/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error("HN radar search failed for query '%s': %s", query, e)
            return []

        docs: list[RawDocument] = []
        for hit in data.get("hits", [])[:limit]:
            docs.append(self._hit_to_raw_doc(hit, origin="radar"))
        return docs

    def _hit_to_raw_doc(self, hit: dict, origin: str = "pro") -> RawDocument:
        """Map an Algolia HN hit to a RawDocument.

        For text-only stories (Ask HN / Show HN without external URL), uses
        story_text as raw_content. For stories with a URL, raw_content is None
        and must be fetched by the caller (via fetch_content() or ContentExtractor).

        The hn_url in metadata always points to the HN discussion thread at
        https://news.ycombinator.com/item?id={hn_id}, regardless of whether
        the story has an external URL.

        Args:
            hit: A single Algolia hit dict from the HN Search API response.
            origin: The ingest origin ('pro' or 'radar').

        Returns:
            A RawDocument populated from the hit fields.
        """
        hn_id = hit.get("objectID", "")
        # The canonical HN discussion URL — always included in metadata
        hn_url = f"https://news.ycombinator.com/item?id={hn_id}"
        # For external articles the story URL differs from the HN discussion URL
        story_url = hit.get("url") or hn_url
        story_text = hit.get("story_text") or ""

        # Parse ISO-8601 timestamp from Algolia (format: "2025-01-15T10:00:00Z")
        published_at: datetime | None = None
        if ts := hit.get("created_at"):
            try:
                published_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                logger.debug("Could not parse created_at timestamp: %s", ts)

        # For Ask HN / text posts (no external URL): use story_text as raw_content.
        # For stories with an external URL: leave raw_content=None for the caller
        # to populate via ContentExtractor.
        has_external_url = bool(hit.get("url"))
        raw_content = None if has_external_url else (story_text or None)

        return RawDocument(
            url=story_url,
            title=hit.get("title"),
            author=hit.get("author"),
            raw_content=raw_content,
            content_type="post",
            published_at=published_at,
            source_type="hn",
            origin=origin,
            metadata={
                "hn_id": hn_id,
                # Use `or 0` to coerce None values from the API to 0
                "points": hit.get("points") or 0,
                "comment_count": hit.get("num_comments") or 0,
                "hn_url": hn_url,
            },
        )

    async def __aenter__(self) -> "HackerNewsIngestor":
        """Enter async context manager, returning self."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager, closing the httpx client."""
        await self._client.aclose()
