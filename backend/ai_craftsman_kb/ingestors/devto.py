"""DEV.to ingestor using the public DEV.to REST API."""
import asyncio
import logging
from datetime import datetime

import httpx

from ..config.models import AppConfig
from .base import BaseIngestor, RawDocument

logger = logging.getLogger(__name__)

BASE_URL = "https://dev.to/api"

# Maximum concurrent individual article fetches to stay polite to the API.
_MAX_CONCURRENT_FETCHES = 5

# Delay between requests in seconds to avoid hammering the API.
_REQUEST_DELAY = 0.2


class DevtoIngestor(BaseIngestor):
    """Ingestor for DEV.to articles via the public REST API.

    Pro mode: fetches articles for each configured tag, then fetches the full
    body_markdown from individual article endpoints (max 5 concurrent).
    Radar mode: uses the ?q= search parameter to find articles by keyword.

    DEV.to's public API requires no authentication for read operations.
    """

    source_type = "devto"

    def __init__(self, config: AppConfig) -> None:
        """Initialize the ingestor with application config and httpx client.

        Args:
            config: The application configuration (see config.models.AppConfig).
        """
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=30.0,
            headers={"User-Agent": "ai-craftsman-kb/1.0"},
        )

    async def fetch_pro(self) -> list[RawDocument]:
        """Fetch articles for all configured DEV.to tags.

        For each tag in config.sources.devto.tags:
        1. Fetches the article list via /articles?tag={tag}&per_page={limit}
        2. For each unique article ID: fetches the full article to get body_markdown
           (max 5 concurrent fetches)
        3. Converts each article to a RawDocument

        Returns deduplicated results across all tags (same article may appear
        under multiple tags).

        Returns:
            List of RawDocuments. Returns empty list if devto config is None,
            no tags are configured, or all API calls fail.
        """
        devto_cfg = self.config.sources.devto
        if devto_cfg is None:
            return []
        if not devto_cfg.tags:
            return []

        limit = devto_cfg.limit
        # Collect article summaries across all tags, deduplicated by article ID.
        seen_ids: set[int] = set()
        article_summaries: list[dict] = []

        for tag in devto_cfg.tags:
            params = {
                "tag": tag,
                "per_page": limit,
                "page": 1,
            }
            try:
                resp = await self._client.get("/articles", params=params)
                resp.raise_for_status()
                articles = resp.json()
            except httpx.HTTPError as e:
                logger.error("DEV.to pro fetch failed for tag '%s': %s", tag, e)
                continue

            await asyncio.sleep(_REQUEST_DELAY)

            for article in articles:
                article_id = article.get("id")
                if article_id is not None and article_id not in seen_ids:
                    seen_ids.add(article_id)
                    article_summaries.append(article)

        if not article_summaries:
            return []

        # Fetch full article details (including body_markdown) with concurrency limit.
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)
        tasks = [
            self._fetch_full_article(summary, semaphore)
            for summary in article_summaries
        ]
        full_articles = await asyncio.gather(*tasks)

        docs: list[RawDocument] = []
        for article in full_articles:
            if article is not None:
                docs.append(self._article_to_raw_doc(article))

        return docs

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search DEV.to for articles matching the query string.

        Uses the GET /articles?q={query} search endpoint.
        Note: DEV.to search returns article summaries without body_markdown,
        so raw_content is populated from body_markdown fetched from the
        individual article endpoint.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            List of RawDocuments with origin='radar'. Returns empty list on error.
        """
        params = {
            "q": query,
            "per_page": min(limit, 30),
        }
        try:
            resp = await self._client.get("/articles", params=params)
            resp.raise_for_status()
            articles = resp.json()
        except httpx.HTTPError as e:
            logger.error("DEV.to radar search failed for query '%s': %s", query, e)
            return []

        if not articles:
            return []

        # Fetch full article content for each result with concurrency limit.
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)
        tasks = [
            self._fetch_full_article(summary, semaphore)
            for summary in articles[:limit]
        ]
        full_articles = await asyncio.gather(*tasks)

        docs: list[RawDocument] = []
        for article in full_articles:
            if article is not None:
                docs.append(self._article_to_raw_doc(article, origin="radar"))

        return docs

    async def _fetch_full_article(
        self,
        summary: dict,
        semaphore: asyncio.Semaphore,
    ) -> dict | None:
        """Fetch the full article detail including body_markdown.

        Merges the summary fields with the full article data so that fields
        available only in the list endpoint (if any) are not lost.

        Args:
            summary: The article summary dict from the list endpoint.
            semaphore: Asyncio semaphore to cap concurrent requests.

        Returns:
            The full article dict (summary merged with detail), or None on error.
        """
        article_id = summary.get("id")
        if article_id is None:
            return summary

        async with semaphore:
            await asyncio.sleep(_REQUEST_DELAY)
            try:
                resp = await self._client.get(f"/articles/{article_id}")
                resp.raise_for_status()
                full = resp.json()
                # Merge: start with summary, overlay with full article data.
                merged = {**summary, **full}
                return merged
            except httpx.HTTPError as e:
                logger.warning(
                    "DEV.to fetch full article %s failed: %s", article_id, e
                )
                # Fall back to summary without body_markdown.
                return summary

    def _article_to_raw_doc(self, article: dict, origin: str = "pro") -> RawDocument:
        """Map a DEV.to article dict to a RawDocument.

        Uses canonical_url as the document URL (preferred over url field).
        raw_content is populated from body_markdown when available, falling back
        to description (the short excerpt from the list endpoint).

        The metadata dict includes DEV.to-specific fields:
        - devto_id: int
        - tags: list[str]
        - reactions: int (positive_reactions_count)
        - comments: int (comments_count)
        - reading_time_minutes: int

        Args:
            article: A DEV.to article dict (preferably from the detail endpoint
                     so body_markdown is present).
            origin: The ingest origin ('pro' or 'radar').

        Returns:
            A RawDocument populated from the article fields.
        """
        # Prefer canonical_url; fall back to url field.
        url = article.get("canonical_url") or article.get("url", "")

        # Author name from nested user object.
        user = article.get("user") or {}
        author_name = user.get("name") or user.get("username")

        # Parse the published_at timestamp.
        published_at: datetime | None = None
        if ts := article.get("published_at"):
            try:
                published_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                logger.debug("Could not parse published_at timestamp: %s", ts)

        # Content: use body_markdown when available (from detail endpoint),
        # otherwise fall back to description (short excerpt from list endpoint).
        body_markdown = article.get("body_markdown")
        description = article.get("description") or ""

        if body_markdown:
            raw_content: str | None = body_markdown
        elif description:
            raw_content = description
        else:
            raw_content = None

        # Compute word count from raw_content if available.
        word_count: int | None = None
        if raw_content:
            word_count = len(raw_content.split())

        return RawDocument(
            url=url,
            title=article.get("title"),
            author=author_name,
            raw_content=raw_content,
            content_type="article",
            published_at=published_at,
            source_type="devto",
            origin=origin,  # type: ignore[arg-type]
            word_count=word_count,
            metadata={
                "devto_id": article.get("id"),
                "tags": article.get("tags") or [],
                "reactions": article.get("positive_reactions_count") or 0,
                "comments": article.get("comments_count") or 0,
                "reading_time_minutes": article.get("reading_time_minutes") or 0,
                # Include description as summary even when full content is present.
                "summary": description,
            },
        )

    async def __aenter__(self) -> "DevtoIngestor":
        """Enter async context manager, returning self."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager, closing the httpx client."""
        await self._client.aclose()
