"""RSS/Atom feed ingestor using feedparser."""
import asyncio
import logging
import re
from calendar import timegm
from datetime import datetime, timedelta, timezone

import feedparser

from ..config.models import AppConfig, RSSSource
from .base import BaseIngestor, RawDocument

logger = logging.getLogger(__name__)

# Default age threshold: entries older than this many days are skipped
MAX_AGE_DAYS = 30


class RSSIngestor(BaseIngestor):
    """Ingestor for generic RSS 2.0 and Atom 1.0 feeds.

    Pro mode: fetches all feeds listed in config.sources.rss, filtering out
    entries older than MAX_AGE_DAYS days. Attempts to extract full content
    from content:encoded fields; falls back to summary; falls back to
    ContentExtractor for full-page scraping.

    Radar mode: Not supported for RSS (no search API). Returns empty list.
    """

    source_type = "rss"

    def __init__(self, config: AppConfig) -> None:
        """Initialize the RSS ingestor with application config.

        Args:
            config: The application configuration (see config.models.AppConfig).
        """
        super().__init__(config)

    async def fetch_pro(self) -> list[RawDocument]:
        """Fetch and parse all RSS/Atom feeds configured in config.sources.rss.

        For each feed URL:
        1. Fetches the feed via feedparser (run in thread executor for async compat)
        2. Skips entries older than MAX_AGE_DAYS (30 days)
        3. Extracts content: content:encoded -> summary -> None (caller uses ContentExtractor)
        4. Uses the feed's display_name from config as the source identifier

        Returns:
            Combined list of RawDocuments across all configured feeds.
            Network/parse errors per feed are logged and skipped without crashing.
        """
        feeds = self.config.sources.rss
        if not feeds:
            return []

        all_docs: list[RawDocument] = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)

        for feed_source in feeds:
            try:
                docs = await self._fetch_feed(feed_source, cutoff)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(
                    "[rss] Failed to fetch feed '%s' (%s): %s",
                    feed_source.name,
                    feed_source.url,
                    e,
                )

        return all_docs

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """RSS has no search API — radar mode is not supported.

        Args:
            query: The search query string (ignored).
            limit: Maximum number of results (ignored).

        Returns:
            Always returns an empty list.
        """
        return []

    async def _fetch_feed(
        self, feed_source: RSSSource, cutoff: datetime
    ) -> list[RawDocument]:
        """Fetch and parse a single RSS/Atom feed.

        Runs feedparser.parse() in a thread executor to avoid blocking the
        event loop (feedparser is synchronous).

        Args:
            feed_source: The RSSSource config containing the feed URL and name.
            cutoff: Entries with a published date older than this are skipped.

        Returns:
            List of RawDocuments parsed from the feed entries.
        """
        loop = asyncio.get_event_loop()
        feed = await loop.run_in_executor(None, feedparser.parse, feed_source.url)

        # feedparser signals HTTP errors via bozo + bozo_exception when entries are empty
        if feed.bozo and not feed.entries:
            exc = getattr(feed, "bozo_exception", None)
            raise RuntimeError(
                f"feedparser failed to parse feed: {exc or 'unknown error'}"
            )

        docs: list[RawDocument] = []
        for entry in feed.entries:
            doc = self._entry_to_raw_doc(entry, feed_source.name, feed_source.url)

            # Skip entries older than the cutoff (only if we have a date)
            if doc.published_at is not None and doc.published_at < cutoff:
                logger.debug(
                    "[rss] Skipping old entry '%s' from feed '%s' (published %s)",
                    doc.title,
                    feed_source.name,
                    doc.published_at.date(),
                )
                continue

            docs.append(doc)

        logger.info(
            "[rss] Fetched %d entries from feed '%s' (%s)",
            len(docs),
            feed_source.name,
            feed_source.url,
        )
        return docs

    def _entry_to_raw_doc(
        self, entry: object, feed_name: str, feed_url: str
    ) -> RawDocument:
        """Parse a feedparser entry object into a RawDocument.

        Content extraction priority:
        1. entry.content (content:encoded in RSS 2.0, content in Atom)
        2. entry.summary (RSS <description> or Atom <summary>)
        3. None — caller uses ContentExtractor to scrape the linked page

        Args:
            entry: A feedparser entry object (supports attribute access like a dict).
            feed_name: Human-readable name of the feed (from config).
            feed_url: The feed URL (from config), stored in metadata.

        Returns:
            A RawDocument populated from the feedparser entry.
        """
        # feedparser entry objects support attribute-style access
        entry_url: str = getattr(entry, "link", "") or ""
        title: str | None = getattr(entry, "title", None) or None
        author: str | None = getattr(entry, "author", None) or None
        entry_id: str = getattr(entry, "id", "") or entry_url

        # Parse published date: feedparser stores it as struct_time in published_parsed
        published_at: datetime | None = None
        published_parsed = getattr(entry, "published_parsed", None)
        if published_parsed is not None:
            try:
                # Convert struct_time (UTC) to timezone-aware datetime using timegm
                # timegm() treats the struct_time as UTC, matching feedparser behavior
                timestamp = timegm(published_parsed)
                published_at = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (TypeError, ValueError, OverflowError) as e:
                logger.debug(
                    "[rss] Could not parse published_parsed for entry '%s': %s",
                    entry_id,
                    e,
                )

        # Extract content with priority: content:encoded -> summary -> None
        # Discard trivially short content (< 50 words) so that fetch_content()
        # will scrape the linked page instead.  Many feeds (e.g. Lobste.rs)
        # only include a one-line "Comments" link as the summary.
        raw_content: str | None = None
        content_list = getattr(entry, "content", None)
        if content_list:
            # feedparser normalizes content:encoded into entry.content as a list of dicts
            # Each dict has 'value' (the HTML/text) and 'type' (mime type)
            raw_content = content_list[0].get("value") or None

        if raw_content is None:
            summary = getattr(entry, "summary", None)
            raw_content = summary or None

        # Strip HTML tags for word counting; discard if too short to be useful
        if raw_content:
            plain = re.sub(r"<[^>]+>", " ", raw_content).strip()
            if len(plain.split()) < 50:
                raw_content = None

        return RawDocument(
            url=entry_url,
            title=title,
            author=author,
            raw_content=raw_content,
            content_type="article",
            published_at=published_at,
            source_type="rss",
            origin="pro",
            metadata={
                "feed_name": feed_name,
                "feed_url": feed_url,
                "entry_id": entry_id,
            },
        )
