"""Substack ingestor using RSS/Atom feeds via feedparser."""
import asyncio
import calendar
import logging
from datetime import datetime, timezone
from typing import Any

import feedparser
import html2text

from ..config.models import AppConfig
from .base import BaseIngestor, RawDocument

logger = logging.getLogger(__name__)

# Rate limit: 1 request per second per feed is safe for Substack RSS
_FEED_RATE_LIMIT_SECONDS = 1.0

# Maximum entries to process per feed when no last-fetch timestamp is known
_DEFAULT_MAX_ENTRIES = 20

# html2text converter instance (module-level to avoid repeated init overhead)
_h2t = html2text.HTML2Text()
_h2t.ignore_links = False
_h2t.ignore_images = True
_h2t.body_width = 0  # do not wrap lines


class SubstackIngestor(BaseIngestor):
    """Ingestor for Substack newsletters via RSS/Atom feeds.

    Pro mode: fetches posts from all configured slugs in config.sources.substack.
    Radar mode: not well-supported (Substack has no public search API) — returns
    empty list with a log warning.

    Substack RSS feeds typically include full post HTML in the `content:encoded`
    field, so no additional HTTP fetch is required for most posts. When only a
    summary is available, the base class ContentExtractor is used as a fallback.
    """

    source_type = "substack"

    def __init__(self, config: AppConfig) -> None:
        """Initialize the ingestor with application config.

        Args:
            config: The application configuration (see config.models.AppConfig).
        """
        super().__init__(config)

    async def fetch_pro(self) -> list[RawDocument]:
        """Fetch posts from all configured Substack publications.

        For each slug in config.sources.substack:
        1. Fetches {slug}.substack.com/feed via feedparser (in a thread pool to
           avoid blocking the event loop, since feedparser is synchronous).
        2. Limits results to at most _DEFAULT_MAX_ENTRIES per feed.
        3. Extracts full post text from content:encoded when available.
        4. Falls back to ContentExtractor if only a summary is present.

        Returns:
            Combined deduplicated list of RawDocuments from all configured
            publications. Returns empty list if no substack sources are
            configured or all feeds fail.
        """
        slugs = self.config.sources.substack
        if not slugs:
            return []

        all_docs: list[RawDocument] = []
        seen_urls: set[str] = set()

        for i, source in enumerate(slugs):
            slug = source.slug
            feed_url = f"https://{slug}.substack.com/feed"

            try:
                # feedparser.parse is synchronous — run in executor to avoid blocking
                loop = asyncio.get_running_loop()
                feed = await loop.run_in_executor(None, feedparser.parse, feed_url)

                if feed.get("bozo") and not feed.get("entries"):
                    logger.warning(
                        "[substack] Feed for slug '%s' appears malformed or empty: %s",
                        slug,
                        feed.get("bozo_exception", "unknown error"),
                    )
                    continue

                entries = feed.get("entries", [])
                if not entries:
                    logger.info(
                        "[substack] No entries found in feed for slug '%s'", slug
                    )
                    continue

                # Limit to most recent entries
                entries = entries[:_DEFAULT_MAX_ENTRIES]

                for entry in entries:
                    try:
                        doc = self._entry_to_raw_doc(entry, slug)
                        # Deduplicate by URL within this run
                        if doc.url not in seen_urls:
                            seen_urls.add(doc.url)
                            all_docs.append(doc)
                    except Exception as e:
                        logger.warning(
                            "[substack] Failed to parse entry from slug '%s': %s",
                            slug,
                            e,
                        )
                        continue

            except Exception as e:
                logger.error(
                    "[substack] Failed to fetch feed for slug '%s' (%s): %s",
                    slug,
                    feed_url,
                    e,
                )
                continue

            # Rate limit between feeds (not after the last one)
            if i < len(slugs) - 1:
                await asyncio.sleep(_FEED_RATE_LIMIT_SECONDS)

        return all_docs

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search Substack for a query (Radar mode).

        Substack does not provide a public search API. This method logs a
        warning and returns an empty list. Radar support for Substack may be
        improved in a future phase (e.g., via DuckDuckGo site-search).

        Args:
            query: The search query string (not used in current implementation).
            limit: Maximum number of results (not used in current implementation).

        Returns:
            Always returns an empty list.
        """
        logger.warning(
            "[substack] search_radar() called with query=%r, but Substack has no "
            "public search API. Returning empty list. "
            "Consider using DuckDuckGo site search as a future improvement.",
            query,
        )
        return []

    def _entry_to_raw_doc(self, entry: Any, slug: str) -> RawDocument:
        """Parse a feedparser entry dict into a RawDocument.

        Extracts full HTML content from content:encoded when available (the
        common case for Substack feeds that include full post content). Falls
        back to the entry summary if content:encoded is absent or empty.
        Converts HTML to plain text using html2text.

        The post GUID from the feed is extracted and stored in metadata alongside
        the Substack slug. The canonical link from the feed is used as the URL.

        Args:
            entry: A feedparser entry dict from a Substack RSS/Atom feed.
            slug: The Substack publication slug (e.g. 'stratechery').

        Returns:
            A RawDocument with source_type='substack', content_type='article',
            and metadata containing substack_slug and post_id.
        """
        # --- URL ---
        # feedparser normalizes the entry link field across RSS and Atom formats
        url: str = entry.get("link", "") or ""

        # --- Title ---
        title: str | None = entry.get("title") or None

        # --- Author ---
        # RSS 2.0 may use dc:creator (mapped to "author" by feedparser)
        # or the <author> element
        author: str | None = None
        if entry.get("author"):
            author = entry.get("author")
        elif entry.get("author_detail"):
            author = entry.get("author_detail", {}).get("name")

        # --- Published date ---
        published_at: datetime | None = None
        # feedparser provides parsed_time structs; prefer "published_parsed" over "updated_parsed"
        time_struct = entry.get("published_parsed") or entry.get("updated_parsed")
        if time_struct is not None:
            try:
                # time.struct_time from calendar.timegm → UTC epoch → datetime
                epoch = calendar.timegm(time_struct)
                published_at = datetime.fromtimestamp(epoch, tz=timezone.utc)
            except Exception as e:
                logger.debug(
                    "[substack] Could not convert time struct for entry in '%s': %s",
                    slug,
                    e,
                )

        # --- Content: prefer content:encoded (full HTML), fall back to summary ---
        content_html: str = ""
        # feedparser maps content:encoded to entry.content[0].value for RSS
        # and entry.content[0].value for Atom content elements
        content_list = entry.get("content", [])
        if content_list:
            # content is a list of dicts with "value" and "type" keys
            content_html = content_list[0].get("value", "") or ""

        if not content_html:
            # Fall back to summary (may be truncated for paywalled posts)
            content_html = entry.get("summary", "") or ""

        # Convert HTML to plain text
        raw_content: str | None = None
        word_count: int | None = None
        if content_html:
            try:
                text = _h2t.handle(content_html).strip()
                if text:
                    raw_content = text
                    word_count = len(text.split())
            except Exception as e:
                logger.debug(
                    "[substack] html2text conversion failed for entry in '%s': %s",
                    slug,
                    e,
                )

        # --- Post ID ---
        # feedparser normalizes id/guid to entry.id
        post_id: str = entry.get("id", "") or ""

        return RawDocument(
            url=url,
            title=title,
            author=author,
            raw_content=raw_content,
            content_type="article",
            published_at=published_at,
            source_type="substack",
            origin="pro",
            word_count=word_count,
            metadata={
                "substack_slug": slug,
                "post_id": post_id,
            },
        )
