"""Radar engine: async fan-out search across all enabled sources.

The RadarEngine accepts a topic query, fans out concurrently to all enabled
radar sources, deduplicates results by URL, and stores them in the documents
table with origin='radar'.
"""
import asyncio
import logging
from typing import TYPE_CHECKING

import aiosqlite
from pydantic import BaseModel, Field

from ..db.models import DocumentRow
from ..db.queries import get_document_by_url, upsert_document
from ..ingestors.base import BaseIngestor, RawDocument

if TYPE_CHECKING:
    from ..config.models import AppConfig

logger = logging.getLogger(__name__)

# Sources that support meaningful radar search (non-trivial search_radar()).
# Used to indicate which source types are expected to return results.
# Sources with limited/no radar support (substack, rss) return [] from search_radar().
RADAR_CAPABLE_SOURCES = {"hn", "reddit", "arxiv", "devto", "youtube"}


class RadarResult(BaseModel):
    """A single document result from a radar search.

    Wraps a DocumentRow with additional metadata about the source and
    whether the document was newly ingested or already existed in the DB.
    """

    document: DocumentRow
    source_type: str
    is_new: bool  # True if not already in DB at search time


class RadarReport(BaseModel):
    """Summary report of a radar search run.

    Contains aggregate counts, which sources were searched, and any
    per-source errors that occurred during fan-out.
    """

    query: str
    total_found: int = 0
    new_documents: int = 0
    sources_searched: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)  # {source_type: error_message}


class RadarEngine:
    """On-demand topic search across all enabled sources with async fan-out.

    The engine fans out concurrently to all provided ingestors (or a filtered
    subset), deduplicates results by URL, and stores new documents with
    origin='radar'. Failing sources are logged and recorded in RadarReport.errors
    without interrupting results from other sources.

    Usage::

        engine = RadarEngine(config, ingestors)
        async with aiosqlite.connect(db_path) as conn:
            conn.row_factory = aiosqlite.Row
            report = await engine.search(conn, "LLM fine-tuning", limit_per_source=10)
    """

    def __init__(
        self,
        config: "AppConfig",
        ingestors: dict[str, BaseIngestor],
    ) -> None:
        """Initialize the radar engine.

        Args:
            config: The application configuration.
            ingestors: Pre-instantiated ingestors keyed by source_type string
                       (e.g. {'hn': HackerNewsIngestor(config), 'reddit': ...}).
        """
        self.config = config
        self.ingestors = ingestors

    async def search(
        self,
        conn: aiosqlite.Connection,
        query: str,
        sources: list[str] | None = None,
        limit_per_source: int = 10,
    ) -> RadarReport:
        """Async fan-out search across all enabled (or specified) sources.

        Steps:
        1. Determine which sources to search (all ingestors if sources=None,
           otherwise only those listed in sources).
        2. Fan out concurrently via asyncio.gather with return_exceptions=True.
        3. Collect results; record per-source exceptions in report.errors.
        4. Deduplicate all results by URL (first occurrence wins).
        5. Check DB for existing documents; store new ones with origin='radar'.
        6. Return a RadarReport with aggregate counts.

        Args:
            conn: An open aiosqlite connection with row_factory=aiosqlite.Row.
            query: The search query string.
            sources: Optional list of source_type strings to restrict the search.
                     None means search all ingestors.
            limit_per_source: Maximum number of results to request from each source.

        Returns:
            A RadarReport summarising total_found, new_documents, sources_searched,
            and any per-source errors.
        """
        report = RadarReport(query=query)

        # Step 1: Determine which ingestors to search
        if sources is not None:
            active_ingestors: dict[str, BaseIngestor] = {
                st: ing
                for st, ing in self.ingestors.items()
                if st in sources
            }
        else:
            active_ingestors = dict(self.ingestors)

        if not active_ingestors:
            logger.warning("RadarEngine.search: no active ingestors for query %r", query)
            return report

        report.sources_searched = list(active_ingestors.keys())

        # Step 2: Fan out concurrently — asyncio.gather with return_exceptions=True
        # so a failing source does not cancel other in-flight tasks.
        tasks = [
            self._search_one_source(ingestor, query, limit_per_source)
            for ingestor in active_ingestors.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Collect results and record per-source exceptions
        all_docs: list[RawDocument] = []
        for source_type, result in zip(active_ingestors.keys(), results):
            if isinstance(result, BaseException):
                err_msg = str(result)
                logger.error(
                    "RadarEngine: source %r raised exception during search: %s",
                    source_type,
                    err_msg,
                )
                report.errors[source_type] = err_msg
            else:
                all_docs.extend(result)

        # Step 4: Deduplicate by URL — keep the first occurrence across all sources
        deduplicated = _deduplicate_by_url(all_docs)
        report.total_found = len(deduplicated)

        # Step 5: Store new documents with origin='radar'
        # _store_results returns (new_count, duplicate_count)
        new_count, _ = await self._store_results(conn, deduplicated)
        report.new_documents = new_count

        return report

    async def _search_one_source(
        self,
        ingestor: BaseIngestor,
        query: str,
        limit: int,
    ) -> list[RawDocument]:
        """Search a single source and return its results.

        On exception, logs the error and returns an empty list so the caller's
        asyncio.gather continues with other sources. The exception is re-raised
        as a standard Exception so return_exceptions=True can capture it.

        Note: This method does NOT catch exceptions — it intentionally lets them
        propagate to asyncio.gather(return_exceptions=True) so the engine can
        record per-source failures in RadarReport.errors. The try/except here
        is only for logging before re-raising.

        Args:
            ingestor: The source ingestor to query.
            query: The search query string.
            limit: Maximum results to request from this source.

        Returns:
            A list of RawDocument results with origin='radar'.

        Raises:
            Exception: Any exception from ingestor.search_radar() is re-raised
                       after logging so asyncio.gather can capture it.
        """
        try:
            docs = await ingestor.search_radar(query, limit=limit)
            # Ensure all returned docs have origin='radar'
            radar_docs = [
                doc.model_copy(update={"origin": "radar"})
                if doc.origin != "radar"
                else doc
                for doc in docs
            ]
            logger.debug(
                "RadarEngine: source %r returned %d results for query %r",
                ingestor.source_type,
                len(radar_docs),
                query,
            )
            return radar_docs
        except Exception as exc:
            logger.error(
                "RadarEngine: source %r search_radar(%r) failed: %s",
                ingestor.source_type,
                query,
                exc,
            )
            raise

    async def _store_results(
        self,
        conn: aiosqlite.Connection,
        docs: list[RawDocument],
    ) -> tuple[int, int]:
        """Deduplicate against DB and store new documents with origin='radar'.

        For each document:
        - Check the DB for an existing document with the same URL.
        - If it exists, count it as a duplicate (skip insert).
        - If new, convert to DocumentRow with origin='radar' and upsert.

        Args:
            conn: An open aiosqlite connection.
            docs: Deduplicated list of RawDocument objects to store.

        Returns:
            A tuple of (new_count, duplicate_count).
        """
        new_count = 0
        duplicate_count = 0

        for doc in docs:
            try:
                existing = await get_document_by_url(conn, doc.url)
                if existing is not None:
                    duplicate_count += 1
                    logger.debug(
                        "RadarEngine: URL already in DB, skipping: %s", doc.url
                    )
                    continue

                # Ensure origin is 'radar' before storing
                if doc.origin != "radar":
                    doc = doc.model_copy(update={"origin": "radar"})

                doc_row = doc.to_document_row(source_id=None)
                await upsert_document(conn, doc_row)
                new_count += 1
                logger.debug("RadarEngine: stored new document %s", doc.url)

            except Exception as exc:
                logger.error(
                    "RadarEngine: failed to store document %s: %s", doc.url, exc
                )

        return new_count, duplicate_count


def _deduplicate_by_url(docs: list[RawDocument]) -> list[RawDocument]:
    """Deduplicate a list of RawDocuments by URL, keeping the first occurrence.

    When the same URL appears from multiple sources, the first one encountered
    (in the original list order) is retained. This is the primary cross-source
    deduplication step before any DB checks.

    Args:
        docs: A list of RawDocument objects, potentially with duplicate URLs.

    Returns:
        A new list with duplicate URLs removed (first occurrence preserved).
    """
    seen_urls: set[str] = set()
    unique: list[RawDocument] = []
    for doc in docs:
        if doc.url not in seen_urls:
            seen_urls.add(doc.url)
            unique.append(doc)
    return unique
