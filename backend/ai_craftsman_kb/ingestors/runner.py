"""IngestRunner: orchestrates fetch -> filter -> deduplicate -> store -> process."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from ..config.models import AppConfig
from ..db.models import DocumentRow, SourceRow
from ..db.queries import (
    get_document_by_url,
    update_source_fetch_status,
    upsert_document,
    upsert_source,
)
from ..db.sqlite import get_db
from ..llm.router import LLMRouter
from ..processing.filter import ContentFilter
from .arxiv import ArxivIngestor
from .base import BaseIngestor, RawDocument
from .devto import DevtoIngestor
from .hackernews import HackerNewsIngestor
from .reddit import RedditIngestor
from .rss import RSSIngestor
from .substack import SubstackIngestor
from .youtube import YouTubeIngestor

if TYPE_CHECKING:
    from ..processing.pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)


# Registry of available ingestors — all 7 sources
INGESTORS: dict[str, type[BaseIngestor]] = {
    "hn": HackerNewsIngestor,
    "substack": SubstackIngestor,
    "rss": RSSIngestor,
    "youtube": YouTubeIngestor,
    "reddit": RedditIngestor,
    "arxiv": ArxivIngestor,
    "devto": DevtoIngestor,
}


class IngestReport(BaseModel):
    """Summary of one ingestor run.

    Attributes:
        source_type: The source type string for this run (e.g. 'hn').
        fetched: Total documents fetched from the source.
        passed_filter: Documents that passed the content filter.
        stored: Documents stored to the DB (new, non-duplicate).
        skipped_duplicate: Documents that already existed in the DB.
        skipped_old: Documents filtered out by incremental fetch (before last run).
        embedded: Documents successfully embedded into Qdrant.
        entities_extracted: Documents that had entity extraction run.
        keywords_extracted: Documents that had keyword extraction run.
        errors: List of error message strings.
    """

    source_type: str
    fetched: int = 0
    passed_filter: int = 0
    stored: int = 0
    skipped_duplicate: int = 0
    skipped_old: int = 0
    embedded: int = 0
    entities_extracted: int = 0
    keywords_extracted: int = 0
    errors: list[str] = []


class IngestRunner:
    """Orchestrates: fetch -> filter -> deduplicate -> store -> process.

    Usage::

        runner = IngestRunner(config, llm_router, db_path)
        report = await runner.run_source(HackerNewsIngestor(config))

    Args:
        config: Fully loaded AppConfig.
        llm_router: Configured LLM router for filter scoring. May be None when
                    LLM-based filtering is not needed (e.g. adhoc ingestion).
        db_path: Path to the craftsman.db file. The parent directory is used
                 as the data_dir for get_db().
        pipeline: Optional ProcessingPipeline for post-ingest embedding and
                  entity extraction. When None, the pipeline step is skipped
                  (documents are stored but not embedded or entity-extracted).
    """

    def __init__(
        self,
        config: AppConfig,
        llm_router: LLMRouter | None,
        db_path: Path,
        pipeline: "ProcessingPipeline | None" = None,
    ) -> None:
        self.config = config
        self.llm_router = llm_router
        self.db_path = db_path
        self.pipeline = pipeline
        self._filter = ContentFilter(config, llm_router)  # type: ignore[arg-type]

    async def run_source(
        self,
        ingestor: BaseIngestor,
        origin: Literal["pro", "radar", "adhoc"] = "pro",
    ) -> IngestReport:
        """Run one ingestor: fetch -> incremental filter -> content filter -> dedup -> store.

        Steps:
        1. Get last_fetched_at from the sources table (for incremental fetch).
        2. Call ingestor.fetch_pro() to get raw documents.
        3. Apply incremental filter: skip docs published before last_fetched_at.
        4. Filter each doc via ContentFilter (batch, concurrency=5).
        5. Open DB connection and upsert the source row so it is tracked.
        6. Deduplicate by URL against existing DB records.
        7. For passed docs: call ingestor.fetch_content() to fill raw_content.
        8. Store to DB via upsert_document().
        9. If a ProcessingPipeline is configured, run process_batch() on all
           newly stored documents to embed and extract entities.
        10. Update source last_fetched_at timestamp.
        11. On fetch error: update fetch_error in sources table.

        Args:
            ingestor: The ingestor instance to run.
            origin: Ingest origin label stored on each document row.

        Returns:
            An IngestReport summarising the run counts and any errors.
        """
        report = IngestReport(source_type=ingestor.source_type)
        data_dir = self.db_path.parent

        # Step 1: Get last_fetched_at for incremental fetch.
        # Open DB briefly to read the existing source row, then close it
        # before the long-running fetch to avoid holding the connection open.
        last_fetched_at: datetime | None = None
        source_id_existing: str | None = None
        try:
            async with get_db(data_dir) as conn:
                async with conn.execute(
                    "SELECT id, last_fetched_at FROM sources "
                    "WHERE source_type = ? AND identifier = ?",
                    (ingestor.source_type, ingestor.source_type),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is not None:
                    source_id_existing = row["id"]
                    if row["last_fetched_at"]:
                        try:
                            last_fetched_at = datetime.fromisoformat(
                                row["last_fetched_at"]
                            )
                            # Ensure timezone-aware for comparison
                            if last_fetched_at.tzinfo is None:
                                last_fetched_at = last_fetched_at.replace(
                                    tzinfo=timezone.utc
                                )
                        except (ValueError, TypeError) as e:
                            logger.debug(
                                "[%s] Could not parse last_fetched_at: %s",
                                ingestor.source_type,
                                e,
                            )
        except Exception as e:
            logger.warning(
                "[%s] Could not read last_fetched_at from sources table: %s",
                ingestor.source_type,
                e,
            )

        # Step 2: Fetch raw documents from the source
        try:
            raw_docs = await ingestor.fetch_pro()
            report.fetched = len(raw_docs)
        except Exception as e:
            msg = f"fetch_pro failed: {e}"
            logger.error("[%s] %s", ingestor.source_type, msg)
            report.errors.append(msg)
            # Update fetch_error on the source row if it already exists
            if source_id_existing:
                try:
                    async with get_db(data_dir) as conn:
                        await update_source_fetch_status(
                            conn, source_id_existing, fetch_error=str(e)
                        )
                        await conn.commit()
                except Exception:
                    pass
            return report

        # Step 3: Incremental filter — skip docs published before last_fetched_at.
        # Docs without a published_at are always included (we cannot determine age).
        if last_fetched_at is not None:
            new_docs: list[RawDocument] = []
            for doc in raw_docs:
                if doc.published_at is None:
                    new_docs.append(doc)
                elif doc.published_at > last_fetched_at:
                    new_docs.append(doc)
                else:
                    report.skipped_old += 1
            if report.skipped_old > 0:
                logger.debug(
                    "[%s] Incremental fetch: skipped %d old docs (before %s)",
                    ingestor.source_type,
                    report.skipped_old,
                    last_fetched_at.isoformat(),
                )
            raw_docs = new_docs

        # Step 4: Filter batch — on filter failure, pass all docs through
        try:
            filter_results = await self._filter.filter_batch(
                raw_docs, ingestor.source_type, concurrency=5
            )
        except Exception as e:
            msg = f"filter_batch failed: {e}"
            logger.error("[%s] %s", ingestor.source_type, msg)
            report.errors.append(msg)
            # Pass all documents through so a filter outage does not silently drop content
            filter_results = [(doc, None) for doc in raw_docs]

        # Collect docs that passed the filter (or were passed-through on error)
        passed_docs: list[tuple] = []
        for doc, result in filter_results:
            if result is None or result.passed:
                doc_with_origin = doc.model_copy(update={
                    "filter_score": result.score if result else None,
                    "origin": origin,
                })
                passed_docs.append((doc_with_origin, result))

        report.passed_filter = len(passed_docs)

        async with get_db(data_dir) as conn:
            # Step 5: Ensure source row exists, reusing existing ID if present
            source_id = await self._get_or_create_source_id(
                conn, ingestor.source_type
            )

            now_iso = datetime.now(timezone.utc).isoformat()

            # Collect newly stored document rows for post-ingest processing
            stored_doc_rows: list[DocumentRow] = []

            for doc, result in passed_docs:
                try:
                    # Step 6: Deduplicate by URL
                    existing = await get_document_by_url(conn, doc.url)
                    if existing is not None:
                        report.skipped_duplicate += 1
                        continue

                    # Step 7: Fetch full content if not already present
                    if not doc.raw_content:
                        try:
                            doc = await ingestor.fetch_content(doc)
                        except Exception as e:
                            logger.warning(
                                "[%s] fetch_content failed for %s: %s",
                                ingestor.source_type,
                                doc.url,
                                e,
                            )

                    # Step 8: Convert to DB row and store
                    doc_row = doc.to_document_row(source_id=source_id)
                    if result is not None:
                        doc_row = doc_row.model_copy(update={
                            "filter_score": result.score,
                            "filter_passed": result.passed,
                        })
                    await upsert_document(conn, doc_row)
                    report.stored += 1
                    stored_doc_rows.append(doc_row)

                except Exception as e:
                    msg = f"store failed for {doc.url}: {e}"
                    logger.error("[%s] %s", ingestor.source_type, msg)
                    report.errors.append(msg)

            # Step 9: Run post-ingest processing pipeline if configured
            if self.pipeline is not None and stored_doc_rows:
                try:
                    processing_report = await self.pipeline.process_batch(
                        conn, stored_doc_rows
                    )
                    report.embedded = processing_report.embedded
                    report.entities_extracted = processing_report.entity_extracted
                    report.keywords_extracted = processing_report.keywords_extracted
                    if processing_report.errors:
                        report.errors.extend(processing_report.errors)
                except Exception as e:
                    msg = f"processing pipeline failed: {e}"
                    logger.error("[%s] %s", ingestor.source_type, msg)
                    report.errors.append(msg)

            # Step 10: Update source last_fetched_at timestamp, clear any previous error
            await update_source_fetch_status(conn, source_id, last_fetched_at=now_iso)
            await conn.commit()

        return report

    async def _get_or_create_source_id(
        self,
        conn,  # aiosqlite.Connection
        source_type: str,
    ) -> str:
        """Return the existing source ID for source_type, or create a new row.

        Querying first avoids the INSERT OR REPLACE DELETE+INSERT cycle that
        would null out source_id on all existing documents via the ON DELETE
        SET NULL foreign key constraint.

        Args:
            conn: An open aiosqlite connection.
            source_type: The source type string (e.g. 'hn').

        Returns:
            The UUID string of the source row.
        """
        # Check if a source row already exists for this source_type/identifier
        async with conn.execute(
            "SELECT id FROM sources WHERE source_type = ? AND identifier = ?",
            (source_type, source_type),
        ) as cursor:
            row = await cursor.fetchone()

        if row is not None:
            return row[0]

        # No existing row — create a new one
        source_id = str(uuid.uuid4())
        source_row = SourceRow(
            id=source_id,
            source_type=source_type,
            identifier=source_type,
            display_name=source_type.upper(),
        )
        await upsert_source(conn, source_row)
        return source_id

    async def run_all(self) -> tuple[list[IngestReport], list[str]]:
        """Run all enabled ingestors in sequence. Return reports and skipped sources.

        Sources listed in ``config.sources.disabled`` are skipped and logged.
        Ingestors are run sequentially rather than concurrently to avoid
        overloading SQLite with concurrent writes.

        Returns:
            A tuple of (reports, skipped) where *reports* is a list of
            IngestReport objects (one per source that was actually run) and
            *skipped* is a list of source type keys that were disabled.
        """
        disabled = set(self.config.sources.disabled)
        reports: list[IngestReport] = []
        skipped: list[str] = []

        for source_type, ingestor_cls in INGESTORS.items():
            if source_type in disabled:
                logger.info("Skipping disabled source: %s", source_type)
                skipped.append(source_type)
                continue
            try:
                ingestor = ingestor_cls(self.config)
                report = await self.run_source(ingestor)
                reports.append(report)
            except Exception as e:
                logger.error("Ingestor %s failed to initialize: %s", source_type, e)
                reports.append(IngestReport(
                    source_type=source_type,
                    errors=[f"init failed: {e}"],
                ))
        return reports, skipped

    async def ingest_url(
        self,
        url: str,
        tags: list[str] | None = None,
    ) -> IngestReport:
        """Ingest a single URL using AdhocIngestor. Skip content filter.

        Detects the URL type (YouTube, ArXiv, article) and delegates to the
        appropriate handler. Stores the document with origin='adhoc'.

        Unlike run_source(), this method does NOT apply content filtering —
        the user explicitly chose to ingest this URL, so filtering would be
        counterproductive.

        Args:
            url: The URL to ingest.
            tags: Optional list of user-supplied tag strings applied to the document.

        Returns:
            An IngestReport with source_type='adhoc' indicating the outcome.
            report.stored == 1 on success, report.skipped_duplicate == 1 if the
            URL was already in the database.
        """
        from .adhoc import AdhocIngestor

        report = IngestReport(source_type="adhoc")
        data_dir = self.db_path.parent

        ingestor = AdhocIngestor(self.config)
        try:
            doc = await ingestor.ingest_url(url, tags=tags)
            report.fetched = 1
        except Exception as e:
            msg = f"ingest_url failed for {url}: {e}"
            logger.error("[adhoc] %s", msg)
            report.errors.append(msg)
            return report

        async with get_db(data_dir) as conn:
            # Deduplicate: skip if already in DB
            existing = await get_document_by_url(conn, doc.url)
            if existing is not None:
                report.skipped_duplicate = 1
                return report

            # Convert to DB row
            doc_row = doc.to_document_row(source_id=None)

            # Apply user tags from adhoc_tags metadata field
            adhoc_tags = doc.metadata.get("adhoc_tags", [])
            if adhoc_tags:
                doc_row = doc_row.model_copy(update={"user_tags": adhoc_tags})

            await upsert_document(conn, doc_row)
            await conn.commit()
            report.stored = 1

        return report


def get_ingestor(source_type: str, config: AppConfig) -> BaseIngestor:
    """Factory: return an ingestor by source_type string.

    Args:
        source_type: The source type key (e.g. 'hn').
        config: The application configuration.

    Returns:
        An instantiated BaseIngestor subclass.

    Raises:
        ValueError: For unknown source types.
    """
    cls = INGESTORS.get(source_type)
    if cls is None:
        raise ValueError(
            f"Unknown source type: {source_type!r}. "
            f"Available: {list(INGESTORS.keys())}"
        )
    return cls(config)
