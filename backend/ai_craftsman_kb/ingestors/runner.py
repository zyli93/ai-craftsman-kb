"""IngestRunner: orchestrates fetch -> filter -> deduplicate -> store."""
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..config.models import AppConfig
from ..db.models import SourceRow
from ..db.queries import (
    get_document_by_url,
    update_source_fetch_status,
    upsert_document,
    upsert_source,
)
from ..db.sqlite import get_db
from ..llm.router import LLMRouter
from ..processing.filter import ContentFilter
from .base import BaseIngestor
from .hackernews import HackerNewsIngestor

logger = logging.getLogger(__name__)


# Registry of available ingestors — extended by later tasks (10-15)
INGESTORS: dict[str, type[BaseIngestor]] = {
    "hn": HackerNewsIngestor,
}


class IngestReport(BaseModel):
    """Summary of one ingestor run."""

    source_type: str
    fetched: int = 0
    passed_filter: int = 0
    stored: int = 0
    skipped_duplicate: int = 0
    errors: list[str] = []


class IngestRunner:
    """Orchestrates: fetch -> filter -> deduplicate -> store.

    Usage::

        runner = IngestRunner(config, llm_router, db_path)
        report = await runner.run_source(HackerNewsIngestor(config))

    Args:
        config: Fully loaded AppConfig.
        llm_router: Configured LLM router for filter scoring.
        db_path: Path to the craftsman.db file. The parent directory is used
                 as the data_dir for get_db().
    """

    def __init__(
        self,
        config: AppConfig,
        llm_router: LLMRouter,
        db_path: Path,
    ) -> None:
        self.config = config
        self.llm_router = llm_router
        self.db_path = db_path
        self._filter = ContentFilter(config, llm_router)

    async def run_source(
        self,
        ingestor: BaseIngestor,
        origin: Literal["pro", "radar", "adhoc"] = "pro",
    ) -> IngestReport:
        """Run one ingestor: fetch -> filter -> dedup -> store.

        Steps:
        1. Call ingestor.fetch_pro() to get raw documents.
        2. Filter each doc via ContentFilter (batch, concurrency=5).
        3. Open DB connection and upsert the source row so it is tracked.
        4. Deduplicate by URL against existing DB records.
        5. For passed docs: call ingestor.fetch_content() to fill raw_content.
        6. Store to DB via upsert_document().
        7. Update source last_fetched_at timestamp.

        Args:
            ingestor: The ingestor instance to run.
            origin: Ingest origin label stored on each document row.

        Returns:
            An IngestReport summarising the run counts and any errors.
        """
        report = IngestReport(source_type=ingestor.source_type)
        data_dir = self.db_path.parent

        # Step 1: Fetch raw documents from the source
        try:
            raw_docs = await ingestor.fetch_pro()
            report.fetched = len(raw_docs)
        except Exception as e:
            msg = f"fetch_pro failed: {e}"
            logger.error("[%s] %s", ingestor.source_type, msg)
            report.errors.append(msg)
            return report

        # Step 2: Filter batch — on filter failure, pass all docs through
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
            # Step 3: Ensure source row exists, reusing existing ID if present
            source_id = await self._get_or_create_source_id(
                conn, ingestor.source_type
            )

            now_iso = datetime.now(timezone.utc).isoformat()

            for doc, result in passed_docs:
                try:
                    # Step 4: Deduplicate by URL
                    existing = await get_document_by_url(conn, doc.url)
                    if existing is not None:
                        report.skipped_duplicate += 1
                        continue

                    # Step 5: Fetch full content if not already present
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

                    # Step 6: Convert to DB row and store
                    doc_row = doc.to_document_row(source_id=source_id)
                    if result is not None:
                        doc_row = doc_row.model_copy(update={
                            "filter_score": result.score,
                            "filter_passed": result.passed,
                        })
                    await upsert_document(conn, doc_row)
                    report.stored += 1

                except Exception as e:
                    msg = f"store failed for {doc.url}: {e}"
                    logger.error("[%s] %s", ingestor.source_type, msg)
                    report.errors.append(msg)

            # Step 7: Update source last_fetched_at timestamp
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

    async def run_all(self) -> list[IngestReport]:
        """Run all enabled ingestors in sequence. Return one report per source.

        Ingestors are run sequentially rather than concurrently to avoid
        overloading SQLite with concurrent writes.

        Returns:
            A list of IngestReport objects, one per registered source type.
        """
        reports: list[IngestReport] = []
        for source_type, ingestor_cls in INGESTORS.items():
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
        return reports


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
