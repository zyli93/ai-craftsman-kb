"""Post-ingest processing pipeline: chunk -> embed -> vector store + entity/keyword extraction.

This module wires together the Chunker, Embedder, VectorStore, EntityExtractor,
and KeywordExtractor into a single orchestrated pipeline that processes documents
after they are stored in SQLite.

Errors in embedding do not block entity or keyword extraction, and vice versa —
each step is independently fault-tolerant. Document flags (is_embedded,
is_entities_extracted, is_keywords_extracted) are updated in the DB after each
successful step.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import aiosqlite
from pydantic import BaseModel

from ..db.models import DocumentRow
from ..db.queries import update_document_flags

if TYPE_CHECKING:
    from ..config.models import AppConfig
    from ..processing.chunker import Chunker
    from ..processing.embedder import Embedder
    from ..processing.entity_extractor import EntityExtractor
    from ..processing.keyword_extractor import KeywordExtractor
    from ..search.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Minimum word count to attempt embedding; documents shorter than this are
# likely stubs or YouTube videos without transcripts.
_MIN_WORD_COUNT = 50


class ProcessingReport(BaseModel):
    """Summary of a batch processing run.

    Attributes:
        total: Total number of documents submitted for processing.
        embedded: Number of documents successfully embedded into Qdrant.
        entity_extracted: Number of documents that had entity extraction run.
        keywords_extracted: Number of documents that had keyword extraction run.
        failed_embedding: Number of documents where embedding failed.
        failed_entities: Number of documents where entity extraction failed.
        failed_keywords: Number of documents where keyword extraction failed.
        errors: List of error message strings for failed operations.
    """

    total: int = 0
    embedded: int = 0
    entity_extracted: int = 0
    keywords_extracted: int = 0
    failed_embedding: int = 0
    failed_entities: int = 0
    failed_keywords: int = 0
    errors: list[str] = []


class ProcessingPipeline:
    """Post-ingest processing: chunk -> embed -> vector store + entity/keyword extraction.

    Orchestrates the full processing flow for newly-ingested documents.
    Designed to be called from IngestRunner after documents are stored in SQLite.

    Processing per document:
    1. Skip entirely if is_embedded, is_entities_extracted, and
       is_keywords_extracted are all True.
    2. If not embedded: chunk text -> embed chunks -> upsert to Qdrant
       -> update document.is_embedded = True in DB.
    3. If not entities extracted: extract entities -> store to DB
       -> update document.is_entities_extracted = True in DB.
    4. If not keywords extracted and a KeywordExtractor is configured:
       extract keywords -> store to DB
       -> update document.is_keywords_extracted = True in DB.

    Errors in any step are logged but do not fail the other steps.

    Usage::

        pipeline = ProcessingPipeline(
            config=config,
            embedder=embedder,
            chunker=chunker,
            vector_store=vector_store,
            entity_extractor=entity_extractor,
            keyword_extractor=keyword_extractor,
        )
        report = await pipeline.process_batch(conn, documents)

    Args:
        config: Fully loaded AppConfig.
        embedder: Configured Embedder instance for generating vectors.
        chunker: Chunker instance for splitting text into chunks.
        vector_store: VectorStore instance for persisting vectors to Qdrant.
        entity_extractor: EntityExtractor instance for LLM-based entity extraction.
        keyword_extractor: Optional KeywordExtractor instance for LLM-based
            keyword extraction. When None, the keyword extraction step is skipped.
    """

    def __init__(
        self,
        config: "AppConfig",
        embedder: "Embedder",
        chunker: "Chunker",
        vector_store: "VectorStore",
        entity_extractor: "EntityExtractor | None" = None,
        keyword_extractor: "KeywordExtractor | None" = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.chunker = chunker
        self.vector_store = vector_store
        self.entity_extractor = entity_extractor
        self.keyword_extractor = keyword_extractor

    async def process_document(
        self,
        conn: aiosqlite.Connection,
        document: DocumentRow,
    ) -> None:
        """Run full processing for one document.

        Processing steps:
        1. Skip if already fully processed (is_embedded AND
           is_entities_extracted AND is_keywords_extracted or no extractor).
        2. If not embedded: chunk -> embed -> upsert to Qdrant
           -> update document.is_embedded = True in DB.
        3. If not entities extracted: extract entities -> store to DB
           -> update document.is_entities_extracted = True in DB.
        4. If not keywords extracted (and KeywordExtractor is configured):
           extract keywords -> store to DB
           -> update document.is_keywords_extracted = True in DB.

        Errors in any step are logged but do not fail the other steps.
        Documents with None raw_content or fewer than 50 words are skipped
        for embedding (but entity/keyword extraction may still run on their title).

        Args:
            conn: Open aiosqlite connection with row_factory configured.
            document: The DocumentRow to process.
        """
        entities_done = document.is_entities_extracted or self.entity_extractor is None
        keywords_done = document.is_keywords_extracted or self.keyword_extractor is None
        if document.is_embedded and entities_done and keywords_done:
            logger.debug(
                "Document %s already fully processed — skipping",
                document.id,
            )
            return

        # Step 2: Embedding
        if not document.is_embedded:
            await self._embed_document(conn, document)

        # Step 3: Entity extraction
        if self.entity_extractor is not None and not document.is_entities_extracted:
            await self._extract_entities(conn, document)

        # Step 4: Keyword extraction
        if self.keyword_extractor is not None and not document.is_keywords_extracted:
            await self._extract_keywords(conn, document)

    async def _embed_document(
        self,
        conn: aiosqlite.Connection,
        document: DocumentRow,
    ) -> bool:
        """Chunk, embed, and store vectors for a document.

        Skips documents with no raw_content or fewer than 50 words.
        On success, updates is_embedded=True in the DB.

        Args:
            conn: Open aiosqlite connection.
            document: The document to embed.

        Returns:
            True if embedding was performed and succeeded, False otherwise.
        """
        # Skip if no content to embed
        if not document.raw_content:
            logger.info(
                "Document %s (%s) has no raw_content — skipping embedding",
                document.id,
                document.title or "untitled",
            )
            return False

        # Skip if content is too short (< 50 words)
        word_count = document.word_count
        if word_count is None:
            word_count = len(document.raw_content.split())
        if word_count < _MIN_WORD_COUNT:
            logger.info(
                "Document %s (%s) word_count=%d < %d — skipping embedding",
                document.id,
                document.title or "untitled",
                word_count,
                _MIN_WORD_COUNT,
            )
            return False

        try:
            # Chunk the raw content
            chunks = self.chunker.chunk(document.raw_content)
            if not chunks:
                logger.info(
                    "Document %s (%s) produced 0 chunks — skipping embedding",
                    document.id,
                    document.title or "untitled",
                )
                return False

            provider = getattr(
                getattr(self.embedder, "embedding_cfg", None), "provider", "unknown"
            )
            logger.info(
                "Document %s: chunked into %d pieces, embedding via %s...",
                document.id,
                len(chunks),
                provider,
            )

            # Embed all chunk texts in one batch call
            chunk_texts = [c.text for c in chunks]
            embedding_results = await self.embedder.embed_texts(chunk_texts)
            vectors = [er.vector for er in embedding_results]

            # Build the payload stored alongside each vector in Qdrant
            payload = {
                "source_type": document.source_type,
                "origin": document.origin,
                "title": document.title,
                "author": document.author,
                "published_at": document.published_at,
            }

            # Upsert all chunk vectors into Qdrant
            await self.vector_store.upsert_vectors(
                document_id=document.id,
                chunks=chunks,
                vectors=vectors,
                payload=payload,
            )

            # Mark document as embedded in the DB
            await update_document_flags(conn, doc_id=document.id, is_embedded=True)
            logger.info(
                "Embedded document %s: %d chunks into Qdrant",
                document.id,
                len(chunks),
            )
            return True

        except Exception as exc:
            logger.error(
                "Embedding failed for document %s: %s",
                document.id,
                exc,
            )
            return False

    async def _extract_entities(
        self,
        conn: aiosqlite.Connection,
        document: DocumentRow,
    ) -> bool:
        """Run entity extraction and store results for a document.

        Uses raw_content if available, otherwise falls back to title.
        On success (even with zero entities found), updates
        is_entities_extracted=True in the DB.

        Args:
            conn: Open aiosqlite connection.
            document: The document to extract entities from.

        Returns:
            True if extraction was performed (regardless of entity count),
            False if there was no content to extract from.
        """
        # Determine the content to extract from
        content = document.raw_content or document.title
        if not content:
            logger.debug(
                "Document %s has no content or title — skipping entity extraction",
                document.id,
            )
            return False

        try:
            await self.entity_extractor.extract_and_store(
                conn=conn,
                document_id=document.id,
                content=content,
            )
            logger.info(
                "Entity extraction complete for document %s",
                document.id,
            )
            return True

        except Exception as exc:
            logger.error(
                "Entity extraction failed for document %s: %s",
                document.id,
                exc,
            )
            return False

    async def _extract_keywords(
        self,
        conn: aiosqlite.Connection,
        document: DocumentRow,
    ) -> bool:
        """Run keyword extraction and store results for a document.

        Uses raw_content if available, otherwise falls back to title.
        On success (even with zero keywords found), updates
        is_keywords_extracted=True in the DB.

        Gracefully degrades if the LLM fails -- the KeywordExtractor itself
        returns an empty list on LLM errors, so the document is still marked
        as processed to avoid endless retries.

        Args:
            conn: Open aiosqlite connection.
            document: The document to extract keywords from.

        Returns:
            True if extraction was performed (regardless of keyword count),
            False if there was no content to extract from or an error occurred.
        """
        content = document.raw_content or document.title
        if not content:
            logger.debug(
                "Document %s has no content or title — skipping keyword extraction",
                document.id,
            )
            return False

        try:
            assert self.keyword_extractor is not None  # guarded by caller
            await self.keyword_extractor.extract_and_store(
                conn=conn,
                document_id=document.id,
                content=content,
            )
            logger.info(
                "Keyword extraction complete for document %s",
                document.id,
            )
            return True

        except Exception as exc:
            logger.error(
                "Keyword extraction failed for document %s: %s",
                document.id,
                exc,
            )
            return False

    async def process_batch(
        self,
        conn: aiosqlite.Connection,
        documents: list[DocumentRow],
        concurrency: int = 3,
    ) -> ProcessingReport:
        """Process multiple documents concurrently.

        Runs process_document() for each document in the list, respecting
        the concurrency limit via an asyncio.Semaphore. Counts results
        per outcome type into a ProcessingReport.

        Args:
            conn: Open aiosqlite connection with row_factory configured.
            documents: List of DocumentRow instances to process.
            concurrency: Maximum number of documents processed simultaneously.
                Default 3 is conservative to avoid overwhelming the embedding API.

        Returns:
            A ProcessingReport with counts for embedded, entity_extracted,
            keywords_extracted, failed_embedding, failed_entities,
            failed_keywords, and error messages.
        """
        report = ProcessingReport(total=len(documents))

        if not documents:
            logger.info("process_batch called with 0 documents — nothing to do")
            return report

        logger.info(
            "Processing batch of %d documents (concurrency=%d)",
            len(documents),
            concurrency,
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def _process_one(doc: DocumentRow) -> dict[str, bool | str]:
            """Process a single document under the semaphore.

            Returns a dict with 'embedded', 'entity_extracted',
            'keywords_extracted', and optionally 'errors' keys to communicate
            the per-document outcome back to the aggregator.
            """
            async with semaphore:
                outcome: dict[str, bool | str] = {
                    "embedded": False,
                    "entity_extracted": False,
                    "keywords_extracted": False,
                }

                entities_done = (
                    doc.is_entities_extracted
                    or self.entity_extractor is None
                )
                keywords_done = (
                    doc.is_keywords_extracted
                    or self.keyword_extractor is None
                )
                if doc.is_embedded and entities_done and keywords_done:
                    # Already fully processed — skip silently
                    return outcome

                # Track pre-state to detect changes from process_document
                was_embedded = doc.is_embedded
                was_extracted = doc.is_entities_extracted
                was_keywords = doc.is_keywords_extracted

                errors: list[str] = []

                # Embedding step
                if not was_embedded:
                    try:
                        embed_success = await self._embed_document(conn, doc)
                        outcome["embedded"] = embed_success
                    except Exception as exc:
                        msg = f"embed error for {doc.id}: {exc}"
                        logger.error(msg)
                        errors.append(msg)

                # Entity extraction step (independent of embedding)
                if self.entity_extractor is not None and not was_extracted:
                    try:
                        extract_success = await self._extract_entities(conn, doc)
                        outcome["entity_extracted"] = extract_success
                    except Exception as exc:
                        msg = f"entity error for {doc.id}: {exc}"
                        logger.error(msg)
                        errors.append(msg)

                # Keyword extraction step (independent of other steps)
                if self.keyword_extractor is not None and not was_keywords:
                    try:
                        kw_success = await self._extract_keywords(conn, doc)
                        outcome["keywords_extracted"] = kw_success
                    except Exception as exc:
                        msg = f"keyword error for {doc.id}: {exc}"
                        logger.error(msg)
                        errors.append(msg)

                if errors:
                    outcome["errors"] = "; ".join(errors)

                return outcome

        # Run all documents concurrently (limited by semaphore)
        tasks = [_process_one(doc) for doc in documents]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for i, outcome in enumerate(outcomes):
            doc = documents[i]

            if isinstance(outcome, Exception):
                msg = f"unexpected error processing {doc.id}: {outcome}"
                logger.error(msg)
                report.errors.append(msg)
                # Count as failures for all steps (we don't know which succeeded)
                report.failed_embedding += 1
                report.failed_entities += 1
                if self.keyword_extractor is not None:
                    report.failed_keywords += 1
                continue

            # outcome is a dict — tally results
            if outcome.get("embedded"):
                report.embedded += 1
            elif not doc.is_embedded and doc.raw_content:
                # Had content but embedding did not succeed
                report.failed_embedding += 1

            if outcome.get("entity_extracted"):
                report.entity_extracted += 1
            elif not doc.is_entities_extracted and (doc.raw_content or doc.title):
                # Had content but entity extraction did not succeed
                report.failed_entities += 1

            if outcome.get("keywords_extracted"):
                report.keywords_extracted += 1
            elif (
                self.keyword_extractor is not None
                and not doc.is_keywords_extracted
                and (doc.raw_content or doc.title)
            ):
                # Had content but keyword extraction did not succeed
                report.failed_keywords += 1

            if "errors" in outcome:
                report.errors.append(str(outcome["errors"]))

        logger.info(
            "Batch complete: %d/%d embedded, %d/%d entities, %d/%d keywords, "
            "%d embed failures, %d errors",
            report.embedded,
            report.total,
            report.entity_extracted,
            report.total,
            report.keywords_extracted,
            report.total,
            report.failed_embedding,
            len(report.errors),
        )
        return report

    async def reprocess_unembedded(
        self,
        conn: aiosqlite.Connection,
    ) -> ProcessingReport:
        """Find all documents where is_embedded=False and process them.

        Used as a catch-up mechanism after embedding pipeline failures.
        Fetches documents in batches of 100 to avoid memory pressure,
        then processes each batch via process_batch().

        Args:
            conn: Open aiosqlite connection with row_factory configured.

        Returns:
            A ProcessingReport summing all batch results.
        """
        # Collect all unembedded documents in batches to avoid loading
        # the entire table into memory at once.
        total_report = ProcessingReport()
        offset = 0
        batch_size = 100

        while True:
            # Query documents that are not yet embedded and not deleted
            async with conn.execute(
                """
                SELECT * FROM documents
                WHERE is_embedded = FALSE
                  AND deleted_at IS NULL
                ORDER BY fetched_at DESC
                LIMIT ? OFFSET ?
                """,
                (batch_size, offset),
            ) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                break

            # Convert rows to DocumentRow models (handle JSON columns)
            from ..db.queries import _row_to_dict  # local import to avoid circular

            docs: list[DocumentRow] = []
            for row in rows:
                row_dict = _row_to_dict(row)
                docs.append(DocumentRow(**row_dict))

            logger.info(
                "reprocess_unembedded: processing batch of %d documents (offset=%d)",
                len(docs),
                offset,
            )

            batch_report = await self.process_batch(conn, docs)

            # Aggregate counters
            total_report.total += batch_report.total
            total_report.embedded += batch_report.embedded
            total_report.entity_extracted += batch_report.entity_extracted
            total_report.keywords_extracted += batch_report.keywords_extracted
            total_report.failed_embedding += batch_report.failed_embedding
            total_report.failed_entities += batch_report.failed_entities
            total_report.failed_keywords += batch_report.failed_keywords
            total_report.errors.extend(batch_report.errors)

            if len(rows) < batch_size:
                # Last batch — we've processed all unembedded documents
                break

            offset += batch_size

        logger.info(
            "reprocess_unembedded complete: total=%d embedded=%d entity_extracted=%d "
            "keywords_extracted=%d failed_embedding=%d failed_entities=%d "
            "failed_keywords=%d",
            total_report.total,
            total_report.embedded,
            total_report.entity_extracted,
            total_report.keywords_extracted,
            total_report.failed_embedding,
            total_report.failed_entities,
            total_report.failed_keywords,
        )
        return total_report
