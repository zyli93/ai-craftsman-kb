"""Unit tests for ProcessingPipeline — post-ingest chunk/embed/entity pipeline.

All external dependencies (Embedder, VectorStore, EntityExtractor) are mocked
via AsyncMock / MagicMock so no real API calls or Qdrant client is required.
DB operations use an in-memory aiosqlite connection with the full schema.
"""
from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from ai_craftsman_kb.llm import CompletionResult

import aiosqlite
import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    EmbeddingConfig,
    FiltersConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    SettingsConfig,
    SourcesConfig,
)
from ai_craftsman_kb.db.models import DocumentRow
from ai_craftsman_kb.db.sqlite import SCHEMA_SQL
from ai_craftsman_kb.processing.chunker import Chunker, TextChunk
from ai_craftsman_kb.processing.embedder import Embedder, EmbeddingResult
from ai_craftsman_kb.processing.entity_extractor import EntityExtractor
from ai_craftsman_kb.processing.keyword_extractor import KeywordExtractor
from ai_craftsman_kb.processing.pipeline import ProcessingPipeline, ProcessingReport
from ai_craftsman_kb.search.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _make_config() -> AppConfig:
    """Build a minimal AppConfig for pipeline tests."""
    return AppConfig(
        sources=SourcesConfig(),
        settings=SettingsConfig(
            data_dir="/tmp/test-pipeline-kb",
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test"),
                briefing=LLMTaskConfig(provider="anthropic", model="test"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test"),
                keyword_extraction=LLMTaskConfig(provider="openrouter", model="test-model"),
            ),
        ),
        filters=FiltersConfig(),
    )


def _make_doc_id() -> str:
    """Return a fresh UUID string for use as a document ID."""
    return str(uuid.uuid4())


def _make_document(
    raw_content: str | None = None,
    word_count: int | None = None,
    is_embedded: bool = False,
    is_entities_extracted: bool = False,
    title: str = "Test Title",
    source_type: str = "hn",
    origin: str = "pro",
) -> DocumentRow:
    """Create a minimal DocumentRow for testing."""
    doc_id = _make_doc_id()
    effective_wc = word_count
    if effective_wc is None and raw_content is not None:
        effective_wc = len(raw_content.split())
    return DocumentRow(
        id=doc_id,
        source_type=source_type,
        origin=origin,  # type: ignore[arg-type]
        url=f"https://example.com/{doc_id}",
        title=title,
        raw_content=raw_content,
        word_count=effective_wc,
        is_embedded=is_embedded,
        is_entities_extracted=is_entities_extracted,
    )


def _make_long_content(word_count: int = 100) -> str:
    """Generate a content string with approximately word_count words."""
    # Use a realistic sentence rather than a repeated single word so
    # the chunker sees varied sentences and produces chunks.
    sentence = "The quick brown fox jumps over the lazy dog. "
    words_per_sentence = len(sentence.split())
    repeats = max(1, word_count // words_per_sentence + 1)
    return (sentence * repeats).strip()


async def _make_in_memory_db() -> aiosqlite.Connection:
    """Create a fully-initialised in-memory SQLite DB with the full schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.executescript(SCHEMA_SQL)
    await conn.commit()
    return conn


async def _insert_test_document(
    conn: aiosqlite.Connection, doc: DocumentRow
) -> None:
    """Insert a DocumentRow into the in-memory DB so FK constraints pass."""
    await conn.execute(
        """
        INSERT INTO documents (
            id, source_id, origin, source_type, url, title, raw_content,
            word_count, is_embedded, is_entities_extracted
        )
        VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc.id,
            doc.origin,
            doc.source_type,
            doc.url,
            doc.title,
            doc.raw_content,
            doc.word_count,
            doc.is_embedded,
            doc.is_entities_extracted,
        ),
    )
    await conn.commit()


def _make_fake_chunks(n: int = 2) -> list[TextChunk]:
    """Create n fake TextChunk objects."""
    return [
        TextChunk(
            chunk_index=i,
            text=f"chunk text {i} " + "word " * 60,
            token_count=70,
            char_start=i * 100,
            char_end=(i + 1) * 100,
        )
        for i in range(n)
    ]


def _make_fake_embeddings(n: int = 2, dim: int = 1536) -> list[EmbeddingResult]:
    """Create n fake EmbeddingResult objects."""
    return [
        EmbeddingResult(
            text=f"chunk text {i}",
            vector=[0.1] * dim,
            token_count=10,
        )
        for i in range(n)
    ]


def _make_mock_chunker(chunks: list[TextChunk] | None = None) -> MagicMock:
    """Create a mock Chunker that returns the given chunks."""
    if chunks is None:
        chunks = _make_fake_chunks()
    mock = MagicMock(spec=Chunker)
    mock.chunk.return_value = chunks
    return mock


def _make_mock_embedder(embeddings: list[EmbeddingResult] | None = None) -> MagicMock:
    """Create a mock Embedder with async embed_texts."""
    if embeddings is None:
        embeddings = _make_fake_embeddings()
    mock = MagicMock(spec=Embedder)
    mock.embed_texts = AsyncMock(return_value=embeddings)
    return mock


def _make_mock_vector_store() -> MagicMock:
    """Create a mock VectorStore with async upsert_vectors."""
    mock = MagicMock(spec=VectorStore)
    mock.upsert_vectors = AsyncMock(return_value=["point-id-1", "point-id-2"])
    return mock


def _make_mock_entity_extractor() -> MagicMock:
    """Create a mock EntityExtractor with async extract_and_store."""
    mock = MagicMock(spec=EntityExtractor)
    mock.extract_and_store = AsyncMock(return_value=[])
    return mock


def _make_mock_keyword_extractor() -> MagicMock:
    """Create a mock KeywordExtractor with async extract_and_store."""
    mock = MagicMock(spec=KeywordExtractor)
    mock.extract_and_store = AsyncMock(return_value=["python", "machine learning"])
    return mock


def _make_pipeline(
    chunker: MagicMock | None = None,
    embedder: MagicMock | None = None,
    vector_store: MagicMock | None = None,
    entity_extractor: MagicMock | None = None,
    keyword_extractor: MagicMock | None = None,
) -> ProcessingPipeline:
    """Create a ProcessingPipeline with all dependencies mocked."""
    return ProcessingPipeline(
        config=_make_config(),
        embedder=embedder or _make_mock_embedder(),
        chunker=chunker or _make_mock_chunker(),
        vector_store=vector_store or _make_mock_vector_store(),
        entity_extractor=entity_extractor or _make_mock_entity_extractor(),
        keyword_extractor=keyword_extractor,
    )


# ---------------------------------------------------------------------------
# ProcessingReport model
# ---------------------------------------------------------------------------


def test_processing_report_defaults() -> None:
    """ProcessingReport initialises with all-zero counts and empty errors."""
    report = ProcessingReport()
    assert report.total == 0
    assert report.embedded == 0
    assert report.entity_extracted == 0
    assert report.keywords_extracted == 0
    assert report.failed_embedding == 0
    assert report.failed_entities == 0
    assert report.failed_keywords == 0
    assert report.errors == []


def test_processing_report_fields() -> None:
    """ProcessingReport stores explicit field values correctly."""
    report = ProcessingReport(
        total=10,
        embedded=8,
        entity_extracted=7,
        keywords_extracted=6,
        failed_embedding=1,
        failed_entities=2,
        failed_keywords=3,
        errors=["err1", "err2"],
    )
    assert report.total == 10
    assert report.embedded == 8
    assert report.entity_extracted == 7
    assert report.keywords_extracted == 6
    assert report.failed_embedding == 1
    assert report.failed_entities == 2
    assert report.failed_keywords == 3
    assert report.errors == ["err1", "err2"]


# ---------------------------------------------------------------------------
# process_document — skip conditions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_document_skips_already_processed() -> None:
    """process_document skips documents that are both embedded and entities extracted."""
    pipeline = _make_pipeline()
    doc = _make_document(
        raw_content=_make_long_content(200),
        is_embedded=True,
        is_entities_extracted=True,
    )
    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        await pipeline.process_document(conn, doc)
        # Neither chunker nor entity extractor should have been called
        pipeline.chunker.chunk.assert_not_called()
        pipeline.entity_extractor.extract_and_store.assert_not_called()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_document_skips_embedding_if_no_content() -> None:
    """process_document skips embedding when raw_content is None."""
    pipeline = _make_pipeline()
    doc = _make_document(raw_content=None, title="Video without transcript")
    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        await pipeline.process_document(conn, doc)
        # Chunker should not be called — no content to embed
        pipeline.chunker.chunk.assert_not_called()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_document_skips_embedding_if_too_short() -> None:
    """process_document skips embedding when word_count < 50."""
    pipeline = _make_pipeline()
    # 10-word content — well under the 50-word minimum
    doc = _make_document(raw_content="short content here just a few words total ok")
    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        await pipeline.process_document(conn, doc)
        pipeline.chunker.chunk.assert_not_called()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_document_skips_embedding_if_already_embedded() -> None:
    """process_document skips embedding step when is_embedded is already True."""
    pipeline = _make_pipeline()
    doc = _make_document(
        raw_content=_make_long_content(200),
        is_embedded=True,
        is_entities_extracted=False,
    )
    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        await pipeline.process_document(conn, doc)
        # Embedding should be skipped, but entity extraction should run
        pipeline.chunker.chunk.assert_not_called()
        pipeline.entity_extractor.extract_and_store.assert_called_once()
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# process_document — successful embedding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_document_embeds_content() -> None:
    """process_document calls chunker, embedder, and vector_store for new documents."""
    chunks = _make_fake_chunks(3)
    embeddings = _make_fake_embeddings(3)
    chunker = _make_mock_chunker(chunks)
    embedder = _make_mock_embedder(embeddings)
    vector_store = _make_mock_vector_store()
    pipeline = _make_pipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    content = _make_long_content(200)
    doc = _make_document(raw_content=content)

    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        await pipeline.process_document(conn, doc)

        # Chunker should have received the raw content
        chunker.chunk.assert_called_once_with(content)
        # Embedder should have received the chunk texts
        embedder.embed_texts.assert_called_once_with([c.text for c in chunks])
        # VectorStore should have received the document_id, chunks, and vectors
        vector_store.upsert_vectors.assert_called_once()
        call_kwargs = vector_store.upsert_vectors.call_args
        assert call_kwargs[1]["document_id"] == doc.id or call_kwargs[0][0] == doc.id
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_document_updates_is_embedded_flag() -> None:
    """process_document sets is_embedded=True in the DB after successful embedding."""
    chunks = _make_fake_chunks(2)
    embeddings = _make_fake_embeddings(2)
    chunker = _make_mock_chunker(chunks)
    embedder = _make_mock_embedder(embeddings)
    vector_store = _make_mock_vector_store()
    pipeline = _make_pipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    content = _make_long_content(200)
    doc = _make_document(raw_content=content)

    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        await pipeline.process_document(conn, doc)

        # Verify the DB flag was updated
        async with conn.execute(
            "SELECT is_embedded FROM documents WHERE id = ?", (doc.id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert bool(row["is_embedded"]) is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_document_updates_is_entities_extracted_flag() -> None:
    """process_document sets is_entities_extracted=True in DB after extraction."""
    pipeline = _make_pipeline()
    content = _make_long_content(200)
    doc = _make_document(raw_content=content)

    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        # Mock extract_and_store to simulate the DB update
        async def _mock_extract_and_store(conn, document_id, content):  # type: ignore[override]
            from ai_craftsman_kb.db.queries import update_document_flags
            await update_document_flags(conn, doc_id=document_id, is_entities_extracted=True)
            return []

        pipeline.entity_extractor.extract_and_store = AsyncMock(
            side_effect=_mock_extract_and_store
        )

        await pipeline.process_document(conn, doc)

        async with conn.execute(
            "SELECT is_entities_extracted FROM documents WHERE id = ?", (doc.id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert bool(row["is_entities_extracted"]) is True
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# process_document — error isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_failure_does_not_block_entity_extraction() -> None:
    """When embedding fails, entity extraction still runs."""
    chunker = _make_mock_chunker()
    embedder = MagicMock(spec=Embedder)
    # Embedding raises an error
    embedder.embed_texts = AsyncMock(side_effect=RuntimeError("OpenAI API down"))
    entity_extractor = _make_mock_entity_extractor()
    pipeline = _make_pipeline(
        chunker=chunker,
        embedder=embedder,
        entity_extractor=entity_extractor,
    )

    content = _make_long_content(200)
    doc = _make_document(raw_content=content)

    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        # Should not raise — errors are caught internally
        await pipeline.process_document(conn, doc)

        # Entity extraction should still have been called despite embedding failure
        entity_extractor.extract_and_store.assert_called_once()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_entity_extraction_failure_does_not_block_embedding() -> None:
    """When entity extraction fails, embedding still completes."""
    chunks = _make_fake_chunks(2)
    embeddings = _make_fake_embeddings(2)
    chunker = _make_mock_chunker(chunks)
    embedder = _make_mock_embedder(embeddings)
    vector_store = _make_mock_vector_store()
    entity_extractor = MagicMock(spec=EntityExtractor)
    entity_extractor.extract_and_store = AsyncMock(
        side_effect=RuntimeError("LLM timeout")
    )

    pipeline = _make_pipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        entity_extractor=entity_extractor,
    )

    content = _make_long_content(200)
    doc = _make_document(raw_content=content)

    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        # Should not raise
        await pipeline.process_document(conn, doc)

        # Embedding should have succeeded
        async with conn.execute(
            "SELECT is_embedded FROM documents WHERE id = ?", (doc.id,)
        ) as cur:
            row = await cur.fetchone()
        assert bool(row["is_embedded"]) is True
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# process_batch — counts and concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_batch_returns_report() -> None:
    """process_batch returns a ProcessingReport with correct total count."""
    pipeline = _make_pipeline()
    docs = [
        _make_document(raw_content=_make_long_content(200))
        for _ in range(3)
    ]
    conn = await _make_in_memory_db()
    try:
        for doc in docs:
            await _insert_test_document(conn, doc)

        report = await pipeline.process_batch(conn, docs)
        assert report.total == 3
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_batch_empty_list_returns_zero_report() -> None:
    """process_batch with an empty list returns a zero ProcessingReport."""
    pipeline = _make_pipeline()
    conn = await _make_in_memory_db()
    try:
        report = await pipeline.process_batch(conn, [])
        assert report.total == 0
        assert report.embedded == 0
        assert report.entity_extracted == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_batch_embeds_all_eligible_docs() -> None:
    """process_batch embeds all documents with sufficient content."""
    chunks = _make_fake_chunks(2)
    embeddings = _make_fake_embeddings(2)

    # Track how many times embed_texts is called
    embed_call_count = 0

    async def _counting_embed(texts: list[str]) -> list[EmbeddingResult]:
        nonlocal embed_call_count
        embed_call_count += 1
        return _make_fake_embeddings(len(texts))

    chunker = MagicMock(spec=Chunker)
    chunker.chunk.return_value = chunks

    embedder = MagicMock(spec=Embedder)
    embedder.embed_texts = AsyncMock(side_effect=_counting_embed)

    vector_store = _make_mock_vector_store()

    pipeline = _make_pipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    docs = [
        _make_document(raw_content=_make_long_content(200))
        for _ in range(4)
    ]

    conn = await _make_in_memory_db()
    try:
        for doc in docs:
            await _insert_test_document(conn, doc)

        report = await pipeline.process_batch(conn, docs, concurrency=3)

        # All 4 documents should have been embedded
        assert report.embedded == 4
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_batch_skips_docs_without_content() -> None:
    """process_batch does not count documents with no raw_content as embedded."""
    pipeline = _make_pipeline()

    # Mix of docs with and without content
    docs_with_content = [
        _make_document(raw_content=_make_long_content(200))
        for _ in range(2)
    ]
    docs_without_content = [
        _make_document(raw_content=None, title="YouTube video no transcript")
        for _ in range(2)
    ]
    all_docs = docs_with_content + docs_without_content

    conn = await _make_in_memory_db()
    try:
        for doc in all_docs:
            await _insert_test_document(conn, doc)

        report = await pipeline.process_batch(conn, all_docs)

        assert report.total == 4
        # Only the 2 docs with content should be embedded
        assert report.embedded == 2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_batch_respects_concurrency_limit() -> None:
    """process_batch honours the concurrency parameter via asyncio.Semaphore."""
    max_concurrent = 0
    current_concurrent = 0

    original_embed = _make_fake_embeddings

    async def _tracking_embed(texts: list[str]) -> list[EmbeddingResult]:
        nonlocal max_concurrent, current_concurrent
        current_concurrent += 1
        max_concurrent = max(max_concurrent, current_concurrent)
        # Yield to allow other coroutines to run
        import asyncio as _asyncio
        await _asyncio.sleep(0)
        current_concurrent -= 1
        return _make_fake_embeddings(len(texts))

    chunker = MagicMock(spec=Chunker)
    chunker.chunk.return_value = _make_fake_chunks(2)

    embedder = MagicMock(spec=Embedder)
    embedder.embed_texts = AsyncMock(side_effect=_tracking_embed)

    vector_store = _make_mock_vector_store()
    pipeline = _make_pipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    docs = [
        _make_document(raw_content=_make_long_content(200))
        for _ in range(6)
    ]

    conn = await _make_in_memory_db()
    try:
        for doc in docs:
            await _insert_test_document(conn, doc)

        await pipeline.process_batch(conn, docs, concurrency=2)
        # Concurrency should never have exceeded 2
        assert max_concurrent <= 2
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# process_batch — failed counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_batch_counts_failed_embedding() -> None:
    """process_batch counts documents that failed embedding in failed_embedding."""
    chunker = MagicMock(spec=Chunker)
    chunker.chunk.return_value = _make_fake_chunks(2)

    embedder = MagicMock(spec=Embedder)
    embedder.embed_texts = AsyncMock(side_effect=RuntimeError("API error"))

    pipeline = _make_pipeline(chunker=chunker, embedder=embedder)

    docs = [
        _make_document(raw_content=_make_long_content(200))
        for _ in range(3)
    ]

    conn = await _make_in_memory_db()
    try:
        for doc in docs:
            await _insert_test_document(conn, doc)

        report = await pipeline.process_batch(conn, docs)

        assert report.embedded == 0
        assert report.failed_embedding == 3
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# reprocess_unembedded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reprocess_unembedded_finds_unembedded_docs() -> None:
    """reprocess_unembedded finds and processes documents with is_embedded=False."""
    chunks = _make_fake_chunks(2)
    embeddings = _make_fake_embeddings(2)
    chunker = _make_mock_chunker(chunks)
    embedder = _make_mock_embedder(embeddings)
    vector_store = _make_mock_vector_store()
    pipeline = _make_pipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    conn = await _make_in_memory_db()
    try:
        # Insert 3 unembedded docs and 1 already-embedded doc
        unembedded_docs = [
            _make_document(raw_content=_make_long_content(200), is_embedded=False)
            for _ in range(3)
        ]
        embedded_doc = _make_document(
            raw_content=_make_long_content(200),
            is_embedded=True,
            is_entities_extracted=True,
        )

        for doc in unembedded_docs + [embedded_doc]:
            await _insert_test_document(conn, doc)

        report = await pipeline.reprocess_unembedded(conn)

        # Should have processed the 3 unembedded docs
        assert report.total == 3
        assert report.embedded == 3
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_reprocess_unembedded_skips_already_embedded() -> None:
    """reprocess_unembedded does not re-process docs that are already embedded."""
    pipeline = _make_pipeline()
    conn = await _make_in_memory_db()
    try:
        # All docs already embedded
        docs = [
            _make_document(
                raw_content=_make_long_content(200),
                is_embedded=True,
                is_entities_extracted=True,
            )
            for _ in range(3)
        ]
        for doc in docs:
            await _insert_test_document(conn, doc)

        report = await pipeline.reprocess_unembedded(conn)

        # Nothing to process
        assert report.total == 0
        assert report.embedded == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_reprocess_unembedded_empty_db_returns_zero_report() -> None:
    """reprocess_unembedded on empty DB returns a zero ProcessingReport."""
    pipeline = _make_pipeline()
    conn = await _make_in_memory_db()
    try:
        report = await pipeline.reprocess_unembedded(conn)
        assert report.total == 0
        assert report.embedded == 0
        assert report.entity_extracted == 0
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# IngestRunner integration — pipeline wired in
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_runner_calls_pipeline_after_store() -> None:
    """IngestRunner calls pipeline.process_batch() after storing new documents."""
    from ai_craftsman_kb.ingestors.runner import IngestRunner

    # Build a fake pipeline that records what it was called with
    mock_pipeline = MagicMock(spec=ProcessingPipeline)
    processing_report = ProcessingReport(total=1, embedded=1, entity_extracted=1)
    mock_pipeline.process_batch = AsyncMock(return_value=processing_report)

    config = _make_config()
    llm_router = MagicMock()
    llm_router.complete = AsyncMock(return_value=CompletionResult(text="5"))  # Filter returns score 5

    import tempfile
    from pathlib import Path

    from ai_craftsman_kb.db.sqlite import init_db

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "craftsman.db"
        await init_db(Path(tmpdir))

        runner = IngestRunner(
            config=config,
            llm_router=llm_router,
            db_path=db_path,
            pipeline=mock_pipeline,
        )

        # Create a mock ingestor that returns one document
        from ai_craftsman_kb.db.models import DocumentRow

        class _FakeIngestor:
            source_type = "hn"

            async def fetch_pro(self):  # type: ignore[override]
                # Return a minimal object that mimics a fetched document
                return [_FakeDoc()]

            async def fetch_content(self, doc):  # type: ignore[override]
                return doc

        class _FakeDoc:
            url = "https://news.ycombinator.com/test-1234"
            title = "Test Article"
            filter_score = None
            origin = "pro"
            raw_content = _make_long_content(200)
            word_count = 200

            def model_copy(self, *, update=None):
                return self

            def to_document_row(self, source_id):
                return DocumentRow(
                    id=_make_doc_id(),
                    source_type="hn",
                    origin="pro",  # type: ignore[arg-type]
                    url=self.url,
                    title=self.title,
                    raw_content=self.raw_content,
                    word_count=self.word_count,
                )

        # Patch the ContentFilter so it passes all docs through
        with patch(
            "ai_craftsman_kb.processing.filter.ContentFilter.filter_batch",
            new=AsyncMock(return_value=[]),
        ):
            # With filter returning empty, all docs are "passed" (the else branch)
            # We need filter to return results so pass all docs
            pass

        # Simpler approach: mock the whole filter
        runner._filter = MagicMock()
        runner._filter.filter_batch = AsyncMock(
            return_value=[(_FakeDoc(), None)]  # (doc, None) means pass-through
        )

        report = await runner.run_source(_FakeIngestor())  # type: ignore[arg-type]

        # Pipeline should have been called
        assert mock_pipeline.process_batch.called
        # Report should reflect pipeline counts
        assert report.embedded == 1
        assert report.entities_extracted == 1


@pytest.mark.asyncio
async def test_ingest_runner_without_pipeline_works() -> None:
    """IngestRunner without pipeline still stores documents and returns report."""
    from ai_craftsman_kb.ingestors.runner import IngestRunner

    config = _make_config()
    llm_router = MagicMock()
    llm_router.complete = AsyncMock(return_value=CompletionResult(text="5"))

    import tempfile
    from pathlib import Path

    from ai_craftsman_kb.db.sqlite import init_db

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "craftsman.db"
        await init_db(Path(tmpdir))

        # No pipeline passed
        runner = IngestRunner(
            config=config,
            llm_router=llm_router,
            db_path=db_path,
        )
        assert runner.pipeline is None

        # Run with an ingestor that returns no documents
        class _EmptyIngestor:
            source_type = "hn"

            async def fetch_pro(self):  # type: ignore[override]
                return []

            async def fetch_content(self, doc):  # type: ignore[override]
                return doc

        runner._filter = MagicMock()
        runner._filter.filter_batch = AsyncMock(return_value=[])

        report = await runner.run_source(_EmptyIngestor())  # type: ignore[arg-type]

        assert report.fetched == 0
        assert report.stored == 0
        assert report.embedded == 0
        assert report.entities_extracted == 0
        assert report.keywords_extracted == 0


# ---------------------------------------------------------------------------
# Keyword extraction integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_document_runs_keyword_extraction() -> None:
    """process_document calls keyword_extractor.extract_and_store when configured."""
    kw_extractor = _make_mock_keyword_extractor()
    pipeline = _make_pipeline(keyword_extractor=kw_extractor)
    content = _make_long_content(200)
    doc = _make_document(raw_content=content)

    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        await pipeline.process_document(conn, doc)

        kw_extractor.extract_and_store.assert_called_once_with(
            conn=conn,
            document_id=doc.id,
            content=content,
        )
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_document_skips_keywords_if_already_extracted() -> None:
    """process_document skips keyword extraction when is_keywords_extracted=True."""
    kw_extractor = _make_mock_keyword_extractor()
    pipeline = _make_pipeline(keyword_extractor=kw_extractor)
    doc = _make_document(
        raw_content=_make_long_content(200),
        is_embedded=True,
        is_entities_extracted=True,
    )
    # Manually set is_keywords_extracted on the document
    doc = doc.model_copy(update={"is_keywords_extracted": True})

    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        await pipeline.process_document(conn, doc)

        kw_extractor.extract_and_store.assert_not_called()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_document_skips_keywords_if_no_extractor() -> None:
    """process_document skips keyword extraction when no KeywordExtractor configured."""
    pipeline = _make_pipeline(keyword_extractor=None)
    content = _make_long_content(200)
    doc = _make_document(raw_content=content)

    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        await pipeline.process_document(conn, doc)
        # No error, and no keyword extractor to call
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_document_keyword_extraction_uses_title_fallback() -> None:
    """Keyword extraction falls back to title when raw_content is None."""
    kw_extractor = _make_mock_keyword_extractor()
    pipeline = _make_pipeline(keyword_extractor=kw_extractor)
    doc = _make_document(raw_content=None, title="Interesting AI Article")

    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        await pipeline.process_document(conn, doc)

        kw_extractor.extract_and_store.assert_called_once_with(
            conn=conn,
            document_id=doc.id,
            content="Interesting AI Article",
        )
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_keyword_failure_does_not_block_other_steps() -> None:
    """When keyword extraction fails, embedding and entity extraction still run."""
    kw_extractor = MagicMock(spec=KeywordExtractor)
    kw_extractor.extract_and_store = AsyncMock(
        side_effect=RuntimeError("LLM quota exceeded")
    )
    entity_extractor = _make_mock_entity_extractor()
    chunks = _make_fake_chunks(2)
    embeddings = _make_fake_embeddings(2)
    chunker = _make_mock_chunker(chunks)
    embedder = _make_mock_embedder(embeddings)
    vector_store = _make_mock_vector_store()

    pipeline = _make_pipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        entity_extractor=entity_extractor,
        keyword_extractor=kw_extractor,
    )

    content = _make_long_content(200)
    doc = _make_document(raw_content=content)

    conn = await _make_in_memory_db()
    try:
        await _insert_test_document(conn, doc)
        # Should not raise
        await pipeline.process_document(conn, doc)

        # Embedding and entity extraction should still have run
        async with conn.execute(
            "SELECT is_embedded FROM documents WHERE id = ?", (doc.id,)
        ) as cur:
            row = await cur.fetchone()
        assert bool(row["is_embedded"]) is True

        entity_extractor.extract_and_store.assert_called_once()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_batch_counts_keywords() -> None:
    """process_batch correctly counts keywords_extracted in the report."""
    kw_extractor = _make_mock_keyword_extractor()
    pipeline = _make_pipeline(keyword_extractor=kw_extractor)

    docs = [
        _make_document(raw_content=_make_long_content(200))
        for _ in range(3)
    ]

    conn = await _make_in_memory_db()
    try:
        for doc in docs:
            await _insert_test_document(conn, doc)

        report = await pipeline.process_batch(conn, docs)
        assert report.keywords_extracted == 3
        assert report.failed_keywords == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_batch_counts_failed_keywords() -> None:
    """process_batch counts keyword extraction failures in failed_keywords."""
    kw_extractor = MagicMock(spec=KeywordExtractor)
    kw_extractor.extract_and_store = AsyncMock(
        side_effect=RuntimeError("LLM error")
    )
    pipeline = _make_pipeline(keyword_extractor=kw_extractor)

    docs = [
        _make_document(raw_content=_make_long_content(200))
        for _ in range(2)
    ]

    conn = await _make_in_memory_db()
    try:
        for doc in docs:
            await _insert_test_document(conn, doc)

        report = await pipeline.process_batch(conn, docs)
        assert report.keywords_extracted == 0
        assert report.failed_keywords == 2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_process_batch_no_keyword_extractor_zero_counts() -> None:
    """Without a keyword extractor, keywords_extracted and failed_keywords stay 0."""
    pipeline = _make_pipeline(keyword_extractor=None)
    docs = [
        _make_document(raw_content=_make_long_content(200))
        for _ in range(2)
    ]

    conn = await _make_in_memory_db()
    try:
        for doc in docs:
            await _insert_test_document(conn, doc)

        report = await pipeline.process_batch(conn, docs)
        assert report.keywords_extracted == 0
        assert report.failed_keywords == 0
    finally:
        await conn.close()


def test_ingest_report_has_keywords_extracted_field() -> None:
    """IngestReport includes keywords_extracted field defaulting to 0."""
    from ai_craftsman_kb.ingestors.runner import IngestReport

    report = IngestReport(source_type="hn")
    assert report.keywords_extracted == 0

    report2 = IngestReport(source_type="hn", keywords_extracted=5)
    assert report2.keywords_extracted == 5
