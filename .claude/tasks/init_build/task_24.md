# Task 24: Auto-Embed + Extract on Ingest Hook

## Wave
Wave 9 (sequential — depends on tasks 18, 19, 20, 22)
Domain: backend

## Objective
Wire the post-ingest processing pipeline: after a document is stored to SQLite, automatically chunk it, embed the chunks into Qdrant, and extract entities — triggered as part of `IngestRunner.run_source()`.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/processing/pipeline.py` — `ProcessingPipeline` orchestrator
- `backend/ai_craftsman_kb/ingestors/runner.py` — Integrate pipeline into `run_source()`
- `backend/tests/test_processing/test_pipeline.py`

### Key interfaces / implementation details:

**`ProcessingPipeline`** (`processing/pipeline.py`):
```python
class ProcessingPipeline:
    """Post-ingest processing: chunk → embed → vector store + entity extraction."""

    def __init__(
        self,
        config: AppConfig,
        embedder: Embedder,
        chunker: Chunker,
        vector_store: VectorStore,
        entity_extractor: EntityExtractor,
    ) -> None: ...

    async def process_document(
        self,
        conn: aiosqlite.Connection,
        document: DocumentRow,
    ) -> None:
        """Full processing for one document:
        1. Skip if already processed (is_embedded AND is_entities_extracted)
        2. If not embedded: chunk → embed → upsert to Qdrant
           → update document.is_embedded = True in DB
        3. If not entities extracted: extract entities → store to DB
           → update document.is_entities_extracted = True in DB
        Errors in step 2 or 3 are logged but don't fail the other step."""

    async def process_batch(
        self,
        conn: aiosqlite.Connection,
        documents: list[DocumentRow],
        concurrency: int = 3,
    ) -> ProcessingReport:
        """Process multiple documents concurrently.
        Returns counts: embedded, entity_extracted, failed."""

    async def reprocess_unembedded(
        self,
        conn: aiosqlite.Connection,
    ) -> ProcessingReport:
        """Find all documents where is_embedded=False and process them.
        Used for catch-up after embedding pipeline failures."""
```

**Integration into `IngestRunner`** (`runner.py`):
```python
async def run_source(self, ingestor: BaseIngestor, origin='pro') -> IngestReport:
    # ... existing fetch → filter → store logic ...

    # After storing to DB:
    stored_docs = [doc for doc in new_docs]
    processing_report = await self.pipeline.process_batch(conn, stored_docs)
    report.embedded = processing_report.embedded
    report.entities_extracted = processing_report.entity_extracted
```

**Processing flow per document**:
```
document.raw_content
    │
    ▼
Chunker.chunk(raw_content)        # produces list[TextChunk]
    │
    ▼
Embedder.embed_texts([c.text for c in chunks])   # produces list[list[float]]
    │
    ▼
VectorStore.upsert_vectors(document_id, chunks, vectors, payload)
    │
    ▼
DB: UPDATE documents SET is_embedded=TRUE WHERE id=?
    │
    ▼
EntityExtractor.extract_and_store(conn, document_id, raw_content)
    │
    ▼
DB: UPDATE documents SET is_entities_extracted=TRUE WHERE id=?
```

**`ProcessingReport`** model:
```python
class ProcessingReport(BaseModel):
    total: int
    embedded: int
    entity_extracted: int
    failed_embedding: int
    failed_entities: int
    errors: list[str]
```

**Skip conditions**:
- If `document.raw_content` is None or `word_count < 50`: skip embedding (no content to embed)
- If `document.is_embedded = True`: skip embedding step
- If `document.is_entities_extracted = True`: skip entity step
- YouTube videos with no transcript: `raw_content = None` → skip both; store but log

## Dependencies
- Depends on: task_18 (Embedder), task_19 (Chunker), task_20 (VectorStore), task_22 (EntityExtractor)
- Packages needed: none new

## Acceptance Criteria
- [ ] After `cr ingest`, newly stored documents are automatically embedded and entity-extracted
- [ ] Documents with `raw_content = None` are stored but skipped for processing
- [ ] `process_batch()` respects concurrency limit (default 3)
- [ ] Failed embedding does not block entity extraction (and vice versa)
- [ ] `is_embedded` and `is_entities_extracted` flags correctly updated in DB
- [ ] `reprocess_unembedded()` finds and processes any docs that failed previously
- [ ] Unit tests mock Embedder, VectorStore, EntityExtractor; verify DB flag updates
- [ ] `IngestReport` includes `embedded` and `entities_extracted` counts

## Notes
- Concurrency of 3 for batch processing is conservative — avoids overwhelming OpenAI API
- Embedding is the most expensive step (API cost); entity extraction uses cheaper LLM
- `reprocess_unembedded()` can be triggered via `cr ingest --reprocess` flag (add to task_08 stub)
- Pipeline errors are NOT fatal to the ingest run — a document can be stored without being embedded
