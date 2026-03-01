# Task 26: Radar Engine Orchestrator (Async Fan-out)

## Wave
Wave 10 (sequential — depends on task 05)
Domain: backend

## Objective
Implement the Radar engine that accepts a topic query, fans out concurrently to all enabled radar sources, deduplicates results, and stores them in the `documents` table with `origin='radar'`.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/radar/engine.py` — `RadarEngine` class
- `backend/tests/test_radar/test_engine.py`

### Key interfaces / implementation details:

**`RadarEngine`** (`radar/engine.py`):
```python
class RadarResult(BaseModel):
    document: DocumentRow
    source_type: str
    is_new: bool               # True if not already in DB

class RadarReport(BaseModel):
    query: str
    total_found: int
    new_documents: int
    sources_searched: list[str]
    errors: dict[str, str]    # {source_type: error_message}

class RadarEngine:
    """On-demand topic search across all enabled sources with async fan-out."""

    def __init__(
        self,
        config: AppConfig,
        ingestors: dict[str, BaseIngestor],   # pre-instantiated ingestors
    ) -> None:
        self.config = config
        self.ingestors = ingestors

    async def search(
        self,
        conn: aiosqlite.Connection,
        query: str,
        sources: list[str] | None = None,    # None = all enabled sources
        limit_per_source: int = 10,
    ) -> RadarReport:
        """
        Async fan-out pattern:
        1. Determine which sources to search (all if sources=None)
        2. Create tasks: [ingestor.search_radar(query, limit) for each source]
        3. asyncio.gather(*tasks, return_exceptions=True)
        4. Collect results, handle exceptions per source
        5. Deduplicate by URL across all sources
        6. Store new documents with origin='radar' via upsert_document()
        7. Trigger ProcessingPipeline for new docs (if pipeline available)
        8. Return RadarReport
        """

    async def _search_one_source(
        self,
        ingestor: BaseIngestor,
        query: str,
        limit: int,
    ) -> list[RawDocument]:
        """Search one source. Returns [] on exception (exception logged)."""

    async def _store_results(
        self,
        conn: aiosqlite.Connection,
        docs: list[RawDocument],
    ) -> tuple[int, int]:
        """Deduplicate and store. Returns (new_count, duplicate_count)."""
```

**Async fan-out pattern**:
```python
tasks = [
    self._search_one_source(ingestor, query, limit_per_source)
    for source_type, ingestor in active_ingestors.items()
]
results = await asyncio.gather(*tasks, return_exceptions=True)

all_docs: list[RawDocument] = []
for source_type, result in zip(active_ingestors.keys(), results):
    if isinstance(result, Exception):
        report.errors[source_type] = str(result)
    else:
        all_docs.extend(result)
```

**Deduplication**: Group by URL — keep first occurrence across sources. URL is the dedup key.

**Sources that support radar search** (from plan.md):
- `hn` — Algolia search
- `reddit` — Reddit search API
- `arxiv` — ArXiv search API
- `devto` — DEV.to search API
- `youtube` — YouTube search API (task_27 adds this)

**Sources with limited/no radar support** (return `[]`):
- `substack` — No public search
- `rss` — No search

## Dependencies
- Depends on: task_05 (BaseIngestor with search_radar method)
- Packages needed: none new

## Acceptance Criteria
- [ ] All enabled source ingestors searched concurrently via `asyncio.gather`
- [ ] Results deduplicated by URL before storage
- [ ] Documents stored with `origin='radar'`
- [ ] Failing source (exception) logged + recorded in `RadarReport.errors`; other sources continue
- [ ] `sources` parameter limits to specified source types
- [ ] New documents vs duplicates counted accurately in report
- [ ] Unit tests mock all 5 ingestors' `search_radar()` methods
- [ ] `asyncio.gather` used (not sequential awaits) — verifiable via timing in tests

## Notes
- The engine does NOT apply content filters to radar results — user triages manually
- Processing pipeline (task_24) integration is optional here — can trigger or skip based on config
- Results stored with `deleted_at = None`, `is_archived = False` — pending user triage
- The `promoted_at` field is set when user promotes a radar result to pro tier
- Sources list is built from INGESTORS registry filtered by which ones have non-trivial `search_radar()`
