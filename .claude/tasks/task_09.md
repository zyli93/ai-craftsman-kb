# Task 09: Phase 1 Integration + Tests

## Wave
Wave 4 (sequential — depends on wave 3: tasks 06, 07, 08)
Domain: integration

## Objective
Wire together the Phase 1 components (config → HN ingestor → content filter → DB storage) into a working `cr ingest` pipeline, and write integration tests that run the full flow end-to-end with mocked HTTP.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/runner.py` — `IngestRunner` orchestrating fetch → filter → store
- `backend/ai_craftsman_kb/cli.py` — Implement the `ingest` command body (replaces stub from task_08)
- `backend/tests/test_integration/test_phase1.py` — Integration tests
- `backend/tests/conftest.py` — Shared fixtures (in-memory DB, config factory)

### Key interfaces / implementation details:

**`IngestRunner`** (`ingestors/runner.py`):
```python
class IngestReport(BaseModel):
    source_type: str
    fetched: int
    passed_filter: int
    stored: int
    skipped_duplicate: int
    errors: list[str]

class IngestRunner:
    """Orchestrates: fetch → filter → deduplicate → store."""

    def __init__(
        self,
        config: AppConfig,
        llm_router: LLMRouter,
        db_path: Path,
    ) -> None: ...

    async def run_source(
        self,
        ingestor: BaseIngestor,
        origin: Literal['pro', 'radar', 'adhoc'] = 'pro',
    ) -> IngestReport:
        """Run one ingestor:
        1. Call ingestor.fetch_pro()
        2. Filter each doc via ContentFilter
        3. Deduplicate by URL (check existing in DB)
        4. For passed docs: call ingestor.fetch_content() to get full text
        5. Store to DB via upsert_document()
        6. Update sources table last_fetched_at
        """

    async def run_all(self) -> list[IngestReport]:
        """Run all enabled ingestors in sequence. Return one report per source."""
```

**`ingest` CLI command** implementation (`cli.py`):
```python
async def _run_ingest(config: AppConfig, source: str | None) -> None:
    async with get_db(_get_db_path(config)) as conn:
        await init_db(conn)
        llm_router = LLMRouter(config)
        runner = IngestRunner(config, llm_router, _get_db_path(config))
        if source:
            ingestor = _get_ingestor(source, config)
            reports = [await runner.run_source(ingestor)]
        else:
            reports = await runner.run_all()
        for r in reports:
            click.echo(f'{r.source_type}: {r.stored} stored, {r.passed_filter} passed filter, '
                       f'{r.skipped_duplicate} duplicates, {r.errors and len(r.errors)} errors')
```

**`_get_ingestor()` factory** (`runner.py`):
```python
INGESTORS: dict[str, type[BaseIngestor]] = {
    'hn': HackerNewsIngestor,
    # later tasks add: 'substack', 'youtube', 'reddit', 'rss', 'arxiv', 'devto'
}
```

**Integration test scenario** (`test_phase1.py`):
```python
@pytest.mark.asyncio
async def test_hn_ingest_to_db(in_memory_config, mock_hn_api):
    """
    Given: mocked Algolia HN API returning 3 stories
    When: cr ingest runs
    Then: stories pass filter → stored in DB → count = 3
    """
```

**Fixtures** (`conftest.py`):
```python
@pytest.fixture
def in_memory_config() -> AppConfig:
    """Return minimal valid AppConfig for testing."""

@pytest.fixture
def mock_hn_api(httpx_mock):
    """Mock Algolia HN API to return 3 known stories."""
```

## Dependencies
- Depends on: task_06 (HackerNewsIngestor), task_07 (ContentFilter), task_08 (CLI)
- Packages needed: `pytest`, `pytest-asyncio`, `pytest-httpx` (add to dev dependencies)

## Acceptance Criteria
- [ ] `uv run cr ingest` fetches from HN, filters, and stores results to SQLite
- [ ] `uv run cr ingest --source hn` runs only HN ingestor
- [ ] `uv run cr stats` shows document counts after ingest
- [ ] Duplicate URLs are skipped (not stored twice)
- [ ] Integration test: mock HN API → ingest → verify 3 docs in DB
- [ ] Integration test: docs with low points filtered out if filter.min_score set
- [ ] `IngestReport` printed to console with correct counts
- [ ] `uv run pytest backend/tests/ -v` passes with no failures

## Notes
- `pytest-httpx` mocks `httpx.AsyncClient` — no real network calls in tests
- The `run_all()` method in phase 1 only has HN ingestor; other sources added in tasks 10–15
- Error handling: a failing ingestor (exception) should log + add to `IngestReport.errors` and continue to next source — never crash the whole run
- `conftest.py` in `backend/tests/` creates shared fixtures; put `conftest.py` in `test_integration/` for integration-specific fixtures
- After this task, `cr ingest` + `cr stats` are the first working end-to-end CLI flows
