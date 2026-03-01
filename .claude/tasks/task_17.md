# Task 17: Phase 2 Integration + Incremental Fetch

## Wave
Wave 6 (sequential — depends on task 16 completing)
Domain: integration

## Objective
Extend `IngestRunner` to support all 7 ingestors, implement incremental fetch (only new content since last run), wire the `ingest-url` CLI command, and write phase 2 integration tests.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/runner.py` — Extend to register all ingestors + incremental fetch
- `backend/ai_craftsman_kb/cli.py` — Implement `ingest-url` command body
- `backend/tests/test_integration/test_phase2.py` — Integration tests for all sources

### Key interfaces / implementation details:

**Extend `IngestRunner`** (`runner.py`):
```python
# Full ingestor registry — all 7 sources
INGESTORS: dict[str, type[BaseIngestor]] = {
    'hn':       HackerNewsIngestor,
    'substack': SubstackIngestor,
    'rss':      RSSIngestor,
    'youtube':  YouTubeIngestor,
    'reddit':   RedditIngestor,
    'arxiv':    ArxivIngestor,
    'devto':    DevtoIngestor,
}

class IngestRunner:
    async def run_all(self) -> list[IngestReport]:
        """Run all enabled source types sequentially.
        For each source type: instantiate ingestor, call run_source()."""

    async def run_source(self, ingestor: BaseIngestor, origin='pro') -> IngestReport:
        """Fetch → filter → deduplicate → store pipeline.
        Incremental fetch: pass last_fetched_at from sources table to ingestor.
        After success: update sources.last_fetched_at to now."""

    async def ingest_url(self, url: str, tags: list[str] | None = None) -> IngestReport:
        """Use AdhocIngestor to ingest a single URL.
        Skip filter. Store with origin='adhoc'."""
```

**Incremental fetch pattern**:
```python
# In run_source():
source_row = await get_source_by_type_and_id(conn, ingestor.source_type, identifier)
last_fetched = source_row.last_fetched_at if source_row else None
# Pass to ingestor — each ingestor uses this to filter to new content only
docs = await ingestor.fetch_pro(since=last_fetched)
```
- Each ingestor's `fetch_pro()` needs optional `since: datetime | None = None` parameter
- For API-based sources (HN, Reddit, DEV.to): filter by date
- For RSS/Substack feeds: feedparser provides `published_parsed`; skip entries before `since`

**`ingest-url` CLI implementation** (`cli.py`):
```python
async def _run_ingest_url(config: AppConfig, url: str, tags: list[str]) -> None:
    async with get_db(_get_db_path(config)) as conn:
        runner = IngestRunner(config, llm_router=None, db_path=_get_db_path(config))
        report = await runner.ingest_url(url, tags=list(tags))
        if report.stored:
            click.echo(f'Ingested: {url}')
        else:
            click.echo(f'Skipped (duplicate): {url}')
```

**Sources table sync**: on first run of an ingestor, create a row in `sources` table with config snapshot. On subsequent runs, update `last_fetched_at`.

**Phase 2 integration tests** — scenarios:
1. `test_full_ingest_pipeline` — mock all 7 APIs, run `run_all()`, verify docs stored per source
2. `test_incremental_fetch` — ingest twice, verify second run only fetches new items
3. `test_ingest_url_article` — ingest an article URL, verify stored with `origin='adhoc'`
4. `test_ingest_url_youtube` — ingest a YouTube URL, verify transcript stored
5. `test_duplicate_skip` — attempt to ingest same URL twice, verify stored only once

## Dependencies
- Depends on: tasks 10–16 (all ingestors)
- Packages needed: none new

## Acceptance Criteria
- [ ] `cr ingest` runs all 7 enabled source types
- [ ] `cr ingest --source youtube` runs only YouTube
- [ ] `cr ingest-url https://youtube.com/watch?v=xxx` stores with `origin='adhoc'`
- [ ] Second `cr ingest` run only fetches content newer than last run timestamp
- [ ] `sources` table updated with `last_fetched_at` after each successful ingestor run
- [ ] Failed ingestor logs error and continues; `fetch_error` updated in sources table
- [ ] All 5 integration test scenarios pass with mocked HTTP
- [ ] `cr stats` shows accurate counts after ingestion

## Notes
- `since` parameter needs to be added to all ingestor `fetch_pro()` signatures as an optional kwarg
- Substack and RSS use `published_parsed` from feedparser entry — always UTC
- The `sources` table `identifier` = source slug/handle/URL depending on type; for multi-entry types (subreddits), one row per subreddit
- `run_all()` is sequential by source type but each source type can be concurrent internally (e.g. fetching multiple subreddits in parallel)
