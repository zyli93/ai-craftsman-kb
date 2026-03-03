# Task 11: RSS Ingestor

## Wave
Wave 5 (parallel with tasks: 10, 12, 13, 14, 15)
Domain: backend

## Objective
Implement the generic RSS/Atom feed ingestor that handles any feed URL from `config.sources.rss`.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/rss.py` — `RSSIngestor`
- `backend/tests/test_ingestors/test_rss.py`

### Key interfaces / implementation details:

**Implementation**:
```python
class RSSIngestor(BaseIngestor):
    source_type = 'rss'

    async def fetch_pro(self) -> list[RawDocument]:
        """For each feed in config.sources.rss:
        1. Fetch feed URL via feedparser (uses httpx for async, or run in executor)
        2. Parse entries, skip items older than 30 days
        3. Extract content: try content:encoded → summary → fallback to ContentExtractor
        4. Assign display_name from config as source identifier
        Returns combined list across all feeds."""

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """RSS has no search API. Return [] — radar not supported."""

    def _entry_to_raw_doc(self, entry: dict, feed_name: str, feed_url: str) -> RawDocument:
        """Parse feedparser entry to RawDocument.
        metadata: {feed_name, feed_url, entry_id}
        content_type = 'article'"""
```

**feedparser** usage:
```python
# feedparser.parse() is synchronous — run in thread executor for async compat
loop = asyncio.get_event_loop()
feed = await loop.run_in_executor(None, feedparser.parse, url)
```

**Date filtering**:
- Parse `entry.published_parsed` (struct_time) → convert to datetime
- Skip entries older than `max_age_days` (default 30, not configurable per feed)
- If `published_parsed` is None, include the entry anyway

## Dependencies
- Depends on: task_05 (BaseIngestor, ContentExtractor)
- Packages needed: `feedparser` (already in pyproject.toml)

## Acceptance Criteria
- [ ] All feeds in `config.sources.rss` are fetched
- [ ] Entries older than 30 days are skipped
- [ ] Full text extracted from `content:encoded` when available, ContentExtractor otherwise
- [ ] `metadata` includes `feed_name`, `feed_url`, `entry_id`
- [ ] Unit tests use fixture Atom/RSS XML strings passed directly to `feedparser.parse()`
- [ ] Network errors per feed are logged and skipped without crashing the runner

## Notes
- feedparser handles both RSS 2.0 and Atom 1.0 automatically
- Run `feedparser.parse()` in executor to avoid blocking the event loop
- Some feeds (e.g. academic blogs) have full content; others only have summaries — both cases handled
