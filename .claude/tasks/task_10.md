# Task 10: Substack Ingestor

## Wave
Wave 5 (parallel with tasks: 11, 12, 13, 14, 15)
Domain: backend

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/substack.py` — `SubstackIngestor`
- `backend/tests/test_ingestors/test_substack.py`

## Objective
Implement the Substack ingestor that fetches posts from configured publications via RSS feed (no auth required) for pro mode, and searches via feed discovery for radar mode.

### Key interfaces / implementation details:

**API**: Substack RSS feeds (no auth):
```
GET https://{slug}.substack.com/feed
    → Atom/RSS feed with <entry> or <item> elements

Feed entry fields used:
  - title
  - link (canonical URL)
  - author / dc:creator
  - published / pubDate
  - content:encoded or summary (HTML body)
  - id (unique post GUID)
```

**Implementation**:
```python
class SubstackIngestor(BaseIngestor):
    source_type = 'substack'

    async def fetch_pro(self) -> list[RawDocument]:
        """For each slug in config.sources.substack:
        1. Fetch {slug}.substack.com/feed via feedparser
        2. Limit to entries newer than last fetch (or max 20 per feed)
        3. Extract full content from content:encoded if present
        4. Fall back to ContentExtractor if content:encoded missing
        Returns combined deduplicated list."""

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Substack has no public search API.
        Strategy: search for Substack posts via DuckDuckGo or return empty.
        For now: return [] with a log warning — radar not well-supported for Substack."""

    def _entry_to_raw_doc(self, entry: dict, slug: str) -> RawDocument:
        """Parse feedparser entry dict to RawDocument.
        metadata: {substack_slug: str, post_id: str}
        content_type = 'article'"""
```

**feedparser** usage:
```python
import feedparser
feed = feedparser.parse(f'https://{slug}.substack.com/feed')
for entry in feed.entries:
    content_html = entry.get('content', [{}])[0].get('value', '') or entry.get('summary', '')
    # pass to html2text for plain text
```

## Dependencies
- Depends on: task_05 (BaseIngestor, ContentExtractor)
- Packages needed: `feedparser` (already in pyproject.toml)

## Acceptance Criteria
- [ ] Fetches posts from all slugs in `config.sources.substack`
- [ ] Extracts full text from `content:encoded` when available
- [ ] Falls back to `ContentExtractor` when feed only has summary
- [ ] `RawDocument.content_type = 'article'`, `source_type = 'substack'`
- [ ] `metadata` includes `substack_slug` and `post_id`
- [ ] Unit tests mock `feedparser.parse()` with fixture feed data
- [ ] Handles malformed/empty feeds gracefully (log + skip, don't crash)

## Notes
- Most Substack feeds include full HTML in `content:encoded` — prefer this over refetching
- `feedparser` may return `None` for some fields — always use `.get()` with defaults
- Rate limit: 1 request per second per feed is safe
- `search_radar()` returning `[]` is acceptable for phase 2; can improve later
