# Task 06: HN Ingestor (Pro + Radar)

## Wave
Wave 3 (parallel with tasks: 07, 08)
Domain: backend

## Objective
Implement the Hacker News ingestor using the Algolia HN Search API for both pro-tier (fetch top stories) and radar (keyword search) modes.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/hackernews.py` — `HackerNewsIngestor` class
- `backend/tests/test_ingestors/test_hackernews.py` — Unit tests with mocked HTTP

### Key interfaces / implementation details:

**API endpoints** (Algolia HN Search API — no auth required):

Pro mode — fetch top/new/best stories:
```
GET https://hacker-news.firebaseio.com/v0/topstories.json
    → returns list of item IDs (up to 500)

GET https://hacker-news.firebaseio.com/v0/item/{id}.json
    → {id, type, title, url, score, by, time, descendants}
```
Alternatively use Algolia for date-filtered fetch:
```
GET https://hn.algolia.com/api/v1/search?
    tags=story
    &numericFilters=created_at_i>{timestamp},points>{min_points}
    &hitsPerPage={limit}
```

Radar mode — keyword search:
```
GET https://hn.algolia.com/api/v1/search?
    query={query}
    &tags=story
    &numericFilters=points>5
    &hitsPerPage={limit}
```

Algolia response shape:
```json
{
  "hits": [
    {
      "objectID": "12345",
      "title": "...",
      "url": "https://...",
      "author": "username",
      "points": 142,
      "created_at": "2025-01-15T10:00:00Z",
      "story_text": "...",   // for Ask HN / text-only stories
      "_highlightResult": {}
    }
  ]
}
```

**Rate limits**: No official rate limit for Algolia HN API; be polite with 0.2s delays between batch requests.

**Implementation**:
```python
class HackerNewsIngestor(BaseIngestor):
    source_type = 'hn'
    BASE_URL = 'https://hn.algolia.com/api/v1'

    async def fetch_pro(self) -> list[RawDocument]:
        """Fetch stories newer than last fetch, filtered by min_points.
        Uses hackernews config from sources.yaml: {mode, limit}.
        Fetches actual article content via ContentExtractor for non-text stories."""

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search HN Algolia for query. Returns stories sorted by relevance."""

    def _hit_to_raw_doc(self, hit: dict) -> RawDocument:
        """Map Algolia hit to RawDocument.
        metadata: {hn_id, points, comment_count, hn_url}
        For text-only stories (Ask HN), use story_text as raw_content."""
```

**Special handling**:
- Stories with `url` → fetch article via `ContentExtractor`
- Stories without `url` (Ask HN, text posts) → use `story_text` as content, URL = `https://news.ycombinator.com/item?id={id}`
- `content_type = 'post'`
- `metadata = {'hn_id': str, 'points': int, 'comment_count': int, 'hn_url': str}`

## Dependencies
- Depends on: task_02 (AppConfig with hackernews config), task_03 (DocumentRow), task_05 (BaseIngestor, ContentExtractor)
- Packages needed: `httpx` (already in pyproject.toml)

## Acceptance Criteria
- [ ] `fetch_pro()` returns up to `config.sources.hackernews.limit` stories
- [ ] Stories filtered by minimum points if configured
- [ ] Text-only stories (Ask HN) use `story_text` as content without HTTP fetch
- [ ] `search_radar()` passes query to Algolia and returns ranked results
- [ ] `RawDocument.metadata` includes `hn_id`, `points`, `comment_count`, `hn_url`
- [ ] Unit tests mock `httpx` responses; test both pro and radar paths
- [ ] Handles API errors (timeout, 5xx) gracefully with logging + empty return

## Notes
- Use Algolia API (not Firebase) for both modes — it supports date filtering and search
- The `hn_url` in metadata is always `https://news.ycombinator.com/item?id={hn_id}` — useful for referencing the HN discussion even when article has a different URL
- Apply content filter in caller (task_07), not here — just return raw docs
