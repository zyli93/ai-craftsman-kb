# Task 28: Multi-Source Radar (Reddit + HN + ArXiv + DEV.to)

## Wave
Wave 11 (parallel with task 27; depends on tasks 13, 14, 15, 26)
Domain: backend

## Objective
Make `search_radar()` production-ready for Reddit, HN (Algolia), ArXiv, and DEV.to — ensuring each returns well-formed `RawDocument` results with content populated, and integrate correctly with the radar engine.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/hackernews.py` — Complete `search_radar()` with full content
- `backend/ai_craftsman_kb/ingestors/reddit.py` — Complete `search_radar()` with content fetch
- `backend/ai_craftsman_kb/ingestors/arxiv.py` — Verify `search_radar()` uses correct query format
- `backend/ai_craftsman_kb/ingestors/devto.py` — Complete `search_radar()` with full content
- `backend/tests/test_ingestors/test_multi_radar.py`

### Key interfaces / implementation details:

**HN radar** (enhance task_06's stub):
```python
async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
    """Algolia HN search API.
    GET https://hn.algolia.com/api/v1/search?query={query}&tags=story&numericFilters=points>5
    Fetch article content for non-text stories via ContentExtractor.
    Use concurrency=5 for content fetching."""
```

**Reddit radar** (enhance task_13's stub):
```python
async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
    """Reddit search across all configured subreddits OR globally.
    GET https://oauth.reddit.com/search?q={query}&sort=relevance&limit={limit}&type=link
    For self-posts: selftext as content.
    For link posts: ContentExtractor (max concurrency=3 to respect rate limit)."""
```

**ArXiv radar** (verify task_14's implementation):
```python
async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
    """ArXiv Atom API search.
    GET http://export.arxiv.org/api/query?search_query={query}&max_results={limit}&sortBy=relevance
    Content = abstract (already done in task_14)."""
```

**DEV.to radar** (enhance task_15's stub):
```python
async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
    """DEV.to search endpoint.
    GET https://dev.to/api/articles?q={query}&per_page={limit}
    Fetch full body_markdown for each result (concurrency=5)."""
```

**Shared patterns across all radar implementations**:
1. Content fetch with bounded concurrency (`asyncio.Semaphore(5)`)
2. Errors per-document logged and skipped (don't fail entire source)
3. All results tagged `origin='radar'`
4. `source_type` set to the ingestor's `source_type`

**Integration test** (`test_multi_radar.py`):
```python
async def test_radar_fan_out_all_sources(mock_all_apis):
    """
    Given: mocked APIs for HN, Reddit, ArXiv, DEV.to
    When: RadarEngine.search(conn, 'LLM inference') called
    Then:
    - All 4 source search APIs called concurrently
    - Results deduplicated
    - New documents stored with origin='radar'
    """
```

## Dependencies
- Depends on: task_13 (Reddit OAuth pattern), task_14 (ArXiv Atom parsing), task_15 (DEV.to API), task_26 (RadarEngine)
- Packages needed: none new

## Acceptance Criteria
- [ ] All 4 sources return `RawDocument` lists from `search_radar()`
- [ ] HN: article content fetched for link stories
- [ ] Reddit: self-post content from selftext, link-post content from ContentExtractor
- [ ] ArXiv: abstract used as content (no additional fetching needed)
- [ ] DEV.to: full `body_markdown` fetched for each result
- [ ] Content fetching uses bounded concurrency (semaphore)
- [ ] Integration test: all 4 sources mocked and fanned-out simultaneously
- [ ] `RadarEngine.search()` completes all 4 sources concurrently (gather, not sequential)

## Notes
- Reddit radar searches globally (not limited to configured subreddits) — it's an open web search
- For HN radar, `points>5` filter is loose — radar is about discovery, not quality filtering
- ArXiv `search_radar()` may have already been complete in task_14 — verify and note if no changes needed
- DEV.to `?q=` search is basic keyword match — not semantic — but sufficient for radar
