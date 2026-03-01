# Task 27: YouTube Radar Search + Transcript Pull

## Wave
Wave 11 (parallel with task 28; depends on tasks 12 and 26)
Domain: backend

## Objective
Enhance `YouTubeIngestor.search_radar()` to return meaningful results with transcripts, and integrate it properly with the radar engine. Task_12 implemented the skeleton; this task makes radar search production-ready.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/youtube.py` — Complete `search_radar()` implementation
- `backend/tests/test_ingestors/test_youtube_radar.py`

### Key interfaces / implementation details:

**Full `search_radar()` implementation** (`youtube.py`):
```python
async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
    """
    1. Search YouTube Data API v3 for query
    2. For each result: attempt to fetch transcript
    3. Only include results where transcript is available (or include all if --include-no-transcript)
    4. Return sorted by relevance (API default order)
    """
    search_results = await self._search_videos(query, limit * 2)  # over-fetch to account for no-transcript
    docs = []
    transcript_tasks = [self._get_transcript_safe(r['id']['videoId']) for r in search_results]
    transcripts = await asyncio.gather(*transcript_tasks)

    for result, transcript in zip(search_results, transcripts):
        doc = self._snippet_to_raw_doc(result, transcript)
        if transcript is not None or len(docs) < limit // 2:
            docs.append(doc)
        if len(docs) >= limit:
            break

    return docs

async def _search_videos(self, query: str, limit: int) -> list[dict]:
    """YouTube Data API v3 search.list call."""

async def _get_transcript_safe(self, video_id: str) -> str | None:
    """Fetch transcript, return None on any error (no exception propagation)."""

def _snippet_to_raw_doc(self, search_result: dict, transcript: str | None) -> RawDocument:
    """Convert YouTube search result + transcript to RawDocument.
    raw_content = transcript if available, else description snippet.
    content_type = 'video'
    metadata: {video_id, channel_id, channel_title, description, thumbnail_url}"""
```

**Search API call**:
```
GET https://www.googleapis.com/youtube/v3/search
    ?part=snippet
    &q={query}
    &type=video
    &order=relevance
    &maxResults={limit}
    &key={YOUTUBE_API_KEY}
```

**Transcript fallback**: If transcript unavailable for a video, use `snippet.description` as a content stub (often enough for filtering/search). Set `metadata['has_transcript'] = False`.

**Concurrent transcript fetching**: `asyncio.gather()` over all transcript requests — transcript API calls are I/O bound.

**Quota awareness**: `search.list` costs 100 quota units. With 10,000/day limit, max ~100 searches/day. Log a warning when radar search runs and quota is low. Handle `quotaExceeded` error gracefully (return partial results).

## Dependencies
- Depends on: task_12 (YouTubeIngestor base implementation), task_26 (RadarEngine calls search_radar)
- Packages needed: none new

## Acceptance Criteria
- [ ] `search_radar()` returns up to `limit` YouTube videos
- [ ] Transcripts fetched concurrently via `asyncio.gather`
- [ ] Videos without transcripts included with `description` as fallback content
- [ ] `metadata['has_transcript']` accurately set
- [ ] YouTube API key missing → returns `[]` with warning log
- [ ] `quotaExceeded` API error → returns partial results collected so far + logs warning
- [ ] Unit tests mock both YouTube search API and `YouTubeTranscriptApi`

## Notes
- Over-fetch (limit * 2) then filter to account for videos where transcripts fail
- The `since` parameter from pro fetch is NOT relevant for radar — radar always searches the open web
- Radar results from YouTube are stored with `origin='radar'` — user can promote to pro (task_29)
- YouTube search doesn't support date filtering in the same way — `publishedAfter`/`publishedBefore` params available but optional for radar
