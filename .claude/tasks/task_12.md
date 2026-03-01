# Task 12: YouTube Ingestor + Transcripts

## Wave
Wave 5 (parallel with tasks: 10, 11, 13, 14, 15)
Domain: backend

## Objective
Implement the YouTube ingestor that fetches channel videos via the YouTube Data API v3 (pro) and searches videos (radar), then pulls transcripts via `youtube-transcript-api` as the content.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/youtube.py` — `YouTubeIngestor`
- `backend/tests/test_ingestors/test_youtube.py`

### Key interfaces / implementation details:

**API endpoints** (YouTube Data API v3):

Pro mode — channel videos:
```
GET https://www.googleapis.com/youtube/v3/search
    ?part=snippet
    &channelId={channel_id}
    &order=date
    &type=video
    &publishedAfter={iso_timestamp}
    &maxResults=50
    &key={YOUTUBE_API_KEY}

Response: { items: [{ id: {videoId}, snippet: {title, description, channelTitle, publishedAt} }] }
```

Radar mode — keyword search:
```
GET https://www.googleapis.com/youtube/v3/search
    ?part=snippet
    &q={query}
    &type=video
    &order=relevance
    &maxResults={limit}
    &key={YOUTUBE_API_KEY}
```

Transcript fetch (no API key required):
```python
from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id: str, langs: list[str]) -> str:
    """Fetch transcript, join segments into plain text."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
    return ' '.join(seg['text'] for seg in transcript)
```

**Channel ID resolution** — config uses `handle` (e.g. `@AndrejKarpathy`). Need to resolve to channel_id:
```
GET https://www.googleapis.com/youtube/v3/channels
    ?part=id
    &forHandle={handle}
    &key={YOUTUBE_API_KEY}
```
Cache handle→id mapping in memory during session.

**Implementation**:
```python
class YouTubeIngestor(BaseIngestor):
    source_type = 'youtube'
    BASE_URL = 'https://www.googleapis.com/youtube/v3'

    async def fetch_pro(self) -> list[RawDocument]:
        """For each channel in config.sources.youtube_channels:
        1. Resolve handle → channel_id (cached)
        2. Fetch recent videos via search endpoint
        3. For each video: pull transcript (async, with fallback if unavailable)
        4. Return RawDocuments with transcript as raw_content"""

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search YouTube for query. Pull transcripts for results.
        Only include videos with available transcripts."""

    async def _get_transcript(self, video_id: str) -> str | None:
        """Fetch transcript. Returns None if unavailable (non-English, private, etc.)."""

    async def fetch_content(self, doc: RawDocument) -> RawDocument:
        """Override: use transcript instead of HTML extraction."""
        video_id = doc.metadata.get('video_id')
        if not video_id:
            return doc
        transcript = await self._get_transcript(video_id)
        if transcript:
            return doc.model_copy(update={'raw_content': transcript, 'word_count': len(transcript.split())})
        return doc
```

**RawDocument shape**:
- `source_type = 'youtube'`
- `content_type = 'video'`
- `url = f'https://youtube.com/watch?v={video_id}'`
- `metadata = {'video_id': str, 'channel_handle': str, 'channel_id': str, 'description': str}`

**Rate limits**: YouTube Data API v3 — 10,000 quota units/day (free tier). `search.list` costs 100 units per call. Plan for ~50 channels max.

**Transcript languages**: from `settings.youtube.transcript_langs` (default `['en']`)

## Dependencies
- Depends on: task_05 (BaseIngestor)
- Packages needed: `youtube-transcript-api` (add to pyproject.toml)

## Acceptance Criteria
- [ ] Fetches recent videos for all channels in `config.sources.youtube_channels`
- [ ] Handles `@handle` → `channel_id` resolution with in-memory cache
- [ ] Transcripts fetched and stored as `raw_content`; videos without transcripts have `raw_content = None`
- [ ] `search_radar()` returns YouTube search results with transcripts
- [ ] API key missing → log warning and return `[]` (don't crash)
- [ ] Transcript unavailable → log at DEBUG level, continue without crashing
- [ ] Unit tests mock YouTube API + `YouTubeTranscriptApi`
- [ ] `metadata` includes `video_id`, `channel_handle`, `channel_id`, `description`

## Notes
- `youtube-transcript-api` is synchronous — run in executor: `await loop.run_in_executor(None, get_transcript, video_id, langs)`
- YouTube search API returns `snippet.channelTitle` but not `channelHandle` — store both what you have
- Videos without transcripts are still stored (with `raw_content = None`); they just won't be embedded
- Quota management: log remaining quota if API returns `quotaExceeded` error
