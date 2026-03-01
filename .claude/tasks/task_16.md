# Task 16: Adhoc URL Ingestor

## Wave
Wave 6 (sequential — depends on task 15 completing wave 5)
Domain: backend

## Objective
Implement the adhoc URL ingestor that takes a single URL, auto-detects its type (article, YouTube video, ArXiv paper), fetches and extracts content appropriately, and returns a `RawDocument`.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/adhoc.py` — `AdhocIngestor`
- `backend/tests/test_ingestors/test_adhoc.py`

### Key interfaces / implementation details:

**Implementation**:
```python
class AdhocIngestor(BaseIngestor):
    source_type = 'adhoc'

    async def fetch_pro(self) -> list[RawDocument]:
        """Not used for adhoc. Raise NotImplementedError."""

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Not used for adhoc. Raise NotImplementedError."""

    async def ingest_url(self, url: str, tags: list[str] | None = None) -> RawDocument:
        """Main entry point. Detect URL type → delegate to appropriate handler.
        tags: stored in metadata for later user_tags population."""
        url_type = self._detect_url_type(url)
        if url_type == 'youtube':
            return await self._handle_youtube(url)
        elif url_type == 'arxiv':
            return await self._handle_arxiv(url)
        else:
            return await self._handle_article(url)

    def _detect_url_type(self, url: str) -> str:
        """Return 'youtube', 'arxiv', or 'article'."""
        parsed = urlparse(url)
        if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
            return 'youtube'
        if 'arxiv.org' in parsed.netloc:
            return 'arxiv'
        return 'article'

    async def _handle_youtube(self, url: str) -> RawDocument:
        """Extract video_id from URL. Fetch transcript via YouTubeTranscriptApi.
        Fall back to description if transcript unavailable."""

    async def _handle_arxiv(self, url: str) -> RawDocument:
        """Extract paper ID from URL. Fetch abstract via ArXiv Atom API."""

    async def _handle_article(self, url: str) -> RawDocument:
        """Use ContentExtractor to fetch and extract article text."""
```

**YouTube URL parsing**:
- `youtube.com/watch?v=VIDEO_ID`
- `youtu.be/VIDEO_ID`
- `youtube.com/shorts/VIDEO_ID`

**ArXiv URL parsing**:
- `arxiv.org/abs/2501.12345` → ID = `2501.12345`
- `arxiv.org/pdf/2501.12345` → resolve to abs URL

**Origin**: always `'adhoc'`

**RawDocument shape**:
- `source_type = 'adhoc'`
- `origin = 'adhoc'`
- `content_type` set per detection
- `metadata = {'adhoc_tags': tags or [], 'url_type': str}`

## Dependencies
- Depends on: task_05 (BaseIngestor, ContentExtractor), task_12 (YouTube transcript logic — reuse, don't duplicate)
- Packages needed: none new

## Acceptance Criteria
- [ ] YouTube URLs: transcript fetched, metadata includes `video_id`
- [ ] ArXiv URLs: abstract fetched, metadata includes `arxiv_id`
- [ ] All other URLs: ContentExtractor used
- [ ] `origin = 'adhoc'` on all returned documents
- [ ] Graceful handling when transcript/abstract unavailable (`raw_content = None`)
- [ ] CLI `cr ingest-url <url>` calls this ingestor
- [ ] Unit tests: mock each URL type, verify correct handler called

## Notes
- Reuse `_get_transcript()` logic from `YouTubeIngestor` (task_12) — import from there, don't duplicate
- Reuse ArXiv Atom API fetch from `ArxivIngestor` (task_14) — same approach
- Adoc URL ingest skips the content filter by default (user is intentionally ingesting this URL)
- `tags` argument from CLI (`--tag`) stored in `metadata['adhoc_tags']`; `user_tags` on the DB row is populated from these after storage
