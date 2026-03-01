# Task 42: Source Discovery Engine

## Wave
Wave 16 (parallel with tasks 43, 44, 45; depends on task 03)
Domain: backend

## Objective
Implement the source discovery engine that analyzes ingested documents to suggest new sources worth following, storing suggestions in the `discovered_sources` table.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/processing/discoverer.py` — `SourceDiscoverer`
- `config/prompts/source_discovery.md` — LLM prompt for suggestion-based discovery
- `backend/ai_craftsman_kb/api/system.py` — Implement `GET /api/discover` endpoint
- `backend/tests/test_processing/test_discoverer.py`

### Key interfaces / implementation details:

**Discovery methods** (from plan.md):
1. **Outbound links** — parse URLs from `raw_content`; classify by domain → Substack, YouTube, etc.
2. **Citation graph** — ArXiv papers: parse arxiv.org links from content
3. **Video mentions** — YouTube descriptions mention channel handles
4. **Substack mentions** — extract `*.substack.com` URLs from text
5. **LLM suggestions** — "Based on these articles, suggest new sources to follow"

**`SourceDiscoverer`** (`processing/discoverer.py`):
```python
class SourceDiscoverer:
    """Analyze documents to suggest new sources to follow."""

    def __init__(self, config: AppConfig, llm_router: LLMRouter) -> None: ...

    async def discover_from_documents(
        self,
        conn: aiosqlite.Connection,
        documents: list[DocumentRow],
    ) -> list[DiscoveredSourceRow]:
        """Run all discovery methods on a batch of documents.
        Returns list of new suggestions (not already in sources or discovered_sources tables)."""

    def _extract_outbound_links(self, doc: DocumentRow) -> list[DiscoveredSourceRow]:
        """Parse URLs from raw_content using regex.
        Classify: *.substack.com → substack, youtube.com/c/* or @* → youtube,
                  reddit.com/r/* → reddit, arxiv.org/abs/* → arxiv
        Skip already-configured sources."""

    def _extract_youtube_handles(self, doc: DocumentRow) -> list[DiscoveredSourceRow]:
        """Find @handle patterns in YouTube video descriptions or article text."""

    async def _llm_suggestions(
        self,
        conn: aiosqlite.Connection,
        limit: int = 5,
    ) -> list[DiscoveredSourceRow]:
        """Fetch 20 recent documents → send to LLM → parse suggestions."""

    async def run_periodic_discovery(
        self,
        conn: aiosqlite.Connection,
    ) -> int:
        """Run all discovery methods on recent documents (last 7 days).
        Returns count of new suggestions added."""
```

**LLM discovery prompt** (`config/prompts/source_discovery.md`):
```
You are helping someone discover new content sources to follow in their RSS/subscription reader.

Here are 20 recent articles they've been reading:

{article_list}

Based on this reading pattern, suggest 5 new sources they should follow.
For each suggestion:
- source_type: one of [substack, youtube, reddit, rss, arxiv, devto]
- identifier: the specific slug/handle/subreddit/URL
- display_name: friendly name
- reason: one sentence why this source would be valuable

Return ONLY a JSON array with no other text:
[{"source_type": "...", "identifier": "...", "display_name": "...", "reason": "..."}, ...]
```

**Link classification regex patterns**:
```python
PATTERNS = {
    'substack': r'https?://([a-z0-9-]+)\.substack\.com',
    'youtube':  r'(?:youtube\.com/(?:@([a-zA-Z0-9_-]+)|c/([a-zA-Z0-9_-]+))|youtu\.be/)',
    'reddit':   r'reddit\.com/r/([a-zA-Z0-9_]+)',
    'arxiv':    r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d+)',
}
```

**Confidence scoring**:
- Found in 3+ documents: 0.9
- Found in 2 documents: 0.7
- Found in 1 document: 0.4
- LLM suggestion: 0.6

**Integration into ingest runner**: Call `discover_from_documents()` at end of each `run_source()` call (post-ingest hook). Store results to `discovered_sources` table.

**`GET /api/discover`** endpoint:
```python
@router.get('/discover')
async def list_discovered(status: str = 'suggested', limit: int = 20):
    """Return discovered sources with status='suggested' (pending review)."""
```

## Dependencies
- Depends on: task_03 (discovered_sources table + queries), task_04 (LLMRouter for LLM suggestions)
- Packages needed: none new (regex stdlib)

## Acceptance Criteria
- [ ] `_extract_outbound_links()` correctly classifies Substack/YouTube/Reddit/ArXiv URLs
- [ ] Suggestions deduplicated: same (source_type, identifier) not inserted twice
- [ ] Already-configured sources filtered out (not suggested again)
- [ ] LLM suggestions parsed from JSON response; invalid JSON logged and skipped
- [ ] Confidence scores correctly computed based on mention frequency
- [ ] `GET /api/discover` returns pending suggestions from DB
- [ ] `run_periodic_discovery()` processes last 7 days of documents
- [ ] Unit tests: fixture HTML content with known links → verify correct suggestions

## Notes
- Discovery runs as a side effect of ingest — low-priority, can fail silently
- `discovery_method` field in DB distinguishes outbound_link / citation / mention / llm_suggestion
- LLM discovery is expensive (~$0.001 per run) — rate limit: at most once per day via `last_discovery_at` tracking
- Sources with `source_type='rss'` use the full URL as `identifier`; others use slug/handle/subreddit
- The `discovered_sources` table has `UNIQUE(source_type, identifier)` — safe to call repeatedly
