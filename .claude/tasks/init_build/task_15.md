# Task 15: DEV.to Ingestor

## Wave
Wave 5 (parallel with tasks: 10, 11, 12, 13, 14)
Domain: backend

## Objective
Implement the DEV.to ingestor that fetches articles by configured tags (pro) and searches DEV.to for queries (radar) via the public DEV.to API.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/devto.py` — `DevtoIngestor`
- `backend/tests/test_ingestors/test_devto.py`

### Key interfaces / implementation details:

**API endpoints** (DEV.to REST API — no auth required for reads):
```
GET https://dev.to/api/articles
    ?tag={tag}        # filter by tag
    &per_page={limit}
    &page=1
    &top=7            # top articles from last 7 days (alternative to tag)

Response: JSON array of article objects
Article fields:
  - id
  - title
  - url (canonical_url)
  - description (excerpt)
  - body_markdown (not in list endpoint — must fetch individual article)
  - body_html (same — individual article only)
  - published_at
  - user.name, user.username
  - tags: list[str]
  - positive_reactions_count
  - comments_count
  - reading_time_minutes

GET https://dev.to/api/articles/{id}
    → Full article including body_markdown and body_html

GET https://dev.to/api/articles?q={query}
    → Search endpoint (basic keyword match)
```

**Implementation**:
```python
class DevtoIngestor(BaseIngestor):
    source_type = 'devto'
    BASE_URL = 'https://dev.to/api'

    async def fetch_pro(self) -> list[RawDocument]:
        """For each tag in config.sources.devto.tags:
        1. Fetch article list via /articles?tag={tag}&per_page={limit}
        2. For each article ID: fetch full article to get body_markdown
        3. Convert markdown to plain text
        Return combined deduplicated list."""

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search via GET /articles?q={query}&per_page={limit}"""

    def _article_to_raw_doc(self, article: dict) -> RawDocument:
        """Map DEV.to article dict to RawDocument.
        raw_content = body_markdown converted to plain text (via html2text or direct)
        metadata: {devto_id, tags, reactions, comments, reading_time_minutes}
        content_type = 'article'"""
```

**Content extraction**:
- List endpoint returns `description` (short excerpt) only
- Must fetch `/articles/{id}` to get `body_markdown` or `body_html`
- Convert `body_markdown` to plain text: use `html2text` on `body_html` OR treat markdown as plain text directly
- Batch fetching: fetch individual articles concurrently (max 5 concurrent) to stay polite

**Rate limits**: DEV.to has no published rate limit; use 0.2s between requests to be safe.

**Filtering**: `min_reactions` from `filters.devto.min_reactions` in article metadata.

## Dependencies
- Depends on: task_05 (BaseIngestor)
- Packages needed: `httpx`, `html2text` (already in pyproject.toml)

## Acceptance Criteria
- [ ] Articles fetched for all tags in `config.sources.devto.tags`
- [ ] Full `body_markdown` fetched from individual article endpoint
- [ ] `metadata` includes `devto_id`, `tags`, `reactions`, `comments`, `reading_time_minutes`
- [ ] `search_radar()` uses the `?q=` search endpoint
- [ ] Concurrency limited to 5 simultaneous article fetches
- [ ] Unit tests mock both list and individual article endpoints
- [ ] Handles empty tag results gracefully

## Notes
- DEV.to article `url` field may differ from `canonical_url` — use `canonical_url` as the document URL
- The `tags` on a DEV.to article are stored in `DocumentRow.metadata['tags']`, not `user_tags` (those are user-applied)
- `description` field (from list endpoint) can serve as the document `summary` even when full content is available
