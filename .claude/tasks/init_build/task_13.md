# Task 13: Reddit Ingestor

## Wave
Wave 5 (parallel with tasks: 10, 11, 12, 14, 15)
Domain: backend

## Objective
Implement the Reddit ingestor that fetches hot/new posts from configured subreddits (pro) and searches Reddit for queries (radar) using the official Reddit API.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/reddit.py` — `RedditIngestor`
- `backend/tests/test_ingestors/test_reddit.py`

### Key interfaces / implementation details:

**Authentication** — Reddit OAuth2 (script app):
```
POST https://www.reddit.com/api/v1/access_token
    Authorization: Basic base64(client_id:client_secret)
    Body: grant_type=client_credentials&device_id=DO_NOT_TRACK_THIS_DEVICE

Response: {"access_token": "...", "token_type": "bearer", "expires_in": 3600}
```
Cache token and refresh when expired.

**API endpoints**:

Pro mode — subreddit feed:
```
GET https://oauth.reddit.com/r/{subreddit}/{sort}
    ?limit={limit}
    &t=day          # time filter for 'top' sort
    Headers: Authorization: Bearer {token}
             User-Agent: ai-craftsman-kb/1.0

Response: { data: { children: [{ data: {id, title, url, selftext, author,
           score, num_comments, created_utc, permalink, is_self} }] } }
```

Radar mode — search:
```
GET https://oauth.reddit.com/search
    ?q={query}
    &sort=relevance
    &limit={limit}
    &type=link
```

**Implementation**:
```python
class RedditIngestor(BaseIngestor):
    source_type = 'reddit'

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    async def fetch_pro(self) -> list[RawDocument]:
        """For each subreddit in config.sources.subreddits:
        1. Authenticate (get/refresh token)
        2. Fetch posts with configured sort and limit
        3. For self-posts: use selftext as raw_content
        4. For link posts: use ContentExtractor on the linked URL
        Return combined list."""

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search Reddit across all subreddits for query."""

    async def _authenticate(self) -> str:
        """Get OAuth token. Cache until expiry."""

    def _post_to_raw_doc(self, post: dict) -> RawDocument:
        """Map Reddit post data dict to RawDocument.
        For self-posts (is_self=True): raw_content = selftext
        For link posts: raw_content = None (fetched by ContentExtractor)
        url = f'https://reddit.com{permalink}'
        metadata: {subreddit, post_id, upvotes, comment_count, is_self, linked_url}"""
```

**Content handling**:
- `is_self = True` → `raw_content = post['selftext']`, `content_type = 'post'`
- `is_self = False` → `raw_content = None` (fetch via ContentExtractor), `content_type = 'article'`
- Filter: apply `min_upvotes` from `filters.reddit.min_upvotes` in metadata

**Rate limits**: Reddit API — 100 requests per minute per OAuth client. Add 0.6s between requests.

**Credentials** from settings: `config.settings.reddit.client_id`, `config.settings.reddit.client_secret`

## Dependencies
- Depends on: task_05 (BaseIngestor, ContentExtractor)
- Packages needed: `httpx` (already in pyproject.toml); no PRAW (use direct API calls)

## Acceptance Criteria
- [ ] OAuth token fetched and cached; refreshed automatically when expired
- [ ] All subreddits in config fetched with correct sort and limit
- [ ] Self-posts use `selftext` as content; link posts trigger ContentExtractor
- [ ] `metadata` includes `subreddit`, `post_id`, `upvotes`, `comment_count`, `is_self`, `linked_url`
- [ ] `search_radar()` queries Reddit search endpoint
- [ ] Missing credentials → log warning and return `[]`
- [ ] Unit tests mock OAuth token endpoint + post listing endpoint
- [ ] Respects 0.6s rate limit delay between requests

## Notes
- Do NOT use PRAW — implement direct httpx calls to avoid the extra dependency
- User-Agent header is required: Reddit blocks requests without it; use `ai-craftsman-kb/1.0`
- `selftext` for self-posts can be empty string or `[deleted]` — handle these: skip if selftext == '[deleted]' or len < 50
- Some subreddits are NSFW — not filtered here; that's the content filter's job
