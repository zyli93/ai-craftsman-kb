# Task 05: Base Ingestor Interface + Content Extractor

## Wave
Wave 2 (parallel with tasks: 02, 03, 04)
Domain: backend

## Objective
Define the abstract `BaseIngestor` class, the `RawDocument` data model, and the `ContentExtractor` utility that fetches URLs and converts HTML to clean text. All source-specific ingestors (tasks 06, 10–16) implement this interface.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/base.py` — `BaseIngestor` ABC + `RawDocument` model
- `backend/ai_craftsman_kb/processing/extractor.py` — `ContentExtractor` (readability + html2text)
- `backend/tests/test_extractor.py` — Unit tests for content extraction

### Key interfaces / implementation details:

**`RawDocument` model** (`ingestors/base.py`):
```python
class RawDocument(BaseModel):
    """Normalized output from any ingestor before DB storage."""
    url: str
    title: str | None = None
    author: str | None = None
    raw_content: str | None = None     # full extracted text (may be None before extraction)
    content_type: str | None = None    # 'article' | 'video' | 'paper' | 'post'
    published_at: datetime | None = None
    source_type: str                   # 'hn' | 'substack' | 'youtube' | 'reddit' | 'rss' | 'arxiv' | 'devto'
    origin: Literal['pro', 'radar', 'adhoc'] = 'pro'
    word_count: int | None = None
    filter_score: float | None = None  # pre-filter score from ingestor (optional)
    metadata: dict = {}                # source-specific extras (HN points, upvotes, etc.)

    def to_document_row(self, source_id: str | None = None) -> DocumentRow:
        """Convert to DB row model, generating a UUID."""
```

**`BaseIngestor` ABC** (`ingestors/base.py`):
```python
class BaseIngestor(ABC):
    """Abstract base for all source ingestors."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def source_type(self) -> str:
        """e.g. 'hn', 'substack', 'youtube' — matches sources table source_type."""

    @abstractmethod
    async def fetch_pro(self) -> list[RawDocument]:
        """Pull latest content from configured subscriptions.
        Called on a schedule for pro-tier ingestion.
        Returns deduplicated RawDocuments (URL uniqueness enforced by caller)."""

    @abstractmethod
    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search this source for a query (Radar mode).
        Returns up to `limit` results, sorted by relevance."""

    async def fetch_content(self, doc: RawDocument) -> RawDocument:
        """Optional: fetch and populate raw_content for a doc that only has a URL.
        Default implementation uses ContentExtractor. Override for sources that
        return full content in their API (e.g. ArXiv abstracts)."""
        if doc.raw_content:
            return doc
        extractor = ContentExtractor()
        extracted = await extractor.fetch_and_extract(doc.url)
        return doc.model_copy(update={
            'raw_content': extracted.text,
            'word_count': extracted.word_count,
            'title': doc.title or extracted.title,
        })
```

**`ContentExtractor`** (`processing/extractor.py`):
```python
class ExtractedContent(BaseModel):
    url: str
    title: str | None
    text: str                # clean plain text
    word_count: int
    author: str | None = None
    html: str | None = None  # raw HTML (kept for debugging)

class ContentExtractor:
    """Fetch a URL and extract clean text using readability-lxml + html2text."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={'User-Agent': 'ai-craftsman-kb/1.0'},
            follow_redirects=True,
        )

    async def fetch_and_extract(self, url: str) -> ExtractedContent:
        """Fetch URL → extract readable HTML → convert to plain text."""

    def extract_from_html(self, url: str, html: str) -> ExtractedContent:
        """Extract from already-fetched HTML (used when ingestor has raw HTML)."""

    async def __aenter__(self) -> ContentExtractor: ...
    async def __aexit__(self, *args) -> None: ...
```

**Extraction pipeline** (implement in `extractor.py`):
1. `httpx.AsyncClient.get(url)` — fetch with redirect following
2. `readability.Document(html).summary()` — extract main content HTML
3. `html2text.html2text(readable_html)` — convert to markdown/plain text
4. Count words: `len(text.split())`
5. Extract title from `readability.Document(html).title()`

## Dependencies
- Depends on: task_01 (project structure, pyproject.toml)
- Packages needed: `readability-lxml`, `lxml`, `html2text`, `httpx` (all in pyproject.toml)

## Acceptance Criteria
- [ ] `BaseIngestor` is abstract — cannot be instantiated directly
- [ ] `RawDocument.to_document_row()` generates a UUID and maps all fields correctly
- [ ] `ContentExtractor.fetch_and_extract()` returns clean text for a real article URL (integration test optional, mock in unit test)
- [ ] `ContentExtractor.extract_from_html()` works without network (pure parsing)
- [ ] `word_count` is populated on every `ExtractedContent`
- [ ] `ContentExtractor` usable as async context manager (handles client cleanup)
- [ ] Unit tests: HTML extraction with known content, word count, title extraction

## Notes
- `readability-lxml` import is `from readability import Document` — verify the import path
- For YouTube URLs, `fetch_and_extract()` will not work (no readable HTML); the YouTube ingestor (task_12) overrides `fetch_content()` to use transcripts instead
- httpx client should be reused across calls (not recreated per request) — use instance-level client
- Handle non-200 responses gracefully: log and return empty `ExtractedContent` with `text=''`
- Content-type check: skip extraction for PDFs, images, etc. unless HTML/text
