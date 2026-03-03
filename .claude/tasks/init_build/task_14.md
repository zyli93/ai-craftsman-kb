# Task 14: ArXiv Ingestor

## Wave
Wave 5 (parallel with tasks: 10, 11, 12, 13, 15)
Domain: backend

## Objective
Implement the ArXiv ingestor that fetches papers matching configured search queries (pro) and searches ArXiv (radar) via the public Atom API.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/ingestors/arxiv.py` — `ArxivIngestor`
- `backend/tests/test_ingestors/test_arxiv.py`

### Key interfaces / implementation details:

**API endpoint** (ArXiv Atom API — no auth required):
```
GET http://export.arxiv.org/api/query
    ?search_query={query}     # e.g. "cat:cs.CL AND abs:large language model"
    &start=0
    &max_results={max_results}
    &sortBy=submittedDate
    &sortOrder=descending

Response: Atom XML feed
Entry fields:
  - id: http://arxiv.org/abs/2501.12345v1
  - title
  - author (multiple)
  - summary (abstract)
  - published
  - updated
  - category (multiple, scheme contains subject area)
  - link (PDF + abs)
```

**Implementation**:
```python
class ArxivIngestor(BaseIngestor):
    source_type = 'arxiv'
    BASE_URL = 'http://export.arxiv.org/api/query'

    async def fetch_pro(self) -> list[RawDocument]:
        """For each query in config.sources.arxiv.queries:
        1. Fetch papers via Atom API with max_results limit
        2. Filter to papers submitted within the last N days
        3. Use abstract (summary) as raw_content — no PDF needed
        Return combined deduplicated list."""

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search ArXiv for query. Use Atom API with search_query={query}."""

    def _parse_atom_feed(self, xml_text: str) -> list[RawDocument]:
        """Parse ArXiv Atom XML. Use xml.etree.ElementTree.
        Extract: id, title, authors, summary, published, categories."""

    def _entry_to_raw_doc(self, entry_elem) -> RawDocument:
        """Map XML entry element to RawDocument.
        url = canonical abs URL (strip version: /abs/2501.12345)
        raw_content = summary (abstract text)
        content_type = 'paper'
        metadata: {arxiv_id, categories: list[str], pdf_url}"""
```

**Arxiv ID handling**:
- Raw ID from feed: `http://arxiv.org/abs/2501.12345v1`
- Store canonical URL: `https://arxiv.org/abs/2501.12345` (strip version suffix)
- PDF URL: `https://arxiv.org/pdf/2501.12345`

**Authors**: ArXiv entries have multiple `<author>` elements. Join as comma-separated string.

**Rate limits**: ArXiv requests a 3-second delay between API calls. Enforce this.

**Content**: Use the `<summary>` (abstract) as `raw_content` — do NOT fetch PDFs. The abstract is sufficient for filtering, entity extraction, and embedding.

## Dependencies
- Depends on: task_05 (BaseIngestor)
- Packages needed: `httpx` (already in pyproject.toml); `xml.etree.ElementTree` (stdlib)

## Acceptance Criteria
- [ ] All queries in `config.sources.arxiv.queries` are fetched
- [ ] Papers limited to `config.sources.arxiv.max_results` per query
- [ ] Abstract used as `raw_content` — no PDF download
- [ ] Canonical URL stored (no version suffix)
- [ ] Multiple authors joined as comma-separated string in `author` field
- [ ] 3-second delay between ArXiv API calls
- [ ] `search_radar()` searches with arbitrary query string
- [ ] Unit tests parse fixture ArXiv Atom XML
- [ ] `metadata` includes `arxiv_id`, `categories`, `pdf_url`

## Notes
- Parse with `xml.etree.ElementTree` — no extra XML libraries needed
- ArXiv namespaces: `xmlns='http://www.w3.org/2005/Atom'`, `xmlns:arxiv='http://arxiv.org/schemas/atom'`
- Some papers have `<arxiv:comment>` with extra metadata — ignore for now
- Date filtering: `published` field is ISO 8601; compare against `datetime.utcnow() - timedelta(days=lookback)`
- No full-text extraction needed — abstract is the content
