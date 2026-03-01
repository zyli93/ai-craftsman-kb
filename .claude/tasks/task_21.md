# Task 21: Hybrid Search (FTS + Vector + RRF)

## Wave
Wave 9 (sequential — depends on tasks 03 and 20)
Domain: backend

## Objective
Implement the hybrid search engine that merges FTS5 keyword results and Qdrant vector results via Reciprocal Rank Fusion (RRF), producing a single ranked result list.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/search/hybrid.py` — `HybridSearch` class + RRF implementation
- `backend/ai_craftsman_kb/search/keyword.py` — `KeywordSearch` wrapper around FTS5
- `backend/ai_craftsman_kb/search/__init__.py` — Exports `HybridSearch`
- `backend/tests/test_search/test_hybrid.py`

### Key interfaces / implementation details:

**`KeywordSearch`** (`search/keyword.py`):
```python
class KeywordSearch:
    """Wrapper around SQLite FTS5 search."""

    async def search(
        self,
        conn: aiosqlite.Connection,
        query: str,
        limit: int = 50,
        source_types: list[str] | None = None,
        since: str | None = None,
    ) -> list[tuple[str, float]]:
        """FTS5 BM25 search. Returns [(document_id, normalized_score), ...].
        Normalizes BM25 scores to [0, 1] range by dividing by max score."""
```

FTS5 query:
```sql
SELECT d.id, bm25(documents_fts) as rank
FROM documents_fts
JOIN documents d ON d.rowid = documents_fts.rowid
WHERE documents_fts MATCH ?
  AND d.deleted_at IS NULL
  AND (? IS NULL OR d.source_type = ?)
  AND (? IS NULL OR d.published_at >= ?)
ORDER BY rank
LIMIT ?
```

**RRF algorithm** (from plan.md — semantic weight 0.6, keyword weight 0.4):
```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    RRF score for document d = sum(weight_i / (k + rank_i(d)))
    k=60 is the standard constant from the original RRF paper.
    weights: [semantic_weight, keyword_weight] e.g. [0.6, 0.4]
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**`HybridSearch`** (`search/hybrid.py`):
```python
class SearchResult(BaseModel):
    document_id: str
    score: float                   # RRF combined score
    title: str | None
    url: str
    source_type: str
    author: str | None
    published_at: str | None
    excerpt: str | None            # first 300 chars of raw_content
    origin: str

class HybridSearch:
    """Combine FTS5 + vector search via Reciprocal Rank Fusion."""

    def __init__(
        self,
        config: AppConfig,
        vector_store: VectorStore,
        embedder: Embedder,
    ) -> None: ...

    async def search(
        self,
        conn: aiosqlite.Connection,
        query: str,
        mode: Literal['hybrid', 'semantic', 'keyword'] = 'hybrid',
        source_types: list[str] | None = None,
        since: str | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        """
        hybrid: run both FTS + vector, merge via RRF
        semantic: vector search only
        keyword: FTS only
        Fetches document metadata from SQLite for each result.
        """

    async def _fetch_results(
        self,
        conn: aiosqlite.Connection,
        doc_ids: list[str],
        scores: dict[str, float],
    ) -> list[SearchResult]:
        """Batch fetch document rows for result doc_ids, build SearchResult list."""
```

**Weights** from settings.yaml:
- `search.hybrid_weight_semantic = 0.6`
- `search.hybrid_weight_keyword = 0.4`

## Dependencies
- Depends on: task_03 (FTS5 queries), task_20 (VectorStore), task_18 (Embedder for query embedding)
- Packages needed: none new

## Acceptance Criteria
- [ ] `mode='hybrid'` runs both FTS and vector and merges via RRF
- [ ] `mode='semantic'` skips FTS, returns vector results only
- [ ] `mode='keyword'` skips vector embedding, returns FTS results only
- [ ] RRF correctly applies `k=60` and weights `[0.6, 0.4]`
- [ ] `source_types` filter applied to both FTS and vector search
- [ ] `since` date filter applied
- [ ] `SearchResult.excerpt` is first 300 chars of `raw_content`
- [ ] Unit tests: mock VectorStore.search() and FTS results; verify RRF ordering

## Notes
- Embed query with `Embedder.embed_single(query)` for vector search
- Vector search returns document_ids from payload; FTS returns document_ids from rowid join
- RRF naturally handles documents appearing in only one result list (missing rank = no contribution)
- For `mode='keyword'`: no embedding call, so no API cost
- The `k=60` constant dampens the impact of absolute rank position — standard in literature
- FTS5 BM25 scores from SQLite are negative (lower = more relevant); invert/normalize before RRF
