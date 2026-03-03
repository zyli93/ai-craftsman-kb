# Task 20: Qdrant Local Setup + Vector Store

## Wave
Wave 8 (parallel with task 23; depends on task 18)
Domain: backend

## Objective
Set up local Qdrant, create the collection, and implement a `VectorStore` wrapper for inserting, searching, and deleting embedding vectors.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/search/vector_store.py` — `VectorStore` class
- `backend/tests/test_search/test_vector_store.py`

### Key interfaces / implementation details:

**Qdrant collection schema** (from plan.md):
```python
COLLECTION_NAME = 'ai_craftsman_kb'
VECTOR_SIZE = 1536      # OpenAI text-embedding-3-small; or 768 for local nomic-embed-text

# Collection config
VectorParams(
    size=VECTOR_SIZE,
    distance=Distance.COSINE,
)

# Point payload schema (fields stored alongside each vector):
{
    "document_id": str,      # UUID of parent document
    "source_type": str,      # 'hn', 'substack', etc.
    "origin": str,           # 'pro' | 'radar' | 'adhoc'
    "title": str,
    "author": str | None,
    "published_at": str | None,   # ISO 8601 for range filtering
    "chunk_index": int,      # 0 for first chunk, N for Nth
    "total_chunks": int,     # total chunks in this document
}
```

**`VectorStore`** (`search/vector_store.py`):
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

class VectorStore:
    """Local Qdrant vector store for document chunk embeddings."""

    def __init__(self, config: AppConfig) -> None:
        qdrant_path = Path(config.settings.data_dir).expanduser() / 'qdrant'
        self._client = QdrantClient(path=str(qdrant_path))
        self._vector_size = config.settings.embedding.dimensions_for_provider()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        names = [c.name for c in collections]
        if COLLECTION_NAME not in names:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )

    def upsert_vectors(
        self,
        document_id: str,
        chunks: list[TextChunk],
        vectors: list[list[float]],
        payload: dict,
    ) -> list[str]:
        """Insert or update vectors for a document's chunks.
        Generates UUIDs for each point. Returns list of point UUIDs."""

    def search(
        self,
        query_vector: list[float],
        limit: int = 20,
        source_types: list[str] | None = None,
        origin: str | None = None,
        since: str | None = None,     # ISO 8601 date string
    ) -> list[tuple[str, float]]:
        """Cosine similarity search. Returns [(document_id, score), ...].
        Applies source_type, origin, and date filters via Qdrant payload filtering."""

    def delete_document(self, document_id: str) -> None:
        """Delete all vectors for a document by filtering on document_id payload."""

    def get_collection_info(self) -> dict:
        """Return {vectors_count, disk_size_bytes}."""
```

**Qdrant filter** (for search with source_type filter):
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

def _build_filter(source_types, origin, since) -> Filter | None:
    conditions = []
    if source_types:
        conditions.append(FieldCondition(key='source_type', match=MatchAny(any=source_types)))
    if origin:
        conditions.append(FieldCondition(key='origin', match=MatchValue(value=origin)))
    if since:
        conditions.append(FieldCondition(key='published_at', range=Range(gte=since)))
    return Filter(must=conditions) if conditions else None
```

**Point ID generation**: Use `uuid.uuid4()` for each chunk point. Store the UUID back to the DB (task_24 will write it to a chunks tracking column added to `documents` or a separate chunks table).

**Embedding dimensions helper** — add to `EmbeddingConfig`:
```python
def dimensions_for_provider(self) -> int:
    if self.provider == 'openai':
        return 1536
    elif 'nomic' in self.model:
        return 768
    else:
        return 384  # MiniLM default
```

## Dependencies
- Depends on: task_18 (EmbeddingResult, vector format), task_02 (AppConfig with qdrant path)
- Packages needed: `qdrant-client` (already in pyproject.toml)

## Acceptance Criteria
- [ ] `_ensure_collection()` creates collection if not present, no-ops if exists
- [ ] `upsert_vectors()` inserts one Qdrant point per chunk with correct payload
- [ ] `search()` returns `(document_id, score)` tuples sorted by cosine similarity
- [ ] `source_type` filter correctly limits results
- [ ] `delete_document()` removes all points for a given document_id
- [ ] Local Qdrant storage created under `settings.data_dir/qdrant/`
- [ ] Unit tests use Qdrant in-memory mode (`QdrantClient(":memory:")`)

## Notes
- Local Qdrant uses file-based storage at `data_dir/qdrant/` — no Docker needed
- `QdrantClient(path=...)` for local file storage; `QdrantClient(":memory:")` for tests
- `qdrant-client` Python SDK is synchronous — wrap in `asyncio.get_event_loop().run_in_executor()` if needed, or call from sync context via task_24's pipeline
- Qdrant search returns best score per point (chunk), but we want document-level results: group by `document_id`, take max score per document
- For the search output, deduplicate by `document_id` and return the highest-scoring chunk's score
