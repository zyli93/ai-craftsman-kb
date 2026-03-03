# Task 23: Entity Dedup + FTS Search

## Wave
Wave 8 (parallel with task 20; depends on tasks 03 and 22)
Domain: backend

## Objective
Implement entity search and browsing via FTS5 on the entities table, entity deduplication on merge, and query functions for the entity explorer page and CLI.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/db/queries.py` — Add entity search + dedup query functions (extends task_03)
- `backend/ai_craftsman_kb/search/entity_search.py` — `EntitySearch` service class
- `backend/tests/test_search/test_entity_search.py`

### Key interfaces / implementation details:

**`EntitySearch`** (`search/entity_search.py`):
```python
class EntityWithDocs(BaseModel):
    entity: EntityRow
    document_count: int
    top_documents: list[DocumentRow]   # up to 5 most recent

class EntitySearch:
    """Entity search, browse, and dedup."""

    async def search(
        self,
        conn: aiosqlite.Connection,
        query: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[EntityRow]:
        """FTS5 search over entity name + normalized_name + description.
        Optionally filter by entity_type.
        Returns entities sorted by relevance, then mention_count."""

    async def list_entities(
        self,
        conn: aiosqlite.Connection,
        entity_type: str | None = None,
        sort_by: Literal['mention_count', 'first_seen_at', 'name'] = 'mention_count',
        limit: int = 50,
        offset: int = 0,
    ) -> list[EntityRow]:
        """Browse entities with optional type filter. Sorted by mention_count DESC."""

    async def get_entity_with_docs(
        self,
        conn: aiosqlite.Connection,
        entity_id: str,
        limit: int = 20,
    ) -> EntityWithDocs | None:
        """Get entity + documents that mention it, sorted by published_at DESC."""

    async def merge_entities(
        self,
        conn: aiosqlite.Connection,
        canonical_id: str,
        duplicate_ids: list[str],
    ) -> None:
        """Merge duplicate entities into the canonical entity.
        1. Reassign all document_entities from duplicate_ids to canonical_id
        2. Sum mention_counts
        3. Delete duplicate entity rows
        4. Update entities_fts"""
```

**FTS entity search query**:
```sql
SELECT e.id, e.name, e.entity_type, e.mention_count, e.normalized_name,
       bm25(entities_fts) as rank
FROM entities_fts
JOIN entities e ON e.rowid = entities_fts.rowid
WHERE entities_fts MATCH ?
  AND (? IS NULL OR e.entity_type = ?)
ORDER BY rank, e.mention_count DESC
LIMIT ?
```

**Co-occurrence query** (related entities — for entity detail page):
```sql
SELECT e2.id, e2.name, e2.entity_type, COUNT(*) as co_count
FROM document_entities de1
JOIN document_entities de2 ON de1.document_id = de2.document_id AND de2.entity_id != de1.entity_id
JOIN entities e2 ON e2.id = de2.entity_id
WHERE de1.entity_id = ?
GROUP BY e2.id
ORDER BY co_count DESC
LIMIT 10
```

Add `get_co_occurring_entities()` to `EntitySearch` using this query.

**FTS5 trigger sync**: Add entity FTS triggers to schema (extend task_03's `init_db`):
```sql
CREATE TRIGGER IF NOT EXISTS entities_fts_ai AFTER INSERT ON entities ...
CREATE TRIGGER IF NOT EXISTS entities_fts_ad AFTER DELETE ON entities ...
CREATE TRIGGER IF NOT EXISTS entities_fts_au AFTER UPDATE ON entities ...
```

## Dependencies
- Depends on: task_03 (entities + entities_fts schema), task_22 (EntityRow model)
- Packages needed: none new

## Acceptance Criteria
- [ ] `search()` returns entities matching FTS query, filtered by type if provided
- [ ] `list_entities()` returns entities sorted by mention_count
- [ ] `get_entity_with_docs()` returns entity + linked documents
- [ ] `get_co_occurring_entities()` returns top 10 co-occurring entities
- [ ] `merge_entities()` reassigns document links and deletes duplicates
- [ ] FTS5 triggers keep entities_fts in sync (test: insert entity, FTS search finds it)
- [ ] Unit tests with in-memory SQLite cover all search paths

## Notes
- Entity dedup is manual (via `merge_entities()`) triggered from UI or CLI — no automatic dedup
- `normalized_name` alone isn't enough for dedup (e.g. "Apple" company vs "Apple" product) — human confirmation needed
- Co-occurrence data powers the "Related entities" section in the entity detail panel (see wireframe in plan.md)
- Empty query string in `search()` should fall back to `list_entities()` behavior
