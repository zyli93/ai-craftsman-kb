# Task 03: SQLite Schema + Migrations + Queries

## Wave
Wave 2 (parallel with tasks: 02, 04, 05)
Domain: backend

## Objective
Define the full SQLite schema, implement the migration/init runner, and provide async query helper functions covering every read/write pattern used by the app.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/db/sqlite.py` — Schema DDL, `init_db()`, connection helper
- `backend/ai_craftsman_kb/db/queries.py` — All async CRUD + FTS query functions
- `backend/ai_craftsman_kb/db/models.py` — Pydantic row models (DocumentRow, EntityRow, etc.)
- `backend/ai_craftsman_kb/db/__init__.py` — Exports `init_db`, `get_db`, row models
- `backend/tests/test_db.py` — Unit tests using in-memory SQLite

### Key interfaces / implementation details:

**Full SQLite schema** (from plan.md — embed verbatim in `sqlite.py`):

```sql
CREATE TABLE IF NOT EXISTS sources (
    id              TEXT PRIMARY KEY,              -- UUID
    source_type     TEXT NOT NULL,                 -- 'substack','youtube','reddit','rss','hn','arxiv','devto'
    identifier      TEXT NOT NULL,                 -- slug, handle, subreddit name, feed URL, etc.
    display_name    TEXT,
    tier            TEXT NOT NULL DEFAULT 'pro',   -- 'pro' only for now
    enabled         BOOLEAN DEFAULT TRUE,
    last_fetched_at TIMESTAMP,
    fetch_error     TEXT,
    config          JSON,                          -- source-specific config snapshot
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_type, identifier)
);

CREATE TABLE IF NOT EXISTS documents (
    id                      TEXT PRIMARY KEY,      -- UUID
    source_id               TEXT REFERENCES sources(id) ON DELETE SET NULL,
    origin                  TEXT NOT NULL,         -- 'pro' | 'radar' | 'adhoc'
    source_type             TEXT NOT NULL,
    url                     TEXT UNIQUE NOT NULL,
    title                   TEXT,
    author                  TEXT,
    published_at            TIMESTAMP,
    fetched_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_type            TEXT,                  -- 'article' | 'video' | 'paper' | 'post'
    raw_content             TEXT,                  -- full extracted text
    word_count              INTEGER,
    metadata                JSON,
    is_embedded             BOOLEAN DEFAULT FALSE,
    is_entities_extracted   BOOLEAN DEFAULT FALSE,
    filter_score            REAL,
    filter_passed           BOOLEAN,
    is_favorited            BOOLEAN DEFAULT FALSE,
    is_archived             BOOLEAN DEFAULT FALSE,
    user_tags               JSON DEFAULT '[]',
    user_notes              TEXT,
    promoted_at             TIMESTAMP,
    deleted_at              TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_source    ON documents(source_type, origin);
CREATE INDEX IF NOT EXISTS idx_documents_date      ON documents(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_processing ON documents(is_embedded, is_entities_extracted);
CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_origin    ON documents(origin);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title, raw_content, author,
    content='documents',
    tokenize='porter unicode61'
);

-- FTS sync triggers for documents
CREATE TRIGGER IF NOT EXISTS documents_fts_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, raw_content, author)
    VALUES (rowid(new.id), new.title, new.raw_content, new.author);
END;

CREATE TRIGGER IF NOT EXISTS documents_fts_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, raw_content, author)
    VALUES ('delete', rowid(old.id), old.title, old.raw_content, old.author);
END;

CREATE TRIGGER IF NOT EXISTS documents_fts_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, raw_content, author)
    VALUES ('delete', rowid(old.id), old.title, old.raw_content, old.author);
    INSERT INTO documents_fts(rowid, title, raw_content, author)
    VALUES (rowid(new.id), new.title, new.raw_content, new.author);
END;

CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,              -- UUID
    name            TEXT NOT NULL,
    entity_type     TEXT NOT NULL,                 -- 'person'|'company'|'technology'|'event'|'book'|'paper'|'product'
    normalized_name TEXT NOT NULL,
    description     TEXT,
    first_seen_at   TIMESTAMP,
    mention_count   INTEGER DEFAULT 1,
    metadata        JSON,
    UNIQUE(normalized_name, entity_type)
);

CREATE TABLE IF NOT EXISTS document_entities (
    document_id     TEXT REFERENCES documents(id) ON DELETE CASCADE,
    entity_id       TEXT REFERENCES entities(id) ON DELETE CASCADE,
    context         TEXT,
    relevance       TEXT,
    PRIMARY KEY (document_id, entity_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    name, normalized_name, description,
    content='entities',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS discovered_sources (
    id                          TEXT PRIMARY KEY,  -- UUID
    source_type                 TEXT NOT NULL,
    identifier                  TEXT NOT NULL,
    display_name                TEXT,
    discovered_from_document_id TEXT REFERENCES documents(id),
    discovery_method            TEXT,              -- 'outbound_link'|'citation'|'mention'|'llm_suggestion'
    confidence                  REAL,
    status                      TEXT DEFAULT 'suggested', -- 'suggested'|'added'|'dismissed'
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_type, identifier)
);

CREATE TABLE IF NOT EXISTS briefings (
    id                  TEXT PRIMARY KEY,          -- UUID
    title               TEXT NOT NULL,
    query               TEXT,
    content             TEXT NOT NULL,             -- markdown
    source_document_ids JSON DEFAULT '[]',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    format              TEXT DEFAULT 'markdown'
);
```

**Note on FTS5 + UUID primary keys**: FTS5 `content=` tables use `rowid` internally. Since documents use UUID TEXT PKs, maintain a separate rowid mapping OR use `WITHOUT ROWID` tables — simplest solution: keep `rowid` implicit (SQLite assigns it) and use it for FTS, but query by UUID via the content table.

**Connection helper** (`sqlite.py`):
```python
@asynccontextmanager
async def get_db(data_dir: Path) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Yield a configured aiosqlite connection with WAL mode and FK enforcement."""

async def init_db(data_dir: Path) -> None:
    """Create all tables, indexes, triggers. Idempotent (IF NOT EXISTS)."""
```

**Key query signatures** (`queries.py`):
```python
# Documents
async def upsert_document(conn, doc: DocumentRow) -> str: ...    # returns id (UUID)
async def get_document(conn, doc_id: str) -> DocumentRow | None: ...
async def get_document_by_url(conn, url: str) -> DocumentRow | None: ...
async def list_documents(
    conn, origin: str | None = None, source_type: str | None = None,
    limit: int = 50, offset: int = 0,
    include_archived: bool = False, include_deleted: bool = False,
) -> list[DocumentRow]: ...
async def update_document_flags(
    conn, doc_id: str,
    is_embedded: bool | None = None,
    is_entities_extracted: bool | None = None,
    filter_score: float | None = None,
    filter_passed: bool | None = None,
) -> None: ...
async def soft_delete_document(conn, doc_id: str) -> None: ...
async def search_documents_fts(conn, query: str, limit: int = 20) -> list[tuple[str, float]]: ...
    # Returns [(doc_id, bm25_rank), ...]

# Sources
async def upsert_source(conn, source: SourceRow) -> str: ...
async def update_source_fetch_status(
    conn, source_id: str,
    last_fetched_at: str | None = None,
    fetch_error: str | None = None,
) -> None: ...
async def list_sources(conn, enabled_only: bool = False) -> list[SourceRow]: ...

# Entities
async def upsert_entity(conn, entity: EntityRow) -> str: ...
async def link_document_entity(conn, document_id: str, entity_id: str, context: str = '') -> None: ...
async def search_entities_fts(conn, query: str, limit: int = 20) -> list[EntityRow]: ...
async def get_entity_documents(conn, entity_id: str, limit: int = 20) -> list[DocumentRow]: ...

# Discovered sources
async def upsert_discovered_source(conn, source: DiscoveredSourceRow) -> str: ...
async def list_discovered_sources(conn, status: str = 'suggested') -> list[DiscoveredSourceRow]: ...
async def update_discovered_source_status(conn, source_id: str, status: str) -> None: ...

# Briefings
async def insert_briefing(conn, briefing: BriefingRow) -> str: ...
async def list_briefings(conn, limit: int = 20) -> list[BriefingRow]: ...
async def get_briefing(conn, briefing_id: str) -> BriefingRow | None: ...

# Stats
async def get_stats(conn) -> dict[str, int]:
    """Returns: {total_documents, total_entities, total_sources, total_briefings,
                 embedded_documents, unembedded_documents}"""
```

**Row models** (`models.py`):
```python
class DocumentRow(BaseModel):
    id: str                                    # UUID, generated if None
    source_id: str | None = None
    origin: Literal['pro', 'radar', 'adhoc']
    source_type: str
    url: str
    title: str | None = None
    author: str | None = None
    published_at: str | None = None            # ISO 8601
    fetched_at: str = Field(default_factory=utcnow_iso)
    content_type: str | None = None
    raw_content: str | None = None
    word_count: int | None = None
    metadata: dict = {}
    is_embedded: bool = False
    is_entities_extracted: bool = False
    filter_score: float | None = None
    filter_passed: bool | None = None
    is_favorited: bool = False
    is_archived: bool = False
    user_tags: list[str] = []
    user_notes: str | None = None
    promoted_at: str | None = None
    deleted_at: str | None = None
```

## Dependencies
- Depends on: task_01 (project structure exists)
- Packages needed: `aiosqlite`, `uuid` stdlib (already available)

## Acceptance Criteria
- [ ] `init_db()` creates all tables, indexes, triggers idempotently (safe to call on existing DB)
- [ ] WAL mode and foreign keys enabled on every connection via PRAGMA
- [ ] `upsert_document()` uses `INSERT OR REPLACE` and returns the UUID
- [ ] `search_documents_fts()` returns results ranked by BM25 score
- [ ] JSON columns (`metadata`, `user_tags`, `source_document_ids`) serialized/deserialized automatically
- [ ] All queries use parameterized `?` placeholders — no f-string SQL
- [ ] Tests use `:memory:` SQLite and cover all CRUD paths + FTS search
- [ ] `get_stats()` returns accurate counts for all six stat keys

## Notes
- FTS5 with `content='documents'` + `tokenize='porter unicode61'` matches plan.md exactly
- For UUID TEXT PKs + FTS5 rowid: use SQLite's implicit rowid alongside UUID PK — query by UUID, FTS5 uses rowid internally; join on `rowid = documents.rowid`
- Enable with: `PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;`
- JSON columns stored as TEXT; serialize with `json.dumps()` in write, `json.loads()` in read
- `briefings` and `discovered_sources` tables not in original plan.md schema extract but required by later tasks — include them here
