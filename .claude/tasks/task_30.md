# Task 30: FastAPI REST API Layer

## Wave
Wave 13 (parallel with task 32)
Domain: backend

## Objective
Build the complete FastAPI REST API that exposes all backend capabilities (search, ingest, radar, entities, documents, sources, briefings, stats) for consumption by the dashboard and MCP server.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/server.py` — FastAPI app + all routers
- `backend/ai_craftsman_kb/api/` — Router modules (create directory)
  - `backend/ai_craftsman_kb/api/documents.py`
  - `backend/ai_craftsman_kb/api/search.py`
  - `backend/ai_craftsman_kb/api/sources.py`
  - `backend/ai_craftsman_kb/api/entities.py`
  - `backend/ai_craftsman_kb/api/radar.py`
  - `backend/ai_craftsman_kb/api/briefings.py`
  - `backend/ai_craftsman_kb/api/system.py`
- `backend/tests/test_api/test_*.py` — API tests using FastAPI TestClient

### Key interfaces / implementation details:

**All endpoint signatures** (from plan.md):

```python
# server.py — FastAPI app setup
app = FastAPI(title='AI Craftsman KB', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:3000'], ...)

# System
GET  /api/stats           → SystemStats
GET  /api/health          → {"status": "ok", "db": bool, "qdrant": bool}

# Documents
GET  /api/documents       → list[DocumentOut]
     ?origin=pro|radar|adhoc
     &source_type=hn|substack|...
     &limit=50&offset=0
     &include_archived=false
GET  /api/documents/{id}  → DocumentOut
DELETE /api/documents/{id} → {"ok": true}   # soft delete

# Search
GET  /api/search          → list[SearchResultOut]
     ?q=               (required)
     &mode=hybrid|semantic|keyword
     &source_type=...
     &since=2025-01-01
     &limit=20

# Ingest
POST /api/ingest/url      → DocumentOut
     body: {"url": str, "tags": list[str]}
POST /api/ingest/pro      → IngestReportOut
     body: {"source": str | null}

# Sources (reads from + writes to sources.yaml + sources table)
GET  /api/sources         → list[SourceOut]
POST /api/sources         → SourceOut
     body: {"source_type": str, "identifier": str, "display_name": str}
PUT  /api/sources/{id}    → SourceOut
     body: {"enabled": bool, "display_name": str}
DELETE /api/sources/{id}  → {"ok": true}
POST /api/sources/{id}/ingest → IngestReportOut   # trigger single source ingest

# Entities
GET  /api/entities        → list[EntityOut]
     ?q=&entity_type=&limit=50&offset=0
GET  /api/entities/{id}   → EntityWithDocsOut
GET  /api/entities/{id}/documents → list[DocumentOut]

# Radar
GET  /api/radar/results   → list[DocumentOut]
     ?status=pending|promoted|archived
POST /api/radar/search    → RadarReportOut
     body: {"query": str, "sources": list[str] | null, "limit_per_source": int}
POST /api/radar/results/{id}/promote → DocumentOut
POST /api/radar/results/{id}/archive → DocumentOut

# Briefings
GET  /api/briefings       → list[BriefingOut]
POST /api/briefings       → BriefingOut
     body: {"query": str, "limit": int, "run_radar": bool, "run_ingest": bool}
GET  /api/briefings/{id}  → BriefingOut
DELETE /api/briefings/{id} → {"ok": true}

# Discovery
GET  /api/discover        → list[DiscoveredSourceOut]
     ?status=suggested
```

**Response models** (Pydantic, all `Out` suffix):
```python
class DocumentOut(BaseModel):
    id: str
    title: str | None
    url: str
    source_type: str
    origin: str
    author: str | None
    published_at: str | None
    fetched_at: str
    word_count: int | None
    is_embedded: bool
    is_favorited: bool
    is_archived: bool
    user_tags: list[str]
    excerpt: str | None      # first 300 chars of raw_content

class SearchResultOut(BaseModel):
    document: DocumentOut
    score: float
    mode_used: str           # 'hybrid' | 'semantic' | 'keyword'

class SystemStats(BaseModel):
    total_documents: int
    embedded_documents: int
    total_entities: int
    active_sources: int
    total_briefings: int
    vector_count: int
    db_size_bytes: int

class IngestReportOut(BaseModel):
    source_type: str
    fetched: int
    stored: int
    embedded: int
    errors: list[str]
```

**App-level dependency injection**:
```python
# Shared instances across requests
@app.on_event('startup')
async def startup():
    app.state.config = load_config()
    app.state.db_path = _get_db_path(app.state.config)
    await init_db(app.state.db_path)
    app.state.vector_store = VectorStore(app.state.config)
    app.state.embedder = Embedder(app.state.config)
    app.state.llm_router = LLMRouter(app.state.config)
```

## Dependencies
- Depends on: task_03 (DB queries), task_21 (HybridSearch), task_26 (RadarEngine)
- Packages needed: `fastapi`, `uvicorn` (add to pyproject.toml)

## Acceptance Criteria
- [ ] All listed endpoints return correct HTTP status codes (200, 201, 404, 422)
- [ ] CORS configured for `http://localhost:3000`
- [ ] `GET /api/health` returns `{"status": "ok"}` with 200
- [ ] `GET /api/search?q=test` returns `list[SearchResultOut]`
- [ ] `POST /api/ingest/url` ingests a URL and returns `DocumentOut`
- [ ] `POST /api/radar/search` triggers RadarEngine and returns results
- [ ] `POST /api/briefings` generates and stores a briefing
- [ ] API tests use `fastapi.testclient.TestClient` with mocked DB and services
- [ ] OpenAPI docs available at `/docs`

## Notes
- Add `fastapi[standard]` and `uvicorn` to `pyproject.toml` dependencies
- Use `Annotated[..., Depends(...)]` pattern for DB connection per request
- `GET /api/search?q=` with empty query should return 422 (FastAPI validation)
- Pagination: all list endpoints support `limit` (max 100) and `offset`
- The `POST /api/ingest/pro` endpoint is async-triggering (returns immediately with report after completion — may take time)
