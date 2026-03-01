# Task 31: MCP Server (Python MCP SDK)

## Wave
Wave 15 (parallel with tasks 40, 41; depends on task 30)
Domain: backend

## Objective
Implement the MCP server using the Python MCP SDK, exposing all tools needed for AI agents to search, ingest, and manage the knowledge base. The MCP server runs alongside or independently from the FastAPI server.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/mcp_server.py` — MCP server with all tool definitions

### Key interfaces / implementation details:

**All MCP tool definitions** (from plan.md — exact signatures):

```python
from mcp.server import FastMCP

mcp = FastMCP('ai-craftsman-kb')

@mcp.tool()
async def search(
    query: str,
    mode: str = 'hybrid',
    sources: list[str] | None = None,
    since: str | None = None,
    limit: int = 20,
    entity_type: str | None = None,
) -> list[dict]:
    """Search indexed content across all sources.
    mode: 'hybrid' | 'semantic' | 'keyword'
    sources: filter by source types e.g. ['hn', 'arxiv']
    since: ISO 8601 date string e.g. '2025-01-01'
    Returns list of {id, title, url, source_type, author, published_at, excerpt, score}"""

@mcp.tool()
async def radar(
    query: str,
    sources: list[str] | None = None,
    since: str | None = None,
    max_results_per_source: int = 10,
) -> list[dict]:
    """Search the open web for a topic, ingest and index results.
    Returns list of newly found documents."""

@mcp.tool()
async def ingest(
    source_type: str | None = None,
) -> dict:
    """Pull latest content from pro-tier sources.
    source_type: if provided, only ingest this source type.
    Returns {fetched, stored, embedded, errors}"""

@mcp.tool()
async def ingest_url(
    url: str,
    tags: list[str] | None = None,
) -> dict:
    """Ingest a single URL (article, YouTube video, ArXiv paper, etc.) into the index.
    Returns the ingested document {id, title, url, source_type}"""

@mcp.tool()
async def briefing(
    topic: str,
    run_radar: bool = True,
    run_ingest: bool = True,
) -> dict:
    """Generate a content briefing on a topic.
    Optionally runs radar search and fresh ingest before generating.
    Returns {title, content, source_count, created_at}"""

@mcp.tool()
async def get_entities(
    query: str | None = None,
    entity_type: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Search and browse extracted entities.
    entity_type: 'person'|'company'|'technology'|'event'|'book'|'paper'|'product'
    Returns list of {id, name, entity_type, mention_count}"""

@mcp.tool()
async def get_stats() -> dict:
    """Get system stats: document counts, embedding coverage, source health.
    Returns {total_documents, embedded_documents, total_entities, active_sources, vector_count}"""

@mcp.tool()
async def manage_source(
    action: str,
    source_type: str,
    identifier: str,
    display_name: str | None = None,
) -> dict:
    """Add, remove, or modify a source subscription.
    action: 'add' | 'remove' | 'enable' | 'disable'
    Returns updated source info."""

@mcp.tool()
async def tag_document(
    document_id: str,
    tags: list[str],
    action: str = 'add',
) -> dict:
    """Tag or untag a document.
    action: 'add' | 'remove' | 'set'
    Returns updated document."""

@mcp.tool()
async def discover_sources(
    based_on: str = 'recent',
    limit: int = 5,
) -> list[dict]:
    """Suggest new sources based on existing content.
    based_on: 'recent' | 'popular' | 'entities'
    Returns list of {source_type, identifier, display_name, confidence, discovery_method}"""
```

**MCP server startup** (`mcp_server.py`):
```python
def create_mcp_server(config: AppConfig) -> FastMCP:
    """Create and configure the MCP server with shared state."""
    # Tools call the same service layer as FastAPI — reuse IngestRunner, HybridSearch, RadarEngine
    # DB connection opened per tool call (not shared across calls)

def run_mcp_server(config: AppConfig) -> None:
    """Run the MCP server (stdio transport for Claude Desktop)."""
    mcp.run()
```

**Integration with FastAPI**: The MCP server can run in the same process as FastAPI (via `asyncio`) or as a separate `cr mcp` process. Both modes supported.

## Dependencies
- Depends on: task_30 (FastAPI — reuses same service layer)
- Packages needed: `mcp` Python SDK (add to pyproject.toml: `mcp>=1.0.0`)

## Acceptance Criteria
- [ ] All 10 MCP tools defined and callable
- [ ] `search()` tool returns list of dicts with correct fields
- [ ] `ingest_url()` tool stores a new document and returns its metadata
- [ ] `briefing()` tool returns generated content
- [ ] Each tool has accurate docstring (used as tool description by Claude)
- [ ] MCP server starts successfully with `mcp.run()` (stdio transport)
- [ ] Tools tested via MCP Python SDK's test utilities or mocked service layer

## Notes
- `mcp` Python SDK package: install as `mcp[cli]` for development utilities
- MCP tools should return plain dicts (JSON-serializable), not Pydantic models
- Docstrings on each `@mcp.tool()` function are the tool descriptions shown to Claude — make them precise and include parameter explanations
- Tool errors should raise `ValueError` or `RuntimeError` with a descriptive message — MCP SDK surfaces this to the client
- `cr mcp` CLI command (task_40) starts the MCP server via stdio transport
