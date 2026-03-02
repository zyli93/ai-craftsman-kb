"""FastAPI router for search endpoints.

Provides endpoints for hybrid search over indexed documents, with optional
export support via ``?format=`` query parameter or ``Accept`` header.

Supported formats:
- Default (no format param): JSON response with SearchResult objects.
- ``?format=markdown``: Returns a ``text/markdown`` response suitable for
  saving as a ``.md`` file or rendering in a Markdown viewer.
- ``?format=json``: Returns the same JSON as the default but as an explicit
  application/json response with formatted JSON body.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SearchResultResponse(BaseModel):
    """API response model for a single search result.

    Maps directly to the fields on
    :class:`~ai_craftsman_kb.search.hybrid.SearchResult`.
    """

    document_id: str
    score: float
    title: str | None
    url: str
    source_type: str
    author: str | None
    published_at: str | None
    excerpt: str | None
    origin: str


class SearchResponse(BaseModel):
    """Wrapper response containing search results and metadata."""

    query: str
    total: int
    mode: str
    results: list[SearchResultResponse]


# ---------------------------------------------------------------------------
# Dependency: database connection
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path.home() / ".ai-craftsman-kb" / "data"


async def get_connection() -> aiosqlite.Connection:
    """Yield an aiosqlite connection for use in route handlers.

    In production, the data_dir should be injected from AppConfig via
    FastAPI app state or a startup dependency. This default is used when
    the app is started without explicit config injection.

    Yields:
        An open aiosqlite connection.
    """
    from ..db.sqlite import get_db

    async with get_db(_DEFAULT_DATA_DIR) as conn:
        yield conn


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/search")
async def search_documents(
    q: str = Query(..., description="Search query string"),
    mode: str = Query(
        default="hybrid",
        description="Search mode: hybrid, semantic, or keyword",
    ),
    source: list[str] = Query(
        default=[],
        description="Filter by source type (repeatable: ?source=hn&source=arxiv)",
    ),
    since: str | None = Query(
        default=None,
        description="Filter results published after this date (YYYY-MM-DD)",
    ),
    limit: int = Query(default=20, ge=1, le=200, description="Maximum results"),
    format: str | None = Query(
        default=None,
        description="Export format: 'markdown' or 'json' (default: JSON response)",
    ),
    conn: aiosqlite.Connection = Depends(get_connection),
) -> Any:
    """Search indexed documents using hybrid (FTS5 + vector) search.

    Returns results as JSON by default. Pass ``?format=markdown`` to receive
    the results as a Markdown document with ``text/markdown`` content-type,
    suitable for saving as a ``.md`` file.

    Args:
        q: Search query string (required).
        mode: Search strategy — ``'hybrid'`` (default), ``'semantic'``, or
            ``'keyword'``.
        source: List of source types to restrict results to (optional).
        since: ISO date string lower bound for ``published_at`` (optional).
        limit: Maximum number of results to return (1-200, default 20).
        format: Optional export format — ``'markdown'`` or ``'json'``.
            When omitted, a structured JSON response is returned.
        conn: Database connection (injected by FastAPI dependency).

    Returns:
        - If ``format=markdown``: A ``text/markdown`` :class:`~fastapi.Response`.
        - If ``format=json``: An ``application/json`` :class:`~fastapi.Response`
          containing the raw JSON array.
        - Otherwise: A :class:`SearchResponse` Pydantic model (default JSON).

    Raises:
        HTTPException 400: If ``mode`` or ``format`` is not a valid value.
    """
    from fastapi import HTTPException

    # Validate mode
    valid_modes = {"hybrid", "semantic", "keyword"}
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}",
        )

    # Validate format
    valid_formats = {None, "markdown", "json"}
    if format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{format}'. Must be one of: markdown, json",
        )

    # Run the search
    try:
        from ..config import load_config
        from ..processing.embedder import Embedder
        from ..search.hybrid import HybridSearch
        from ..search.vector_store import VectorStore

        config = load_config()
        embedder = Embedder(config)
        vector_store = VectorStore(config)
        searcher = HybridSearch(config, vector_store, embedder)

        results = await searcher.search(
            conn=conn,
            query=q,
            mode=mode,
            source_types=source if source else None,
            since=since,
            limit=limit,
        )
    except Exception as exc:
        logger.error("Search failed for query %r: %s", q, exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    # Export mode: return plain text/markdown or raw JSON string
    if format == "markdown":
        from ..export import search_results_to_markdown

        generated_at = datetime.now(timezone.utc).isoformat()
        content = search_results_to_markdown(results, q, generated_at)
        return Response(content=content, media_type="text/markdown")

    if format == "json":
        from ..export import search_results_to_json

        content = search_results_to_json(results)
        return Response(content=content, media_type="application/json")

    # Default: structured JSON response
    return SearchResponse(
        query=q,
        total=len(results),
        mode=mode,
        results=[
            SearchResultResponse(
                document_id=r.document_id,
                score=r.score,
                title=r.title,
                url=r.url,
                source_type=r.source_type,
                author=r.author,
                published_at=r.published_at,
                excerpt=r.excerpt,
                origin=r.origin,
            )
            for r in results
        ],
    )
