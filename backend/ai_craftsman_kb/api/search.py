"""FastAPI router for document search endpoint.

Provides:
- GET /api/search — hybrid/semantic/keyword search over documents
  with optional export format support (format=markdown|json).

Supported formats:
- Default (no format param): JSON response with SearchResultOut objects.
- ``?format=markdown``: Returns a ``text/markdown`` response suitable for
  saving as a ``.md`` file or rendering in a Markdown viewer.
- ``?format=json``: Returns the same JSON as the default but as an explicit
  application/json response with formatted JSON body.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

import aiosqlite
from fastapi import APIRouter, Depends, Query, Request, Response

from ..db.queries import get_document
from ..search.hybrid import HybridSearch
from .deps import get_conn
from .documents import _doc_row_to_out
from .models import DocumentOut, SearchResultOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=list[SearchResultOut])
async def search_documents(
    q: str = Query(
        ...,
        min_length=1,
        description="Search query (required, non-empty)",
    ),
    mode: Literal["hybrid", "semantic", "keyword"] = Query(
        default="hybrid",
        description="Search mode: hybrid, semantic, or keyword",
    ),
    source_type: str | None = Query(
        default=None,
        description="Comma-separated source types to restrict, e.g. 'hn,arxiv'",
    ),
    since: str | None = Query(
        default=None,
        description="ISO 8601 date lower bound for published_at, e.g. '2025-01-01'",
    ),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results"),
    format: str | None = Query(
        default=None,
        description="Export format: 'markdown' or 'json' (default: structured JSON)",
    ),
    request: Request = None,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> Any:
    """Search documents using hybrid, semantic, or keyword search.

    The default mode is 'hybrid', which combines FTS5 keyword search and
    Qdrant vector search via Reciprocal Rank Fusion (RRF).

    An empty ``q`` parameter returns HTTP 422 (handled automatically by
    FastAPI's ``min_length=1`` constraint).

    Pass ``?format=markdown`` to receive results as a Markdown document,
    suitable for saving as a ``.md`` file.

    Args:
        q: Search query string (required, minimum length 1).
        mode: Search strategy ('hybrid', 'semantic', 'keyword').
        source_type: Comma-separated source type filter (e.g. 'hn,arxiv').
        since: ISO 8601 date string for filtering by published_at.
        limit: Maximum results to return (1-100, default 20).
        format: Optional export format — 'markdown' or 'json'. Default is
                structured JSON with SearchResultOut objects.
        request: FastAPI request object (used to access app state services).
        conn: DB connection (injected).

    Returns:
        Either a list of SearchResultOut (default), or a formatted Response
        when ``format`` is specified.
    """
    # Validate export format
    valid_formats = {None, "markdown", "json"}
    if format not in valid_formats:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{format}'. Must be one of: markdown, json",
        )

    # Parse comma-separated source types
    source_types: list[str] | None = None
    if source_type:
        source_types = [s.strip() for s in source_type.split(",") if s.strip()]

    # Get shared service instances from app state
    vector_store = request.app.state.vector_store
    embedder = request.app.state.embedder
    config = request.app.state.config

    searcher = HybridSearch(config=config, vector_store=vector_store, embedder=embedder)

    results = await searcher.search(
        conn,
        query=q,
        mode=mode,
        source_types=source_types,
        since=since,
        limit=limit,
    )

    # Convert SearchResult objects to SearchResultOut response models.
    # Look up actual document metadata from DB to avoid returning fabricated values.
    output: list[SearchResultOut] = []
    for result in results:
        doc_row = await get_document(conn, result.document_id)
        if doc_row is not None:
            doc_out = _doc_row_to_out(doc_row)
        else:
            # Fallback if document was deleted between search and lookup
            doc_out = DocumentOut(
                id=result.document_id,
                title=result.title,
                url=result.url,
                source_type=result.source_type,
                origin=result.origin,
                author=result.author,
                published_at=result.published_at,
                fetched_at="",
                word_count=None,
                is_embedded=True,
                is_favorited=False,
                is_archived=False,
                user_tags=[],
                excerpt=result.excerpt,
            )
        output.append(
            SearchResultOut(
                document=doc_out,
                score=result.score,
                mode_used=mode,
            )
        )

    # Handle export formats
    if format == "markdown":
        try:
            from ..export import search_results_to_markdown
            # Convert to the SearchResult objects the export module expects
            generated_at = datetime.now(timezone.utc).isoformat()
            content = search_results_to_markdown(results, q, generated_at)
            return Response(content=content, media_type="text/markdown")
        except (ImportError, Exception) as e:
            logger.warning("Markdown export failed, returning JSON: %s", e)

    if format == "json":
        try:
            from ..export import search_results_to_json
            content = search_results_to_json(results)
            return Response(content=content, media_type="application/json")
        except (ImportError, Exception) as e:
            logger.warning("JSON export failed, returning structured response: %s", e)

    return output
