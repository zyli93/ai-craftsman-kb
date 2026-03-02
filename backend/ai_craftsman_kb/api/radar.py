"""FastAPI router for radar search endpoints.

Provides:
- GET  /api/radar/results               — list radar documents by status
- POST /api/radar/search                — run a radar search
- POST /api/radar/results/{id}/promote  — promote a radar document
- POST /api/radar/results/{id}/archive  — archive a radar document
"""
from __future__ import annotations

import logging
from typing import Annotated, Literal

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..db.queries import (
    archive_document,
    get_document,
    list_documents,
    promote_document,
)
from ..ingestors.base import BaseIngestor
from ..ingestors.runner import INGESTORS
from ..radar.engine import RadarEngine
from .deps import get_conn
from .documents import _doc_row_to_out
from .models import DocumentOut, RadarReportOut, RadarSearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["radar"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/radar/results", response_model=list[DocumentOut])
async def list_radar_results(
    status: str = Query(
        default="pending",
        description="Filter by radar status: 'pending', 'promoted', or 'archived'",
    ),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum results"),
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> list[DocumentOut]:
    """List radar documents filtered by their status.

    - ``pending``: radar documents not yet promoted or archived
    - ``promoted``: radar documents that have been promoted (promoted_at IS NOT NULL)
    - ``archived``: radar documents that are archived (is_archived = TRUE)

    Args:
        status: Status filter ('pending', 'promoted', 'archived').
        limit: Maximum results (1-100, default 50).
        conn: DB connection (injected).

    Returns:
        List of DocumentOut objects with origin='radar'.
    """
    allowed = {"pending", "promoted", "archived"}
    if status not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{status}'. Must be one of: {sorted(allowed)}",
        )

    if status == "archived":
        # Archived radar docs: origin=radar, is_archived=TRUE
        async with conn.execute(
            """
            SELECT * FROM documents
            WHERE origin = 'radar'
              AND is_archived = TRUE
              AND deleted_at IS NULL
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    elif status == "promoted":
        # Promoted radar docs: origin=radar, promoted_at IS NOT NULL
        async with conn.execute(
            """
            SELECT * FROM documents
            WHERE origin = 'radar'
              AND promoted_at IS NOT NULL
              AND is_archived = FALSE
              AND deleted_at IS NULL
            ORDER BY promoted_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    else:
        # Pending: origin=radar, not promoted, not archived
        async with conn.execute(
            """
            SELECT * FROM documents
            WHERE origin = 'radar'
              AND promoted_at IS NULL
              AND is_archived = FALSE
              AND deleted_at IS NULL
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()

    from ..db.queries import _row_to_dict
    from ..db.models import DocumentRow

    doc_rows = [DocumentRow(**_row_to_dict(row)) for row in rows]
    return [_doc_row_to_out(d) for d in doc_rows]


@router.post("/radar/search", response_model=RadarReportOut)
async def radar_search(
    body: RadarSearchRequest,
    request: Request = None,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> RadarReportOut:
    """Run an on-demand radar search across sources.

    Fans out concurrently to all configured radar-capable sources (or a
    specified subset), deduplicates results, and stores new documents with
    origin='radar'.

    Args:
        body: Request body with ``query``, optional ``sources`` filter, and
              ``limit_per_source``.
        request: FastAPI request (for app state access).
        conn: DB connection (injected).

    Returns:
        RadarReportOut summarising results.
    """
    config = request.app.state.config

    # Build ingestor instances for all available source types
    ingestors: dict[str, BaseIngestor] = {
        st: cls(config)
        for st, cls in INGESTORS.items()
    }

    engine = RadarEngine(config=config, ingestors=ingestors)
    report = await engine.search(
        conn,
        query=body.query,
        sources=body.sources,
        limit_per_source=body.limit_per_source,
    )

    return RadarReportOut(
        query=report.query,
        total_found=report.total_found,
        new_documents=report.new_documents,
        sources_searched=report.sources_searched,
        errors=report.errors,
    )


@router.post("/radar/results/{doc_id}/promote", response_model=DocumentOut)
async def promote_radar_document(
    doc_id: str,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> DocumentOut:
    """Promote a radar document, surfacing it alongside pro documents.

    Sets ``promoted_at`` to the current timestamp. Promoted documents
    appear in regular search results alongside pro-tier content.

    Args:
        doc_id: The document UUID.
        conn: DB connection (injected).

    Returns:
        The updated DocumentOut.

    Raises:
        HTTPException 404: If the document does not exist.
        HTTPException 422: If the document is not a radar document.
    """
    doc = await get_document(conn, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    if doc.origin != "radar":
        raise HTTPException(
            status_code=422,
            detail=f"Document '{doc_id}' is not a radar document (origin={doc.origin!r})",
        )

    await promote_document(conn, doc_id)
    logger.info("Promoted radar document %s", doc_id)

    updated = await get_document(conn, doc_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve updated document")

    return _doc_row_to_out(updated)


@router.post("/radar/results/{doc_id}/archive", response_model=DocumentOut)
async def archive_radar_document(
    doc_id: str,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> DocumentOut:
    """Archive a radar document, hiding it from the pending radar view.

    Sets ``is_archived`` to True. Archived documents can still be retrieved
    via ``GET /api/radar/results?status=archived``.

    Args:
        doc_id: The document UUID.
        conn: DB connection (injected).

    Returns:
        The updated DocumentOut.

    Raises:
        HTTPException 404: If the document does not exist.
        HTTPException 422: If the document is not a radar document.
    """
    doc = await get_document(conn, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    if doc.origin != "radar":
        raise HTTPException(
            status_code=422,
            detail=f"Document '{doc_id}' is not a radar document (origin={doc.origin!r})",
        )

    await archive_document(conn, doc_id)
    logger.info("Archived radar document %s", doc_id)

    updated = await get_document(conn, doc_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve updated document")

    return _doc_row_to_out(updated)
