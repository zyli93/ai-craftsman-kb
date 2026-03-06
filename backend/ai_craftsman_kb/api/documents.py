"""FastAPI router for document CRUD endpoints.

Provides:
- GET  /api/documents       — list documents with filters
- GET  /api/documents/{id}  — get a single document
- DELETE /api/documents/{id} — soft-delete a document
"""
from __future__ import annotations

import logging
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query

from ..db.models import DocumentRow
from ..db.queries import get_document, list_documents, soft_delete_document, update_document_user_fields
from .deps import get_conn
from .models import DocumentOut, UpdateDocumentRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _doc_row_to_out(doc: DocumentRow) -> DocumentOut:
    """Convert a DocumentRow to a DocumentOut response model.

    Generates the ``excerpt`` field (first 300 chars of raw_content) and
    maps all other fields directly.

    Args:
        doc: The database row model.

    Returns:
        A DocumentOut suitable for JSON serialisation.
    """
    excerpt: str | None = None
    if doc.raw_content:
        excerpt = doc.raw_content[:300]

    return DocumentOut(
        id=doc.id,
        title=doc.title,
        url=doc.url,
        source_type=doc.source_type,
        origin=doc.origin,
        author=doc.author,
        published_at=doc.published_at,
        fetched_at=doc.fetched_at,
        word_count=doc.word_count,
        is_embedded=doc.is_embedded,
        is_favorited=doc.is_favorited,
        is_archived=doc.is_archived,
        user_tags=doc.user_tags,
        excerpt=excerpt,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents_endpoint(
    origin: str | None = Query(
        default=None,
        description="Filter by origin: 'pro', 'radar', or 'adhoc'",
    ),
    source_type: str | None = Query(
        default=None,
        description="Filter by source type, e.g. 'hn', 'substack'",
    ),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Results to skip"),
    is_archived: bool = Query(default=False, description="Include archived documents"),
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> list[DocumentOut]:
    """List documents with optional filters.

    Returns documents ordered by published_at descending. Soft-deleted
    documents are always excluded.

    Args:
        origin: Optional origin filter ('pro', 'radar', 'adhoc').
        source_type: Optional source type filter.
        limit: Maximum results (1-100, default 50).
        offset: Pagination offset.
        is_archived: Include archived documents (default False).
        conn: DB connection (injected).

    Returns:
        List of DocumentOut objects.
    """
    rows = await list_documents(
        conn,
        origin=origin,
        source_type=source_type,
        limit=limit,
        offset=offset,
        include_archived=is_archived,
        include_deleted=False,
    )
    return [_doc_row_to_out(r) for r in rows]


@router.get("/documents/{doc_id}", response_model=DocumentOut)
async def get_document_endpoint(
    doc_id: str,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> DocumentOut:
    """Retrieve a single document by its UUID.

    Args:
        doc_id: The document UUID.
        conn: DB connection (injected).

    Returns:
        A DocumentOut response.

    Raises:
        HTTPException 404: If no document with the given ID exists.
    """
    doc = await get_document(conn, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return _doc_row_to_out(doc)


@router.put("/documents/{doc_id}", response_model=DocumentOut)
async def update_document_endpoint(
    doc_id: str,
    body: UpdateDocumentRequest,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> DocumentOut:
    """Update user-facing fields on a document (archive, favorite, tags).

    Args:
        doc_id: The document UUID.
        body: Fields to update.
        conn: DB connection (injected).

    Returns:
        The updated DocumentOut.

    Raises:
        HTTPException 404: If no document with the given ID exists.
    """
    doc = await get_document(conn, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    await update_document_user_fields(
        conn,
        doc_id,
        is_archived=body.is_archived,
        is_favorited=body.is_favorited,
        user_tags=body.user_tags,
    )

    updated = await get_document(conn, doc_id)
    return _doc_row_to_out(updated)


@router.delete("/documents/{doc_id}", response_model=dict)
async def delete_document_endpoint(
    doc_id: str,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> dict:
    """Soft-delete a document by setting its deleted_at timestamp.

    The document is NOT removed from the database — it is hidden from
    list/search results by default. Pass ``include_deleted=True`` to
    retrieve soft-deleted documents.

    Args:
        doc_id: The document UUID.
        conn: DB connection (injected).

    Returns:
        ``{"ok": True}`` on success.

    Raises:
        HTTPException 404: If no document with the given ID exists.
    """
    doc = await get_document(conn, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    await soft_delete_document(conn, doc_id)
    logger.info("Soft-deleted document %s", doc_id)
    return {"ok": True}
