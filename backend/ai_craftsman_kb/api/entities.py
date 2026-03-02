"""FastAPI router for entity endpoints.

Provides:
- GET /api/entities          — list/search entities
- GET /api/entities/{id}     — get entity with linked documents
- GET /api/entities/{id}/documents — get documents linked to an entity
"""
from __future__ import annotations

import logging
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query

from ..db.models import DocumentRow, EntityRow
from ..db.queries import get_entity_documents, list_documents, search_entities_fts
from .deps import get_conn
from .documents import _doc_row_to_out
from .models import DocumentOut, EntityOut, EntityWithDocsOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["entities"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity_row_to_out(entity: EntityRow) -> EntityOut:
    """Convert an EntityRow to an EntityOut response model.

    Args:
        entity: The DB row model.

    Returns:
        EntityOut suitable for JSON serialisation.
    """
    return EntityOut(
        id=entity.id,
        name=entity.name,
        entity_type=entity.entity_type,
        normalized_name=entity.normalized_name,
        description=entity.description,
        mention_count=entity.mention_count,
        first_seen_at=entity.first_seen_at,
    )


async def _get_entity_by_id(
    conn: aiosqlite.Connection,
    entity_id: str,
) -> EntityRow | None:
    """Fetch a single entity row by its UUID.

    Args:
        conn: An open aiosqlite connection.
        entity_id: The entity's UUID string.

    Returns:
        An EntityRow if found, or None.
    """
    async with conn.execute(
        "SELECT * FROM entities WHERE id = ?",
        (entity_id,),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        return None

    from ..db.queries import _row_to_dict

    return EntityRow(**_row_to_dict(row))


async def _list_entities(
    conn: aiosqlite.Connection,
    q: str | None,
    entity_type: str | None,
    limit: int,
    offset: int,
) -> list[EntityRow]:
    """List entities with optional text search and type filter.

    When ``q`` is provided, uses FTS5 full-text search. Otherwise lists
    all entities ordered by mention_count descending.

    Args:
        conn: An open aiosqlite connection.
        q: Optional FTS5 search query.
        entity_type: Optional entity type filter.
        limit: Maximum results.
        offset: Pagination offset.

    Returns:
        List of EntityRow objects.
    """
    if q:
        # Use FTS5 search — does not support entity_type filter directly
        fts_results = await search_entities_fts(conn, q, limit=limit + offset)
        entities = fts_results[offset:]

        # Apply entity_type filter post-FTS if requested
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]

        return entities[:limit]

    # No text search — list all with optional type filter
    conditions: list[str] = []
    params: list = []

    if entity_type:
        conditions.append("entity_type = ?")
        params.append(entity_type)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.extend([limit, offset])

    async with conn.execute(
        f"SELECT * FROM entities {where} ORDER BY mention_count DESC LIMIT ? OFFSET ?",  # noqa: S608
        params,
    ) as cursor:
        rows = await cursor.fetchall()

    from ..db.queries import _row_to_dict

    return [EntityRow(**_row_to_dict(row)) for row in rows]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/entities", response_model=list[EntityOut])
async def list_entities(
    q: str | None = Query(default=None, description="Search query for entity names"),
    entity_type: str | None = Query(
        default=None,
        description="Filter by entity type (person, company, technology, etc.)",
    ),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Results to skip"),
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> list[EntityOut]:
    """List entities with optional text search and type filter.

    When ``q`` is provided, uses FTS5 full-text search over entity names and
    descriptions. Otherwise lists all entities ordered by mention_count.

    Args:
        q: Optional search query.
        entity_type: Optional type filter (e.g. 'person', 'company').
        limit: Maximum results (1-100, default 50).
        offset: Pagination offset.
        conn: DB connection (injected).

    Returns:
        List of EntityOut objects.
    """
    entities = await _list_entities(conn, q=q, entity_type=entity_type, limit=limit, offset=offset)
    return [_entity_row_to_out(e) for e in entities]


@router.get("/entities/{entity_id}", response_model=EntityWithDocsOut)
async def get_entity(
    entity_id: str,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> EntityWithDocsOut:
    """Retrieve a single entity with its linked documents.

    Args:
        entity_id: The entity UUID.
        conn: DB connection (injected).

    Returns:
        EntityWithDocsOut including the entity and up to 20 linked documents.

    Raises:
        HTTPException 404: If no entity with the given ID exists.
    """
    entity = await _get_entity_by_id(conn, entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")

    doc_rows = await get_entity_documents(conn, entity_id, limit=20)

    return EntityWithDocsOut(
        entity=_entity_row_to_out(entity),
        documents=[_doc_row_to_out(d) for d in doc_rows],
    )


@router.get("/entities/{entity_id}/documents", response_model=list[DocumentOut])
async def get_entity_documents_endpoint(
    entity_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results"),
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> list[DocumentOut]:
    """Retrieve documents linked to a specific entity.

    Args:
        entity_id: The entity UUID.
        limit: Maximum results (1-100, default 20).
        conn: DB connection (injected).

    Returns:
        List of DocumentOut objects linked to the entity.

    Raises:
        HTTPException 404: If no entity with the given ID exists.
    """
    entity = await _get_entity_by_id(conn, entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")

    doc_rows = await get_entity_documents(conn, entity_id, limit=limit)
    return [_doc_row_to_out(d) for d in doc_rows]
