"""FastAPI router for source management endpoints.

Provides:
- GET    /api/sources              — list all sources
- POST   /api/sources              — create a new source
- PUT    /api/sources/{id}         — update a source
- DELETE /api/sources/{id}         — delete a source
- POST   /api/sources/{id}/ingest  — trigger single-source ingestion
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request

from ..db.models import SourceRow
from ..db.queries import list_sources, upsert_source
from ..ingestors.runner import INGESTORS, IngestRunner
from .deps import get_conn
from .models import (
    CreateSourceRequest,
    IngestReportOut,
    SourceOut,
    UpdateSourceRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["sources"])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _source_row_to_out(source: SourceRow) -> SourceOut:
    """Convert a SourceRow to a SourceOut response model.

    Args:
        source: The DB row model.

    Returns:
        SourceOut suitable for JSON serialisation.
    """
    return SourceOut(
        id=source.id,
        source_type=source.source_type,
        identifier=source.identifier,
        display_name=source.display_name,
        enabled=source.enabled,
        last_fetched_at=source.last_fetched_at,
        fetch_error=source.fetch_error,
        created_at=source.created_at,
    )


async def _get_source_by_id(
    conn: aiosqlite.Connection,
    source_id: str,
) -> SourceRow | None:
    """Fetch a single source row by its UUID.

    Args:
        conn: An open aiosqlite connection.
        source_id: The source's UUID string.

    Returns:
        A SourceRow if found, or None.
    """
    async with conn.execute(
        "SELECT * FROM sources WHERE id = ?",
        (source_id,),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        return None

    from ..db.queries import _row_to_dict

    return SourceRow(**_row_to_dict(row))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/sources", response_model=list[SourceOut])
async def list_sources_endpoint(
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> list[SourceOut]:
    """List all configured sources.

    Returns all sources (enabled and disabled) ordered by source_type,
    identifier.

    Args:
        conn: DB connection (injected).

    Returns:
        List of SourceOut objects.
    """
    sources = await list_sources(conn, enabled_only=False)
    return [_source_row_to_out(s) for s in sources]


@router.post("/sources", response_model=SourceOut, status_code=201)
async def create_source(
    body: CreateSourceRequest,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> SourceOut:
    """Create a new source.

    Creates a new source row in the database. Sources represent configured
    feeds, channels, or subreddits used for pro ingestion.

    Args:
        body: Request body with source_type, identifier, display_name.
        conn: DB connection (injected).

    Returns:
        The created SourceOut.

    Raises:
        HTTPException 409: If a source with the same type+identifier already exists.
    """
    # Check for duplicates
    async with conn.execute(
        "SELECT id FROM sources WHERE source_type = ? AND identifier = ?",
        (body.source_type, body.identifier),
    ) as cursor:
        existing = await cursor.fetchone()

    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Source with type='{body.source_type}' and "
                f"identifier='{body.identifier}' already exists"
            ),
        )

    source = SourceRow(
        id=str(uuid.uuid4()),
        source_type=body.source_type,
        identifier=body.identifier,
        display_name=body.display_name,
        enabled=True,
    )
    await upsert_source(conn, source)
    logger.info("Created source %s (%s/%s)", source.id, source.source_type, source.identifier)
    return _source_row_to_out(source)


@router.put("/sources/{source_id}", response_model=SourceOut)
async def update_source(
    source_id: str,
    body: UpdateSourceRequest,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> SourceOut:
    """Update a source's enabled status or display name.

    Only the fields provided in the request body are updated. Use ``enabled``
    to pause or resume pro ingestion for a source.

    Args:
        source_id: The source UUID to update.
        body: Request body with optional ``enabled`` and ``display_name``.
        conn: DB connection (injected).

    Returns:
        The updated SourceOut.

    Raises:
        HTTPException 404: If no source with the given ID exists.
    """
    source = await _get_source_by_id(conn, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")

    # Apply only the fields that were provided
    updates: dict = {}
    if body.enabled is not None:
        updates["enabled"] = body.enabled
    if body.display_name is not None:
        updates["display_name"] = body.display_name

    if updates:
        # Build SET clause dynamically — only update provided fields
        set_parts = [f"{k} = ?" for k in updates]
        set_parts.append("updated_at = CURRENT_TIMESTAMP")
        params = list(updates.values()) + [source_id]
        await conn.execute(
            f"UPDATE sources SET {', '.join(set_parts)} WHERE id = ?",  # noqa: S608
            params,
        )
        await conn.commit()

    updated = await _get_source_by_id(conn, source_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve updated source")

    logger.info("Updated source %s: %s", source_id, updates)
    return _source_row_to_out(updated)


@router.delete("/sources/{source_id}", response_model=dict)
async def delete_source(
    source_id: str,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> dict:
    """Delete a source by its UUID.

    Permanently removes the source row. Documents already ingested from this
    source are not deleted (their source_id FK becomes NULL).

    Args:
        source_id: The source UUID to delete.
        conn: DB connection (injected).

    Returns:
        ``{"ok": True}`` on success.

    Raises:
        HTTPException 404: If no source with the given ID exists.
    """
    source = await _get_source_by_id(conn, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")

    await conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
    await conn.commit()
    logger.info("Deleted source %s", source_id)
    return {"ok": True}


@router.post("/sources/{source_id}/ingest", response_model=IngestReportOut)
async def ingest_source(
    source_id: str,
    request: Request = None,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> IngestReportOut:
    """Trigger pro ingestion for a single source.

    Runs the full ingest pipeline (fetch -> filter -> dedup -> store) for
    the specified source. May take significant time.

    Args:
        source_id: The source UUID to ingest.
        request: FastAPI request (for app state access).
        conn: DB connection (injected).

    Returns:
        IngestReportOut with ingestion results.

    Raises:
        HTTPException 404: If no source with the given ID exists.
        HTTPException 422: If the source type is not supported for pro ingestion.
    """
    source = await _get_source_by_id(conn, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")

    if source.source_type not in INGESTORS:
        raise HTTPException(
            status_code=422,
            detail=f"Source type '{source.source_type}' is not supported for ingestion",
        )

    config = request.app.state.config
    db_path = request.app.state.db_path
    llm_router = request.app.state.llm_router

    runner = IngestRunner(config=config, llm_router=llm_router, db_path=db_path)

    ingestor_cls = INGESTORS[source.source_type]
    ingestor = ingestor_cls(config)
    report = await runner.run_source(ingestor)

    return IngestReportOut(
        source_type=report.source_type,
        fetched=report.fetched,
        stored=report.stored,
        embedded=report.embedded,
        errors=report.errors,
    )
