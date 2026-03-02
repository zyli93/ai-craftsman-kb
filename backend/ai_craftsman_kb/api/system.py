"""FastAPI router for system-level endpoints.

Provides endpoints for:
- Listing discovered sources pending user review (GET /api/discover)
- Updating discovered source status (PATCH /api/discover/{source_id})
"""
from __future__ import annotations

import logging
from typing import Annotated, Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..db.models import DiscoveredSourceRow
from ..db.queries import list_discovered_sources, update_discovered_source_status
from .deps import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["system"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class DiscoveredSourceResponse(BaseModel):
    """API response model for a discovered source suggestion.

    Maps directly to DiscoveredSourceRow fields for API consumption.
    """

    id: str
    source_type: str
    identifier: str
    display_name: str | None
    discovered_from_document_id: str | None
    discovery_method: str | None
    confidence: float | None
    status: str
    created_at: str


class DiscoverListResponse(BaseModel):
    """Response wrapping a list of discovered source suggestions."""

    items: list[DiscoveredSourceResponse]
    total: int
    status_filter: str


class StatusUpdateRequest(BaseModel):
    """Request body for updating a discovered source's status."""

    status: str  # 'suggested' | 'added' | 'dismissed'


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/discover", response_model=DiscoverListResponse)
async def list_discovered(
    status: str = Query(default="suggested", description="Filter by status: suggested, added, dismissed"),
    limit: int = Query(default=20, ge=1, le=200, description="Maximum results to return"),
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> DiscoverListResponse:
    """Return discovered sources filtered by status.

    Retrieves source suggestions from the ``discovered_sources`` table,
    filtered by the given status. The default status is 'suggested', which
    returns sources pending user review.

    Args:
        status: Status filter — one of 'suggested', 'added', 'dismissed'.
        limit: Maximum number of results to return (1-200).
        conn: Database connection (injected by FastAPI dependency).

    Returns:
        A DiscoverListResponse with matched source suggestions.

    Raises:
        HTTPException 400: If the status value is not one of the allowed values.
    """
    allowed_statuses = {"suggested", "added", "dismissed"}
    if status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{status}'. Must be one of: {', '.join(sorted(allowed_statuses))}",
        )

    rows = await list_discovered_sources(conn, status=status)

    # Apply limit after fetching (list_discovered_sources doesn't support limit)
    rows = rows[:limit]

    items = [
        DiscoveredSourceResponse(
            id=row.id,
            source_type=row.source_type,
            identifier=row.identifier,
            display_name=row.display_name,
            discovered_from_document_id=row.discovered_from_document_id,
            discovery_method=row.discovery_method,
            confidence=row.confidence,
            status=row.status,
            created_at=row.created_at,
        )
        for row in rows
    ]

    return DiscoverListResponse(
        items=items,
        total=len(items),
        status_filter=status,
    )


@router.patch("/discover/{source_id}", response_model=DiscoveredSourceResponse)
async def update_discovered_status(
    source_id: str,
    body: StatusUpdateRequest,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> DiscoveredSourceResponse:
    """Update the status of a discovered source suggestion.

    Allows the user to approve ('added') or dismiss ('dismissed') a source
    suggestion, or reset it back to 'suggested'.

    Args:
        source_id: The UUID of the discovered source to update.
        body: Request body containing the new status value.
        conn: Database connection (injected by FastAPI dependency).

    Returns:
        The updated DiscoveredSourceResponse.

    Raises:
        HTTPException 400: If the new status is not a valid value.
        HTTPException 404: If no source with the given ID exists.
    """
    allowed_statuses = {"suggested", "added", "dismissed"}
    if body.status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{body.status}'. Must be one of: {', '.join(sorted(allowed_statuses))}",
        )

    # Check existence first
    async with conn.execute(
        "SELECT * FROM discovered_sources WHERE id = ?",
        (source_id,),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Discovered source '{source_id}' not found",
        )

    await update_discovered_source_status(conn, source_id, body.status)

    # Re-fetch the updated row
    async with conn.execute(
        "SELECT * FROM discovered_sources WHERE id = ?",
        (source_id,),
    ) as cursor:
        updated_row = await cursor.fetchone()

    if updated_row is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve updated source")

    return DiscoveredSourceResponse(
        id=updated_row["id"],
        source_type=updated_row["source_type"],
        identifier=updated_row["identifier"],
        display_name=updated_row["display_name"],
        discovered_from_document_id=updated_row["discovered_from_document_id"],
        discovery_method=updated_row["discovery_method"],
        confidence=updated_row["confidence"],
        status=updated_row["status"],
        created_at=updated_row["created_at"],
    )
