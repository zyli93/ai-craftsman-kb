"""FastAPI router for system stats and health check endpoints.

Provides:
- GET /api/stats   — aggregate system statistics
- GET /api/health  — health check
"""
from __future__ import annotations

import logging
import os
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, Request

from ..db.queries import get_stats
from .deps import get_conn
from .models import HealthOut, SystemStats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
    request: Request = None,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> SystemStats:
    """Return aggregate system statistics.

    Queries the SQLite database for document, entity, source, and briefing
    counts, and queries Qdrant for vector count.

    Args:
        request: FastAPI request (for app state access).
        conn: DB connection (injected).

    Returns:
        A SystemStats response model.
    """
    stats = await get_stats(conn)

    # Get active source count (enabled=TRUE)
    async with conn.execute(
        "SELECT COUNT(*) FROM sources WHERE enabled = TRUE"
    ) as cursor:
        row = await cursor.fetchone()
        active_sources = row[0] if row else 0

    # Get vector count and DB file size from app state
    vector_count = 0
    db_size_bytes = 0

    try:
        vector_store = request.app.state.vector_store
        info = vector_store.get_collection_info()
        vector_count = info.get("vectors_count", 0)
    except Exception as e:
        logger.debug("Could not get vector count: %s", e)

    try:
        db_path = request.app.state.db_path
        if db_path.exists():
            db_size_bytes = os.path.getsize(db_path)
    except Exception as e:
        logger.debug("Could not get DB file size: %s", e)

    return SystemStats(
        total_documents=stats.get("total_documents", 0),
        embedded_documents=stats.get("embedded_documents", 0),
        total_entities=stats.get("total_entities", 0),
        active_sources=active_sources,
        total_briefings=stats.get("total_briefings", 0),
        vector_count=vector_count,
        db_size_bytes=db_size_bytes,
    )


@router.get("/health", response_model=HealthOut)
async def health_check(
    request: Request = None,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> HealthOut:
    """Return the health status of all backend components.

    Checks connectivity to SQLite and Qdrant.

    Args:
        request: FastAPI request (for app state access).
        conn: DB connection (injected).

    Returns:
        A HealthOut response with status and component availability.
    """
    # Check SQLite connectivity
    db_ok = False
    try:
        async with conn.execute("SELECT 1") as cursor:
            await cursor.fetchone()
        db_ok = True
    except Exception as e:
        logger.warning("SQLite health check failed: %s", e)

    # Check Qdrant connectivity
    qdrant_ok = False
    try:
        vector_store = request.app.state.vector_store
        vector_store.get_collection_info()
        qdrant_ok = True
    except Exception as e:
        logger.warning("Qdrant health check failed: %s", e)

    return HealthOut(
        status="ok" if (db_ok and qdrant_ok) else "degraded",
        db=db_ok,
        qdrant=qdrant_ok,
    )
