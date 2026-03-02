"""FastAPI router for system stats and health check endpoints.

Provides:
- GET /api/stats   — aggregate system statistics
- GET /api/health  — health check (basic or full diagnostics)
"""
from __future__ import annotations

import logging
import os
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, Request

from ..db.queries import get_stats
from .deps import get_conn
from .models import HealthCheckResult, HealthOut, SystemStats

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


async def _health_db_check(conn: aiosqlite.Connection) -> tuple[str, str]:
    """Run a lightweight SQLite connectivity check using an existing connection.

    Args:
        conn: An open aiosqlite connection from the request dependency.

    Returns:
        ('ok', doc count message) or ('error', description).
    """
    try:
        async with conn.execute("SELECT COUNT(*) FROM documents") as cursor:
            row = await cursor.fetchone()
            count = row[0] if row else 0
        return ("ok", f"{count} documents")
    except Exception as exc:
        return ("error", f"DB query failed: {exc}")


async def _health_qdrant_check(request: Request) -> tuple[str, str]:
    """Run a lightweight Qdrant connectivity check using app state.

    Args:
        request: FastAPI request carrying the app-level vector_store.

    Returns:
        ('ok', vector count) or ('error', description).
    """
    try:
        vector_store = request.app.state.vector_store
        info = vector_store.get_collection_info()
        count = info.get("vectors_count", 0)
        return ("ok", f"{count} vectors")
    except Exception as exc:
        return ("error", f"Qdrant unavailable: {exc}")


@router.get("/health", response_model=HealthOut)
async def health_check(
    request: Request = None,
    full: bool = False,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> HealthOut:
    """Return the health status of all backend components.

    When ``full=false`` (default) only SQLite and Qdrant are checked.
    When ``full=true`` a broader set of checks is run (same as ``cr doctor``)
    including API key presence, external connectivity, and data directory
    writability.

    Args:
        request: FastAPI request (for app state access).
        full: When True, include extended diagnostics in the response.
        conn: DB connection (injected).

    Returns:
        A HealthOut response with status and component availability.
        When full=True the ``checks`` dict is populated.
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

    checks: dict[str, HealthCheckResult] | None = None

    if full:
        # Import doctor check functions lazily to avoid circular imports
        from ..cli import (
            _check_api_key,
            _check_arxiv_connectivity,
            _check_data_dir,
            _check_hn_connectivity,
            _check_reddit_credentials,
            _check_youtube_key,
        )

        config = request.app.state.config

        raw_checks: list[tuple[str, object]] = [
            ("db", _health_db_check(conn)),
            ("qdrant", _health_qdrant_check(request)),
            ("openai_key", _check_api_key(config, "openai")),
            ("anthropic_key", _check_api_key(config, "anthropic")),
            ("openrouter_key", _check_api_key(config, "openrouter")),
            ("youtube_key", _check_youtube_key(config)),
            ("reddit_credentials", _check_reddit_credentials(config)),
            ("hn_connectivity", _check_hn_connectivity()),
            ("arxiv_connectivity", _check_arxiv_connectivity()),
            ("data_dir", _check_data_dir(config)),
        ]

        checks = {}
        for check_name, coro in raw_checks:
            try:
                status, message = await coro  # type: ignore[misc]
            except Exception as exc:
                status, message = "error", str(exc)
            checks[check_name] = HealthCheckResult(status=status, message=message)

    return HealthOut(
        status="ok" if (db_ok and qdrant_ok) else "degraded",
        db=db_ok,
        qdrant=qdrant_ok,
        checks=checks,
    )
