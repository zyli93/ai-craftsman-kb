"""Usage API endpoints for querying LLM token consumption data.

Provides two endpoints:
- ``GET /api/usage`` -- aggregated stats grouped by provider, model, and task.
- ``GET /api/usage/recent`` -- the most recent raw usage records.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query, Request

from .models import UsageRecordOut, UsageSummaryItem, UsageSummaryOut

router = APIRouter(prefix="/api", tags=["usage"])


@router.get("/usage", response_model=UsageSummaryOut)
async def get_usage_summary(
    request: Request,
    since: str | None = Query(
        default=None,
        description="ISO 8601 timestamp; defaults to 24 hours ago.",
    ),
) -> UsageSummaryOut:
    """Return aggregated LLM usage stats grouped by provider, model, and task.

    Args:
        request: The incoming FastAPI request (provides access to app state).
        since: Optional ISO timestamp string. Defaults to 24 hours ago.

    Returns:
        UsageSummaryOut with summary rows and period boundaries.
    """
    now = datetime.now(timezone.utc)

    if since is not None:
        # Parse the ISO timestamp, treating naive datetimes as UTC
        period_start = datetime.fromisoformat(since)
        if period_start.tzinfo is None:
            period_start = period_start.replace(tzinfo=timezone.utc)
    else:
        period_start = now - timedelta(hours=24)

    tracker = request.app.state.usage_tracker
    rows = await tracker.get_summary(period_start)

    summary_items = [
        UsageSummaryItem(
            provider=row["provider"],
            model=row["model"],
            task=row["task"],
            total_input_tokens=row["total_input_tokens"] or 0,
            total_output_tokens=row["total_output_tokens"] or 0,
            request_count=row["request_count"],
        )
        for row in rows
    ]

    return UsageSummaryOut(
        summary=summary_items,
        period_start=period_start.isoformat(),
        period_end=now.isoformat(),
    )


@router.get("/usage/recent", response_model=list[UsageRecordOut])
async def get_recent_usage(
    request: Request,
    limit: int = Query(default=50, ge=1, le=1000, description="Number of records to return."),
) -> list[UsageRecordOut]:
    """Return the most recent raw LLM usage records.

    Args:
        request: The incoming FastAPI request (provides access to app state).
        limit: Maximum number of records to return (default 50, max 1000).

    Returns:
        List of UsageRecordOut ordered by most recent first.
    """
    tracker = request.app.state.usage_tracker
    rows = await tracker.get_recent(limit)

    return [
        UsageRecordOut(
            id=row["id"],
            timestamp=row["timestamp"],
            provider=row["provider"],
            model=row["model"],
            task=row["task"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            duration_ms=row["duration_ms"],
            success=bool(row["success"]),
        )
        for row in rows
    ]
