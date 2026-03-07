"""FastAPI router for ingest endpoints.

Provides:
- POST /api/ingest/url  — ingest a single URL
- POST /api/ingest/pro  — run pro ingestion (one source or all)
"""
from __future__ import annotations

import logging
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request

from ..db.queries import get_document_by_url
from ..ingestors.runner import INGESTORS, IngestReport, IngestRunner
from ..llm.router import LLMRouter
from .deps import get_conn
from .documents import _doc_row_to_out
from .models import DocumentOut, IngestProRequest, IngestReportOut, IngestURLRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingest"])


def _report_to_out(report: IngestReport) -> IngestReportOut:
    """Convert an IngestReport to an IngestReportOut response model.

    Args:
        report: Internal IngestReport from the runner.

    Returns:
        IngestReportOut suitable for JSON serialisation.
    """
    return IngestReportOut(
        source_type=report.source_type,
        fetched=report.fetched,
        stored=report.stored,
        embedded=report.embedded,
        errors=report.errors,
    )


@router.post("/ingest/url", response_model=DocumentOut, status_code=201)
async def ingest_url(
    body: IngestURLRequest,
    request: Request = None,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> DocumentOut:
    """Ingest a single URL and return the stored document.

    Uses AdhocIngestor to detect the URL type (YouTube, ArXiv, article)
    and fetch content. Content filtering is skipped for adhoc ingestion —
    the user explicitly chose to ingest this URL.

    Args:
        body: Request body with ``url`` and optional ``tags``.
        request: FastAPI request (for app state access).
        conn: DB connection (injected).

    Returns:
        The ingested DocumentOut.

    Raises:
        HTTPException 422: If the URL is invalid.
        HTTPException 409: If the URL already exists in the database.
        HTTPException 500: If ingestion fails.
    """
    config = request.app.state.config
    db_path = request.app.state.db_path
    llm_router = request.app.state.llm_router
    pipeline = getattr(request.app.state, "pipeline", None)

    runner = IngestRunner(config=config, llm_router=llm_router, db_path=db_path, pipeline=pipeline)

    report = await runner.ingest_url(body.url, tags=body.tags if body.tags else None)

    if report.skipped_duplicate == 1:
        # Document already exists — fetch and return it
        existing = await get_document_by_url(conn, body.url)
        if existing is not None:
            return _doc_row_to_out(existing)
        raise HTTPException(status_code=409, detail="URL already exists in the database")

    if report.errors:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {'; '.join(report.errors)}",
        )

    if report.stored == 0:
        raise HTTPException(status_code=500, detail="Document was not stored")

    # Fetch the newly created document to return it
    doc = await get_document_by_url(conn, body.url)
    if doc is None:
        raise HTTPException(status_code=500, detail="Document stored but could not be retrieved")

    return _doc_row_to_out(doc)


@router.post("/ingest/pro", response_model=list[IngestReportOut])
async def ingest_pro(
    body: IngestProRequest,
    request: Request = None,
) -> list[IngestReportOut]:
    """Run pro ingestion for one or all configured sources.

    This is a synchronous endpoint — it runs the ingestion pipeline to
    completion before returning. May take significant time for full ingest.

    Args:
        body: Request body with optional ``source`` to restrict to one type.
        request: FastAPI request (for app state access).

    Returns:
        List of IngestReportOut, one per source ingested.

    Raises:
        HTTPException 422: If ``source`` is not a known source type.
    """
    config = request.app.state.config
    db_path = request.app.state.db_path
    llm_router = request.app.state.llm_router
    pipeline = getattr(request.app.state, "pipeline", None)

    # Validate single source if specified
    if body.source is not None and body.source not in INGESTORS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown source type: '{body.source}'. "
                f"Available: {sorted(INGESTORS.keys())}"
            ),
        )

    runner = IngestRunner(config=config, llm_router=llm_router, db_path=db_path, pipeline=pipeline)

    if body.source is not None:
        # Run a single source
        ingestor_cls = INGESTORS[body.source]
        ingestor = ingestor_cls(config)
        report = await runner.run_source(ingestor)
        return [_report_to_out(report)]
    else:
        # Run all sources (skipped sources are not included in reports)
        reports, _skipped = await runner.run_all()
        return [_report_to_out(r) for r in reports]
