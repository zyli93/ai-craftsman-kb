"""FastAPI router for briefing generation and retrieval.

Implements:
- GET    /api/briefings       — list recent briefings
- POST   /api/briefings       — generate a new briefing
- GET    /api/briefings/{id}  — retrieve a single briefing by ID
- DELETE /api/briefings/{id}  — delete a briefing

The BriefingGenerator is instantiated per-request by pulling components from
``request.app.state``. This keeps the router stateless and fully testable.

Note: ``POST /api/briefings`` with ``run_ingest=True`` can take 30-60 seconds
as it awaits a full pro ingest before generating. Clients should use a long
HTTP timeout for this endpoint.
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..db.models import BriefingRow
from ..db.queries import (
    get_briefing,
    insert_briefing,
    list_briefings,
)
from .deps import get_conn
from .models import BriefingOut, CreateBriefingRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["briefings"])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _briefing_row_to_out(briefing: BriefingRow) -> BriefingOut:
    """Convert a BriefingRow to a BriefingOut response model.

    Args:
        briefing: The DB row model.

    Returns:
        BriefingOut suitable for JSON serialisation.
    """
    return BriefingOut(
        id=briefing.id,
        title=briefing.title,
        query=briefing.query,
        content=briefing.content,
        source_document_ids=briefing.source_document_ids,
        created_at=briefing.created_at,
        format=briefing.format,
    )


async def _generate_briefing_content(
    conn: aiosqlite.Connection,
    request: Request,
    query: str,
    limit: int,
) -> tuple[str, str, list[str]]:
    """Generate a briefing by searching for relevant documents.

    When the BriefingGenerator engine (task_41) is available via app state,
    it is used for LLM-synthesised briefings. Otherwise falls back to a simple
    document aggregation approach.

    Args:
        conn: An open aiosqlite connection.
        request: FastAPI request for app state access.
        query: The search query to find relevant documents.
        limit: Maximum documents to include in the briefing.

    Returns:
        Tuple of (title, content_markdown, source_document_ids).
    """
    # Try to use the BriefingGenerator if it is available in app state
    try:
        from ..briefing.generator import BriefingGenerator
        from ..ingestors.runner import IngestRunner, INGESTORS
        from ..radar.engine import RadarEngine
        from ..search.hybrid import HybridSearch

        config = request.app.state.config
        llm_router = request.app.state.llm_router
        vector_store = request.app.state.vector_store
        embedder = request.app.state.embedder
        db_path = request.app.state.db_path

        hybrid_search = HybridSearch(config=config, vector_store=vector_store, embedder=embedder)

        # Build supporting engines
        ingestors = {st: cls(config) for st, cls in INGESTORS.items()}
        radar_engine = RadarEngine(config=config, ingestors=ingestors)
        ingest_runner = IngestRunner(config=config, llm_router=llm_router, db_path=db_path)

        generator = BriefingGenerator(
            config=config,
            llm_router=llm_router,
            hybrid_search=hybrid_search,
            radar_engine=radar_engine,
            ingest_runner=ingest_runner,
        )
        briefing_row = await generator.generate(
            conn,
            topic=query,
            run_radar=False,  # radar handled separately if body.run_radar=True
            run_ingest=False,  # ingest handled separately if body.run_ingest=True
            limit=limit,
        )
        return briefing_row.title, briefing_row.content, briefing_row.source_document_ids
    except (ImportError, AttributeError) as e:
        # BriefingGenerator not available or app state missing — fall back to simple aggregation
        logger.debug("BriefingGenerator not available, using fallback: %s", e)

    # Fallback: keyword search + simple markdown assembly
    from ..db.queries import search_documents_fts, get_document

    try:
        fts_results = await search_documents_fts(conn, query, limit=limit)
        doc_ids = [doc_id for doc_id, _ in fts_results]
    except Exception:
        doc_ids = []

    if not doc_ids:
        from ..db.queries import list_documents
        docs = await list_documents(conn, limit=limit)
        doc_ids = [d.id for d in docs]

    docs = []
    for doc_id in doc_ids[:limit]:
        doc = await get_document(conn, doc_id)
        if doc is not None:
            docs.append(doc)

    title = f"Briefing: {query}"
    lines: list[str] = [f"# {title}\n"]
    lines.append(f"**Query:** {query}\n")
    lines.append(f"**Documents included:** {len(docs)}\n\n---\n")

    for i, doc in enumerate(docs, 1):
        doc_title = doc.title or "Untitled"
        lines.append(f"## {i}. {doc_title}\n")
        if doc.author:
            lines.append(f"*Author:* {doc.author}  ")
        if doc.published_at:
            lines.append(f"*Published:* {doc.published_at}  ")
        lines.append(f"*Source:* {doc.source_type}  ")
        lines.append(f"*URL:* {doc.url}\n\n")
        if doc.raw_content:
            excerpt = doc.raw_content[:500].strip()
            lines.append(f"{excerpt}...\n\n")

    content = "\n".join(lines)
    source_ids = [d.id for d in docs]
    return title, content, source_ids


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/briefings", response_model=list[BriefingOut])
async def list_briefings_endpoint(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results"),
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> list[BriefingOut]:
    """List all briefings ordered by created_at descending.

    Args:
        limit: Maximum results (1-100, default 20).
        conn: DB connection (injected).

    Returns:
        List of BriefingOut objects.
    """
    briefings = await list_briefings(conn, limit=limit)
    return [_briefing_row_to_out(b) for b in briefings]


@router.post("/briefings", response_model=BriefingOut, status_code=201)
async def create_briefing(
    body: CreateBriefingRequest,
    request: Request = None,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> BriefingOut:
    """Generate and store a new briefing.

    Searches for documents relevant to the query, assembles a markdown
    briefing, and stores it in the database.

    When the BriefingGenerator engine is available in app state, it is used
    for richer LLM-synthesised briefings. Otherwise a simple document
    aggregation fallback is used.

    Args:
        body: Request body with ``query``, ``limit``, ``run_radar``,
              ``run_ingest`` options.
        request: FastAPI request (for app state access).
        conn: DB connection (injected).

    Returns:
        The generated BriefingOut.

    Raises:
        HTTPException 500: If briefing generation fails.
    """
    # Optionally run pro ingest first to get fresh content
    if body.run_ingest:
        try:
            from ..ingestors.runner import IngestRunner

            config = request.app.state.config
            llm_router = request.app.state.llm_router
            db_path = request.app.state.db_path
            runner = IngestRunner(config=config, llm_router=llm_router, db_path=db_path)
            await runner.run_all()
        except Exception as e:
            logger.warning("Pro ingest for briefing failed: %s", e)

    # Optionally run radar search first
    if body.run_radar:
        try:
            from ..ingestors.base import BaseIngestor
            from ..ingestors.runner import INGESTORS
            from ..radar.engine import RadarEngine

            config = request.app.state.config
            ingestors: dict[str, BaseIngestor] = {
                st: cls(config) for st, cls in INGESTORS.items()
            }
            engine = RadarEngine(config=config, ingestors=ingestors)
            await engine.search(conn, query=body.query, limit_per_source=5)
        except Exception as e:
            logger.warning("Radar pre-search for briefing failed: %s", e)

    # Generate briefing content
    try:
        title, content, source_ids = await _generate_briefing_content(
            conn, request, query=body.query, limit=body.limit
        )
    except Exception as e:
        logger.error("Briefing generation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Briefing generation failed: {e}",
        )

    briefing = BriefingRow(
        id=str(uuid.uuid4()),
        title=title,
        query=body.query,
        content=content,
        source_document_ids=source_ids,
        format="markdown",
    )

    try:
        await insert_briefing(conn, briefing)
    except Exception as e:
        logger.error("Failed to store briefing: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to store briefing: {e}")

    logger.info("Created briefing %s for query %r", briefing.id, body.query)
    return _briefing_row_to_out(briefing)


@router.get("/briefings/{briefing_id}", response_model=BriefingOut)
async def get_briefing_endpoint(
    briefing_id: str,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> BriefingOut:
    """Retrieve a single briefing by its UUID.

    Args:
        briefing_id: The briefing UUID.
        conn: DB connection (injected).

    Returns:
        The BriefingOut.

    Raises:
        HTTPException 404: If no briefing with the given ID exists.
    """
    briefing = await get_briefing(conn, briefing_id)
    if briefing is None:
        raise HTTPException(status_code=404, detail=f"Briefing '{briefing_id}' not found")

    return _briefing_row_to_out(briefing)


@router.delete("/briefings/{briefing_id}", response_model=dict)
async def delete_briefing(
    briefing_id: str,
    conn: Annotated[aiosqlite.Connection, Depends(get_conn)] = None,
) -> dict:
    """Permanently delete a briefing.

    Unlike documents, briefings are hard-deleted (not soft-deleted).

    Args:
        briefing_id: The briefing UUID.
        conn: DB connection (injected).

    Returns:
        ``{"ok": True}`` on success.

    Raises:
        HTTPException 404: If no briefing with the given ID exists.
    """
    briefing = await get_briefing(conn, briefing_id)
    if briefing is None:
        raise HTTPException(status_code=404, detail=f"Briefing '{briefing_id}' not found")

    await conn.execute("DELETE FROM briefings WHERE id = ?", (briefing_id,))
    await conn.commit()
    logger.info("Deleted briefing %s", briefing_id)
    return {"ok": True}
