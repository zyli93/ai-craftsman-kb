"""FastAPI router for briefing generation and retrieval.

Implements:
- POST /api/briefings   — generate a new briefing via the BriefingGenerator engine
- GET  /api/briefings   — list recent briefings
- GET  /api/briefings/{id} — retrieve a single briefing by ID

The BriefingGenerator is instantiated per-request by pulling components from
``request.app.state``. This keeps the router stateless and fully testable.

Note: ``POST /api/briefings`` with ``run_ingest=True`` can take 30-60 seconds
as it awaits a full pro ingest before generating. Clients should use a long
HTTP timeout for this endpoint.
"""
from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..briefing.generator import BriefingGenerator
from ..db.models import BriefingRow
from ..db.queries import get_briefing, list_briefings
from ..db.sqlite import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["briefings"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path.home() / ".ai-craftsman-kb" / "data"


class CreateBriefingRequest(BaseModel):
    """Request body for POST /api/briefings.

    Attributes:
        query: The topic or search query to generate a briefing for.
        run_radar: If True, run RadarEngine.search(query) before generating.
        run_ingest: If True, run pro ingest for all sources before generating.
            Note: This can add 30-60 seconds to response time.
        limit: Maximum number of source documents to include in the briefing context.
    """

    query: str = Field(..., min_length=1, description="Topic or search query for the briefing")
    run_radar: bool = Field(default=True, description="Run radar search before generating")
    run_ingest: bool = Field(default=False, description="Run pro ingest before generating (slow)")
    limit: int = Field(default=20, ge=1, le=50, description="Max source documents to include")


class BriefingOut(BaseModel):
    """API response model for a briefing.

    Maps directly to BriefingRow fields for API consumption.
    """

    id: str
    title: str
    query: str | None
    content: str
    source_document_ids: list[str]
    created_at: str
    format: str


def _briefing_to_out(row: BriefingRow) -> BriefingOut:
    """Convert a BriefingRow model to a BriefingOut API response.

    Args:
        row: The BriefingRow from the database.

    Returns:
        A BriefingOut instance suitable for JSON serialization.
    """
    return BriefingOut(
        id=row.id,
        title=row.title,
        query=row.query,
        content=row.content,
        source_document_ids=row.source_document_ids,
        created_at=row.created_at,
        format=row.format,
    )


def _get_data_dir(request: Request) -> Path:
    """Resolve the data directory from app state or use the default.

    Args:
        request: The incoming FastAPI request (provides app state).

    Returns:
        Path to the data directory containing craftsman.db.
    """
    try:
        config = request.app.state.config
        return Path(config.settings.data_dir).expanduser()
    except AttributeError:
        return _DEFAULT_DATA_DIR


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/briefings", response_model=BriefingOut)
async def create_briefing(
    body: CreateBriefingRequest,
    request: Request,
) -> BriefingOut:
    """Generate a new briefing for the given topic.

    Pipeline:
    1. (Optional) Run pro ingest for all sources — can take 30-60 seconds.
    2. (Optional) Run RadarEngine.search(query) to pull fresh open-web content.
    3. Hybrid search for the most relevant documents.
    4. Assemble document context window and call the LLM.
    5. Save the briefing to the database and return it.

    Args:
        body: Request body with query, run_radar, run_ingest, and limit fields.
        request: FastAPI request (provides app state: config, llm_router, etc.).

    Returns:
        The generated BriefingOut response.

    Raises:
        HTTPException 503: If required app state components are not available.
        HTTPException 500: If briefing generation fails.
    """
    # Pull required components from app state
    try:
        config = request.app.state.config
        llm_router = request.app.state.llm_router
        hybrid_search = request.app.state.hybrid_search
        radar_engine = request.app.state.radar_engine
        ingest_runner = request.app.state.ingest_runner
    except AttributeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"App state component not available: {exc}. "
                   f"Ensure the server was started with all components initialized.",
        ) from exc

    generator = BriefingGenerator(
        config=config,
        llm_router=llm_router,
        hybrid_search=hybrid_search,
        radar_engine=radar_engine,
        ingest_runner=ingest_runner,
    )

    data_dir = _get_data_dir(request)

    try:
        async with get_db(data_dir) as conn:
            briefing = await generator.generate(
                conn,
                topic=body.query,
                run_radar=body.run_radar,
                run_ingest=body.run_ingest,
                limit=body.limit,
            )
    except Exception as exc:
        logger.exception("Briefing generation failed for query %r", body.query)
        raise HTTPException(
            status_code=500,
            detail=f"Briefing generation failed: {exc}",
        ) from exc

    return _briefing_to_out(briefing)


@router.get("/briefings", response_model=list[BriefingOut])
async def list_briefings_endpoint(
    request: Request,
    limit: int = 20,
) -> list[BriefingOut]:
    """List recent briefings ordered by creation date.

    Args:
        request: FastAPI request (provides app state).
        limit: Maximum number of briefings to return (default 20).

    Returns:
        A list of BriefingOut objects ordered by created_at descending.
    """
    data_dir = _get_data_dir(request)

    async with get_db(data_dir) as conn:
        rows = await list_briefings(conn, limit=limit)

    return [_briefing_to_out(row) for row in rows]


@router.get("/briefings/{briefing_id}", response_model=BriefingOut)
async def get_briefing_endpoint(
    briefing_id: str,
    request: Request,
) -> BriefingOut:
    """Retrieve a single briefing by its UUID.

    Args:
        briefing_id: The UUID of the briefing to retrieve.
        request: FastAPI request (provides app state).

    Returns:
        The BriefingOut for the requested briefing.

    Raises:
        HTTPException 404: If no briefing with the given ID exists.
    """
    data_dir = _get_data_dir(request)

    async with get_db(data_dir) as conn:
        row = await get_briefing(conn, briefing_id)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Briefing '{briefing_id}' not found",
        )

    return _briefing_to_out(row)
