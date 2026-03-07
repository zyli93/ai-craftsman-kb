"""FastAPI application for AI Craftsman KB.

Exposes all backend capabilities — search, ingest, radar, entities, documents,
sources, briefings, and system stats — as a RESTful API consumed by the
dashboard and MCP server.

The application is initialised via a lifespan context manager that:
- Loads configuration from YAML
- Initialises the SQLite database schema
- Creates shared VectorStore, Embedder, and LLMRouter instances
stored on ``app.state``.

Run locally with::

    uv run uvicorn ai_craftsman_kb.server:app --reload --port 8000

Or via the CLI::

    cr server --reload
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.briefings import router as briefings_router
from .api.documents import router as documents_router
from .api.entities import router as entities_router
from .api.ingest import router as ingest_router
from .api.radar import router as radar_router
from .api.search import router as search_router
from .api.sources import router as sources_router
from .api.stats import router as stats_router
from .api.system import router as system_router
from .api.usage import router as usage_router
from .config.loader import load_config
from .db.sqlite import init_db
from .llm.router import LLMRouter
from .llm.usage_tracker import UsageTracker
from .processing.embedder import Embedder
from .search.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Path to the built dashboard static files (produced by `pnpm build`)
DASHBOARD_DIST = Path(__file__).parent.parent.parent / "dashboard" / "dist"


def _get_db_path(config) -> Path:
    """Resolve the absolute path to craftsman.db from the config data_dir.

    Args:
        config: Loaded AppConfig instance.

    Returns:
        Absolute Path to the craftsman.db file.
    """
    data_dir = Path(config.settings.data_dir).expanduser().resolve()
    return data_dir / "craftsman.db"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager for startup and shutdown.

    Initialises shared resources on startup (config, DB, vector store,
    embedder, LLM router) and stores them on ``app.state`` so routers
    can access them via ``request.app.state``.

    Args:
        app: The FastAPI application instance.

    Yields:
        Nothing — just suspends while the app is running.
    """
    # Startup
    logger.info("AI Craftsman KB API starting up...")

    config = load_config()
    app.state.config = config

    db_path = _get_db_path(config)
    app.state.db_path = db_path

    # Ensure DB schema exists
    data_dir = db_path.parent
    await init_db(data_dir)
    logger.info("Database initialised at %s", db_path)

    # Shared service instances — created once, reused across requests
    app.state.vector_store = VectorStore(config)
    app.state.embedder = Embedder(config)
    app.state.usage_tracker = UsageTracker(data_dir)

    # Pass usage_tracker to LLMRouter if it accepts the parameter (added by AIKB-15)
    import inspect

    llm_router_params = inspect.signature(LLMRouter.__init__).parameters
    if "usage_tracker" in llm_router_params:
        app.state.llm_router = LLMRouter(config, usage_tracker=app.state.usage_tracker)
    else:
        app.state.llm_router = LLMRouter(config)

    # Construct ProcessingPipeline for post-ingest embedding
    try:
        from .config.models import LLMGatewayConfig
        from .processing.chunker import Chunker
        from .processing.entity_extractor import EntityExtractor
        from .processing.keyword_extractor import KeywordExtractor
        from .processing.pipeline import ProcessingPipeline

        chunker = Chunker(
            chunk_size=config.settings.embedding.chunk_size,
            chunk_overlap=config.settings.embedding.chunk_overlap,
        )
        entity_extractor = EntityExtractor(config, app.state.llm_router)

        keyword_extractor = None
        if config.settings.llm is not None:
            if isinstance(config.settings.llm, LLMGatewayConfig):
                if "keyword_extraction" in config.settings.llm.tasks:
                    keyword_extractor = KeywordExtractor(config, app.state.llm_router)
            elif config.settings.llm.keyword_extraction is not None:
                keyword_extractor = KeywordExtractor(config, app.state.llm_router)

        app.state.pipeline = ProcessingPipeline(
            config=config,
            embedder=app.state.embedder,
            chunker=chunker,
            vector_store=app.state.vector_store,
            entity_extractor=entity_extractor,
            keyword_extractor=keyword_extractor,
        )
    except Exception as e:
        logger.warning("Could not create processing pipeline: %s", e)
        app.state.pipeline = None

    logger.info("AI Craftsman KB API ready on port %d", config.settings.server.backend_port)

    yield  # App is running

    # Shutdown (nothing to clean up for now)
    logger.info("AI Craftsman KB API shutting down.")


def mount_dashboard(app: FastAPI) -> None:
    """Mount built dashboard static files at the root path.

    If ``dashboard/dist/`` exists (i.e., the React app has been built), serve
    it via FastAPI's StaticFiles handler with HTML mode enabled so that the
    SPA's client-side router works correctly for deep links.

    Falls back gracefully — if the dist directory is missing a lightweight JSON
    endpoint at ``/`` tells the user how to build the dashboard.

    Args:
        app: The FastAPI application instance to mount the static files on.
    """
    if DASHBOARD_DIST.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount(
            "/",
            StaticFiles(directory=DASHBOARD_DIST, html=True),
            name="dashboard",
        )
        logger.info("Dashboard static files mounted from %s", DASHBOARD_DIST)
    else:

        @app.get("/")
        async def dashboard_not_built() -> dict[str, str]:  # type: ignore[return]
            """Placeholder route shown when the dashboard has not been built yet."""
            return {"message": "Dashboard not built. Run: cd dashboard && pnpm build"}

        logger.info(
            "Dashboard dist not found at %s — serving placeholder at /", DASHBOARD_DIST
        )


def create_app(serve_dashboard: bool = True) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        serve_dashboard: When True (default), mount built dashboard static
            files at ``/`` if ``dashboard/dist/`` exists.  Pass False when
            running in ``--no-dashboard`` mode or during testing.

    Returns:
        A configured FastAPI instance with CORS middleware and all routers
        registered under the ``/api`` prefix.
    """
    app = FastAPI(
        title="AI Craftsman KB",
        version="1.0.0",
        description=(
            "Local-first API for aggregating, indexing, and semantically searching "
            "AI content across HN, Substack, YouTube, Reddit, ArXiv, RSS, and DEV.to."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow the local dashboard dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register all API routers first so they take priority over static files
    app.include_router(stats_router)
    app.include_router(system_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(ingest_router)
    app.include_router(sources_router)
    app.include_router(entities_router)
    app.include_router(radar_router)
    app.include_router(briefings_router)
    app.include_router(usage_router)

    # Mount dashboard static files last — StaticFiles acts as a catch-all
    if serve_dashboard:
        mount_dashboard(app)

    return app


# Module-level singleton used by uvicorn when launched via `cr server` or directly.
# Respects the CRAFTSMAN_NO_DASHBOARD env var set by `cr server --no-dashboard`.
_serve_dashboard = os.environ.get("CRAFTSMAN_NO_DASHBOARD") != "1"
app = create_app(serve_dashboard=_serve_dashboard)
