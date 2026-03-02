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
"""
from __future__ import annotations

import logging
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
from .config.loader import load_config
from .db.sqlite import init_db
from .llm.router import LLMRouter
from .processing.embedder import Embedder
from .search.vector_store import VectorStore

logger = logging.getLogger(__name__)


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
    app.state.llm_router = LLMRouter(config)

    logger.info("AI Craftsman KB API ready on port %d", config.settings.server.backend_port)

    yield  # App is running

    # Shutdown (nothing to clean up for now)
    logger.info("AI Craftsman KB API shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

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

    # Register all routers
    app.include_router(stats_router)
    app.include_router(system_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(ingest_router)
    app.include_router(sources_router)
    app.include_router(entities_router)
    app.include_router(radar_router)
    app.include_router(briefings_router)

    return app


# Module-level singleton used by uvicorn
app = create_app()
