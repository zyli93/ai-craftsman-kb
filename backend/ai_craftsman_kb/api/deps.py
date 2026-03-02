"""FastAPI dependency functions shared across routers.

Provides a per-request database connection using the app state's configured
db_path. Routers import ``get_conn`` and inject it via ``Depends(get_conn)``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite
from fastapi import Request

from ..db.sqlite import get_db

logger = logging.getLogger(__name__)

# Fallback data dir when not running in a full lifespan context (e.g. tests).
_DEFAULT_DATA_DIR = Path.home() / ".ai-craftsman-kb" / "data"


async def get_conn(request: Request) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Yield an aiosqlite connection for the current request.

    Reads the db_path from ``request.app.state.db_path`` set during startup.
    Falls back to the default data dir if app state is not available (e.g.
    during testing without a full lifespan context).

    Args:
        request: The incoming FastAPI request, used to access app state.

    Yields:
        An open aiosqlite connection with WAL mode and FK enforcement.
    """
    try:
        db_path: Path = request.app.state.db_path
        data_dir = db_path.parent
    except AttributeError:
        # Fallback for test environments that bypass the lifespan
        data_dir = _DEFAULT_DATA_DIR

    async with get_db(data_dir) as conn:
        yield conn
