"""Database package for AI Craftsman KB.

Provides SQLite connection management, schema initialization, and
async CRUD/FTS query helpers for all tables.
"""
from .models import BriefingRow, DiscoveredSourceRow, DocumentRow, EntityRow, SourceRow
from .sqlite import get_db, init_db

__all__ = [
    "get_db",
    "init_db",
    "DocumentRow",
    "SourceRow",
    "EntityRow",
    "DiscoveredSourceRow",
    "BriefingRow",
]
