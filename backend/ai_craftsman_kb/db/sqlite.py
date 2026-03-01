"""SQLite database initialization and connection management.

This module defines the full schema DDL and provides a connection helper
that always enables WAL mode and foreign key enforcement.

FTS5 note: FTS5 content tables (documents_fts, entities_fts) use SQLite's
implicit rowid for internal indexing, even though the primary key on the
content table is a UUID TEXT. The triggers keep the FTS index in sync with
the content table by using `new.rowid` / `old.rowid` which refers to SQLite's
implicit integer rowid that every table has alongside its declared PK.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    id              TEXT PRIMARY KEY,
    source_type     TEXT NOT NULL,
    identifier      TEXT NOT NULL,
    display_name    TEXT,
    tier            TEXT NOT NULL DEFAULT 'pro',
    enabled         BOOLEAN DEFAULT TRUE,
    last_fetched_at TIMESTAMP,
    fetch_error     TEXT,
    config          JSON,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_type, identifier)
);

CREATE TABLE IF NOT EXISTS documents (
    id                      TEXT PRIMARY KEY,
    source_id               TEXT REFERENCES sources(id) ON DELETE SET NULL,
    origin                  TEXT NOT NULL,
    source_type             TEXT NOT NULL,
    url                     TEXT UNIQUE NOT NULL,
    title                   TEXT,
    author                  TEXT,
    published_at            TIMESTAMP,
    fetched_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_type            TEXT,
    raw_content             TEXT,
    word_count              INTEGER,
    metadata                JSON,
    is_embedded             BOOLEAN DEFAULT FALSE,
    is_entities_extracted   BOOLEAN DEFAULT FALSE,
    filter_score            REAL,
    filter_passed           BOOLEAN,
    is_favorited            BOOLEAN DEFAULT FALSE,
    is_archived             BOOLEAN DEFAULT FALSE,
    user_tags               JSON DEFAULT '[]',
    user_notes              TEXT,
    promoted_at             TIMESTAMP,
    deleted_at              TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_source    ON documents(source_type, origin);
CREATE INDEX IF NOT EXISTS idx_documents_date      ON documents(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_processing ON documents(is_embedded, is_entities_extracted);
CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_origin    ON documents(origin);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title, raw_content, author,
    content='documents',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS documents_fts_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, raw_content, author)
    VALUES (new.rowid, new.title, new.raw_content, new.author);
END;

CREATE TRIGGER IF NOT EXISTS documents_fts_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, raw_content, author)
    VALUES ('delete', old.rowid, old.title, old.raw_content, old.author);
END;

CREATE TRIGGER IF NOT EXISTS documents_fts_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, raw_content, author)
    VALUES ('delete', old.rowid, old.title, old.raw_content, old.author);
    INSERT INTO documents_fts(rowid, title, raw_content, author)
    VALUES (new.rowid, new.title, new.raw_content, new.author);
END;

CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    description     TEXT,
    first_seen_at   TIMESTAMP,
    mention_count   INTEGER DEFAULT 1,
    metadata        JSON,
    UNIQUE(normalized_name, entity_type)
);

CREATE TABLE IF NOT EXISTS document_entities (
    document_id     TEXT REFERENCES documents(id) ON DELETE CASCADE,
    entity_id       TEXT REFERENCES entities(id) ON DELETE CASCADE,
    context         TEXT,
    relevance       TEXT,
    PRIMARY KEY (document_id, entity_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    name, normalized_name, description,
    content='entities',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS entities_fts_ai AFTER INSERT ON entities BEGIN
    INSERT INTO entities_fts(rowid, name, normalized_name, description)
    VALUES (new.rowid, new.name, new.normalized_name, new.description);
END;

CREATE TRIGGER IF NOT EXISTS entities_fts_ad AFTER DELETE ON entities BEGIN
    INSERT INTO entities_fts(entities_fts, rowid, name, normalized_name, description)
    VALUES ('delete', old.rowid, old.name, old.normalized_name, old.description);
END;

CREATE TRIGGER IF NOT EXISTS entities_fts_au AFTER UPDATE ON entities BEGIN
    INSERT INTO entities_fts(entities_fts, rowid, name, normalized_name, description)
    VALUES ('delete', old.rowid, old.name, old.normalized_name, old.description);
    INSERT INTO entities_fts(rowid, name, normalized_name, description)
    VALUES (new.rowid, new.name, new.normalized_name, new.description);
END;

CREATE TABLE IF NOT EXISTS discovered_sources (
    id                          TEXT PRIMARY KEY,
    source_type                 TEXT NOT NULL,
    identifier                  TEXT NOT NULL,
    display_name                TEXT,
    discovered_from_document_id TEXT REFERENCES documents(id),
    discovery_method            TEXT,
    confidence                  REAL,
    status                      TEXT DEFAULT 'suggested',
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_type, identifier)
);

CREATE TABLE IF NOT EXISTS briefings (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    query               TEXT,
    content             TEXT NOT NULL,
    source_document_ids JSON DEFAULT '[]',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    format              TEXT DEFAULT 'markdown'
);
"""


@asynccontextmanager
async def get_db(data_dir: Path) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Yield a configured aiosqlite connection with WAL mode and FK enforcement.

    Creates the data directory if it does not exist. Always enables:
    - WAL journal mode for better concurrent read performance
    - Foreign key enforcement (disabled by default in SQLite)
    - Row factory set to aiosqlite.Row for named column access

    Args:
        data_dir: Directory where the craftsman.db file is stored.

    Yields:
        An open aiosqlite.Connection with WAL mode and foreign keys enabled.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "craftsman.db"
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        yield conn


async def init_db(data_dir: Path) -> None:
    """Create all tables, indexes, triggers, and virtual tables.

    This function is idempotent — all DDL statements use IF NOT EXISTS so
    it is safe to call on an existing database without data loss.

    Args:
        data_dir: Directory where the craftsman.db file will be stored.
    """
    async with get_db(data_dir) as conn:
        # executescript runs the full DDL in a single transaction
        await conn.executescript(SCHEMA_SQL)
        await conn.commit()
    logger.info("Database initialized at %s", data_dir / "craftsman.db")
