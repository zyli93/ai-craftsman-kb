"""Async CRUD and FTS query helpers for every SQLite table.

All queries use parameterized ? placeholders — never f-string SQL.
JSON columns (metadata, user_tags, config, source_document_ids) are
serialized with json.dumps() on write and deserialized with json.loads()
on read via the _row_to_dict() helper.
"""
import json
import logging
from typing import Any

import aiosqlite

from .models import (
    BriefingRow,
    DiscoveredSourceRow,
    DocumentRow,
    EntityRow,
    SourceRow,
)

logger = logging.getLogger(__name__)

# JSON column names that need serialization/deserialization
_JSON_COLUMNS = {"metadata", "user_tags", "config", "source_document_ids"}

# Default values for JSON columns when the DB value is NULL.
# These mirror the Pydantic model field defaults so that
# DocumentRow(**row_dict) and similar model constructions succeed
# even when a JSON column has never been written (NULL in SQLite).
_JSON_COLUMN_DEFAULTS: dict[str, Any] = {
    "metadata": {},
    "user_tags": [],
    "config": {},
    "source_document_ids": [],
}


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict, deserializing JSON columns.

    JSON columns are stored as TEXT in SQLite. This helper parses them back
    to Python objects (dict or list). Non-JSON text columns are returned as-is.
    NULL values for JSON columns are replaced with the appropriate default
    (empty dict or empty list) so that Pydantic model construction succeeds.

    Args:
        row: An aiosqlite.Row object from a cursor fetch.

    Returns:
        A dict mapping column names to Python values, with JSON columns parsed.
    """
    result: dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if key in _JSON_COLUMNS:
            if value is None:
                # NULL in DB — use the Pydantic model's default value
                result[key] = _JSON_COLUMN_DEFAULTS.get(key)
            elif isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Fall back to raw value if JSON parsing fails
                    result[key] = value
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def _serialize_doc(doc: DocumentRow) -> dict[str, Any]:
    """Convert a DocumentRow to a flat dict ready for SQLite insertion.

    JSON fields are serialized to strings. Boolean fields are passed as-is
    (SQLite stores them as integers 0/1).

    Args:
        doc: The DocumentRow model instance.

    Returns:
        A dict with all fields ready for parameterized SQL insertion.
    """
    d = doc.model_dump()
    d["metadata"] = json.dumps(d["metadata"])
    d["user_tags"] = json.dumps(d["user_tags"])
    return d


def _serialize_source(source: SourceRow) -> dict[str, Any]:
    """Convert a SourceRow to a flat dict ready for SQLite insertion."""
    d = source.model_dump()
    d["config"] = json.dumps(d["config"])
    return d


def _serialize_entity(entity: EntityRow) -> dict[str, Any]:
    """Convert an EntityRow to a flat dict ready for SQLite insertion."""
    d = entity.model_dump()
    d["metadata"] = json.dumps(d["metadata"])
    return d


def _serialize_briefing(briefing: BriefingRow) -> dict[str, Any]:
    """Convert a BriefingRow to a flat dict ready for SQLite insertion."""
    d = briefing.model_dump()
    d["source_document_ids"] = json.dumps(d["source_document_ids"])
    return d


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


async def upsert_document(conn: aiosqlite.Connection, doc: DocumentRow) -> str:
    """Insert or replace a document row, returning its UUID.

    Uses INSERT OR REPLACE semantics — if a document with the same URL
    already exists, it will be fully replaced. This is intentional for
    re-ingestion and update scenarios.

    Args:
        conn: An open aiosqlite connection.
        doc: The DocumentRow to insert or replace.

    Returns:
        The document's UUID string (doc.id).
    """
    d = _serialize_doc(doc)
    await conn.execute(
        """
        INSERT OR REPLACE INTO documents (
            id, source_id, origin, source_type, url, title, author,
            published_at, fetched_at, content_type, raw_content, word_count,
            metadata, is_embedded, is_entities_extracted, is_keywords_extracted,
            filter_score, filter_passed, is_favorited, is_archived, user_tags,
            user_notes, promoted_at, deleted_at
        ) VALUES (
            :id, :source_id, :origin, :source_type, :url, :title, :author,
            :published_at, :fetched_at, :content_type, :raw_content, :word_count,
            :metadata, :is_embedded, :is_entities_extracted, :is_keywords_extracted,
            :filter_score, :filter_passed, :is_favorited, :is_archived, :user_tags,
            :user_notes, :promoted_at, :deleted_at
        )
        """,
        d,
    )
    await conn.commit()
    logger.debug("Upserted document %s (%s)", doc.id, doc.url)
    return doc.id


async def get_document(conn: aiosqlite.Connection, doc_id: str) -> DocumentRow | None:
    """Fetch a single document by its UUID.

    Args:
        conn: An open aiosqlite connection.
        doc_id: The document's UUID string.

    Returns:
        A DocumentRow if found, or None if no matching document exists.
    """
    async with conn.execute(
        "SELECT * FROM documents WHERE id = ?",
        (doc_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return DocumentRow(**_row_to_dict(row))


async def get_document_by_url(conn: aiosqlite.Connection, url: str) -> DocumentRow | None:
    """Fetch a single document by its URL.

    Args:
        conn: An open aiosqlite connection.
        url: The document's unique URL.

    Returns:
        A DocumentRow if found, or None if no matching document exists.
    """
    async with conn.execute(
        "SELECT * FROM documents WHERE url = ?",
        (url,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return DocumentRow(**_row_to_dict(row))


async def list_documents(
    conn: aiosqlite.Connection,
    origin: str | None = None,
    source_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    include_archived: bool = False,
    include_deleted: bool = False,
) -> list[DocumentRow]:
    """List documents with optional filters, sorted by published_at descending.

    By default, archived and soft-deleted documents are excluded. Pass
    include_archived=True or include_deleted=True to include them.

    Args:
        conn: An open aiosqlite connection.
        origin: Filter by origin ('pro', 'radar', 'adhoc'). None = all.
        source_type: Filter by source type (e.g., 'hn', 'substack'). None = all.
        limit: Maximum number of results to return.
        offset: Number of results to skip (for pagination).
        include_archived: If False, excludes is_archived=TRUE rows.
        include_deleted: If False, excludes rows where deleted_at IS NOT NULL.

    Returns:
        A list of DocumentRow objects.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if origin is not None:
        conditions.append("origin = ?")
        params.append(origin)

    if source_type is not None:
        conditions.append("source_type = ?")
        params.append(source_type)

    if not include_archived:
        conditions.append("is_archived = FALSE")

    if not include_deleted:
        conditions.append("deleted_at IS NULL")

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    params.extend([limit, offset])
    query = f"""
        SELECT * FROM documents
        {where_clause}
        ORDER BY published_at DESC
        LIMIT ? OFFSET ?
    """  # noqa: S608 — no user input in table/column names; only ? params for values

    async with conn.execute(query, params) as cursor:
        rows = await cursor.fetchall()

    return [DocumentRow(**_row_to_dict(row)) for row in rows]


async def update_document_flags(
    conn: aiosqlite.Connection,
    doc_id: str,
    is_embedded: bool | None = None,
    is_entities_extracted: bool | None = None,
    is_keywords_extracted: bool | None = None,
    filter_score: float | None = None,
    filter_passed: bool | None = None,
) -> None:
    """Update processing-status flags on a document row.

    Only the fields that are explicitly passed (not None) are updated.
    This avoids overwriting flags that were not changed.

    Args:
        conn: An open aiosqlite connection.
        doc_id: The document's UUID string.
        is_embedded: If provided, update the embedding status flag.
        is_entities_extracted: If provided, update the entity extraction flag.
        is_keywords_extracted: If provided, update the keyword extraction flag.
        filter_score: If provided, update the content filter score.
        filter_passed: If provided, update the content filter decision.
    """
    sets: list[str] = []
    params: list[Any] = []

    if is_embedded is not None:
        sets.append("is_embedded = ?")
        params.append(is_embedded)

    if is_entities_extracted is not None:
        sets.append("is_entities_extracted = ?")
        params.append(is_entities_extracted)

    if is_keywords_extracted is not None:
        sets.append("is_keywords_extracted = ?")
        params.append(is_keywords_extracted)

    if filter_score is not None:
        sets.append("filter_score = ?")
        params.append(filter_score)

    if filter_passed is not None:
        sets.append("filter_passed = ?")
        params.append(filter_passed)

    if not sets:
        return  # Nothing to update

    params.append(doc_id)
    await conn.execute(
        f"UPDATE documents SET {', '.join(sets)} WHERE id = ?",  # noqa: S608
        params,
    )
    await conn.commit()


async def update_document_user_fields(
    conn: aiosqlite.Connection,
    doc_id: str,
    is_archived: bool | None = None,
    is_favorited: bool | None = None,
    user_tags: list[str] | None = None,
) -> None:
    """Update user-facing fields on a document (archive, favorite, tags).

    Only the fields that are explicitly passed (not None) are updated.

    Args:
        conn: An open aiosqlite connection.
        doc_id: The document's UUID string.
        is_archived: If provided, update the archive status.
        is_favorited: If provided, update the favorite status.
        user_tags: If provided, replace the user tags (stored as JSON).
    """
    sets: list[str] = []
    params: list[Any] = []

    if is_archived is not None:
        sets.append("is_archived = ?")
        params.append(is_archived)

    if is_favorited is not None:
        sets.append("is_favorited = ?")
        params.append(is_favorited)

    if user_tags is not None:
        import json
        sets.append("user_tags = ?")
        params.append(json.dumps(user_tags))

    if not sets:
        return

    params.append(doc_id)
    await conn.execute(
        f"UPDATE documents SET {', '.join(sets)} WHERE id = ?",  # noqa: S608
        params,
    )
    await conn.commit()


async def promote_document(conn: aiosqlite.Connection, doc_id: str) -> None:
    """Promote a radar document to the pro tier by setting promoted_at timestamp.

    Promoted documents appear alongside pro-tier documents in search results.
    The promoted_at timestamp records when the promotion occurred.

    Args:
        conn: An open aiosqlite connection.
        doc_id: The document's UUID string.
    """
    await conn.execute(
        "UPDATE documents SET promoted_at = CURRENT_TIMESTAMP WHERE id = ?",
        (doc_id,),
    )
    await conn.commit()


async def archive_document(conn: aiosqlite.Connection, doc_id: str) -> None:
    """Archive a document by setting is_archived to True.

    Archived documents are hidden from default views but are not deleted.
    Use include_archived=True in list_documents to see archived documents.

    Args:
        conn: An open aiosqlite connection.
        doc_id: The document's UUID string.
    """
    await conn.execute(
        "UPDATE documents SET is_archived = TRUE WHERE id = ?",
        (doc_id,),
    )
    await conn.commit()


async def soft_delete_document(conn: aiosqlite.Connection, doc_id: str) -> None:
    """Soft-delete a document by setting deleted_at to the current timestamp.

    The document is NOT removed from the database, allowing it to be
    recovered or filtered out by list_documents. Use include_deleted=True
    to see soft-deleted documents.

    Args:
        conn: An open aiosqlite connection.
        doc_id: The document's UUID string.
    """
    await conn.execute(
        "UPDATE documents SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?",
        (doc_id,),
    )
    await conn.commit()


async def search_documents_fts(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 20,
) -> list[tuple[str, float]]:
    """Full-text search over documents using FTS5 BM25 ranking.

    Uses the FTS5 bm25() function to rank results. Lower bm25 scores
    indicate better matches (FTS5 returns negative values where more
    negative = better match).

    The search is performed on the title, raw_content, and author fields
    as configured in the documents_fts virtual table.

    Args:
        conn: An open aiosqlite connection.
        query: The FTS5 query string (supports phrase queries, AND/OR, etc.).
        limit: Maximum number of results to return.

    Returns:
        A list of (doc_id, bm25_rank) tuples ordered by relevance rank.
    """
    async with conn.execute(
        """
        SELECT d.id, bm25(documents_fts) AS rank
        FROM documents_fts
        JOIN documents d ON d.rowid = documents_fts.rowid
        WHERE documents_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    ) as cursor:
        rows = await cursor.fetchall()

    return [(row["id"], row["rank"]) for row in rows]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


async def upsert_source(conn: aiosqlite.Connection, source: SourceRow) -> str:
    """Insert or replace a source row, returning its UUID.

    Args:
        conn: An open aiosqlite connection.
        source: The SourceRow to insert or replace.

    Returns:
        The source's UUID string (source.id).
    """
    d = _serialize_source(source)
    await conn.execute(
        """
        INSERT OR REPLACE INTO sources (
            id, source_type, identifier, display_name, tier, enabled,
            last_fetched_at, fetch_error, config, created_at, updated_at
        ) VALUES (
            :id, :source_type, :identifier, :display_name, :tier, :enabled,
            :last_fetched_at, :fetch_error, :config, :created_at, :updated_at
        )
        """,
        d,
    )
    await conn.commit()
    logger.debug("Upserted source %s (%s/%s)", source.id, source.source_type, source.identifier)
    return source.id


async def update_source_fetch_status(
    conn: aiosqlite.Connection,
    source_id: str,
    last_fetched_at: str | None = None,
    fetch_error: str | None = None,
) -> None:
    """Update the fetch timestamp and/or error message on a source row.

    At least one of last_fetched_at or fetch_error should be provided.
    The updated_at column is always refreshed.

    Args:
        conn: An open aiosqlite connection.
        source_id: The source's UUID string.
        last_fetched_at: ISO 8601 timestamp of the most recent successful fetch.
        fetch_error: Error message from the most recent failed fetch, or None
                     to clear a previous error.
    """
    sets: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
    params: list[Any] = []

    if last_fetched_at is not None:
        sets.append("last_fetched_at = ?")
        params.append(last_fetched_at)

    if fetch_error is not None:
        sets.append("fetch_error = ?")
        params.append(fetch_error)
    else:
        # Explicitly clear any previous error when a successful fetch is recorded
        sets.append("fetch_error = NULL")

    params.append(source_id)
    await conn.execute(
        f"UPDATE sources SET {', '.join(sets)} WHERE id = ?",  # noqa: S608
        params,
    )
    await conn.commit()


async def list_sources(
    conn: aiosqlite.Connection,
    enabled_only: bool = False,
) -> list[SourceRow]:
    """List all sources, optionally filtered to enabled ones only.

    Args:
        conn: An open aiosqlite connection.
        enabled_only: If True, only return sources where enabled=TRUE.

    Returns:
        A list of SourceRow objects ordered by source_type, identifier.
    """
    where = "WHERE enabled = TRUE" if enabled_only else ""
    async with conn.execute(
        f"SELECT * FROM sources {where} ORDER BY source_type, identifier",  # noqa: S608
    ) as cursor:
        rows = await cursor.fetchall()
    return [SourceRow(**_row_to_dict(row)) for row in rows]


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


async def upsert_entity(conn: aiosqlite.Connection, entity: EntityRow) -> str:
    """Insert or replace an entity row, returning its UUID.

    Entity deduplication happens at the (normalized_name, entity_type) level.
    Use INSERT OR REPLACE to handle re-extraction of the same entity.

    Args:
        conn: An open aiosqlite connection.
        entity: The EntityRow to insert or replace.

    Returns:
        The entity's UUID string (entity.id).
    """
    d = _serialize_entity(entity)
    await conn.execute(
        """
        INSERT OR REPLACE INTO entities (
            id, name, entity_type, normalized_name, description,
            first_seen_at, mention_count, metadata
        ) VALUES (
            :id, :name, :entity_type, :normalized_name, :description,
            :first_seen_at, :mention_count, :metadata
        )
        """,
        d,
    )
    await conn.commit()
    return entity.id


async def link_document_entity(
    conn: aiosqlite.Connection,
    document_id: str,
    entity_id: str,
    context: str = "",
) -> None:
    """Create a link between a document and an entity.

    Uses INSERT OR IGNORE so that re-linking the same pair is safe.

    Args:
        conn: An open aiosqlite connection.
        document_id: The document's UUID string.
        entity_id: The entity's UUID string.
        context: Optional surrounding text snippet for quick reference.
    """
    await conn.execute(
        """
        INSERT OR IGNORE INTO document_entities (document_id, entity_id, context)
        VALUES (?, ?, ?)
        """,
        (document_id, entity_id, context),
    )
    await conn.commit()


async def search_entities_fts(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 20,
) -> list[EntityRow]:
    """Full-text search over entities using FTS5.

    Searches across name, normalized_name, and description fields.

    Args:
        conn: An open aiosqlite connection.
        query: The FTS5 query string.
        limit: Maximum number of results to return.

    Returns:
        A list of EntityRow objects ordered by FTS5 BM25 relevance.
    """
    async with conn.execute(
        """
        SELECT e.*
        FROM entities_fts
        JOIN entities e ON e.rowid = entities_fts.rowid
        WHERE entities_fts MATCH ?
        ORDER BY bm25(entities_fts)
        LIMIT ?
        """,
        (query, limit),
    ) as cursor:
        rows = await cursor.fetchall()
    return [EntityRow(**_row_to_dict(row)) for row in rows]


async def get_entity_documents(
    conn: aiosqlite.Connection,
    entity_id: str,
    limit: int = 20,
) -> list[DocumentRow]:
    """Fetch documents linked to a specific entity.

    Returns documents ordered by published_at descending, excluding
    soft-deleted documents.

    Args:
        conn: An open aiosqlite connection.
        entity_id: The entity's UUID string.
        limit: Maximum number of documents to return.

    Returns:
        A list of DocumentRow objects linked to the entity.
    """
    async with conn.execute(
        """
        SELECT d.*
        FROM documents d
        JOIN document_entities de ON de.document_id = d.id
        WHERE de.entity_id = ?
          AND d.deleted_at IS NULL
        ORDER BY d.published_at DESC
        LIMIT ?
        """,
        (entity_id, limit),
    ) as cursor:
        rows = await cursor.fetchall()
    return [DocumentRow(**_row_to_dict(row)) for row in rows]


# ---------------------------------------------------------------------------
# Discovered Sources
# ---------------------------------------------------------------------------


async def upsert_discovered_source(
    conn: aiosqlite.Connection,
    source: DiscoveredSourceRow,
) -> str:
    """Insert or replace a discovered source row, returning its UUID.

    Args:
        conn: An open aiosqlite connection.
        source: The DiscoveredSourceRow to insert or replace.

    Returns:
        The discovered source's UUID string (source.id).
    """
    d = source.model_dump()
    await conn.execute(
        """
        INSERT OR REPLACE INTO discovered_sources (
            id, source_type, identifier, display_name,
            discovered_from_document_id, discovery_method, confidence,
            status, created_at
        ) VALUES (
            :id, :source_type, :identifier, :display_name,
            :discovered_from_document_id, :discovery_method, :confidence,
            :status, :created_at
        )
        """,
        d,
    )
    await conn.commit()
    return source.id


async def list_discovered_sources(
    conn: aiosqlite.Connection,
    status: str = "suggested",
) -> list[DiscoveredSourceRow]:
    """List discovered sources filtered by status.

    Args:
        conn: An open aiosqlite connection.
        status: Status filter ('suggested', 'added', 'dismissed').

    Returns:
        A list of DiscoveredSourceRow objects ordered by created_at descending.
    """
    async with conn.execute(
        "SELECT * FROM discovered_sources WHERE status = ? ORDER BY created_at DESC",
        (status,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [DiscoveredSourceRow(**_row_to_dict(row)) for row in rows]


async def update_discovered_source_status(
    conn: aiosqlite.Connection,
    source_id: str,
    status: str,
) -> None:
    """Update the status of a discovered source.

    Args:
        conn: An open aiosqlite connection.
        source_id: The discovered source's UUID string.
        status: The new status value ('suggested', 'added', 'dismissed').
    """
    await conn.execute(
        "UPDATE discovered_sources SET status = ? WHERE id = ?",
        (status, source_id),
    )
    await conn.commit()


# ---------------------------------------------------------------------------
# Briefings
# ---------------------------------------------------------------------------


async def insert_briefing(conn: aiosqlite.Connection, briefing: BriefingRow) -> str:
    """Insert a briefing row (briefings are never updated in-place).

    Args:
        conn: An open aiosqlite connection.
        briefing: The BriefingRow to insert.

    Returns:
        The briefing's UUID string (briefing.id).
    """
    d = _serialize_briefing(briefing)
    await conn.execute(
        """
        INSERT INTO briefings (id, title, query, content, source_document_ids, created_at, format)
        VALUES (:id, :title, :query, :content, :source_document_ids, :created_at, :format)
        """,
        d,
    )
    await conn.commit()
    return briefing.id


async def list_briefings(
    conn: aiosqlite.Connection,
    limit: int = 20,
) -> list[BriefingRow]:
    """List briefings ordered by created_at descending.

    Args:
        conn: An open aiosqlite connection.
        limit: Maximum number of briefings to return.

    Returns:
        A list of BriefingRow objects.
    """
    async with conn.execute(
        "SELECT * FROM briefings ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [BriefingRow(**_row_to_dict(row)) for row in rows]


async def get_briefing(conn: aiosqlite.Connection, briefing_id: str) -> BriefingRow | None:
    """Fetch a single briefing by its UUID.

    Args:
        conn: An open aiosqlite connection.
        briefing_id: The briefing's UUID string.

    Returns:
        A BriefingRow if found, or None if no matching briefing exists.
    """
    async with conn.execute(
        "SELECT * FROM briefings WHERE id = ?",
        (briefing_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return BriefingRow(**_row_to_dict(row))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


async def get_stats(conn: aiosqlite.Connection) -> dict[str, int]:
    """Return aggregate counts across all main tables.

    Returns a dict with the following keys:
    - total_documents: all documents (including archived, excluding deleted)
    - total_entities: all entities
    - total_sources: all configured sources
    - total_briefings: all briefings
    - embedded_documents: documents with is_embedded=TRUE
    - unembedded_documents: documents with is_embedded=FALSE (and not deleted)

    Returns:
        A dict mapping stat name to integer count.
    """
    stats: dict[str, int] = {}

    async with conn.execute(
        "SELECT COUNT(*) FROM documents WHERE deleted_at IS NULL"
    ) as cursor:
        row = await cursor.fetchone()
        stats["total_documents"] = row[0] if row else 0

    async with conn.execute("SELECT COUNT(*) FROM entities") as cursor:
        row = await cursor.fetchone()
        stats["total_entities"] = row[0] if row else 0

    async with conn.execute("SELECT COUNT(*) FROM sources") as cursor:
        row = await cursor.fetchone()
        stats["total_sources"] = row[0] if row else 0

    async with conn.execute("SELECT COUNT(*) FROM briefings") as cursor:
        row = await cursor.fetchone()
        stats["total_briefings"] = row[0] if row else 0

    async with conn.execute(
        "SELECT COUNT(*) FROM documents WHERE is_embedded = TRUE AND deleted_at IS NULL"
    ) as cursor:
        row = await cursor.fetchone()
        stats["embedded_documents"] = row[0] if row else 0

    async with conn.execute(
        "SELECT COUNT(*) FROM documents WHERE is_embedded = FALSE AND deleted_at IS NULL"
    ) as cursor:
        row = await cursor.fetchone()
        stats["unembedded_documents"] = row[0] if row else 0

    return stats
