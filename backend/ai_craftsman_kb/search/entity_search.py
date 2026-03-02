"""Entity search, browse, and deduplication service.

Provides FTS5-based entity search, entity browsing with sorting/filtering,
entity detail retrieval with linked documents, co-occurrence queries for
the Related Entities panel, and manual entity merge (dedup) for the UI/CLI.

All methods accept an open aiosqlite connection and are fully async.
"""
import json
import logging
from typing import Any, Literal

import aiosqlite
from pydantic import BaseModel

from ..db.models import DocumentRow, EntityRow

# JSON column names that need deserialization when converting aiosqlite.Row to dict.
# Must stay in sync with the set defined in db/queries.py.
_JSON_COLUMNS = {"metadata", "user_tags", "config", "source_document_ids"}


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict, deserializing JSON columns.

    Mirrors the private helper in db/queries.py so this module does not
    depend on a private symbol across package boundaries.

    Args:
        row: An aiosqlite.Row object from a cursor fetch.

    Returns:
        A dict mapping column names to Python values, with JSON columns parsed.
    """
    result: dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if key in _JSON_COLUMNS and isinstance(value, str):
            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                result[key] = value
        else:
            result[key] = value
    return result

logger = logging.getLogger(__name__)


class EntityWithDocs(BaseModel):
    """An entity paired with the documents that mention it.

    Attributes:
        entity: The EntityRow for the entity.
        document_count: Total number of documents that mention this entity.
        top_documents: Up to 5 most recently published documents that mention it.
    """

    entity: EntityRow
    document_count: int
    top_documents: list[DocumentRow]


class CoOccurringEntity(BaseModel):
    """An entity that co-occurs with a target entity in documents.

    Attributes:
        id: The entity's UUID string.
        name: Human-readable entity name.
        entity_type: One of the 7 valid entity types.
        co_count: Number of documents where both entities appear together.
    """

    id: str
    name: str
    entity_type: str
    co_count: int


class EntitySearch:
    """Entity search, browse, and dedup.

    Provides methods to search entities via FTS5, browse them with sorting
    and filtering, get entity details with linked documents, find co-occurring
    entities, and merge duplicate entities into a canonical one.

    All public methods accept a live aiosqlite.Connection. Callers are
    responsible for creating and closing the connection.
    """

    async def search(
        self,
        conn: aiosqlite.Connection,
        query: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[EntityRow]:
        """FTS5 search over entity name, normalized_name, and description.

        When query is a non-empty string, performs a BM25-ranked FTS5 search
        over the entities_fts virtual table. When query is empty or whitespace,
        falls back to list_entities() behavior (sorted by mention_count DESC).

        Optionally filters results to a specific entity_type.

        Args:
            conn: An open aiosqlite connection.
            query: FTS5 query string. Empty string triggers list fallback.
            entity_type: If provided, only return entities of this type.
            limit: Maximum number of results to return.

        Returns:
            A list of EntityRow objects sorted by FTS5 relevance then
            mention_count DESC (or by mention_count alone on empty query).
        """
        # Fall back to list_entities when the query is blank
        if not query or not query.strip():
            return await self.list_entities(
                conn,
                entity_type=entity_type,
                sort_by="mention_count",
                limit=limit,
            )

        # FTS5 search with optional entity_type filter
        # bm25() returns negative values; lower = better match, so ORDER BY rank ASC
        async with conn.execute(
            """
            SELECT e.id, e.name, e.entity_type, e.mention_count,
                   e.normalized_name, e.description,
                   e.first_seen_at, e.metadata,
                   bm25(entities_fts) AS rank
            FROM entities_fts
            JOIN entities e ON e.rowid = entities_fts.rowid
            WHERE entities_fts MATCH ?
              AND (? IS NULL OR e.entity_type = ?)
            ORDER BY rank, e.mention_count DESC
            LIMIT ?
            """,
            (query, entity_type, entity_type, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        return [EntityRow(**_row_to_dict(row)) for row in rows]

    async def list_entities(
        self,
        conn: aiosqlite.Connection,
        entity_type: str | None = None,
        sort_by: Literal["mention_count", "first_seen_at", "name"] = "mention_count",
        limit: int = 50,
        offset: int = 0,
    ) -> list[EntityRow]:
        """Browse entities with optional type filter and configurable sort.

        Returns all (non-deleted) entities, optionally filtered by entity_type,
        sorted by the specified column.  mention_count and first_seen_at sort
        descending (highest/newest first); name sorts ascending (A-Z).

        Args:
            conn: An open aiosqlite connection.
            entity_type: If provided, only return entities of this type.
            sort_by: Column to sort by ('mention_count', 'first_seen_at', 'name').
            limit: Maximum number of results to return.
            offset: Number of results to skip (pagination).

        Returns:
            A list of EntityRow objects.
        """
        # Map sort_by to SQL ORDER BY clause
        order_map: dict[str, str] = {
            "mention_count": "mention_count DESC",
            "first_seen_at": "first_seen_at DESC",
            "name": "name ASC",
        }
        order_clause = order_map.get(sort_by, "mention_count DESC")

        if entity_type is not None:
            async with conn.execute(
                f"""
                SELECT * FROM entities
                WHERE entity_type = ?
                ORDER BY {order_clause}
                LIMIT ? OFFSET ?
                """,  # noqa: S608 — order_clause is from a fixed map, not user input
                (entity_type, limit, offset),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with conn.execute(
                f"""
                SELECT * FROM entities
                ORDER BY {order_clause}
                LIMIT ? OFFSET ?
                """,  # noqa: S608 — order_clause is from a fixed map, not user input
                (limit, offset),
            ) as cursor:
                rows = await cursor.fetchall()

        return [EntityRow(**_row_to_dict(row)) for row in rows]

    async def get_entity_with_docs(
        self,
        conn: aiosqlite.Connection,
        entity_id: str,
        limit: int = 20,
    ) -> EntityWithDocs | None:
        """Get entity + documents that mention it, sorted by published_at DESC.

        Returns None if the entity does not exist.  Fetches a total count
        of linked documents and the top `limit` most recently published ones
        (excluding soft-deleted documents).

        Args:
            conn: An open aiosqlite connection.
            entity_id: The entity's UUID string.
            limit: Maximum number of documents to include in top_documents.

        Returns:
            An EntityWithDocs model, or None if entity_id is not found.
        """
        # Fetch the entity row first
        async with conn.execute(
            "SELECT * FROM entities WHERE id = ?",
            (entity_id,),
        ) as cursor:
            entity_row = await cursor.fetchone()

        if entity_row is None:
            return None

        entity = EntityRow(**_row_to_dict(entity_row))

        # Count total linked documents (excluding soft-deleted)
        async with conn.execute(
            """
            SELECT COUNT(*)
            FROM document_entities de
            JOIN documents d ON d.id = de.document_id
            WHERE de.entity_id = ?
              AND d.deleted_at IS NULL
            """,
            (entity_id,),
        ) as cursor:
            count_row = await cursor.fetchone()
        document_count: int = count_row[0] if count_row else 0

        # Fetch the top `limit` documents sorted by published_at DESC
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
            doc_rows = await cursor.fetchall()

        top_documents = [DocumentRow(**_row_to_dict(r)) for r in doc_rows]

        return EntityWithDocs(
            entity=entity,
            document_count=document_count,
            top_documents=top_documents,
        )

    async def get_co_occurring_entities(
        self,
        conn: aiosqlite.Connection,
        entity_id: str,
        limit: int = 10,
    ) -> list[CoOccurringEntity]:
        """Find entities that frequently appear in the same documents.

        Uses a self-join on document_entities to count shared documents between
        the target entity and every other entity.  Results are sorted by
        co-occurrence count descending — powers the "Related entities" panel.

        Args:
            conn: An open aiosqlite connection.
            entity_id: The target entity's UUID string.
            limit: Maximum number of co-occurring entities to return.

        Returns:
            A list of CoOccurringEntity objects ordered by co_count DESC.
        """
        async with conn.execute(
            """
            SELECT e2.id, e2.name, e2.entity_type, COUNT(*) AS co_count
            FROM document_entities de1
            JOIN document_entities de2
                ON de1.document_id = de2.document_id
               AND de2.entity_id != de1.entity_id
            JOIN entities e2 ON e2.id = de2.entity_id
            WHERE de1.entity_id = ?
            GROUP BY e2.id
            ORDER BY co_count DESC
            LIMIT ?
            """,
            (entity_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            CoOccurringEntity(
                id=row["id"],
                name=row["name"],
                entity_type=row["entity_type"],
                co_count=row["co_count"],
            )
            for row in rows
        ]

    async def merge_entities(
        self,
        conn: aiosqlite.Connection,
        canonical_id: str,
        duplicate_ids: list[str],
    ) -> None:
        """Merge duplicate entities into the canonical entity.

        Steps performed in a single transaction:
        1. Re-assign all document_entities links from each duplicate to canonical_id,
           using INSERT OR IGNORE to avoid PK conflicts when both entities already
           appear in the same document.
        2. Sum the mention_counts from all duplicate rows and add them to canonical.
        3. Delete the duplicate entity rows (cascades to document_entities via FK).
        4. The FTS5 triggers on entities (entities_fts_ad and entities_fts_au)
           automatically keep entities_fts in sync.

        Args:
            conn: An open aiosqlite connection.
            canonical_id: UUID of the entity to keep.
            duplicate_ids: List of UUIDs for entities to merge and delete.

        Raises:
            ValueError: If canonical_id or any duplicate_id is not found in DB.
        """
        if not duplicate_ids:
            logger.debug("merge_entities called with empty duplicate_ids — no-op")
            return

        # Validate canonical entity exists
        async with conn.execute(
            "SELECT id, mention_count FROM entities WHERE id = ?",
            (canonical_id,),
        ) as cursor:
            canonical_row = await cursor.fetchone()

        if canonical_row is None:
            raise ValueError(f"Canonical entity not found: {canonical_id}")

        canonical_mention_count: int = canonical_row["mention_count"]
        total_extra_mentions = 0

        for dup_id in duplicate_ids:
            # Fetch duplicate mention_count
            async with conn.execute(
                "SELECT mention_count FROM entities WHERE id = ?",
                (dup_id,),
            ) as cursor:
                dup_row = await cursor.fetchone()

            if dup_row is None:
                logger.warning("Duplicate entity not found (skipping): %s", dup_id)
                continue

            total_extra_mentions += dup_row["mention_count"]

            # Re-assign document_entities links to the canonical entity.
            # INSERT OR IGNORE: if the canonical already has a link to the same
            # document, the duplicate's link is silently dropped (the document
            # is already associated with the canonical).
            async with conn.execute(
                """
                SELECT document_id, context, relevance
                FROM document_entities
                WHERE entity_id = ?
                """,
                (dup_id,),
            ) as cursor:
                dup_links = await cursor.fetchall()

            for link in dup_links:
                await conn.execute(
                    """
                    INSERT OR IGNORE INTO document_entities
                        (document_id, entity_id, context, relevance)
                    VALUES (?, ?, ?, ?)
                    """,
                    (link["document_id"], canonical_id, link["context"], link["relevance"]),
                )

            # Delete document_entities for duplicate before deleting entity
            # (FK ON DELETE CASCADE would handle this, but we want explicit control
            # to avoid deleting links we just moved to the canonical)
            await conn.execute(
                "DELETE FROM document_entities WHERE entity_id = ?",
                (dup_id,),
            )

            # Delete the duplicate entity row; the entities_fts_ad trigger will
            # automatically remove it from the FTS index
            await conn.execute(
                "DELETE FROM entities WHERE id = ?",
                (dup_id,),
            )
            logger.debug("Merged entity %s into canonical %s", dup_id, canonical_id)

        # Update the canonical entity's mention_count to include all merged counts
        new_mention_count = canonical_mention_count + total_extra_mentions
        await conn.execute(
            "UPDATE entities SET mention_count = ? WHERE id = ?",
            (new_mention_count, canonical_id),
        )
        # The entities_fts_au trigger will update the FTS index for this change

        await conn.commit()
        logger.info(
            "Merged %d duplicate(s) into canonical entity %s (new mention_count=%d)",
            len(duplicate_ids),
            canonical_id,
            new_mention_count,
        )
