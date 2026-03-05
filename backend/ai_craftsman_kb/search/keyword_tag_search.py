"""Keyword tag search using SQLite FTS5 over the keywords_fts virtual table.

Provides three operations:

- **search**: FTS5 full-text search across extracted keywords, returning
  ``(document_id, normalized_score)`` pairs ranked by BM25 relevance.
- **get_keywords_for_document**: Retrieve all keywords associated with a document.
- **get_documents_for_keyword**: Retrieve all document IDs tagged with a keyword.

The ``keywords_fts`` virtual table mirrors the ``document_keywords`` table via
triggers. BM25 scores are negative in SQLite (more negative = more relevant);
this module inverts and normalizes them to the [0, 1] range.
"""
from __future__ import annotations

import logging

import aiosqlite

logger = logging.getLogger(__name__)


class KeywordTagSearch:
    """FTS5 search and bidirectional lookup over extracted keyword tags.

    Operates on the ``keywords_fts`` and ``document_keywords`` tables.
    All methods are async and accept an open aiosqlite connection.

    Usage::

        kts = KeywordTagSearch()
        results = await kts.search(conn, "machine learning", limit=20)
        # [("doc-uuid-1", 1.0), ("doc-uuid-2", 0.73), ...]

        keywords = await kts.get_keywords_for_document(conn, "doc-uuid-1")
        # ["machine learning", "transformers", "nlp"]

        doc_ids = await kts.get_documents_for_keyword(conn, "transformers", limit=50)
        # ["doc-uuid-1", "doc-uuid-3"]
    """

    async def search(
        self,
        conn: aiosqlite.Connection,
        query: str,
        limit: int = 50,
    ) -> list[tuple[str, float]]:
        """Run an FTS5 BM25 search on keywords_fts and return normalized scores.

        Searches the ``keywords_fts`` virtual table, joins back to
        ``document_keywords`` to obtain document IDs, and aggregates scores
        per document (a document may match via multiple keywords). Scores
        are normalized to [0, 1] where 1.0 is the most relevant result.

        Args:
            conn: Active aiosqlite connection with the craftsman DB schema.
            query: Free-text search query passed to FTS5 MATCH.
            limit: Maximum number of ``(document_id, score)`` pairs to return.

        Returns:
            A list of ``(document_id, normalized_score)`` tuples sorted by
            descending relevance. Returns an empty list if the query is blank
            or produces no matches.
        """
        if not query or not query.strip():
            return []

        # FTS5 BM25 returns negative scores (more negative = more relevant).
        # We use the built-in `rank` column (equivalent to bm25()) which
        # works correctly in subqueries and aggregations. We aggregate per
        # document_id since a document may match via multiple keywords.
        sql = """
            SELECT dk.document_id, SUM(keywords_fts.rank) AS total_rank
            FROM keywords_fts
            JOIN document_keywords dk ON dk.rowid = keywords_fts.rowid
            WHERE keywords_fts MATCH ?
            GROUP BY dk.document_id
            ORDER BY total_rank
            LIMIT ?
        """

        rows: list[tuple[str, float]] = []
        try:
            async with conn.execute(sql, (query, limit)) as cursor:
                async for row in cursor:
                    rows.append((str(row[0]), float(row[1])))
        except Exception as exc:
            logger.warning("keywords_fts search error for query %r: %s", query, exc)
            return []

        return self._normalize(rows)

    async def get_keywords_for_document(
        self,
        conn: aiosqlite.Connection,
        doc_id: str,
    ) -> list[str]:
        """Retrieve all keywords associated with a document.

        Args:
            conn: Active aiosqlite connection.
            doc_id: The document's UUID string.

        Returns:
            A sorted list of keyword strings for the document.
            Returns an empty list if the document has no keywords.
        """
        sql = """
            SELECT keyword
            FROM document_keywords
            WHERE document_id = ?
            ORDER BY keyword
        """
        async with conn.execute(sql, (doc_id,)) as cursor:
            rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]

    async def get_documents_for_keyword(
        self,
        conn: aiosqlite.Connection,
        keyword: str,
        limit: int = 100,
    ) -> list[str]:
        """Retrieve all document IDs tagged with a specific keyword.

        Performs an exact (case-sensitive) match on the ``keyword`` column
        in ``document_keywords``.

        Args:
            conn: Active aiosqlite connection.
            keyword: The keyword string to look up.
            limit: Maximum number of document IDs to return.

        Returns:
            A list of document UUID strings associated with the keyword.
            Returns an empty list if no documents have the keyword.
        """
        sql = """
            SELECT document_id
            FROM document_keywords
            WHERE keyword = ?
            LIMIT ?
        """
        async with conn.execute(sql, (keyword, limit)) as cursor:
            rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]

    @staticmethod
    def _normalize(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Normalize raw BM25 scores to [0, 1].

        FTS5 BM25 scores are negative (more negative = more relevant). This
        method inverts the sign (so higher = better) then divides by the
        maximum so the top result gets a score of 1.0.

        Args:
            rows: List of ``(doc_id, raw_bm25_score)`` tuples sorted by
                ascending BM25 score (most relevant first in SQLite convention).

        Returns:
            List of ``(doc_id, normalized_score)`` tuples sorted descending
            by normalized score (best first).
        """
        if not rows:
            return []

        # Invert signs: BM25 is negative, more negative = more relevant
        inverted = [(doc_id, -score) for doc_id, score in rows]

        max_score = max(score for _, score in inverted)
        if max_score == 0.0:
            return [(doc_id, 0.0) for doc_id, _ in inverted]

        return [(doc_id, score / max_score) for doc_id, score in inverted]
