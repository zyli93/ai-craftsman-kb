"""Keyword search wrapper around SQLite FTS5 for AI Craftsman KB.

Uses FTS5's built-in BM25 ranking to retrieve relevant documents from the
``documents_fts`` virtual table. BM25 scores returned by SQLite are negative
(more negative = more relevant), so scores are inverted and normalized to
the [0, 1] range before returning.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aiosqlite

logger = logging.getLogger(__name__)


class KeywordSearch:
    """Wrapper around SQLite FTS5 for BM25 keyword search.

    Searches the ``documents_fts`` virtual table using FTS5's built-in BM25
    ranking function. Supports optional filtering by source type and
    publication date. Scores are normalized to [0, 1] by dividing each
    score by the maximum score in the result set.

    Usage::

        ks = KeywordSearch()
        results = await ks.search(conn, "machine learning", limit=20)
        # Returns: [("doc-uuid-1", 1.0), ("doc-uuid-2", 0.87), ...]
    """

    async def search(
        self,
        conn: aiosqlite.Connection,
        query: str,
        limit: int = 50,
        source_types: list[str] | None = None,
        since: str | None = None,
    ) -> list[tuple[str, float]]:
        """Run an FTS5 BM25 search and return normalized (document_id, score) pairs.

        FTS5 BM25 scores are negative — more negative means more relevant.
        This method inverts the sign (making higher = better) then normalizes
        all scores by dividing by the maximum so the top result always has
        score 1.0.

        Args:
            conn: Active aiosqlite connection with FTS5 schema initialized.
            query: Free-text search query. Passed directly to FTS5 MATCH.
            limit: Maximum number of results to return (default 50).
            source_types: Optional list of ``source_type`` values to restrict
                results (e.g. ``['hn', 'substack']``).
            since: Optional ISO 8601 date string. Only documents with
                ``published_at >= since`` are included.

        Returns:
            A list of ``(document_id, normalized_score)`` tuples sorted by
            descending relevance.  Returns an empty list if the query produces
            no FTS5 matches or if the query string is empty.
        """
        if not query or not query.strip():
            return []

        # Build the SQL with optional filter clauses.
        # FTS5 BM25 values are negative — lower is more relevant.
        # The JOIN connects FTS rowid back to documents.id (UUID).
        sql = """
            SELECT d.id, bm25(documents_fts) AS rank
            FROM documents_fts
            JOIN documents d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
              AND d.deleted_at IS NULL
              AND (? IS NULL OR d.source_type = ?)
              AND (? IS NULL OR d.published_at >= ?)
            ORDER BY rank
            LIMIT ?
        """

        # source_types filter: if multiple types, run per-type and merge;
        # if single type or None, use the parameterized form.
        # For simplicity (and to avoid dynamic SQL), run one query per
        # source_type when multiple are requested, then merge.
        if source_types and len(source_types) > 1:
            all_rows: list[tuple[str, float]] = []
            seen: set[str] = set()
            for st in source_types:
                rows = await self._run_query(conn, query, limit, st, since)
                for doc_id, score in rows:
                    if doc_id not in seen:
                        all_rows.append((doc_id, score))
                        seen.add(doc_id)
            # Re-sort the combined results (scores are already raw BM25 at this point)
            all_rows.sort(key=lambda x: x[1])
            # Take top limit
            all_rows = all_rows[:limit]
            return self._normalize(all_rows)
        else:
            single_type = source_types[0] if source_types else None
            rows = await self._run_query(conn, query, limit, single_type, since)
            return self._normalize(rows)

    async def _run_query(
        self,
        conn: aiosqlite.Connection,
        query: str,
        limit: int,
        source_type: str | None,
        since: str | None,
    ) -> list[tuple[str, float]]:
        """Execute the FTS5 SQL query and return raw (doc_id, bm25_score) pairs.

        BM25 scores here are still negative (SQLite convention). Normalization
        is handled by the caller.

        Args:
            conn: Active aiosqlite connection.
            query: FTS5 MATCH expression.
            limit: Maximum number of rows to fetch.
            source_type: Single source type filter, or ``None`` for all.
            since: ISO 8601 date lower bound for ``published_at``, or ``None``.

        Returns:
            List of ``(document_id, raw_bm25_score)`` tuples sorted ascending
            by BM25 score (most relevant first in SQLite's convention).
        """
        sql = """
            SELECT d.id, bm25(documents_fts) AS rank
            FROM documents_fts
            JOIN documents d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
              AND d.deleted_at IS NULL
              AND (? IS NULL OR d.source_type = ?)
              AND (? IS NULL OR d.published_at >= ?)
            ORDER BY rank
            LIMIT ?
        """
        params = (query, source_type, source_type, since, since, limit)

        rows: list[tuple[str, float]] = []
        try:
            async with conn.execute(sql, params) as cursor:
                async for row in cursor:
                    rows.append((str(row[0]), float(row[1])))
        except Exception as exc:
            # FTS5 can raise if the query syntax is invalid; log and return empty
            logger.warning("FTS5 search error for query %r: %s", query, exc)
            return []

        return rows

    @staticmethod
    def _normalize(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Normalize raw BM25 scores to [0, 1].

        FTS5 BM25 scores are negative (more negative = more relevant). This
        method:
        1. Inverts the sign: score = -bm25  (so higher = better).
        2. Divides by the maximum inverted score so the top result is 1.0.

        If all scores are zero or the list is empty the original list is
        returned unchanged.

        Args:
            rows: List of ``(doc_id, raw_bm25_score)`` tuples sorted by
                ascending BM25 score (most relevant first).

        Returns:
            List of ``(doc_id, normalized_score)`` tuples sorted descending
            by normalized score (best first).
        """
        if not rows:
            return []

        # Invert signs (BM25 is negative, more negative = more relevant)
        inverted = [(doc_id, -score) for doc_id, score in rows]

        max_score = max(score for _, score in inverted)
        if max_score == 0.0:
            return [(doc_id, 0.0) for doc_id, _ in inverted]

        return [(doc_id, score / max_score) for doc_id, score in inverted]
