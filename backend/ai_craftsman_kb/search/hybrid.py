"""Hybrid search engine combining FTS5 keyword search and Qdrant vector search.

Merges results from both search modalities using Reciprocal Rank Fusion (RRF)
with configurable weights. Supports three modes:

- ``'hybrid'``: Run both FTS5 and vector search, merge via RRF.
- ``'semantic'``: Vector search only (no FTS5 query, no API keyword cost).
- ``'keyword'``: FTS5 only (no embedding call, no API cost).

RRF formula (from the original paper, Cormack et al. 2009):
    score(d) = sum(weight_i / (k + rank_i(d)))
where k=60 dampens the absolute rank position.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import aiosqlite
from pydantic import BaseModel

from .keyword import KeywordSearch
from .keyword_tag_search import KeywordTagSearch

if TYPE_CHECKING:
    from ..config.models import AppConfig
    from ..processing.embedder import Embedder
    from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Default RRF k constant from the original paper
_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float],
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists into a single ranking via Reciprocal Rank Fusion.

    For each document appearing in any of the ranked lists, its RRF score is:

        score(d) = sum_i( weight_i / (k + rank_i(d)) )

    Documents that appear in more lists accumulate higher scores. Documents
    absent from a list contribute nothing from that list. The ``k`` constant
    (default 60, from the original RRF paper) dampens the influence of
    absolute rank position.

    Args:
        ranked_lists: A list of ranked result lists. Each inner list contains
            ``(document_id, score)`` tuples in descending score order (best
            first). The score values are not used by RRF — only the rank order
            matters.
        weights: A list of non-negative floats, one per ranked list. Controls
            the relative contribution of each list. Typically ``[0.6, 0.4]``
            for ``[semantic, keyword]``.
        k: RRF constant (default 60). Higher values reduce the influence of
            being top-ranked; lower values amplify the benefit of top ranks.

    Returns:
        A list of ``(document_id, rrf_score)`` tuples sorted by descending
        RRF score. Documents not found in any list are excluded.

    Raises:
        ValueError: If ``ranked_lists`` and ``weights`` have different lengths.
    """
    if len(ranked_lists) != len(weights):
        raise ValueError(
            f"ranked_lists and weights must have the same length "
            f"(got {len(ranked_lists)} and {len(weights)})"
        )

    scores: dict[str, float] = defaultdict(float)
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] += weight / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class SearchResult(BaseModel):
    """A single search result returned by :class:`HybridSearch`.

    Attributes:
        document_id: UUID string of the matching document.
        score: Combined RRF score (hybrid), or normalized vector/BM25 score
            (semantic/keyword modes).
        title: Document title, or ``None`` if not available.
        url: Canonical URL of the document.
        source_type: Source platform (e.g. ``'hn'``, ``'substack'``).
        author: Author name, or ``None`` if not available.
        published_at: ISO 8601 publication timestamp, or ``None``.
        excerpt: First 300 characters of ``raw_content``, or ``None``.
        origin: Ingestion origin: ``'pro'``, ``'radar'``, or ``'adhoc'``.
    """

    document_id: str
    score: float
    title: str | None
    url: str
    source_type: str
    author: str | None
    published_at: str | None
    excerpt: str | None
    origin: str


class HybridSearch:
    """Combine FTS5 keyword search and Qdrant vector search via RRF.

    Orchestrates three search modes:

    - **hybrid**: Runs both FTS5 and vector search in parallel, then merges
      results using Reciprocal Rank Fusion with configurable weights.
    - **semantic**: Vector search only. No FTS5 query is executed.
    - **keyword**: FTS5 search only. No embedding call is made (saves API cost).

    After merging/selecting document IDs, document metadata is batch-fetched
    from SQLite to populate :class:`SearchResult` objects.

    Args:
        config: Fully loaded :class:`~ai_craftsman_kb.config.models.AppConfig`.
            ``config.settings.search`` provides default weights and limits.
        vector_store: A :class:`~ai_craftsman_kb.search.VectorStore` instance
            used for semantic search.
        embedder: An :class:`~ai_craftsman_kb.processing.embedder.Embedder`
            instance used to embed query strings for vector search.

    Usage::

        search = HybridSearch(config, vector_store, embedder)
        results = await search.search(conn, "transformer architecture", limit=10)
    """

    def __init__(
        self,
        config: "AppConfig",
        vector_store: "VectorStore",
        embedder: "Embedder",
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._embedder = embedder
        self._keyword_search = KeywordSearch()
        self._keyword_tag_search = KeywordTagSearch()

    async def search(
        self,
        conn: aiosqlite.Connection,
        query: str,
        mode: Literal["hybrid", "semantic", "keyword"] = "hybrid",
        source_types: list[str] | None = None,
        since: str | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Search documents and return ranked :class:`SearchResult` objects.

        Dispatches to the appropriate search strategy based on *mode*:

        - ``'hybrid'``: FTS5 + vector → RRF merge.
        - ``'semantic'``: vector search only.
        - ``'keyword'``: FTS5 only (no embedding API call).

        For hybrid mode, both searches are fetched with a higher internal
        candidate limit (``limit * 5``) before RRF scoring, so that documents
        appearing in only one result list still have a fair chance of making
        the final top-*limit* cut.

        Args:
            conn: Active aiosqlite connection with the craftsman DB schema.
            query: Free-text search query string.
            mode: Search strategy. One of ``'hybrid'``, ``'semantic'``,
                ``'keyword'``. Defaults to ``'hybrid'``.
            source_types: Optional list of source type strings to restrict
                results (e.g. ``['hn', 'arxiv']``).
            since: Optional ISO 8601 date lower bound for ``published_at``.
            limit: Maximum number of results to return (default 20).

        Returns:
            A list of at most *limit* :class:`SearchResult` objects sorted by
            descending score.
        """
        search_cfg = self._config.settings.search
        semantic_weight = search_cfg.hybrid_weight_semantic
        keyword_weight = search_cfg.hybrid_weight_keyword
        keyword_tags_weight = search_cfg.hybrid_weight_keyword_tags

        # Internal candidate pool size — fetch more to ensure good coverage
        candidate_limit = limit * 5

        if mode == "keyword":
            keyword_results = await self._keyword_search.search(
                conn,
                query,
                limit=limit,
                source_types=source_types,
                since=since,
            )
            ranked: list[tuple[str, float]] = keyword_results[:limit]

        elif mode == "semantic":
            query_vector = await self._embedder.embed_single(query)
            vector_results = await self._vector_store.search(
                query_vector,
                limit=limit,
                source_types=source_types,
                since=since,
            )
            ranked = vector_results[:limit]

        else:  # hybrid
            # Run both searches with larger candidate pools
            query_vector = await self._embedder.embed_single(query)

            import asyncio

            vector_task = self._vector_store.search(
                query_vector,
                limit=candidate_limit,
                source_types=source_types,
                since=since,
            )
            keyword_task = self._keyword_search.search(
                conn,
                query,
                limit=candidate_limit,
                source_types=source_types,
                since=since,
            )

            tasks = [vector_task, keyword_task]

            # Optionally include keyword tag search as a third ranked list
            if keyword_tags_weight > 0:
                keyword_tags_task = self._keyword_tag_search.search(
                    conn,
                    query,
                    limit=candidate_limit,
                )
                tasks.append(keyword_tags_task)

            gathered = await asyncio.gather(*tasks)

            vector_results = gathered[0]
            keyword_results = gathered[1]

            ranked_lists = [vector_results, keyword_results]
            weights = [semantic_weight, keyword_weight]

            if keyword_tags_weight > 0:
                keyword_tags_results = gathered[2]
                ranked_lists.append(keyword_tags_results)
                weights.append(keyword_tags_weight)

            # Merge via RRF with configured weights
            merged = reciprocal_rank_fusion(
                ranked_lists=ranked_lists,
                weights=weights,
            )
            ranked = merged[:limit]

        if not ranked:
            return []

        # Build a score lookup for the final fetch
        scores = dict(ranked)
        doc_ids = [doc_id for doc_id, _ in ranked]

        return await self._fetch_results(conn, doc_ids, scores)

    async def _fetch_results(
        self,
        conn: aiosqlite.Connection,
        doc_ids: list[str],
        scores: dict[str, float],
    ) -> list[SearchResult]:
        """Batch-fetch document metadata from SQLite and build SearchResult objects.

        Uses a single IN-clause query to retrieve all requested documents.
        Results are returned in the same order as *doc_ids* (i.e. descending
        score order). Documents that cannot be found in SQLite are silently
        skipped.

        Args:
            conn: Active aiosqlite connection.
            doc_ids: Ordered list of document UUID strings to fetch.
            scores: Dict mapping ``document_id`` to its combined search score.

        Returns:
            List of :class:`SearchResult` objects in descending score order,
            one per document that was found in SQLite.
        """
        if not doc_ids:
            return []

        # Build a single query with IN clause — parameterized
        placeholders = ",".join("?" * len(doc_ids))
        sql = f"""
            SELECT id, title, url, source_type, author, published_at,
                   raw_content, origin
            FROM documents
            WHERE id IN ({placeholders})
              AND deleted_at IS NULL
        """

        rows_by_id: dict[str, dict] = {}
        async with conn.execute(sql, doc_ids) as cursor:
            async for row in cursor:
                doc_id = str(row[0])
                raw_content = row[6]
                excerpt = raw_content[:300] if raw_content else None
                rows_by_id[doc_id] = {
                    "document_id": doc_id,
                    "title": row[1],
                    "url": str(row[2]),
                    "source_type": str(row[3]),
                    "author": row[4],
                    "published_at": row[5],
                    "excerpt": excerpt,
                    "origin": str(row[7]),
                }

        # Rebuild in original order (preserves score ranking)
        results: list[SearchResult] = []
        for doc_id in doc_ids:
            if doc_id not in rows_by_id:
                logger.debug("Document %s not found in SQLite, skipping.", doc_id)
                continue
            row_data = rows_by_id[doc_id]
            results.append(
                SearchResult(
                    score=scores[doc_id],
                    **row_data,
                )
            )

        return results
