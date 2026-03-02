"""Qdrant local vector store for AI Craftsman KB.

Provides a ``VectorStore`` wrapper around the Qdrant client for inserting,
searching, and deleting embedding vectors. Uses local file-based storage
(no Docker required). Collection is created automatically on first use.

The Qdrant client SDK is synchronous; heavy operations (upsert, search) are
wrapped in ``asyncio.get_event_loop().run_in_executor`` so they can be called
from async contexts without blocking the event loop.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

if TYPE_CHECKING:
    from ..config.models import AppConfig
    from ..processing.chunker import TextChunk

logger = logging.getLogger(__name__)

# Name of the single Qdrant collection used by this application.
COLLECTION_NAME = "ai_craftsman_kb"

# Fallback dimension used when the configured model is not recognised.
_DEFAULT_VECTOR_SIZE = 1536


def _dimensions_for_provider(config: "AppConfig") -> int:
    """Return the embedding vector dimension for the configured provider/model.

    Inspects ``config.settings.embedding`` to select the correct dimension.
    Falls back to 1536 (OpenAI text-embedding-3-small) for unrecognised models.

    Args:
        config: Fully loaded :class:`~ai_craftsman_kb.config.models.AppConfig`.

    Returns:
        Integer vector dimension.
    """
    embedding_cfg = config.settings.embedding
    provider = embedding_cfg.provider
    model = embedding_cfg.model

    if provider == "openai":
        # OpenAI embedding model dimensions
        openai_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return openai_dims.get(model, _DEFAULT_VECTOR_SIZE)
    elif "nomic" in model:
        return 768
    else:
        # MiniLM and other sentence-transformers defaults
        local_dims = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
        }
        return local_dims.get(model, 384)


def _build_filter(
    source_types: list[str] | None,
    origin: str | None,
    since: str | None,
) -> Filter | None:
    """Construct a Qdrant :class:`Filter` from optional search constraints.

    Builds ``must`` conditions from whichever of the three filter dimensions
    are specified. Returns ``None`` when no constraints are given so that the
    caller can pass it directly to the Qdrant search without extra checks.

    Args:
        source_types: Optional list of source type strings to match
            (e.g. ``['hn', 'substack']``). Uses ``MatchAny`` semantics.
        origin: Optional single origin string to match (``'pro'``, ``'radar'``,
            or ``'adhoc'``).
        since: Optional ISO 8601 date string for a ``>=`` range filter on
            the ``published_at`` payload field.

    Returns:
        A :class:`~qdrant_client.models.Filter` if any constraints are given,
        otherwise ``None``.
    """
    conditions: list[FieldCondition] = []

    if source_types:
        conditions.append(
            FieldCondition(key="source_type", match=MatchAny(any=source_types))
        )
    if origin:
        conditions.append(
            FieldCondition(key="origin", match=MatchValue(value=origin))
        )
    if since:
        # Use DatetimeRange for ISO 8601 date strings — Range only accepts numerics
        conditions.append(
            FieldCondition(key="published_at", range=DatetimeRange(gte=since))
        )

    return Filter(must=conditions) if conditions else None  # type: ignore[arg-type]


class VectorStore:
    """Local Qdrant vector store for document chunk embeddings.

    Wraps the synchronous ``QdrantClient`` and exposes an async-friendly API
    by running blocking calls in a thread executor.  All vectors are stored in
    a single collection (``ai_craftsman_kb``) with a payload schema that allows
    filtering by source type, origin, and publication date.

    Typical usage::

        store = VectorStore(config)
        point_ids = await store.upsert_vectors(
            document_id=doc_id,
            chunks=chunks,
            vectors=vectors,
            payload={"source_type": "hn", "origin": "pro", ...},
        )
        results = await store.search(query_vector, limit=10)

    Args:
        config: Fully loaded :class:`~ai_craftsman_kb.config.models.AppConfig`.
            ``config.settings.data_dir`` determines where Qdrant persists data.
            Pass ``None`` to use an in-memory client (testing only).

    Attributes:
        COLLECTION_NAME: Name of the Qdrant collection used by this store.
    """

    COLLECTION_NAME = COLLECTION_NAME

    def __init__(
        self,
        config: "AppConfig | None" = None,
        *,
        _client: QdrantClient | None = None,
        _vector_size: int | None = None,
    ) -> None:
        """Initialise the vector store and ensure the collection exists.

        Args:
            config: App configuration.  If *None*, an in-memory Qdrant client
                is created (for tests).
            _client: Optional pre-constructed :class:`QdrantClient` to use
                instead (for testing with the in-memory client).
            _vector_size: Optional explicit vector size override.  Used in
                tests that inject a small-dimensioned in-memory client.
        """
        if _client is not None:
            # Injected client — used in tests
            self._client = _client
            self._vector_size = _vector_size if _vector_size is not None else _DEFAULT_VECTOR_SIZE
        elif config is not None:
            qdrant_path = Path(config.settings.data_dir).expanduser() / "qdrant"
            qdrant_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(qdrant_path))
            # Prefer the method on EmbeddingConfig if available, fall back to module helper
            embedding_cfg = config.settings.embedding
            if hasattr(embedding_cfg, "dimensions_for_provider"):
                self._vector_size = embedding_cfg.dimensions_for_provider()
            else:
                self._vector_size = _dimensions_for_provider(config)
        else:
            # Fallback: in-memory (no persistence)
            self._client = QdrantClient(":memory:")
            self._vector_size = _DEFAULT_VECTOR_SIZE

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist.

        Safe to call multiple times — is a no-op when the collection already
        exists.  Uses cosine distance, which is appropriate for normalised
        sentence-embedding vectors.
        """
        existing = self._client.get_collections().collections
        existing_names = [c.name for c in existing]

        if COLLECTION_NAME not in existing_names:
            logger.info(
                "Creating Qdrant collection '%s' with vector_size=%d",
                COLLECTION_NAME,
                self._vector_size,
            )
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.debug("Qdrant collection '%s' already exists.", COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def upsert_vectors(
        self,
        document_id: str,
        chunks: list["TextChunk"],
        vectors: list[list[float]],
        payload: dict,
    ) -> list[str]:
        """Insert or update vectors for a document's chunks.

        Creates one :class:`~qdrant_client.models.PointStruct` per chunk with
        a freshly generated UUID as the point ID.  The supplied ``payload``
        dict is merged with per-chunk metadata (``chunk_index``,
        ``total_chunks``, ``document_id``) before storage.

        Args:
            document_id: UUID string identifying the parent document.
            chunks: List of :class:`~ai_craftsman_kb.processing.chunker.TextChunk`
                objects produced by the chunker.
            vectors: Parallel list of embedding vectors; must be the same length
                as *chunks*.
            payload: Dict of fields to store alongside each vector.  Expected
                keys: ``source_type``, ``origin``, ``title``, ``author``,
                ``published_at``.  ``document_id``, ``chunk_index``, and
                ``total_chunks`` are added automatically.

        Returns:
            List of UUID strings for the created/updated points, one per chunk.

        Raises:
            ValueError: If *chunks* and *vectors* have different lengths.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have the same length "
                f"(got {len(chunks)} chunks and {len(vectors)} vectors)"
            )

        if not chunks:
            return []

        total_chunks = len(chunks)
        points: list[PointStruct] = []
        point_ids: list[str] = []

        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            # Merge base payload with chunk-specific fields
            chunk_payload = {
                **payload,
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "total_chunks": total_chunks,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk_payload,
                )
            )

        logger.debug(
            "Upserting %d vectors for document %s into collection '%s'",
            len(points),
            document_id,
            COLLECTION_NAME,
        )

        # Qdrant client is synchronous — run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            ),
        )

        return point_ids

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def search(
        self,
        query_vector: list[float],
        limit: int = 20,
        source_types: list[str] | None = None,
        origin: str | None = None,
        since: str | None = None,
    ) -> list[tuple[str, float]]:
        """Cosine similarity search over the vector collection.

        Retrieves the top ``limit`` matching chunks and deduplicates them by
        ``document_id``, keeping only the highest-scoring chunk per document.
        This converts chunk-level results into document-level results.

        Args:
            query_vector: The query embedding vector.
            limit: Maximum number of *documents* to return (default 20).
                Internally fetches ``limit * 5`` chunk results to allow for
                adequate deduplication headroom.
            source_types: Optional list of source type strings to filter by
                (e.g. ``['hn', 'substack']``).
            origin: Optional origin filter (``'pro'``, ``'radar'``, ``'adhoc'``).
            since: Optional ISO 8601 date string; only documents published on
                or after this date are returned.

        Returns:
            List of ``(document_id, score)`` tuples sorted by descending
            cosine similarity score.
        """
        query_filter = _build_filter(source_types, origin, since)

        # Fetch more chunk-level results than needed to allow deduplication
        fetch_limit = limit * 5

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=fetch_limit,
                with_payload=True,
            ),
        )

        # Deduplicate by document_id, keeping the highest score per document
        best_scores: dict[str, float] = {}
        for hit in response.points:
            payload = hit.payload or {}
            doc_id = payload.get("document_id", "")
            if not doc_id:
                logger.warning("Search hit missing document_id in payload: %s", hit.id)
                continue

            score = hit.score
            if doc_id not in best_scores or score > best_scores[doc_id]:
                best_scores[doc_id] = score

        # Sort by score descending and limit to requested count
        sorted_results = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    async def delete_document(self, document_id: str) -> None:
        """Delete all vectors associated with a given document.

        Filters points by the ``document_id`` payload field and deletes them
        in a single batch operation.

        Args:
            document_id: UUID string of the document whose vectors should be
                removed.
        """
        delete_filter = Filter(
            must=[  # type: ignore[arg-type]
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

        logger.debug(
            "Deleting all vectors for document %s from collection '%s'",
            document_id,
            COLLECTION_NAME,
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=delete_filter,
            ),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_collection_info(self) -> dict:
        """Return basic statistics about the Qdrant collection.

        Retrieves the collection info from Qdrant and extracts the vector
        count and approximate disk usage.

        Returns:
            Dict with keys:
            - ``vectors_count``: Total number of vectors stored.
            - ``disk_size_bytes``: Approximate disk usage in bytes (0 if
              unavailable, e.g. in-memory mode).
        """
        info = self._client.get_collection(collection_name=COLLECTION_NAME)

        vectors_count: int = 0
        disk_size_bytes: int = 0

        # Modern qdrant-client uses points_count; older versions used vectors_count
        if hasattr(info, "points_count") and info.points_count is not None:
            vectors_count = int(info.points_count)
        elif hasattr(info, "vectors_count") and info.vectors_count is not None:
            vectors_count = int(info.vectors_count)

        # disk_data_size is available in some Qdrant versions / deployment modes
        if hasattr(info, "disk_data_size") and info.disk_data_size is not None:
            disk_size_bytes = int(info.disk_data_size)

        return {
            "vectors_count": vectors_count,
            "disk_size_bytes": disk_size_bytes,
        }
