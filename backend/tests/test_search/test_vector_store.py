"""Tests for the VectorStore Qdrant wrapper.

All tests use Qdrant's in-memory mode so no local file system or Docker
dependency is required.  An in-memory QdrantClient is injected via the
``_client`` constructor parameter to keep the tests fully isolated.
"""
from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from ai_craftsman_kb.processing.chunker import TextChunk
from ai_craftsman_kb.search.vector_store import COLLECTION_NAME, VectorStore


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_store() -> VectorStore:
    """Return a VectorStore backed by an in-memory Qdrant client.

    Uses 3-dimensional vectors to keep test data tiny.
    The ``_vector_size`` parameter is passed at construction time so the
    Qdrant collection is created with the correct dimension from the start.
    """
    client = QdrantClient(":memory:")
    return VectorStore(_client=client, _vector_size=3)


def _chunk(idx: int, text: str = "hello world") -> TextChunk:
    """Create a minimal TextChunk for testing."""
    return TextChunk(
        chunk_index=idx,
        text=text,
        token_count=2,
        char_start=0,
        char_end=len(text),
    )


def _vec(val: float) -> list[float]:
    """Return a 3-dimensional unit vector with all components set to *val*."""
    return [val, val, val]


_BASE_PAYLOAD: dict = {
    "source_type": "hn",
    "origin": "pro",
    "title": "Test Article",
    "author": "tester",
    "published_at": "2025-01-01T00:00:00Z",
}


# ---------------------------------------------------------------------------
# _ensure_collection tests
# ---------------------------------------------------------------------------


def test_ensure_collection_creates_collection() -> None:
    """_ensure_collection() should create the collection on first call."""
    client = QdrantClient(":memory:")
    store = VectorStore(_client=client)

    collections = client.get_collections().collections
    names = [c.name for c in collections]
    assert COLLECTION_NAME in names


def test_ensure_collection_is_idempotent() -> None:
    """_ensure_collection() called twice must not raise."""
    client = QdrantClient(":memory:")
    store = VectorStore(_client=client)
    store._ensure_collection()  # second call — should be a no-op

    collections = client.get_collections().collections
    names = [c.name for c in collections]
    assert names.count(COLLECTION_NAME) == 1


# ---------------------------------------------------------------------------
# upsert_vectors tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_vectors_returns_point_ids() -> None:
    """upsert_vectors() should return one UUID per chunk."""
    store = _make_store()
    chunks = [_chunk(0), _chunk(1)]
    vectors = [_vec(0.1), _vec(0.2)]

    ids = await store.upsert_vectors("doc-1", chunks, vectors, _BASE_PAYLOAD)

    assert len(ids) == 2
    # Each ID should be a non-empty string (UUID)
    for id_ in ids:
        assert isinstance(id_, str)
        assert len(id_) == 36  # UUID4 canonical format


@pytest.mark.asyncio
async def test_upsert_vectors_stores_correct_payload() -> None:
    """upsert_vectors() should persist document_id, chunk_index, total_chunks."""
    store = _make_store()
    chunks = [_chunk(0, "chunk zero"), _chunk(1, "chunk one")]
    vectors = [_vec(0.5), _vec(0.6)]

    await store.upsert_vectors("doc-abc", chunks, vectors, _BASE_PAYLOAD)

    # Retrieve the stored points to verify payload
    points = store._client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        limit=10,
    )[0]

    assert len(points) == 2
    for pt in points:
        payload = pt.payload or {}
        assert payload["document_id"] == "doc-abc"
        assert payload["total_chunks"] == 2
        assert payload["source_type"] == "hn"
        assert payload["origin"] == "pro"


@pytest.mark.asyncio
async def test_upsert_vectors_empty_chunks_returns_empty_list() -> None:
    """upsert_vectors() with empty input should return an empty list."""
    store = _make_store()
    ids = await store.upsert_vectors("doc-empty", [], [], _BASE_PAYLOAD)
    assert ids == []


@pytest.mark.asyncio
async def test_upsert_vectors_mismatched_lengths_raises() -> None:
    """upsert_vectors() raises ValueError when chunks and vectors have different lengths."""
    store = _make_store()
    with pytest.raises(ValueError, match="same length"):
        await store.upsert_vectors("doc-x", [_chunk(0)], [], _BASE_PAYLOAD)


# ---------------------------------------------------------------------------
# search tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_document_ids_and_scores() -> None:
    """search() should return (document_id, score) tuples."""
    store = _make_store()
    await store.upsert_vectors("doc-1", [_chunk(0)], [_vec(1.0)], _BASE_PAYLOAD)
    await store.upsert_vectors("doc-2", [_chunk(0)], [_vec(0.5)], _BASE_PAYLOAD)

    results = await store.search(_vec(1.0), limit=10)

    assert len(results) == 2
    doc_ids = [r[0] for r in results]
    assert "doc-1" in doc_ids
    assert "doc-2" in doc_ids

    # Scores should be floats
    for _, score in results:
        assert isinstance(score, float)


@pytest.mark.asyncio
async def test_search_results_sorted_by_score_descending() -> None:
    """search() results should be ordered highest score first.

    Uses orthogonal vectors so cosine similarity clearly distinguishes them:
    [1, 0, 0] queried against [1, 0, 0] scores 1.0, while [0, 1, 0] scores 0.0.
    """
    store = _make_store()
    # doc-high points in the same direction as the query vector
    await store.upsert_vectors("doc-high", [_chunk(0)], [[1.0, 0.0, 0.0]], _BASE_PAYLOAD)
    # doc-low points in a perpendicular direction — cosine similarity ~0
    await store.upsert_vectors("doc-low", [_chunk(0)], [[0.0, 1.0, 0.0]], _BASE_PAYLOAD)

    results = await store.search([1.0, 0.0, 0.0], limit=10)

    assert results[0][0] == "doc-high"
    assert results[0][1] >= results[1][1]


@pytest.mark.asyncio
async def test_search_deduplicates_by_document_id() -> None:
    """search() should return one entry per document even with multiple chunks."""
    store = _make_store()
    # Insert two chunks for the same document
    await store.upsert_vectors(
        "doc-multi",
        [_chunk(0), _chunk(1)],
        [_vec(0.9), _vec(0.8)],
        _BASE_PAYLOAD,
    )

    results = await store.search(_vec(1.0), limit=10)

    doc_ids = [r[0] for r in results]
    # Should appear exactly once
    assert doc_ids.count("doc-multi") == 1


@pytest.mark.asyncio
async def test_search_dedup_keeps_highest_score() -> None:
    """search() deduplication should keep the highest-scoring chunk's score."""
    store = _make_store()
    await store.upsert_vectors(
        "doc-multi",
        [_chunk(0), _chunk(1)],
        [_vec(0.9), _vec(0.1)],
        _BASE_PAYLOAD,
    )

    results = await store.search(_vec(1.0), limit=10)

    doc_id, score = results[0]
    assert doc_id == "doc-multi"
    # Score should correspond to the closer vector (0.9 triple is closer to 1.0 triple)
    assert score > 0.5


@pytest.mark.asyncio
async def test_search_with_source_type_filter() -> None:
    """search() with source_types filter should only return matching documents."""
    store = _make_store()

    hn_payload = {**_BASE_PAYLOAD, "source_type": "hn"}
    substack_payload = {**_BASE_PAYLOAD, "source_type": "substack"}

    await store.upsert_vectors("doc-hn", [_chunk(0)], [_vec(0.9)], hn_payload)
    await store.upsert_vectors("doc-substack", [_chunk(0)], [_vec(0.9)], substack_payload)

    results = await store.search(_vec(1.0), limit=10, source_types=["hn"])

    doc_ids = [r[0] for r in results]
    assert "doc-hn" in doc_ids
    assert "doc-substack" not in doc_ids


@pytest.mark.asyncio
async def test_search_with_origin_filter() -> None:
    """search() with origin filter should only return matching documents."""
    store = _make_store()

    pro_payload = {**_BASE_PAYLOAD, "origin": "pro"}
    radar_payload = {**_BASE_PAYLOAD, "origin": "radar"}

    await store.upsert_vectors("doc-pro", [_chunk(0)], [_vec(0.9)], pro_payload)
    await store.upsert_vectors("doc-radar", [_chunk(0)], [_vec(0.9)], radar_payload)

    results = await store.search(_vec(1.0), limit=10, origin="pro")

    doc_ids = [r[0] for r in results]
    assert "doc-pro" in doc_ids
    assert "doc-radar" not in doc_ids


@pytest.mark.asyncio
async def test_search_with_since_filter() -> None:
    """search() with since filter should exclude documents published before the date."""
    store = _make_store()

    old_payload = {**_BASE_PAYLOAD, "published_at": "2020-01-01T00:00:00Z"}
    new_payload = {**_BASE_PAYLOAD, "published_at": "2025-06-01T00:00:00Z"}

    await store.upsert_vectors("doc-old", [_chunk(0)], [_vec(0.9)], old_payload)
    await store.upsert_vectors("doc-new", [_chunk(0)], [_vec(0.9)], new_payload)

    results = await store.search(_vec(1.0), limit=10, since="2024-01-01T00:00:00Z")

    doc_ids = [r[0] for r in results]
    assert "doc-new" in doc_ids
    assert "doc-old" not in doc_ids


@pytest.mark.asyncio
async def test_search_limit_respected() -> None:
    """search() should not return more results than the limit."""
    store = _make_store()

    for i in range(5):
        await store.upsert_vectors(
            f"doc-{i}", [_chunk(0)], [_vec(float(i) / 10)], _BASE_PAYLOAD
        )

    results = await store.search(_vec(1.0), limit=3)

    assert len(results) <= 3


@pytest.mark.asyncio
async def test_search_empty_collection_returns_empty_list() -> None:
    """search() on an empty collection should return an empty list."""
    store = _make_store()
    results = await store.search(_vec(1.0), limit=10)
    assert results == []


# ---------------------------------------------------------------------------
# delete_document tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_removes_all_chunks() -> None:
    """delete_document() should remove all vectors for the target document."""
    store = _make_store()

    await store.upsert_vectors(
        "doc-to-delete",
        [_chunk(0), _chunk(1)],
        [_vec(0.5), _vec(0.6)],
        _BASE_PAYLOAD,
    )
    await store.upsert_vectors(
        "doc-to-keep",
        [_chunk(0)],
        [_vec(0.7)],
        _BASE_PAYLOAD,
    )

    # Verify both docs are present
    results_before = await store.search(_vec(1.0), limit=10)
    doc_ids_before = [r[0] for r in results_before]
    assert "doc-to-delete" in doc_ids_before
    assert "doc-to-keep" in doc_ids_before

    await store.delete_document("doc-to-delete")

    results_after = await store.search(_vec(1.0), limit=10)
    doc_ids_after = [r[0] for r in results_after]
    assert "doc-to-delete" not in doc_ids_after
    assert "doc-to-keep" in doc_ids_after


@pytest.mark.asyncio
async def test_delete_document_nonexistent_does_not_raise() -> None:
    """delete_document() on a non-existent document_id should not raise."""
    store = _make_store()
    # Should complete without error even when nothing matches
    await store.delete_document("nonexistent-doc-id")


# ---------------------------------------------------------------------------
# get_collection_info tests
# ---------------------------------------------------------------------------


def test_get_collection_info_returns_expected_keys() -> None:
    """get_collection_info() should return a dict with vectors_count and disk_size_bytes."""
    store = _make_store()
    info = store.get_collection_info()

    assert "vectors_count" in info
    assert "disk_size_bytes" in info
    assert isinstance(info["vectors_count"], int)
    assert isinstance(info["disk_size_bytes"], int)


@pytest.mark.asyncio
async def test_get_collection_info_counts_vectors() -> None:
    """get_collection_info() vectors_count should reflect inserted vectors."""
    store = _make_store()

    await store.upsert_vectors(
        "doc-count",
        [_chunk(0), _chunk(1), _chunk(2)],
        [_vec(0.1), _vec(0.2), _vec(0.3)],
        _BASE_PAYLOAD,
    )

    info = store.get_collection_info()
    assert info["vectors_count"] == 3
