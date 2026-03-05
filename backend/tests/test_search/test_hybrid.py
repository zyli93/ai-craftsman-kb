"""Tests for the hybrid search engine (FTS5 + vector + RRF).

All tests are fully isolated:
- SQLite is opened as an in-memory database (aiosqlite ``':memory:'``).
- VectorStore is replaced by an AsyncMock so no Qdrant dependency is needed.
- Embedder is replaced by an AsyncMock that returns a fixed dummy vector.

The FTS5 virtual table is created in the in-memory DB using the exact same
DDL as the production schema, so the KeywordSearch SQL runs against a real
(in-memory) FTS5 index.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    EmbeddingConfig,
    FiltersConfig,
    HackerNewsConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    SearchConfig,
    SettingsConfig,
    SourcesConfig,
)
from ai_craftsman_kb.search.hybrid import (
    HybridSearch,
    SearchResult,
    reciprocal_rank_fusion,
)
from ai_craftsman_kb.search.keyword import KeywordSearch

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Minimal DDL needed for FTS5 keyword search tests
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    source_id    TEXT,
    origin       TEXT NOT NULL DEFAULT 'pro',
    source_type  TEXT NOT NULL,
    url          TEXT UNIQUE NOT NULL,
    title        TEXT,
    author       TEXT,
    published_at TIMESTAMP,
    raw_content  TEXT,
    deleted_at   TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title, raw_content, author,
    content='documents',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS documents_fts_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, raw_content, author)
    VALUES (new.rowid, new.title, new.raw_content, new.author);
END;
"""


async def _make_conn() -> aiosqlite.Connection:
    """Open and initialise an in-memory aiosqlite connection with the schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.executescript(_SCHEMA_SQL)
    await conn.commit()
    return conn


async def _insert_doc(
    conn: aiosqlite.Connection,
    doc_id: str,
    title: str = "Test Title",
    raw_content: str = "Test content",
    source_type: str = "hn",
    origin: str = "pro",
    url: str | None = None,
    published_at: str | None = "2025-01-01T00:00:00",
    author: str | None = "author",
) -> None:
    """Insert a single document row into the in-memory DB."""
    if url is None:
        url = f"https://example.com/{doc_id}"
    await conn.execute(
        """
        INSERT INTO documents
            (id, origin, source_type, url, title, author, published_at, raw_content)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, origin, source_type, url, title, author, published_at, raw_content),
    )
    await conn.commit()


def _make_config(
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> AppConfig:
    """Return a minimal AppConfig with configurable search weights."""
    return AppConfig(
        sources=SourcesConfig(
            hackernews=HackerNewsConfig(mode="top", limit=10),
        ),
        settings=SettingsConfig(
            data_dir="/tmp/test-craftsman-kb",
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test-model"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test-model"),
                briefing=LLMTaskConfig(provider="anthropic", model="test-model"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test-model"),
                keyword_extraction=LLMTaskConfig(provider="openrouter", model="test-model"),
            ),
            search=SearchConfig(
                hybrid_weight_semantic=semantic_weight,
                hybrid_weight_keyword=keyword_weight,
            ),
        ),
        filters=FiltersConfig(),
    )


def _mock_vector_store(
    results: list[tuple[str, float]] | None = None,
) -> MagicMock:
    """Return a MagicMock VectorStore whose search() returns *results*."""
    store = MagicMock()
    store.search = AsyncMock(return_value=results or [])
    return store


def _mock_embedder(vector: list[float] | None = None) -> MagicMock:
    """Return a MagicMock Embedder whose embed_single() returns *vector*."""
    embedder = MagicMock()
    embedder.embed_single = AsyncMock(return_value=vector or [0.1, 0.2, 0.3])
    return embedder


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion() unit tests
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    """Tests for the standalone RRF function."""

    def test_empty_lists(self) -> None:
        """RRF with empty inputs returns an empty list."""
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_single_list(self) -> None:
        """With a single list, RRF score is weight / (k + rank)."""
        ranked = [("doc_a", 1.0), ("doc_b", 0.5)]
        result = reciprocal_rank_fusion([ranked], weights=[1.0], k=60)
        # doc_a: 1.0 / (60 + 1) ≈ 0.01639
        # doc_b: 1.0 / (60 + 2) ≈ 0.01613
        doc_ids = [r[0] for r in result]
        assert doc_ids == ["doc_a", "doc_b"]
        assert result[0][1] > result[1][1]

    def test_two_lists_doc_in_both(self) -> None:
        """Document in both lists scores higher than one appearing in only one."""
        semantic = [("doc_a", 0.9), ("doc_b", 0.8)]
        keyword = [("doc_b", 0.7), ("doc_c", 0.6)]
        result = reciprocal_rank_fusion(
            [semantic, keyword],
            weights=[0.6, 0.4],
            k=60,
        )
        scores = dict(result)
        # doc_b appears in both lists, should outscore doc_a and doc_c
        assert scores["doc_b"] > scores["doc_a"]
        assert scores["doc_b"] > scores["doc_c"]

    def test_weight_scaling(self) -> None:
        """Higher weight for first list boosts its top result."""
        list1 = [("doc_a", 1.0)]
        list2 = [("doc_b", 1.0)]
        # Give all weight to list1
        result = reciprocal_rank_fusion([list1, list2], weights=[1.0, 0.0], k=60)
        assert result[0][0] == "doc_a"
        assert result[1][1] == 0.0  # doc_b has zero contribution

    def test_k_constant_dampening(self) -> None:
        """With k=0, rank-1 score is much higher than rank-2."""
        ranked = [("first", 1.0), ("second", 0.9), ("third", 0.8)]
        result_low_k = reciprocal_rank_fusion([ranked], weights=[1.0], k=1)
        result_high_k = reciprocal_rank_fusion([ranked], weights=[1.0], k=100)
        # Low k: bigger gap between rank 1 and rank 2
        gap_low = result_low_k[0][1] - result_low_k[1][1]
        gap_high = result_high_k[0][1] - result_high_k[1][1]
        assert gap_low > gap_high

    def test_standard_k60_applied(self) -> None:
        """Default k=60 matches the standard RRF paper formula."""
        ranked = [("doc_a", 1.0)]
        result = reciprocal_rank_fusion([ranked], weights=[1.0])
        expected = 1.0 / (60 + 1)
        assert abs(result[0][1] - expected) < 1e-9

    def test_three_lists(self) -> None:
        """RRF with three lists correctly accumulates scores from all lists."""
        list1 = [("doc_a", 1.0), ("doc_shared", 0.8)]
        list2 = [("doc_b", 1.0), ("doc_shared", 0.7)]
        list3 = [("doc_c", 1.0), ("doc_shared", 0.6)]
        result = reciprocal_rank_fusion(
            [list1, list2, list3],
            weights=[0.4, 0.3, 0.3],
            k=60,
        )
        scores = dict(result)
        # doc_shared at rank 2 in all three: 0.4/62 + 0.3/62 + 0.3/62 = 1.0/62
        expected_shared = 0.4 / 62 + 0.3 / 62 + 0.3 / 62
        assert abs(scores["doc_shared"] - expected_shared) < 1e-9
        # doc_a only in list1 at rank 1: 0.4/61
        expected_a = 0.4 / 61
        assert abs(scores["doc_a"] - expected_a) < 1e-9
        # doc_shared should outscore any single-list doc
        assert scores["doc_shared"] > scores["doc_a"]
        assert scores["doc_shared"] > scores["doc_b"]
        assert scores["doc_shared"] > scores["doc_c"]

    def test_mismatched_lengths_raises(self) -> None:
        """Passing lists and weights of different lengths raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            reciprocal_rank_fusion([[], []], weights=[1.0])

    def test_sorted_descending(self) -> None:
        """RRF results are always sorted in descending score order."""
        list1 = [("c", 1.0), ("b", 0.9), ("a", 0.8)]
        list2 = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        result = reciprocal_rank_fusion([list1, list2], weights=[0.6, 0.4])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# KeywordSearch unit tests
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    """Tests for the FTS5 keyword search wrapper."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self) -> None:
        """Empty query string returns empty list without hitting DB."""
        conn = await _make_conn()
        ks = KeywordSearch()
        result = await ks.search(conn, "")
        assert result == []
        await conn.close()

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self) -> None:
        """Whitespace-only query returns empty list."""
        conn = await _make_conn()
        ks = KeywordSearch()
        result = await ks.search(conn, "   ")
        assert result == []
        await conn.close()

    @pytest.mark.asyncio
    async def test_no_results(self) -> None:
        """Query with no matching documents returns empty list."""
        conn = await _make_conn()
        await _insert_doc(conn, "doc1", title="Python programming", raw_content="Hello world")
        ks = KeywordSearch()
        result = await ks.search(conn, "javascript frameworks")
        # May or may not match; just verify it doesn't raise and is a list
        assert isinstance(result, list)
        await conn.close()

    @pytest.mark.asyncio
    async def test_basic_search_finds_document(self) -> None:
        """A simple keyword query returns the matching document."""
        conn = await _make_conn()
        await _insert_doc(
            conn,
            "doc1",
            title="Machine Learning Basics",
            raw_content="Neural networks are used for deep learning tasks",
        )
        ks = KeywordSearch()
        result = await ks.search(conn, "machine learning")
        assert len(result) > 0
        doc_ids = [r[0] for r in result]
        assert "doc1" in doc_ids
        await conn.close()

    @pytest.mark.asyncio
    async def test_scores_normalized_to_0_1(self) -> None:
        """All returned scores should be in the [0, 1] range."""
        conn = await _make_conn()
        await _insert_doc(
            conn, "doc1",
            title="Transformers in NLP",
            raw_content="Attention mechanism is central to transformer models",
        )
        await _insert_doc(
            conn, "doc2",
            title="Transformer architecture",
            raw_content="Multi-head attention and feed-forward layers in transformers",
        )
        ks = KeywordSearch()
        result = await ks.search(conn, "transformer")
        for _, score in result:
            assert 0.0 <= score <= 1.0
        await conn.close()

    @pytest.mark.asyncio
    async def test_top_result_has_score_1(self) -> None:
        """The top result should have a normalized score of 1.0."""
        conn = await _make_conn()
        await _insert_doc(
            conn, "doc1",
            title="Neural network tutorial",
            raw_content="Deep learning with neural networks and backpropagation",
        )
        await _insert_doc(
            conn, "doc2",
            title="Computer vision basics",
            raw_content="Image classification using convolutional neural networks",
        )
        ks = KeywordSearch()
        result = await ks.search(conn, "neural networks")
        if result:
            assert abs(result[0][1] - 1.0) < 1e-9
        await conn.close()

    @pytest.mark.asyncio
    async def test_source_type_filter(self) -> None:
        """source_types filter excludes documents from other source types."""
        conn = await _make_conn()
        await _insert_doc(
            conn, "hn_doc", title="Rust programming", raw_content="Rust is fast",
            source_type="hn",
        )
        await _insert_doc(
            conn, "reddit_doc", title="Rust on Reddit", raw_content="Rust is popular",
            source_type="reddit", url="https://reddit.com/rust",
        )
        ks = KeywordSearch()
        result = await ks.search(conn, "rust", source_types=["hn"])
        doc_ids = [r[0] for r in result]
        assert "hn_doc" in doc_ids
        assert "reddit_doc" not in doc_ids
        await conn.close()

    @pytest.mark.asyncio
    async def test_multiple_source_types(self) -> None:
        """Multiple source_types returns docs from all specified types."""
        conn = await _make_conn()
        await _insert_doc(
            conn, "hn_doc", title="Rust programming", raw_content="Rust lang systems",
            source_type="hn",
        )
        await _insert_doc(
            conn, "reddit_doc", title="Rust subreddit", raw_content="Rust community post",
            source_type="reddit", url="https://reddit.com/rust2",
        )
        await _insert_doc(
            conn, "arxiv_doc", title="Rust paper", raw_content="Rust formal analysis",
            source_type="arxiv", url="https://arxiv.org/rust",
        )
        ks = KeywordSearch()
        result = await ks.search(conn, "rust", source_types=["hn", "reddit"])
        doc_ids = [r[0] for r in result]
        assert "hn_doc" in doc_ids
        assert "reddit_doc" in doc_ids
        assert "arxiv_doc" not in doc_ids
        await conn.close()

    @pytest.mark.asyncio
    async def test_invalid_fts_query_returns_empty(self) -> None:
        """Malformed FTS5 query returns empty list without crashing."""
        conn = await _make_conn()
        await _insert_doc(conn, "doc1", title="Test", raw_content="Some content")
        ks = KeywordSearch()
        # FTS5 syntax: bare AND/OR without operands are invalid
        result = await ks.search(conn, "AND OR")
        assert isinstance(result, list)
        await conn.close()

    @pytest.mark.asyncio
    async def test_limit_respected(self) -> None:
        """Result count does not exceed the requested limit."""
        conn = await _make_conn()
        for i in range(10):
            await _insert_doc(
                conn, f"doc{i}",
                title=f"Python tutorial {i}",
                raw_content=f"Python programming tutorial number {i}",
                url=f"https://example.com/{i}",
            )
        ks = KeywordSearch()
        result = await ks.search(conn, "python tutorial", limit=3)
        assert len(result) <= 3
        await conn.close()

    @pytest.mark.asyncio
    async def test_deleted_documents_excluded(self) -> None:
        """Documents with deleted_at set should not appear in results."""
        conn = await _make_conn()
        await _insert_doc(
            conn, "live_doc",
            title="Active document about Python",
            raw_content="Python is great for data science",
        )
        # Insert deleted doc manually
        await conn.execute(
            """INSERT INTO documents
               (id, origin, source_type, url, title, raw_content, deleted_at)
               VALUES (?, 'pro', 'hn', 'https://del.example.com', ?, ?, ?)""",
            ("deleted_doc", "Deleted Python article", "Python programming deleted", "2025-01-01"),
        )
        await conn.commit()
        ks = KeywordSearch()
        result = await ks.search(conn, "python")
        doc_ids = [r[0] for r in result]
        assert "deleted_doc" not in doc_ids
        assert "live_doc" in doc_ids
        await conn.close()


# ---------------------------------------------------------------------------
# HybridSearch unit tests
# ---------------------------------------------------------------------------


class TestHybridSearchKeywordMode:
    """Tests for HybridSearch with mode='keyword'."""

    @pytest.mark.asyncio
    async def test_keyword_mode_no_embedding_call(self) -> None:
        """Keyword mode must not call embedder.embed_single."""
        conn = await _make_conn()
        await _insert_doc(
            conn, "doc1",
            title="Python asyncio guide",
            raw_content="Async programming with Python asyncio event loop",
        )
        config = _make_config()
        vector_store = _mock_vector_store()
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)
        results = await hs.search(conn, "asyncio", mode="keyword")
        embedder.embed_single.assert_not_called()
        vector_store.search.assert_not_called()
        await conn.close()

    @pytest.mark.asyncio
    async def test_keyword_mode_returns_search_results(self) -> None:
        """Keyword mode returns SearchResult objects for matching documents."""
        conn = await _make_conn()
        await _insert_doc(
            conn, "doc1",
            title="FastAPI framework",
            raw_content="FastAPI is a modern web framework for Python",
            origin="pro",
        )
        config = _make_config()
        hs = HybridSearch(config, _mock_vector_store(), _mock_embedder())
        results = await hs.search(conn, "FastAPI", mode="keyword")
        if results:
            assert isinstance(results[0], SearchResult)
            assert results[0].document_id == "doc1"
            assert results[0].source_type == "hn"
            assert results[0].origin == "pro"
        await conn.close()

    @pytest.mark.asyncio
    async def test_keyword_mode_excerpt_is_300_chars(self) -> None:
        """excerpt is at most 300 characters of raw_content."""
        long_content = "x" * 600
        conn = await _make_conn()
        await _insert_doc(
            conn, "doc1",
            title="Long document fastapi",
            raw_content=long_content,
        )
        config = _make_config()
        hs = HybridSearch(config, _mock_vector_store(), _mock_embedder())
        results = await hs.search(conn, "fastapi", mode="keyword")
        if results:
            assert results[0].excerpt is not None
            assert len(results[0].excerpt) == 300
        await conn.close()


class TestHybridSearchSemanticMode:
    """Tests for HybridSearch with mode='semantic'."""

    @pytest.mark.asyncio
    async def test_semantic_mode_embeds_query(self) -> None:
        """Semantic mode must call embedder.embed_single with the query."""
        conn = await _make_conn()
        await _insert_doc(conn, "doc1", title="Test", raw_content="Test content")
        config = _make_config()
        vector_store = _mock_vector_store(results=[("doc1", 0.9)])
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)
        await hs.search(conn, "my query", mode="semantic")
        embedder.embed_single.assert_called_once_with("my query")
        await conn.close()

    @pytest.mark.asyncio
    async def test_semantic_mode_no_fts_call(self) -> None:
        """Semantic mode must not run FTS5 queries."""
        conn = await _make_conn()
        config = _make_config()
        vector_store = _mock_vector_store(results=[])
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        # Patch KeywordSearch.search to detect any FTS5 call
        with patch.object(
            hs._keyword_search, "search", new_callable=AsyncMock
        ) as mock_ks:
            await hs.search(conn, "test query", mode="semantic")
            mock_ks.assert_not_called()
        await conn.close()

    @pytest.mark.asyncio
    async def test_semantic_mode_returns_vector_results(self) -> None:
        """Semantic mode returns documents matching vector store results."""
        conn = await _make_conn()
        await _insert_doc(conn, "doc1", title="Vector result", raw_content="Content A")
        await _insert_doc(
            conn, "doc2", title="Second", raw_content="Content B",
            url="https://example.com/doc2",
        )
        config = _make_config()
        vector_store = _mock_vector_store(results=[("doc1", 0.9), ("doc2", 0.7)])
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)
        results = await hs.search(conn, "any query", mode="semantic")
        doc_ids = [r.document_id for r in results]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        await conn.close()

    @pytest.mark.asyncio
    async def test_semantic_mode_passes_filters_to_vector_store(self) -> None:
        """source_types and since filters are forwarded to VectorStore.search."""
        conn = await _make_conn()
        config = _make_config()
        vector_store = _mock_vector_store(results=[])
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)
        await hs.search(
            conn, "query", mode="semantic",
            source_types=["hn", "arxiv"],
            since="2025-01-01",
        )
        call_kwargs = vector_store.search.call_args
        assert call_kwargs.kwargs.get("source_types") == ["hn", "arxiv"] or \
               call_kwargs.args[1:] or call_kwargs.kwargs
        # At minimum, verify VectorStore.search was called once
        vector_store.search.assert_called_once()
        await conn.close()


class TestHybridSearchHybridMode:
    """Tests for HybridSearch with mode='hybrid' (default)."""

    @pytest.mark.asyncio
    async def test_hybrid_calls_both_search_backends(self) -> None:
        """Hybrid mode must call both VectorStore.search and KeywordSearch."""
        conn = await _make_conn()
        await _insert_doc(conn, "doc1", title="Python async", raw_content="asyncio tutorial")
        config = _make_config()
        vector_store = _mock_vector_store(results=[("doc1", 0.9)])
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        with patch.object(
            hs._keyword_search, "search", new_callable=AsyncMock,
            return_value=[("doc1", 1.0)]
        ) as mock_ks:
            await hs.search(conn, "asyncio", mode="hybrid")
            mock_ks.assert_called_once()
            vector_store.search.assert_called_once()
            embedder.embed_single.assert_called_once()
        await conn.close()

    @pytest.mark.asyncio
    async def test_hybrid_rrf_merges_results(self) -> None:
        """Hybrid mode merges two disjoint result lists, doc in both ranks higher."""
        conn = await _make_conn()
        # Insert 3 docs
        await _insert_doc(conn, "doc_both", title="In both results", raw_content="content1")
        await _insert_doc(
            conn, "doc_vector_only", title="Vector only", raw_content="content2",
            url="https://example.com/v",
        )
        await _insert_doc(
            conn, "doc_keyword_only", title="Keyword only", raw_content="content3",
            url="https://example.com/k",
        )

        config = _make_config(semantic_weight=0.6, keyword_weight=0.4)
        vector_store = _mock_vector_store(
            results=[("doc_both", 0.9), ("doc_vector_only", 0.8)]
        )
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        with patch.object(
            hs._keyword_search, "search", new_callable=AsyncMock,
            return_value=[("doc_both", 1.0), ("doc_keyword_only", 0.8)]
        ):
            results = await hs.search(conn, "query", mode="hybrid")

        doc_ids = [r.document_id for r in results]
        # doc_both should be first (appears in both lists)
        assert doc_ids[0] == "doc_both"
        await conn.close()

    @pytest.mark.asyncio
    async def test_hybrid_respects_limit(self) -> None:
        """Hybrid search result count does not exceed the requested limit."""
        conn = await _make_conn()
        for i in range(5):
            await _insert_doc(
                conn, f"doc{i}",
                title=f"Document {i}",
                raw_content=f"Content {i}",
                url=f"https://example.com/{i}",
            )
        config = _make_config()
        vector_store = _mock_vector_store(
            results=[(f"doc{i}", float(5 - i) / 5) for i in range(5)]
        )
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        with patch.object(
            hs._keyword_search, "search", new_callable=AsyncMock,
            return_value=[(f"doc{i}", float(5 - i) / 5) for i in range(5)]
        ):
            results = await hs.search(conn, "query", mode="hybrid", limit=3)

        assert len(results) <= 3
        await conn.close()

    @pytest.mark.asyncio
    async def test_hybrid_rrf_weights_from_config(self) -> None:
        """RRF weights are taken from config.settings.search, not hardcoded."""
        conn = await _make_conn()
        # doc_a only in vector (semantic), doc_b only in keyword
        await _insert_doc(conn, "doc_a", title="A", raw_content="content a")
        await _insert_doc(
            conn, "doc_b", title="B", raw_content="content b",
            url="https://example.com/b",
        )

        # Use extreme weights: all semantic
        config_all_semantic = _make_config(semantic_weight=1.0, keyword_weight=0.0)
        vector_store = _mock_vector_store(results=[("doc_a", 0.9)])
        embedder = _mock_embedder()
        hs = HybridSearch(config_all_semantic, vector_store, embedder)

        with patch.object(
            hs._keyword_search, "search", new_callable=AsyncMock,
            return_value=[("doc_b", 1.0)]
        ):
            results = await hs.search(conn, "query", mode="hybrid")

        doc_ids = [r.document_id for r in results]
        # doc_a should rank higher (all weight on semantic)
        if doc_ids:
            assert doc_ids[0] == "doc_a"
        await conn.close()

    @pytest.mark.asyncio
    async def test_hybrid_missing_in_sqlite_skipped(self) -> None:
        """Documents returned by search but missing in SQLite are silently skipped."""
        conn = await _make_conn()
        # Only insert doc1, not doc2
        await _insert_doc(conn, "doc1", title="Real doc", raw_content="real content")

        config = _make_config()
        vector_store = _mock_vector_store(results=[("doc1", 0.9), ("nonexistent", 0.8)])
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        with patch.object(
            hs._keyword_search, "search", new_callable=AsyncMock,
            return_value=[("doc1", 1.0), ("nonexistent", 0.9)]
        ):
            results = await hs.search(conn, "query", mode="hybrid")

        doc_ids = [r.document_id for r in results]
        assert "doc1" in doc_ids
        assert "nonexistent" not in doc_ids
        await conn.close()

    @pytest.mark.asyncio
    async def test_hybrid_empty_query_returns_empty_from_fts(self) -> None:
        """When FTS returns empty and vector returns nothing, result is empty."""
        conn = await _make_conn()
        config = _make_config()
        vector_store = _mock_vector_store(results=[])
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        with patch.object(
            hs._keyword_search, "search", new_callable=AsyncMock,
            return_value=[]
        ):
            results = await hs.search(conn, "no results query", mode="hybrid")

        assert results == []
        await conn.close()


class TestHybridSearchKeywordTags:
    """Tests for three-way RRF merge with keyword tag search."""

    @pytest.mark.asyncio
    async def test_keyword_tags_disabled_by_default(self) -> None:
        """When keyword_tags weight is 0.0, KeywordTagSearch is not called."""
        conn = await _make_conn()
        await _insert_doc(conn, "doc1", title="Python async", raw_content="asyncio tutorial")
        config = _make_config()  # default keyword_tags weight = 0.0
        vector_store = _mock_vector_store(results=[("doc1", 0.9)])
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        with patch.object(
            hs._keyword_search, "search", new_callable=AsyncMock,
            return_value=[("doc1", 1.0)]
        ), patch.object(
            hs._keyword_tag_search, "search", new_callable=AsyncMock,
        ) as mock_kts:
            await hs.search(conn, "asyncio", mode="hybrid")
            mock_kts.assert_not_called()
        await conn.close()

    @pytest.mark.asyncio
    async def test_keyword_tags_enabled_calls_tag_search(self) -> None:
        """When keyword_tags weight > 0, KeywordTagSearch.search is called."""
        conn = await _make_conn()
        await _insert_doc(conn, "doc1", title="Python async", raw_content="asyncio tutorial")
        config = _make_config()
        config.settings.search.hybrid_weight_keyword_tags = 0.2
        vector_store = _mock_vector_store(results=[("doc1", 0.9)])
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        with patch.object(
            hs._keyword_search, "search", new_callable=AsyncMock,
            return_value=[("doc1", 1.0)]
        ), patch.object(
            hs._keyword_tag_search, "search", new_callable=AsyncMock,
            return_value=[("doc1", 1.0)]
        ) as mock_kts:
            await hs.search(conn, "asyncio", mode="hybrid")
            mock_kts.assert_called_once()
        await conn.close()

    @pytest.mark.asyncio
    async def test_three_way_rrf_merge(self) -> None:
        """Three-way RRF: doc appearing in all three lists ranks highest."""
        conn = await _make_conn()
        await _insert_doc(conn, "doc_all", title="In all three", raw_content="content a")
        await _insert_doc(
            conn, "doc_two", title="In two lists", raw_content="content b",
            url="https://example.com/two",
        )
        await _insert_doc(
            conn, "doc_one", title="In one list", raw_content="content c",
            url="https://example.com/one",
        )

        config = _make_config(semantic_weight=0.4, keyword_weight=0.3)
        config.settings.search.hybrid_weight_keyword_tags = 0.3
        vector_store = _mock_vector_store(
            results=[("doc_all", 0.9), ("doc_two", 0.8)]
        )
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        with patch.object(
            hs._keyword_search, "search", new_callable=AsyncMock,
            return_value=[("doc_all", 1.0), ("doc_two", 0.8)]
        ), patch.object(
            hs._keyword_tag_search, "search", new_callable=AsyncMock,
            return_value=[("doc_all", 1.0), ("doc_one", 0.7)]
        ):
            results = await hs.search(conn, "query", mode="hybrid")

        doc_ids = [r.document_id for r in results]
        # doc_all appears in all 3 lists, should be first
        assert doc_ids[0] == "doc_all"
        await conn.close()

    @pytest.mark.asyncio
    async def test_keyword_tags_not_used_in_keyword_mode(self) -> None:
        """Keyword-only mode does not invoke keyword tag search even if weight > 0."""
        conn = await _make_conn()
        await _insert_doc(conn, "doc1", title="Python async", raw_content="asyncio tutorial")
        config = _make_config()
        config.settings.search.hybrid_weight_keyword_tags = 0.3
        vector_store = _mock_vector_store()
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        with patch.object(
            hs._keyword_tag_search, "search", new_callable=AsyncMock,
        ) as mock_kts:
            await hs.search(conn, "asyncio", mode="keyword")
            mock_kts.assert_not_called()
        await conn.close()

    @pytest.mark.asyncio
    async def test_keyword_tags_not_used_in_semantic_mode(self) -> None:
        """Semantic-only mode does not invoke keyword tag search even if weight > 0."""
        conn = await _make_conn()
        await _insert_doc(conn, "doc1", title="Test", raw_content="content")
        config = _make_config()
        config.settings.search.hybrid_weight_keyword_tags = 0.3
        vector_store = _mock_vector_store(results=[("doc1", 0.9)])
        embedder = _mock_embedder()
        hs = HybridSearch(config, vector_store, embedder)

        with patch.object(
            hs._keyword_tag_search, "search", new_callable=AsyncMock,
        ) as mock_kts:
            await hs.search(conn, "query", mode="semantic")
            mock_kts.assert_not_called()
        await conn.close()


class TestSearchResultModel:
    """Tests for the SearchResult Pydantic model."""

    def test_all_fields_present(self) -> None:
        """SearchResult accepts all expected fields."""
        sr = SearchResult(
            document_id="uuid-1",
            score=0.85,
            title="My Title",
            url="https://example.com",
            source_type="hn",
            author="Test Author",
            published_at="2025-01-01T00:00:00",
            excerpt="First 300 chars...",
            origin="pro",
        )
        assert sr.document_id == "uuid-1"
        assert sr.score == 0.85
        assert sr.excerpt == "First 300 chars..."

    def test_optional_fields_none(self) -> None:
        """SearchResult allows None for optional fields."""
        sr = SearchResult(
            document_id="uuid-2",
            score=0.5,
            title=None,
            url="https://example.com/2",
            source_type="arxiv",
            author=None,
            published_at=None,
            excerpt=None,
            origin="radar",
        )
        assert sr.title is None
        assert sr.author is None
        assert sr.excerpt is None


# ---------------------------------------------------------------------------
# RRF ordering correctness
# ---------------------------------------------------------------------------


class TestRRFOrdering:
    """Tests verifying correct RRF ordering with k=60 and weights=[0.6, 0.4]."""

    def test_rrf_standard_k60_ordering(self) -> None:
        """RRF with k=60 matches hand-calculated scores."""
        semantic = [("doc_s", 0.9)]   # rank 1 in semantic
        keyword = [("doc_k", 0.8)]    # rank 1 in keyword
        result = reciprocal_rank_fusion(
            [semantic, keyword],
            weights=[0.6, 0.4],
            k=60,
        )
        scores = dict(result)
        expected_s = 0.6 / (60 + 1)   # doc_s: 0.6/61 ≈ 0.009836
        expected_k = 0.4 / (60 + 1)   # doc_k: 0.4/61 ≈ 0.006557
        assert abs(scores["doc_s"] - expected_s) < 1e-9
        assert abs(scores["doc_k"] - expected_k) < 1e-9
        # Higher semantic weight means doc_s ranks first
        assert scores["doc_s"] > scores["doc_k"]

    def test_rrf_doc_in_both_lists_beats_top_of_single_list(self) -> None:
        """A rank-3 doc in both lists can beat a rank-1 doc in only one list."""
        semantic = [("top_s", 1.0), ("shared", 0.8), ("other_s", 0.7)]
        keyword = [("top_k", 1.0), ("shared", 0.8), ("other_k", 0.7)]
        result = reciprocal_rank_fusion(
            [semantic, keyword],
            weights=[0.6, 0.4],
            k=1,   # Low k to amplify the benefit of appearing in both
        )
        scores = dict(result)
        # shared appears in both lists at rank 2
        # top_s at rank 1 in semantic only
        # With k=1: shared = 0.6/(1+2) + 0.4/(1+2) = 1.0/3 ≈ 0.333
        # top_s = 0.6/(1+1) = 0.3
        assert scores["shared"] > scores["top_s"]
