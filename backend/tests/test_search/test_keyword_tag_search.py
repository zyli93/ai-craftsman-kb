"""Tests for the KeywordTagSearch class (FTS5 over keywords_fts).

All tests use an in-memory SQLite database with the minimal schema needed
for document_keywords, keywords_fts, and the supporting documents table.
"""
from __future__ import annotations

import pytest
import aiosqlite

from ai_craftsman_kb.search.keyword_tag_search import KeywordTagSearch

# Minimal schema for keyword tag search tests
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

CREATE TABLE IF NOT EXISTS document_keywords (
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    keyword         TEXT NOT NULL,
    UNIQUE(document_id, keyword)
);

CREATE INDEX IF NOT EXISTS idx_document_keywords_keyword ON document_keywords(keyword);
CREATE INDEX IF NOT EXISTS idx_document_keywords_document ON document_keywords(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS keywords_fts USING fts5(
    keyword,
    content='document_keywords',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS keywords_fts_ai AFTER INSERT ON document_keywords BEGIN
    INSERT INTO keywords_fts(rowid, keyword)
    VALUES (new.rowid, new.keyword);
END;

CREATE TRIGGER IF NOT EXISTS keywords_fts_ad AFTER DELETE ON document_keywords BEGIN
    INSERT INTO keywords_fts(keywords_fts, rowid, keyword)
    VALUES ('delete', old.rowid, old.keyword);
END;

CREATE TRIGGER IF NOT EXISTS keywords_fts_au AFTER UPDATE ON document_keywords BEGIN
    INSERT INTO keywords_fts(keywords_fts, rowid, keyword)
    VALUES ('delete', old.rowid, old.keyword);
    INSERT INTO keywords_fts(rowid, keyword)
    VALUES (new.rowid, new.keyword);
END;
"""


@pytest.fixture
async def conn():
    """Create an in-memory SQLite connection with keyword search schema."""
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys = ON")
    await db.executescript(_SCHEMA_SQL)

    # Insert sample documents
    docs = [
        ("doc-1", "hn", "https://example.com/1", "Machine Learning Intro"),
        ("doc-2", "arxiv", "https://example.com/2", "Deep Learning Survey"),
        ("doc-3", "substack", "https://example.com/3", "Python Tutorial"),
    ]
    for doc_id, src, url, title in docs:
        await db.execute(
            "INSERT INTO documents (id, source_type, url, title) VALUES (?, ?, ?, ?)",
            (doc_id, src, url, title),
        )

    # Insert keyword tags
    keywords = [
        ("doc-1", "machine learning"),
        ("doc-1", "neural networks"),
        ("doc-1", "python"),
        ("doc-2", "deep learning"),
        ("doc-2", "neural networks"),
        ("doc-2", "transformers"),
        ("doc-3", "python"),
        ("doc-3", "tutorial"),
    ]
    for doc_id, kw in keywords:
        await db.execute(
            "INSERT INTO document_keywords (document_id, keyword) VALUES (?, ?)",
            (doc_id, kw),
        )

    await db.commit()
    yield db
    await db.close()


@pytest.fixture
def kts() -> KeywordTagSearch:
    """Create a KeywordTagSearch instance."""
    return KeywordTagSearch()


# ---------------------------------------------------------------------------
# search() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_results(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """FTS5 search on keywords_fts returns matching document IDs with scores."""
    results = await kts.search(conn, "neural networks", limit=10)
    assert len(results) > 0
    doc_ids = [doc_id for doc_id, _ in results]
    assert "doc-1" in doc_ids
    assert "doc-2" in doc_ids


@pytest.mark.asyncio
async def test_search_scores_normalized(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """Scores are normalized to [0, 1] with the top result at 1.0."""
    results = await kts.search(conn, "neural networks", limit=10)
    assert len(results) > 0
    scores = [score for _, score in results]
    assert scores[0] == 1.0
    for score in scores:
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_search_respects_limit(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """Limit parameter caps the number of results."""
    results = await kts.search(conn, "neural networks", limit=1)
    assert len(results) <= 1


@pytest.mark.asyncio
async def test_search_empty_query(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """Empty or whitespace-only query returns an empty list."""
    assert await kts.search(conn, "", limit=10) == []
    assert await kts.search(conn, "   ", limit=10) == []


@pytest.mark.asyncio
async def test_search_no_matches(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """Query with no FTS5 matches returns an empty list."""
    results = await kts.search(conn, "nonexistentkeyword12345", limit=10)
    assert results == []


@pytest.mark.asyncio
async def test_search_partial_match(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """FTS5 with porter stemming matches partial/stemmed terms."""
    results = await kts.search(conn, "learn", limit=10)
    assert len(results) > 0
    doc_ids = [doc_id for doc_id, _ in results]
    # "machine learning" and "deep learning" should match via porter stemming
    assert "doc-1" in doc_ids or "doc-2" in doc_ids


# ---------------------------------------------------------------------------
# get_keywords_for_document() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_keywords_for_document(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """Returns all keywords for a given document, sorted alphabetically."""
    keywords = await kts.get_keywords_for_document(conn, "doc-1")
    assert keywords == ["machine learning", "neural networks", "python"]


@pytest.mark.asyncio
async def test_get_keywords_for_document_empty(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """Returns empty list for a document with no keywords."""
    keywords = await kts.get_keywords_for_document(conn, "nonexistent-doc")
    assert keywords == []


# ---------------------------------------------------------------------------
# get_documents_for_keyword() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_documents_for_keyword(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """Returns all document IDs for a given keyword."""
    doc_ids = await kts.get_documents_for_keyword(conn, "python")
    assert set(doc_ids) == {"doc-1", "doc-3"}


@pytest.mark.asyncio
async def test_get_documents_for_keyword_single(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """Returns single document for a unique keyword."""
    doc_ids = await kts.get_documents_for_keyword(conn, "transformers")
    assert doc_ids == ["doc-2"]


@pytest.mark.asyncio
async def test_get_documents_for_keyword_not_found(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """Returns empty list for a keyword that does not exist."""
    doc_ids = await kts.get_documents_for_keyword(conn, "nonexistent")
    assert doc_ids == []


@pytest.mark.asyncio
async def test_get_documents_for_keyword_respects_limit(conn: aiosqlite.Connection, kts: KeywordTagSearch) -> None:
    """Limit parameter caps the number of document IDs returned."""
    doc_ids = await kts.get_documents_for_keyword(conn, "python", limit=1)
    assert len(doc_ids) == 1


# ---------------------------------------------------------------------------
# _normalize() tests
# ---------------------------------------------------------------------------


def test_normalize_empty() -> None:
    """Normalizing an empty list returns an empty list."""
    assert KeywordTagSearch._normalize([]) == []


def test_normalize_single() -> None:
    """Single result normalizes to 1.0."""
    result = KeywordTagSearch._normalize([("doc-1", -2.5)])
    assert len(result) == 1
    assert result[0] == ("doc-1", 1.0)


def test_normalize_multiple() -> None:
    """Multiple results are normalized with max = 1.0."""
    # BM25 scores: more negative = more relevant
    rows = [("doc-1", -4.0), ("doc-2", -2.0)]
    result = KeywordTagSearch._normalize(rows)
    assert result[0] == ("doc-1", 1.0)
    assert result[1] == ("doc-2", 0.5)


def test_normalize_zero_scores() -> None:
    """All-zero scores produce 0.0 for every document."""
    rows = [("doc-1", 0.0), ("doc-2", 0.0)]
    result = KeywordTagSearch._normalize(rows)
    assert all(score == 0.0 for _, score in result)
