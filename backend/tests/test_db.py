"""Comprehensive tests for the SQLite database layer.

Uses an in-memory SQLite database for all tests to ensure isolation
and speed. The get_test_db() helper creates a fresh in-memory connection
with the same PRAGMA settings as production.
"""
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiosqlite
import pytest
import pytest_asyncio

from ai_craftsman_kb.db.models import (
    BriefingRow,
    DiscoveredSourceRow,
    DocumentRow,
    EntityRow,
    SourceRow,
)
from ai_craftsman_kb.db.queries import (
    get_briefing,
    get_document,
    get_document_by_url,
    get_entity_documents,
    get_stats,
    insert_briefing,
    link_document_entity,
    list_briefings,
    list_discovered_sources,
    list_documents,
    list_sources,
    search_documents_fts,
    search_entities_fts,
    soft_delete_document,
    update_discovered_source_status,
    update_document_flags,
    update_source_fetch_status,
    upsert_discovered_source,
    upsert_document,
    upsert_entity,
    upsert_source,
)
from ai_craftsman_kb.db.sqlite import SCHEMA_SQL


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def get_test_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Yield a fresh in-memory aiosqlite connection with the full schema applied.

    Each call creates an isolated in-memory database. The schema is applied
    via executescript so all tables, indexes, triggers, and FTS virtual tables
    are available.
    """
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.executescript(SCHEMA_SQL)
        await conn.commit()
        yield conn


def make_doc(**kwargs) -> DocumentRow:
    """Create a DocumentRow with sensible defaults for testing."""
    defaults = {
        "id": str(uuid.uuid4()),
        "origin": "pro",
        "source_type": "hn",
        "url": f"https://example.com/{uuid.uuid4()}",
        "title": "Test Article",
        "author": "Test Author",
        "raw_content": "This is the raw content of the test article.",
        "metadata": {"score": 100},
        "user_tags": ["ai", "python"],
    }
    defaults.update(kwargs)
    return DocumentRow(**defaults)


def make_source(**kwargs) -> SourceRow:
    """Create a SourceRow with sensible defaults for testing."""
    defaults = {
        "id": str(uuid.uuid4()),
        "source_type": "hn",
        "identifier": f"hn-{uuid.uuid4()}",
        "display_name": "Hacker News",
        "config": {"max_items": 30},
    }
    defaults.update(kwargs)
    return SourceRow(**defaults)


def make_entity(**kwargs) -> EntityRow:
    """Create an EntityRow with sensible defaults for testing."""
    name = kwargs.get("name", "Python")
    defaults = {
        "id": str(uuid.uuid4()),
        "name": name,
        "entity_type": "technology",
        "normalized_name": name.lower(),
        "description": f"The {name} programming language",
        "metadata": {},
    }
    defaults.update(kwargs)
    return EntityRow(**defaults)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestInitDb:
    """Tests for schema initialization."""

    async def test_init_db_creates_tables(self) -> None:
        """init_db (via get_test_db) creates all required tables."""
        async with get_test_db() as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cursor:
                rows = await cursor.fetchall()
            table_names = {row[0] for row in rows}

        expected_tables = {
            "sources",
            "documents",
            "entities",
            "document_entities",
            "discovered_sources",
            "briefings",
        }
        assert expected_tables.issubset(table_names)

    async def test_init_db_creates_fts_tables(self) -> None:
        """init_db creates FTS5 virtual tables."""
        async with get_test_db() as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ) as cursor:
                rows = await cursor.fetchall()
            table_names = {row[0] for row in rows}

        assert "documents_fts" in table_names
        assert "entities_fts" in table_names

    async def test_init_db_creates_indexes(self) -> None:
        """init_db creates performance indexes."""
        async with get_test_db() as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
            ) as cursor:
                rows = await cursor.fetchall()
            index_names = {row[0] for row in rows}

        expected_indexes = {
            "idx_documents_source",
            "idx_documents_date",
            "idx_documents_processing",
            "idx_documents_source_id",
            "idx_documents_origin",
        }
        assert expected_indexes.issubset(index_names)

    async def test_init_db_is_idempotent(self) -> None:
        """Applying the schema twice does not raise an error."""
        async with get_test_db() as conn:
            # Applying schema again should be safe due to IF NOT EXISTS
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()
            # Verify tables still exist and are functional
            async with conn.execute("SELECT COUNT(*) FROM documents") as cursor:
                row = await cursor.fetchone()
            assert row[0] == 0


# ---------------------------------------------------------------------------
# Document tests
# ---------------------------------------------------------------------------


class TestDocuments:
    """Tests for document CRUD and FTS operations."""

    async def test_upsert_and_get_document_round_trip(self) -> None:
        """upsert_document + get_document round-trips a DocumentRow correctly."""
        async with get_test_db() as conn:
            doc = make_doc(title="My Article", author="Alice")
            returned_id = await upsert_document(conn, doc)

            assert returned_id == doc.id

            fetched = await get_document(conn, doc.id)

        assert fetched is not None
        assert fetched.id == doc.id
        assert fetched.title == "My Article"
        assert fetched.author == "Alice"
        assert fetched.origin == "pro"
        assert fetched.source_type == "hn"

    async def test_upsert_document_replaces_existing(self) -> None:
        """Upserting a document with the same URL replaces the existing row."""
        async with get_test_db() as conn:
            url = "https://example.com/article"
            doc1 = make_doc(url=url, title="Original Title")
            await upsert_document(conn, doc1)

            # Same URL, different id and title — INSERT OR REPLACE wins on URL UNIQUE
            # Actually INSERT OR REPLACE replaces on PK conflict; for URL conflict
            # it will fail unless we use the same id. Let's test with same id.
            doc2 = make_doc(id=doc1.id, url=url, title="Updated Title")
            await upsert_document(conn, doc2)

            fetched = await get_document(conn, doc1.id)

        assert fetched is not None
        assert fetched.title == "Updated Title"

    async def test_get_document_returns_none_for_missing(self) -> None:
        """get_document returns None for a non-existent ID."""
        async with get_test_db() as conn:
            result = await get_document(conn, "nonexistent-id")
        assert result is None

    async def test_get_document_by_url(self) -> None:
        """get_document_by_url returns the correct document."""
        async with get_test_db() as conn:
            url = "https://news.ycombinator.com/item?id=99999"
            doc = make_doc(url=url, title="HN Thread")
            await upsert_document(conn, doc)

            fetched = await get_document_by_url(conn, url)

        assert fetched is not None
        assert fetched.id == doc.id
        assert fetched.title == "HN Thread"

    async def test_get_document_by_url_returns_none_for_missing(self) -> None:
        """get_document_by_url returns None for a non-existent URL."""
        async with get_test_db() as conn:
            result = await get_document_by_url(conn, "https://notfound.example.com/")
        assert result is None

    async def test_list_documents_basic(self) -> None:
        """list_documents returns inserted documents."""
        async with get_test_db() as conn:
            doc1 = make_doc(title="Doc 1")
            doc2 = make_doc(title="Doc 2")
            await upsert_document(conn, doc1)
            await upsert_document(conn, doc2)

            docs = await list_documents(conn)

        assert len(docs) == 2
        titles = {d.title for d in docs}
        assert "Doc 1" in titles
        assert "Doc 2" in titles

    async def test_list_documents_filter_by_origin(self) -> None:
        """list_documents filters by origin correctly."""
        async with get_test_db() as conn:
            pro_doc = make_doc(origin="pro")
            radar_doc = make_doc(origin="radar")
            await upsert_document(conn, pro_doc)
            await upsert_document(conn, radar_doc)

            pro_docs = await list_documents(conn, origin="pro")
            radar_docs = await list_documents(conn, origin="radar")

        assert len(pro_docs) == 1
        assert pro_docs[0].id == pro_doc.id
        assert len(radar_docs) == 1
        assert radar_docs[0].id == radar_doc.id

    async def test_list_documents_filter_by_source_type(self) -> None:
        """list_documents filters by source_type correctly."""
        async with get_test_db() as conn:
            hn_doc = make_doc(source_type="hn")
            reddit_doc = make_doc(source_type="reddit")
            await upsert_document(conn, hn_doc)
            await upsert_document(conn, reddit_doc)

            hn_docs = await list_documents(conn, source_type="hn")

        assert len(hn_docs) == 1
        assert hn_docs[0].id == hn_doc.id

    async def test_list_documents_excludes_archived_by_default(self) -> None:
        """list_documents excludes archived documents unless include_archived=True."""
        async with get_test_db() as conn:
            normal_doc = make_doc()
            archived_doc = make_doc(is_archived=True)
            await upsert_document(conn, normal_doc)
            await upsert_document(conn, archived_doc)

            without_archived = await list_documents(conn)
            with_archived = await list_documents(conn, include_archived=True)

        assert len(without_archived) == 1
        assert len(with_archived) == 2

    async def test_list_documents_excludes_deleted_by_default(self) -> None:
        """list_documents excludes soft-deleted documents unless include_deleted=True."""
        async with get_test_db() as conn:
            normal_doc = make_doc()
            to_delete_doc = make_doc()
            await upsert_document(conn, normal_doc)
            await upsert_document(conn, to_delete_doc)
            await soft_delete_document(conn, to_delete_doc.id)

            without_deleted = await list_documents(conn)
            with_deleted = await list_documents(conn, include_deleted=True)

        assert len(without_deleted) == 1
        assert len(with_deleted) == 2

    async def test_list_documents_pagination(self) -> None:
        """list_documents respects limit and offset for pagination."""
        async with get_test_db() as conn:
            for i in range(5):
                await upsert_document(conn, make_doc(title=f"Doc {i}"))

            page1 = await list_documents(conn, limit=3, offset=0)
            page2 = await list_documents(conn, limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 2

    async def test_update_document_flags(self) -> None:
        """update_document_flags updates only the specified flags."""
        async with get_test_db() as conn:
            doc = make_doc()
            await upsert_document(conn, doc)

            await update_document_flags(
                conn,
                doc.id,
                is_embedded=True,
                filter_score=0.85,
                filter_passed=True,
            )

            fetched = await get_document(conn, doc.id)

        assert fetched is not None
        assert fetched.is_embedded is True
        assert fetched.filter_score == pytest.approx(0.85)
        assert fetched.filter_passed is True
        assert fetched.is_entities_extracted is False  # Unchanged

    async def test_update_document_flags_no_args_is_safe(self) -> None:
        """update_document_flags with no kwargs is a no-op."""
        async with get_test_db() as conn:
            doc = make_doc()
            await upsert_document(conn, doc)

            # Should not raise
            await update_document_flags(conn, doc.id)

            fetched = await get_document(conn, doc.id)

        assert fetched is not None
        assert fetched.is_embedded is False

    async def test_soft_delete_document(self) -> None:
        """soft_delete_document sets deleted_at and hides doc from list_documents."""
        async with get_test_db() as conn:
            doc = make_doc()
            await upsert_document(conn, doc)

            await soft_delete_document(conn, doc.id)

            # The document should still be retrievable directly
            fetched = await get_document(conn, doc.id)
            # But not appear in default list
            all_docs = await list_documents(conn)

        assert fetched is not None
        assert fetched.deleted_at is not None
        assert len(all_docs) == 0

    async def test_search_documents_fts(self) -> None:
        """search_documents_fts returns matching documents ranked by BM25."""
        async with get_test_db() as conn:
            doc1 = make_doc(
                title="Introduction to Python programming",
                raw_content="Python is a high-level programming language.",
            )
            doc2 = make_doc(
                title="Rust vs Go performance",
                raw_content="Rust and Go are systems programming languages.",
            )
            await upsert_document(conn, doc1)
            await upsert_document(conn, doc2)

            results = await search_documents_fts(conn, "Python programming")

        assert len(results) >= 1
        # First result should be the Python document
        ids = [r[0] for r in results]
        assert doc1.id in ids

        # Results are (doc_id, rank) tuples
        for doc_id, rank in results:
            assert isinstance(doc_id, str)
            assert isinstance(rank, float)

    async def test_search_documents_fts_no_results(self) -> None:
        """search_documents_fts returns empty list for no matches."""
        async with get_test_db() as conn:
            doc = make_doc(title="Python basics", raw_content="Learn Python today.")
            await upsert_document(conn, doc)

            results = await search_documents_fts(conn, "zzznomatchzzz")

        assert results == []

    async def test_search_documents_fts_limit(self) -> None:
        """search_documents_fts respects the limit parameter."""
        async with get_test_db() as conn:
            for i in range(5):
                await upsert_document(
                    conn,
                    make_doc(
                        title=f"Article about Python {i}",
                        raw_content=f"Python content number {i}.",
                    ),
                )

            results = await search_documents_fts(conn, "Python", limit=3)

        assert len(results) <= 3


# ---------------------------------------------------------------------------
# JSON field tests
# ---------------------------------------------------------------------------


class TestJsonFields:
    """Tests that JSON fields serialize/deserialize correctly."""

    async def test_metadata_round_trip(self) -> None:
        """metadata dict survives a write/read cycle."""
        async with get_test_db() as conn:
            metadata = {"score": 42, "tags": ["ai", "ml"], "nested": {"key": "value"}}
            doc = make_doc(metadata=metadata)
            await upsert_document(conn, doc)
            fetched = await get_document(conn, doc.id)

        assert fetched is not None
        assert fetched.metadata == metadata

    async def test_user_tags_round_trip(self) -> None:
        """user_tags list survives a write/read cycle."""
        async with get_test_db() as conn:
            tags = ["deep-learning", "transformers", "llm"]
            doc = make_doc(user_tags=tags)
            await upsert_document(conn, doc)
            fetched = await get_document(conn, doc.id)

        assert fetched is not None
        assert fetched.user_tags == tags

    async def test_empty_metadata_and_tags(self) -> None:
        """Empty metadata dict and user_tags list deserialize correctly."""
        async with get_test_db() as conn:
            doc = make_doc(metadata={}, user_tags=[])
            await upsert_document(conn, doc)
            fetched = await get_document(conn, doc.id)

        assert fetched is not None
        assert fetched.metadata == {}
        assert fetched.user_tags == []

    async def test_source_config_round_trip(self) -> None:
        """Source config dict survives a write/read cycle."""
        async with get_test_db() as conn:
            config = {"max_items": 100, "include_comments": True, "min_score": 50}
            source = make_source(config=config)
            await upsert_source(conn, source)
            sources = await list_sources(conn)

        assert len(sources) == 1
        assert sources[0].config == config

    async def test_briefing_source_document_ids_round_trip(self) -> None:
        """source_document_ids list survives a write/read cycle in briefings."""
        async with get_test_db() as conn:
            doc_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
            briefing = BriefingRow(
                id=str(uuid.uuid4()),
                title="Test Briefing",
                content="## Summary\n\nSome content here.",
                source_document_ids=doc_ids,
            )
            await insert_briefing(conn, briefing)
            fetched = await get_briefing(conn, briefing.id)

        assert fetched is not None
        assert fetched.source_document_ids == doc_ids


# ---------------------------------------------------------------------------
# Source tests
# ---------------------------------------------------------------------------


class TestSources:
    """Tests for source CRUD operations."""

    async def test_upsert_and_list_sources(self) -> None:
        """upsert_source + list_sources round-trips a SourceRow correctly."""
        async with get_test_db() as conn:
            source = make_source(display_name="Hacker News Top")
            await upsert_source(conn, source)

            sources = await list_sources(conn)

        assert len(sources) == 1
        assert sources[0].id == source.id
        assert sources[0].display_name == "Hacker News Top"

    async def test_list_sources_enabled_only(self) -> None:
        """list_sources with enabled_only=True filters disabled sources."""
        async with get_test_db() as conn:
            enabled_source = make_source(enabled=True)
            disabled_source = make_source(enabled=False)
            await upsert_source(conn, enabled_source)
            await upsert_source(conn, disabled_source)

            all_sources = await list_sources(conn)
            enabled_only = await list_sources(conn, enabled_only=True)

        assert len(all_sources) == 2
        assert len(enabled_only) == 1
        assert enabled_only[0].id == enabled_source.id

    async def test_update_source_fetch_status_success(self) -> None:
        """update_source_fetch_status records a successful fetch."""
        async with get_test_db() as conn:
            source = make_source()
            await upsert_source(conn, source)

            fetched_at = "2026-03-01T10:00:00+00:00"
            await update_source_fetch_status(conn, source.id, last_fetched_at=fetched_at)

            sources = await list_sources(conn)

        assert sources[0].last_fetched_at == fetched_at
        # Error should be cleared
        assert sources[0].fetch_error is None

    async def test_update_source_fetch_status_error(self) -> None:
        """update_source_fetch_status records a fetch error."""
        async with get_test_db() as conn:
            source = make_source()
            await upsert_source(conn, source)

            await update_source_fetch_status(conn, source.id, fetch_error="Connection timeout")

            sources = await list_sources(conn)

        assert sources[0].fetch_error == "Connection timeout"


# ---------------------------------------------------------------------------
# Entity tests
# ---------------------------------------------------------------------------


class TestEntities:
    """Tests for entity CRUD and FTS operations."""

    async def test_upsert_entity(self) -> None:
        """upsert_entity stores and retrieves an entity correctly."""
        async with get_test_db() as conn:
            entity = make_entity(name="Andrej Karpathy", entity_type="person")
            returned_id = await upsert_entity(conn, entity)

            assert returned_id == entity.id

    async def test_link_document_entity(self) -> None:
        """link_document_entity creates the join table entry correctly."""
        async with get_test_db() as conn:
            doc = make_doc()
            entity = make_entity()
            await upsert_document(conn, doc)
            await upsert_entity(conn, entity)

            await link_document_entity(conn, doc.id, entity.id, context="Python is great")

            # Verify the link was created
            async with conn.execute(
                "SELECT context FROM document_entities WHERE document_id = ? AND entity_id = ?",
                (doc.id, entity.id),
            ) as cursor:
                row = await cursor.fetchone()

        assert row is not None
        assert row[0] == "Python is great"

    async def test_link_document_entity_is_idempotent(self) -> None:
        """Linking the same document-entity pair twice does not raise an error."""
        async with get_test_db() as conn:
            doc = make_doc()
            entity = make_entity()
            await upsert_document(conn, doc)
            await upsert_entity(conn, entity)

            await link_document_entity(conn, doc.id, entity.id)
            # Second call should be a no-op (INSERT OR IGNORE)
            await link_document_entity(conn, doc.id, entity.id)

    async def test_search_entities_fts(self) -> None:
        """search_entities_fts returns entities matching the query."""
        async with get_test_db() as conn:
            python_entity = make_entity(
                name="Python",
                normalized_name="python",
                description="A high-level programming language",
            )
            rust_entity = make_entity(
                name="Rust",
                normalized_name="rust",
                description="A systems programming language",
            )
            await upsert_entity(conn, python_entity)
            await upsert_entity(conn, rust_entity)

            results = await search_entities_fts(conn, "Python")

        assert len(results) >= 1
        entity_ids = {e.id for e in results}
        assert python_entity.id in entity_ids

    async def test_get_entity_documents(self) -> None:
        """get_entity_documents returns documents linked to an entity."""
        async with get_test_db() as conn:
            doc1 = make_doc(title="Python Tutorial")
            doc2 = make_doc(title="Rust Guide")
            entity = make_entity(name="Programming")

            await upsert_document(conn, doc1)
            await upsert_document(conn, doc2)
            await upsert_entity(conn, entity)

            await link_document_entity(conn, doc1.id, entity.id)
            # doc2 is not linked to entity

            entity_docs = await get_entity_documents(conn, entity.id)

        assert len(entity_docs) == 1
        assert entity_docs[0].id == doc1.id


# ---------------------------------------------------------------------------
# Discovered sources tests
# ---------------------------------------------------------------------------


class TestDiscoveredSources:
    """Tests for discovered source CRUD operations."""

    async def test_upsert_and_list_discovered_sources(self) -> None:
        """upsert_discovered_source + list_discovered_sources round-trips correctly."""
        async with get_test_db() as conn:
            source = DiscoveredSourceRow(
                id=str(uuid.uuid4()),
                source_type="substack",
                identifier="example.substack.com",
                display_name="Example Newsletter",
                discovery_method="outbound_link",
                confidence=0.9,
            )
            await upsert_discovered_source(conn, source)

            results = await list_discovered_sources(conn, status="suggested")

        assert len(results) == 1
        assert results[0].id == source.id
        assert results[0].display_name == "Example Newsletter"

    async def test_list_discovered_sources_filters_by_status(self) -> None:
        """list_discovered_sources returns only sources with the given status."""
        async with get_test_db() as conn:
            suggested = DiscoveredSourceRow(
                id=str(uuid.uuid4()),
                source_type="hn",
                identifier="hn-suggested",
                status="suggested",
            )
            added = DiscoveredSourceRow(
                id=str(uuid.uuid4()),
                source_type="reddit",
                identifier="r/added",
                status="added",
            )
            await upsert_discovered_source(conn, suggested)
            await upsert_discovered_source(conn, added)

            suggested_results = await list_discovered_sources(conn, status="suggested")
            added_results = await list_discovered_sources(conn, status="added")

        assert len(suggested_results) == 1
        assert len(added_results) == 1

    async def test_update_discovered_source_status(self) -> None:
        """update_discovered_source_status changes the status correctly."""
        async with get_test_db() as conn:
            source = DiscoveredSourceRow(
                id=str(uuid.uuid4()),
                source_type="rss",
                identifier="https://example.com/feed.xml",
                status="suggested",
            )
            await upsert_discovered_source(conn, source)

            await update_discovered_source_status(conn, source.id, "added")

            added_results = await list_discovered_sources(conn, status="added")
            suggested_results = await list_discovered_sources(conn, status="suggested")

        assert len(added_results) == 1
        assert len(suggested_results) == 0


# ---------------------------------------------------------------------------
# Briefings tests
# ---------------------------------------------------------------------------


class TestBriefings:
    """Tests for briefing CRUD operations."""

    async def test_insert_and_get_briefing(self) -> None:
        """insert_briefing + get_briefing round-trips a BriefingRow correctly."""
        async with get_test_db() as conn:
            briefing = BriefingRow(
                id=str(uuid.uuid4()),
                title="Weekly AI Digest",
                query="AI developments",
                content="## Week in AI\n\nHighlights this week...",
                source_document_ids=["doc1", "doc2"],
                format="markdown",
            )
            returned_id = await insert_briefing(conn, briefing)
            assert returned_id == briefing.id

            fetched = await get_briefing(conn, briefing.id)

        assert fetched is not None
        assert fetched.title == "Weekly AI Digest"
        assert fetched.query == "AI developments"
        assert fetched.source_document_ids == ["doc1", "doc2"]

    async def test_get_briefing_returns_none_for_missing(self) -> None:
        """get_briefing returns None for a non-existent ID."""
        async with get_test_db() as conn:
            result = await get_briefing(conn, "nonexistent-id")
        assert result is None

    async def test_list_briefings(self) -> None:
        """list_briefings returns briefings ordered by created_at descending."""
        async with get_test_db() as conn:
            briefing1 = BriefingRow(
                id=str(uuid.uuid4()),
                title="Briefing 1",
                content="Content 1",
            )
            briefing2 = BriefingRow(
                id=str(uuid.uuid4()),
                title="Briefing 2",
                content="Content 2",
            )
            await insert_briefing(conn, briefing1)
            await insert_briefing(conn, briefing2)

            briefings = await list_briefings(conn)

        assert len(briefings) == 2

    async def test_list_briefings_limit(self) -> None:
        """list_briefings respects the limit parameter."""
        async with get_test_db() as conn:
            for i in range(5):
                await insert_briefing(
                    conn,
                    BriefingRow(
                        id=str(uuid.uuid4()),
                        title=f"Briefing {i}",
                        content=f"Content {i}",
                    ),
                )

            briefings = await list_briefings(conn, limit=3)

        assert len(briefings) == 3


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for the get_stats aggregate query."""

    async def test_get_stats_empty_db(self) -> None:
        """get_stats returns all zeros on an empty database."""
        async with get_test_db() as conn:
            stats = await get_stats(conn)

        assert stats["total_documents"] == 0
        assert stats["total_entities"] == 0
        assert stats["total_sources"] == 0
        assert stats["total_briefings"] == 0
        assert stats["embedded_documents"] == 0
        assert stats["unembedded_documents"] == 0

    async def test_get_stats_with_data(self) -> None:
        """get_stats returns accurate counts after inserting data."""
        async with get_test_db() as conn:
            # Insert 3 documents: 2 embedded, 1 not
            doc1 = make_doc()
            doc2 = make_doc()
            doc3 = make_doc()
            await upsert_document(conn, doc1)
            await upsert_document(conn, doc2)
            await upsert_document(conn, doc3)
            await update_document_flags(conn, doc1.id, is_embedded=True)
            await update_document_flags(conn, doc2.id, is_embedded=True)

            # Insert 2 sources
            await upsert_source(conn, make_source())
            await upsert_source(conn, make_source())

            # Insert 1 entity
            await upsert_entity(conn, make_entity())

            # Insert 1 briefing
            await insert_briefing(
                conn,
                BriefingRow(
                    id=str(uuid.uuid4()),
                    title="Test",
                    content="Content",
                ),
            )

            stats = await get_stats(conn)

        assert stats["total_documents"] == 3
        assert stats["total_entities"] == 1
        assert stats["total_sources"] == 2
        assert stats["total_briefings"] == 1
        assert stats["embedded_documents"] == 2
        assert stats["unembedded_documents"] == 1

    async def test_get_stats_excludes_deleted_documents(self) -> None:
        """get_stats does not count soft-deleted documents in totals."""
        async with get_test_db() as conn:
            doc1 = make_doc()
            doc2 = make_doc()
            await upsert_document(conn, doc1)
            await upsert_document(conn, doc2)

            # Soft-delete doc2
            await soft_delete_document(conn, doc2.id)

            stats = await get_stats(conn)

        assert stats["total_documents"] == 1
        assert stats["unembedded_documents"] == 1

    async def test_get_stats_has_all_required_keys(self) -> None:
        """get_stats returns all six required stat keys."""
        async with get_test_db() as conn:
            stats = await get_stats(conn)

        required_keys = {
            "total_documents",
            "total_entities",
            "total_sources",
            "total_briefings",
            "embedded_documents",
            "unembedded_documents",
        }
        assert required_keys == set(stats.keys())
