"""Tests for entity search, browse, co-occurrence, and dedup via EntitySearch.

All tests use an in-memory SQLite database to ensure isolation and speed.
The get_test_db() helper mirrors the one in test_db.py — it creates a fresh
connection with the full schema applied (including FTS5 triggers).
"""
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiosqlite
import pytest

from ai_craftsman_kb.db.models import DocumentRow, EntityRow
from ai_craftsman_kb.db.queries import (
    link_document_entity,
    upsert_document,
    upsert_entity,
)
from ai_craftsman_kb.db.sqlite import SCHEMA_SQL
from ai_craftsman_kb.search.entity_search import (
    CoOccurringEntity,
    EntitySearch,
    EntityWithDocs,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def get_test_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Yield a fresh in-memory aiosqlite connection with the full schema applied.

    Creates an isolated in-memory database identical to the production setup,
    including FTS5 virtual tables and all triggers.
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
        "raw_content": "This is the raw content.",
        "metadata": {},
        "user_tags": [],
    }
    defaults.update(kwargs)
    return DocumentRow(**defaults)


def make_entity(**kwargs) -> EntityRow:
    """Create an EntityRow with sensible defaults for testing."""
    name = kwargs.pop("name", "Python")
    defaults = {
        "id": str(uuid.uuid4()),
        "name": name,
        "entity_type": "technology",
        "normalized_name": name.lower(),
        "description": f"The {name} programming language",
        "metadata": {},
        "mention_count": 1,
    }
    defaults.update(kwargs)
    return EntityRow(**defaults)


# ---------------------------------------------------------------------------
# EntitySearch.search() tests
# ---------------------------------------------------------------------------


class TestEntitySearchSearch:
    """Tests for EntitySearch.search()."""

    async def test_search_returns_matching_entity(self) -> None:
        """search() returns entities whose name matches the FTS query."""
        service = EntitySearch()
        async with get_test_db() as conn:
            python_entity = make_entity(name="Python", description="A high-level language")
            rust_entity = make_entity(name="Rust", description="A systems language")
            await upsert_entity(conn, python_entity)
            await upsert_entity(conn, rust_entity)

            results = await service.search(conn, "Python")

        ids = {e.id for e in results}
        assert python_entity.id in ids
        # Rust should not appear in results for query "Python"
        assert rust_entity.id not in ids

    async def test_search_returns_empty_for_no_match(self) -> None:
        """search() returns an empty list when no entity matches the query."""
        service = EntitySearch()
        async with get_test_db() as conn:
            await upsert_entity(conn, make_entity(name="Python"))

            results = await service.search(conn, "zzznomatchzzz")

        assert results == []

    async def test_search_filters_by_entity_type(self) -> None:
        """search() respects the entity_type filter."""
        service = EntitySearch()
        async with get_test_db() as conn:
            tech_entity = make_entity(
                name="Python", entity_type="technology", normalized_name="python_tech"
            )
            person_entity = make_entity(
                name="Python Guido", entity_type="person", normalized_name="python guido"
            )
            await upsert_entity(conn, tech_entity)
            await upsert_entity(conn, person_entity)

            tech_results = await service.search(conn, "Python", entity_type="technology")
            person_results = await service.search(conn, "Python", entity_type="person")

        tech_ids = {e.id for e in tech_results}
        person_ids = {e.id for e in person_results}

        assert tech_entity.id in tech_ids
        assert tech_entity.id not in person_ids
        assert person_entity.id in person_ids
        assert person_entity.id not in tech_ids

    async def test_search_falls_back_to_list_on_empty_query(self) -> None:
        """search() with an empty query returns entities sorted by mention_count."""
        service = EntitySearch()
        async with get_test_db() as conn:
            low_entity = make_entity(name="Rust", mention_count=1)
            high_entity = make_entity(name="Python", mention_count=50)
            await upsert_entity(conn, low_entity)
            await upsert_entity(conn, high_entity)

            results = await service.search(conn, "")

        assert len(results) == 2
        # The entity with higher mention_count should come first
        assert results[0].id == high_entity.id

    async def test_search_falls_back_on_whitespace_query(self) -> None:
        """search() with a whitespace-only query falls back to list_entities."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity = make_entity(name="Python")
            await upsert_entity(conn, entity)

            results = await service.search(conn, "   ")

        assert len(results) == 1

    async def test_search_respects_limit(self) -> None:
        """search() respects the limit parameter."""
        service = EntitySearch()
        async with get_test_db() as conn:
            for i in range(5):
                await upsert_entity(
                    conn,
                    make_entity(
                        name=f"Python {i}",
                        normalized_name=f"python {i}",
                    ),
                )

            results = await service.search(conn, "Python", limit=3)

        assert len(results) <= 3

    async def test_search_fts_trigger_keeps_index_in_sync(self) -> None:
        """Inserting an entity via upsert_entity makes it findable via FTS search."""
        service = EntitySearch()
        async with get_test_db() as conn:
            # Insert after schema is already applied — FTS trigger should fire
            entity = make_entity(
                name="TensorFlow",
                normalized_name="tensorflow",
                description="Machine learning framework by Google",
            )
            await upsert_entity(conn, entity)

            results = await service.search(conn, "TensorFlow")

        ids = {e.id for e in results}
        assert entity.id in ids

    async def test_search_description_is_searchable(self) -> None:
        """search() can match entities by their description field."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity = make_entity(
                name="MyLib",
                normalized_name="mylib",
                description="A quantum computing framework",
            )
            await upsert_entity(conn, entity)

            results = await service.search(conn, "quantum")

        ids = {e.id for e in results}
        assert entity.id in ids


# ---------------------------------------------------------------------------
# EntitySearch.list_entities() tests
# ---------------------------------------------------------------------------


class TestEntitySearchListEntities:
    """Tests for EntitySearch.list_entities()."""

    async def test_list_entities_returns_all(self) -> None:
        """list_entities() returns all inserted entities."""
        service = EntitySearch()
        async with get_test_db() as conn:
            e1 = make_entity(name="Python", normalized_name="python")
            e2 = make_entity(name="Rust", normalized_name="rust")
            await upsert_entity(conn, e1)
            await upsert_entity(conn, e2)

            results = await service.list_entities(conn)

        assert len(results) == 2

    async def test_list_entities_sorted_by_mention_count(self) -> None:
        """list_entities() sorts by mention_count DESC by default."""
        service = EntitySearch()
        async with get_test_db() as conn:
            low = make_entity(name="Rare", normalized_name="rare", mention_count=2)
            high = make_entity(name="Popular", normalized_name="popular", mention_count=100)
            await upsert_entity(conn, low)
            await upsert_entity(conn, high)

            results = await service.list_entities(conn, sort_by="mention_count")

        assert results[0].id == high.id
        assert results[1].id == low.id

    async def test_list_entities_sorted_by_name_ascending(self) -> None:
        """list_entities() sorted by name is alphabetical ascending."""
        service = EntitySearch()
        async with get_test_db() as conn:
            z_entity = make_entity(name="Zebra", normalized_name="zebra")
            a_entity = make_entity(name="Apple", normalized_name="apple")
            await upsert_entity(conn, z_entity)
            await upsert_entity(conn, a_entity)

            results = await service.list_entities(conn, sort_by="name")

        names = [e.name for e in results]
        assert names == sorted(names)

    async def test_list_entities_filter_by_type(self) -> None:
        """list_entities() filters correctly by entity_type."""
        service = EntitySearch()
        async with get_test_db() as conn:
            tech = make_entity(name="Python", entity_type="technology", normalized_name="python_t")
            person = make_entity(
                name="Sam Altman", entity_type="person", normalized_name="sam altman"
            )
            await upsert_entity(conn, tech)
            await upsert_entity(conn, person)

            tech_results = await service.list_entities(conn, entity_type="technology")
            person_results = await service.list_entities(conn, entity_type="person")

        assert len(tech_results) == 1
        assert tech_results[0].id == tech.id
        assert len(person_results) == 1
        assert person_results[0].id == person.id

    async def test_list_entities_pagination(self) -> None:
        """list_entities() respects limit and offset for pagination."""
        service = EntitySearch()
        async with get_test_db() as conn:
            for i in range(5):
                await upsert_entity(
                    conn,
                    make_entity(name=f"Entity {i}", normalized_name=f"entity {i}"),
                )

            page1 = await service.list_entities(conn, limit=3, offset=0)
            page2 = await service.list_entities(conn, limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 2

    async def test_list_entities_empty_db(self) -> None:
        """list_entities() returns empty list on an empty database."""
        service = EntitySearch()
        async with get_test_db() as conn:
            results = await service.list_entities(conn)

        assert results == []


# ---------------------------------------------------------------------------
# EntitySearch.get_entity_with_docs() tests
# ---------------------------------------------------------------------------


class TestEntitySearchGetEntityWithDocs:
    """Tests for EntitySearch.get_entity_with_docs()."""

    async def test_returns_none_for_missing_entity(self) -> None:
        """get_entity_with_docs() returns None for a non-existent entity_id."""
        service = EntitySearch()
        async with get_test_db() as conn:
            result = await service.get_entity_with_docs(conn, "nonexistent-id")

        assert result is None

    async def test_returns_entity_with_linked_docs(self) -> None:
        """get_entity_with_docs() returns EntityWithDocs with correct top_documents."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity = make_entity(name="Python")
            doc1 = make_doc(title="Python Tutorial")
            doc2 = make_doc(title="Python Tips")
            unrelated_doc = make_doc(title="Rust Guide")

            await upsert_entity(conn, entity)
            await upsert_document(conn, doc1)
            await upsert_document(conn, doc2)
            await upsert_document(conn, unrelated_doc)

            await link_document_entity(conn, doc1.id, entity.id)
            await link_document_entity(conn, doc2.id, entity.id)
            # unrelated_doc is NOT linked to entity

            result = await service.get_entity_with_docs(conn, entity.id)

        assert result is not None
        assert isinstance(result, EntityWithDocs)
        assert result.entity.id == entity.id
        assert result.document_count == 2
        doc_ids = {d.id for d in result.top_documents}
        assert doc1.id in doc_ids
        assert doc2.id in doc_ids
        assert unrelated_doc.id not in doc_ids

    async def test_document_count_excludes_soft_deleted(self) -> None:
        """get_entity_with_docs() excludes soft-deleted documents from count."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity = make_entity(name="Python")
            doc1 = make_doc(title="Live Doc")
            doc2 = make_doc(title="Deleted Doc")

            await upsert_entity(conn, entity)
            await upsert_document(conn, doc1)
            await upsert_document(conn, doc2)
            await link_document_entity(conn, doc1.id, entity.id)
            await link_document_entity(conn, doc2.id, entity.id)

            # Soft-delete doc2
            await conn.execute(
                "UPDATE documents SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?",
                (doc2.id,),
            )
            await conn.commit()

            result = await service.get_entity_with_docs(conn, entity.id)

        assert result is not None
        assert result.document_count == 1
        assert len(result.top_documents) == 1
        assert result.top_documents[0].id == doc1.id

    async def test_top_documents_respect_limit(self) -> None:
        """get_entity_with_docs() respects the limit parameter for top_documents."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity = make_entity(name="Python")
            await upsert_entity(conn, entity)

            for i in range(10):
                doc = make_doc(title=f"Doc {i}")
                await upsert_document(conn, doc)
                await link_document_entity(conn, doc.id, entity.id)

            result = await service.get_entity_with_docs(conn, entity.id, limit=5)

        assert result is not None
        assert result.document_count == 10
        assert len(result.top_documents) <= 5

    async def test_returns_entity_with_no_docs(self) -> None:
        """get_entity_with_docs() works for an entity with no linked documents."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity = make_entity(name="Lonely Entity")
            await upsert_entity(conn, entity)

            result = await service.get_entity_with_docs(conn, entity.id)

        assert result is not None
        assert result.document_count == 0
        assert result.top_documents == []


# ---------------------------------------------------------------------------
# EntitySearch.get_co_occurring_entities() tests
# ---------------------------------------------------------------------------


class TestEntitySearchCoOccurrence:
    """Tests for EntitySearch.get_co_occurring_entities()."""

    async def test_returns_co_occurring_entities(self) -> None:
        """get_co_occurring_entities() returns entities in the same docs."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity_a = make_entity(name="Python", normalized_name="python_a")
            entity_b = make_entity(name="Machine Learning", normalized_name="machine learning")
            entity_c = make_entity(name="Rust", normalized_name="rust_a")

            await upsert_entity(conn, entity_a)
            await upsert_entity(conn, entity_b)
            await upsert_entity(conn, entity_c)

            doc1 = make_doc(title="Python ML Article")
            doc2 = make_doc(title="Python ML Advanced")
            doc3 = make_doc(title="Rust Article")

            await upsert_document(conn, doc1)
            await upsert_document(conn, doc2)
            await upsert_document(conn, doc3)

            # doc1 and doc2: Python + ML co-occur
            await link_document_entity(conn, doc1.id, entity_a.id)
            await link_document_entity(conn, doc1.id, entity_b.id)
            await link_document_entity(conn, doc2.id, entity_a.id)
            await link_document_entity(conn, doc2.id, entity_b.id)
            # doc3: only Rust (no co-occurrence with Python)
            await link_document_entity(conn, doc3.id, entity_c.id)

            results = await service.get_co_occurring_entities(conn, entity_a.id)

        assert len(results) >= 1
        result_ids = {r.id for r in results}
        # ML co-occurs with Python in 2 docs
        assert entity_b.id in result_ids
        # Rust never appears in a doc with Python
        assert entity_c.id not in result_ids

    async def test_co_occurrence_count_is_correct(self) -> None:
        """get_co_occurring_entities() returns the correct co_count."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity_a = make_entity(name="Python", normalized_name="python_b")
            entity_b = make_entity(name="Django", normalized_name="django")

            await upsert_entity(conn, entity_a)
            await upsert_entity(conn, entity_b)

            # Link both entities to 3 separate documents
            for _ in range(3):
                doc = make_doc()
                await upsert_document(conn, doc)
                await link_document_entity(conn, doc.id, entity_a.id)
                await link_document_entity(conn, doc.id, entity_b.id)

            results = await service.get_co_occurring_entities(conn, entity_a.id)

        assert len(results) == 1
        assert results[0].id == entity_b.id
        assert results[0].co_count == 3

    async def test_returns_empty_for_isolated_entity(self) -> None:
        """get_co_occurring_entities() returns [] if entity never co-occurs."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity = make_entity(name="Isolated")
            await upsert_entity(conn, entity)
            doc = make_doc()
            await upsert_document(conn, doc)
            await link_document_entity(conn, doc.id, entity.id)

            results = await service.get_co_occurring_entities(conn, entity.id)

        assert results == []

    async def test_respects_limit(self) -> None:
        """get_co_occurring_entities() respects the limit parameter."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity_a = make_entity(name="Hub", normalized_name="hub")
            await upsert_entity(conn, entity_a)

            # Create 15 other entities each co-occurring with entity_a
            for i in range(15):
                other = make_entity(name=f"Other {i}", normalized_name=f"other {i}")
                await upsert_entity(conn, other)
                doc = make_doc()
                await upsert_document(conn, doc)
                await link_document_entity(conn, doc.id, entity_a.id)
                await link_document_entity(conn, doc.id, other.id)

            results = await service.get_co_occurring_entities(conn, entity_a.id, limit=10)

        assert len(results) <= 10

    async def test_co_occurring_entity_has_correct_fields(self) -> None:
        """get_co_occurring_entities() returns CoOccurringEntity with all fields."""
        service = EntitySearch()
        async with get_test_db() as conn:
            entity_a = make_entity(name="Python", normalized_name="python_c")
            entity_b = make_entity(name="NumPy", entity_type="technology", normalized_name="numpy")
            await upsert_entity(conn, entity_a)
            await upsert_entity(conn, entity_b)
            doc = make_doc()
            await upsert_document(conn, doc)
            await link_document_entity(conn, doc.id, entity_a.id)
            await link_document_entity(conn, doc.id, entity_b.id)

            results = await service.get_co_occurring_entities(conn, entity_a.id)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, CoOccurringEntity)
        assert result.id == entity_b.id
        assert result.name == "NumPy"
        assert result.entity_type == "technology"
        assert result.co_count == 1


# ---------------------------------------------------------------------------
# EntitySearch.merge_entities() tests
# ---------------------------------------------------------------------------


class TestEntitySearchMergeEntities:
    """Tests for EntitySearch.merge_entities()."""

    async def test_merge_reassigns_document_links(self) -> None:
        """merge_entities() moves document_entities links to the canonical."""
        service = EntitySearch()
        async with get_test_db() as conn:
            canonical = make_entity(name="OpenAI", normalized_name="openai_canonical")
            duplicate = make_entity(name="Open AI", normalized_name="open ai")
            await upsert_entity(conn, canonical)
            await upsert_entity(conn, duplicate)

            doc = make_doc(title="OpenAI GPT Article")
            await upsert_document(conn, doc)
            await link_document_entity(conn, doc.id, duplicate.id)

            await service.merge_entities(conn, canonical.id, [duplicate.id])

            # The link should now point to canonical
            async with conn.execute(
                "SELECT entity_id FROM document_entities WHERE document_id = ?",
                (doc.id,),
            ) as cursor:
                rows = await cursor.fetchall()
            entity_ids = {row[0] for row in rows}

        assert canonical.id in entity_ids
        assert duplicate.id not in entity_ids

    async def test_merge_deletes_duplicate_entities(self) -> None:
        """merge_entities() removes duplicate entity rows from the DB."""
        service = EntitySearch()
        async with get_test_db() as conn:
            canonical = make_entity(name="GPT-4", normalized_name="gpt-4 canonical")
            dup1 = make_entity(name="GPT4", normalized_name="gpt4 dup1")
            dup2 = make_entity(name="gpt 4", normalized_name="gpt 4 dup2")
            await upsert_entity(conn, canonical)
            await upsert_entity(conn, dup1)
            await upsert_entity(conn, dup2)

            await service.merge_entities(conn, canonical.id, [dup1.id, dup2.id])

            async with conn.execute("SELECT id FROM entities") as cursor:
                rows = await cursor.fetchall()
            remaining_ids = {row[0] for row in rows}

        assert canonical.id in remaining_ids
        assert dup1.id not in remaining_ids
        assert dup2.id not in remaining_ids

    async def test_merge_sums_mention_counts(self) -> None:
        """merge_entities() adds duplicate mention_counts to canonical."""
        service = EntitySearch()
        async with get_test_db() as conn:
            canonical = make_entity(
                name="ML", normalized_name="ml canonical", mention_count=5
            )
            dup1 = make_entity(name="ml", normalized_name="ml dup1", mention_count=3)
            dup2 = make_entity(name="Machine Learning", normalized_name="ml dup2", mention_count=7)
            await upsert_entity(conn, canonical)
            await upsert_entity(conn, dup1)
            await upsert_entity(conn, dup2)

            await service.merge_entities(conn, canonical.id, [dup1.id, dup2.id])

            async with conn.execute(
                "SELECT mention_count FROM entities WHERE id = ?",
                (canonical.id,),
            ) as cursor:
                row = await cursor.fetchone()

        assert row is not None
        # 5 + 3 + 7 = 15
        assert row[0] == 15

    async def test_merge_handles_overlapping_doc_links(self) -> None:
        """merge_entities() handles docs already linked to both canonical and dup."""
        service = EntitySearch()
        async with get_test_db() as conn:
            canonical = make_entity(name="Python", normalized_name="python merge1")
            duplicate = make_entity(name="python", normalized_name="python merge2")
            await upsert_entity(conn, canonical)
            await upsert_entity(conn, duplicate)

            doc = make_doc(title="Shared Article")
            await upsert_document(conn, doc)
            # Both entities are linked to the same document
            await link_document_entity(conn, doc.id, canonical.id)
            await link_document_entity(conn, doc.id, duplicate.id)

            # Should not raise any error (INSERT OR IGNORE handles PK conflict)
            await service.merge_entities(conn, canonical.id, [duplicate.id])

            async with conn.execute(
                "SELECT entity_id FROM document_entities WHERE document_id = ?",
                (doc.id,),
            ) as cursor:
                rows = await cursor.fetchall()
            entity_ids = {row[0] for row in rows}

        # Only the canonical link should remain
        assert canonical.id in entity_ids
        assert duplicate.id not in entity_ids

    async def test_merge_raises_for_unknown_canonical(self) -> None:
        """merge_entities() raises ValueError if canonical_id is not in DB."""
        service = EntitySearch()
        async with get_test_db() as conn:
            dup = make_entity(name="Dup")
            await upsert_entity(conn, dup)

            with pytest.raises(ValueError, match="Canonical entity not found"):
                await service.merge_entities(conn, "nonexistent-canonical", [dup.id])

    async def test_merge_no_op_on_empty_duplicate_list(self) -> None:
        """merge_entities() with empty duplicate_ids is a no-op and does not fail."""
        service = EntitySearch()
        async with get_test_db() as conn:
            canonical = make_entity(name="Python", normalized_name="python noop")
            await upsert_entity(conn, canonical)

            # Should complete without error
            await service.merge_entities(conn, canonical.id, [])

            async with conn.execute("SELECT COUNT(*) FROM entities") as cursor:
                row = await cursor.fetchone()

        assert row[0] == 1

    async def test_merge_updates_fts_index(self) -> None:
        """merge_entities() ensures deleted duplicates are removed from FTS index."""
        service = EntitySearch()
        async with get_test_db() as conn:
            canonical = make_entity(
                name="TensorFlow", normalized_name="tensorflow canonical"
            )
            duplicate = make_entity(
                name="Tensorflow", normalized_name="tensorflow dup"
            )
            await upsert_entity(conn, canonical)
            await upsert_entity(conn, duplicate)

            await service.merge_entities(conn, canonical.id, [duplicate.id])

            # FTS search should find canonical but return exactly 1 result
            results = await service.search(conn, "TensorFlow")

        assert len(results) == 1
        assert results[0].id == canonical.id

    async def test_merge_multiple_duplicates(self) -> None:
        """merge_entities() handles multiple duplicates in a single call."""
        service = EntitySearch()
        async with get_test_db() as conn:
            canonical = make_entity(name="AI", normalized_name="ai canonical", mention_count=10)
            dups = [
                make_entity(name=f"AI{i}", normalized_name=f"ai dup {i}", mention_count=i)
                for i in range(1, 6)
            ]
            await upsert_entity(conn, canonical)
            for dup in dups:
                await upsert_entity(conn, dup)

            dup_ids = [d.id for d in dups]
            await service.merge_entities(conn, canonical.id, dup_ids)

            async with conn.execute(
                "SELECT mention_count FROM entities WHERE id = ?",
                (canonical.id,),
            ) as cursor:
                row = await cursor.fetchone()

            async with conn.execute("SELECT COUNT(*) FROM entities") as cursor:
                count_row = await cursor.fetchone()

        # canonical (10) + 1 + 2 + 3 + 4 + 5 = 25
        assert row[0] == 25
        # Only the canonical should remain
        assert count_row[0] == 1
