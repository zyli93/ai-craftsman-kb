"""Unit tests for EntityExtractor — LLM-based entity extraction pipeline.

All LLM calls are mocked via AsyncMock on LLMRouter.complete().
DB operations use an in-memory aiosqlite connection with the full schema.
No network access is required.
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    FiltersConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    SettingsConfig,
    SourcesConfig,
)
from ai_craftsman_kb.db.sqlite import SCHEMA_SQL
from ai_craftsman_kb.llm import CompletionResult
from ai_craftsman_kb.llm.router import LLMRouter
from ai_craftsman_kb.processing.entity_extractor import (
    VALID_ENTITY_TYPES,
    EntityExtractor,
    ExtractedEntity,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_app_config() -> AppConfig:
    """Build a minimal AppConfig for EntityExtractor tests."""
    return AppConfig(
        sources=SourcesConfig(),
        settings=SettingsConfig(
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test-model"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test-model"),
                briefing=LLMTaskConfig(provider="anthropic", model="test-model"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test-model"),
                keyword_extraction=LLMTaskConfig(provider="openrouter", model="test-model"),
            )
        ),
        filters=FiltersConfig(),
    )


def _make_llm_router(complete_return: str = "[]") -> LLMRouter:
    """Build a mock LLMRouter whose complete() returns a fixed string."""
    router = MagicMock(spec=LLMRouter)
    router.complete = AsyncMock(return_value=CompletionResult(text=complete_return))
    return router


def _make_extractor(llm_return: str = "[]") -> tuple[EntityExtractor, LLMRouter]:
    """Create an EntityExtractor and its mock LLMRouter together."""
    config = _make_app_config()
    router = _make_llm_router(llm_return)
    extractor = EntityExtractor(config=config, llm_router=router)
    return extractor, router


async def _make_in_memory_db() -> aiosqlite.Connection:
    """Create a fully-initialised in-memory SQLite DB with the full schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.executescript(SCHEMA_SQL)
    await conn.commit()
    return conn


def _make_doc_id() -> str:
    """Return a fresh UUID string for use as a document ID."""
    return str(uuid.uuid4())


async def _insert_test_document(conn: aiosqlite.Connection, doc_id: str) -> None:
    """Insert a minimal document row so foreign-key constraints are satisfied."""
    await conn.execute(
        """
        INSERT INTO documents (id, origin, source_type, url)
        VALUES (?, 'pro', 'hn', ?)
        """,
        (doc_id, f"https://example.com/{doc_id}"),
    )
    await conn.commit()


# ---------------------------------------------------------------------------
# Test: ExtractedEntity model
# ---------------------------------------------------------------------------


def test_extracted_entity_model() -> None:
    """ExtractedEntity correctly stores name, type, and context."""
    entity = ExtractedEntity(name="OpenAI", type="company", context="OpenAI released GPT-4")
    assert entity.name == "OpenAI"
    assert entity.type == "company"
    assert entity.context == "OpenAI released GPT-4"


def test_valid_entity_types_set() -> None:
    """VALID_ENTITY_TYPES contains exactly the 7 expected types."""
    expected = {"person", "company", "technology", "event", "book", "paper", "product"}
    assert VALID_ENTITY_TYPES == expected


# ---------------------------------------------------------------------------
# Test: _normalize_name
# ---------------------------------------------------------------------------


def test_normalize_name_lowercase() -> None:
    """_normalize_name converts to lowercase."""
    extractor, _ = _make_extractor()
    assert extractor._normalize_name("OpenAI") == "openai"


def test_normalize_name_strip_whitespace() -> None:
    """_normalize_name strips leading and trailing whitespace."""
    extractor, _ = _make_extractor()
    assert extractor._normalize_name("  Meta  ") == "meta"


def test_normalize_name_combined() -> None:
    """_normalize_name strips whitespace and lowercases simultaneously."""
    extractor, _ = _make_extractor()
    assert extractor._normalize_name("  GPT-4  ") == "gpt-4"


def test_normalize_name_already_normalized() -> None:
    """_normalize_name is idempotent on already-normalized strings."""
    extractor, _ = _make_extractor()
    assert extractor._normalize_name("pytorch") == "pytorch"


# ---------------------------------------------------------------------------
# Test: _parse_llm_response — valid JSON
# ---------------------------------------------------------------------------


def test_parse_llm_response_valid_array() -> None:
    """Valid JSON array with all 3 required fields is parsed correctly."""
    extractor, _ = _make_extractor()
    response = json.dumps([
        {"name": "OpenAI", "type": "company", "context": "OpenAI released GPT-4"},
        {"name": "Andrej Karpathy", "type": "person", "context": "Karpathy left OpenAI"},
        {"name": "PyTorch", "type": "technology", "context": "built with PyTorch"},
    ])
    entities = extractor._parse_llm_response(response)
    assert len(entities) == 3
    assert entities[0].name == "OpenAI"
    assert entities[0].type == "company"
    assert entities[1].name == "Andrej Karpathy"
    assert entities[1].type == "person"
    assert entities[2].name == "PyTorch"
    assert entities[2].type == "technology"


def test_parse_llm_response_all_seven_types() -> None:
    """All 7 valid entity types are parsed without error."""
    extractor, _ = _make_extractor()
    items = [
        {"name": "Alice", "type": "person", "context": "Alice spoke"},
        {"name": "Acme Corp", "type": "company", "context": "Acme announced"},
        {"name": "TensorFlow", "type": "technology", "context": "used TensorFlow"},
        {"name": "NeurIPS 2024", "type": "event", "context": "at NeurIPS"},
        {"name": "SICP", "type": "book", "context": "recommended SICP"},
        {"name": "Attention Is All You Need", "type": "paper", "context": "citing the paper"},
        {"name": "ChatGPT", "type": "product", "context": "using ChatGPT"},
    ]
    entities = extractor._parse_llm_response(json.dumps(items))
    assert len(entities) == 7
    types = {e.type for e in entities}
    assert types == VALID_ENTITY_TYPES


def test_parse_llm_response_empty_array() -> None:
    """Empty JSON array returns empty list without error."""
    extractor, _ = _make_extractor()
    entities = extractor._parse_llm_response("[]")
    assert entities == []


# ---------------------------------------------------------------------------
# Test: _parse_llm_response — markdown code fences
# ---------------------------------------------------------------------------


def test_parse_llm_response_strips_json_code_fence() -> None:
    """Markdown ```json code fence is stripped before parsing."""
    extractor, _ = _make_extractor()
    response = '```json\n[{"name": "Meta", "type": "company", "context": "Meta AI"}]\n```'
    entities = extractor._parse_llm_response(response)
    assert len(entities) == 1
    assert entities[0].name == "Meta"


def test_parse_llm_response_strips_plain_code_fence() -> None:
    """Markdown ``` (without json language tag) is also stripped."""
    extractor, _ = _make_extractor()
    response = '```\n[{"name": "Rust", "type": "technology", "context": "written in Rust"}]\n```'
    entities = extractor._parse_llm_response(response)
    assert len(entities) == 1
    assert entities[0].name == "Rust"


# ---------------------------------------------------------------------------
# Test: _parse_llm_response — invalid entity types discarded
# ---------------------------------------------------------------------------


def test_parse_llm_response_invalid_type_discarded() -> None:
    """Entities with invalid types are discarded with a warning."""
    extractor, _ = _make_extractor()
    response = json.dumps([
        {"name": "Python", "type": "programming_language", "context": "written in Python"},
        {"name": "OpenAI", "type": "company", "context": "OpenAI model"},
    ])
    entities = extractor._parse_llm_response(response)
    # "programming_language" is not a valid type — only "company" survives
    assert len(entities) == 1
    assert entities[0].name == "OpenAI"


def test_parse_llm_response_all_invalid_types_returns_empty() -> None:
    """All-invalid entity types result in empty list."""
    extractor, _ = _make_extractor()
    response = json.dumps([
        {"name": "Bad Entity", "type": "invalid_type", "context": "..."},
        {"name": "Another", "type": "concept", "context": "..."},
    ])
    entities = extractor._parse_llm_response(response)
    assert entities == []


# ---------------------------------------------------------------------------
# Test: _parse_llm_response — JSON errors handled gracefully
# ---------------------------------------------------------------------------


def test_parse_llm_response_invalid_json_returns_empty() -> None:
    """Malformed JSON returns empty list, does not raise."""
    extractor, _ = _make_extractor()
    entities = extractor._parse_llm_response("not valid json at all")
    assert entities == []


def test_parse_llm_response_non_array_json_returns_empty() -> None:
    """JSON object (not array) returns empty list with warning."""
    extractor, _ = _make_extractor()
    response = json.dumps({"name": "OpenAI", "type": "company", "context": "..."})
    entities = extractor._parse_llm_response(response)
    assert entities == []


def test_parse_llm_response_skips_non_dict_entries() -> None:
    """Non-dict entries within the array are skipped gracefully."""
    extractor, _ = _make_extractor()
    response = json.dumps([
        "this is a string",
        {"name": "OpenAI", "type": "company", "context": "valid entry"},
        42,
    ])
    entities = extractor._parse_llm_response(response)
    assert len(entities) == 1
    assert entities[0].name == "OpenAI"


def test_parse_llm_response_skips_empty_name() -> None:
    """Entries with empty name strings are skipped."""
    extractor, _ = _make_extractor()
    response = json.dumps([
        {"name": "", "type": "company", "context": "no name"},
        {"name": "  ", "type": "company", "context": "whitespace name"},
        {"name": "Real Entity", "type": "technology", "context": "valid"},
    ])
    entities = extractor._parse_llm_response(response)
    assert len(entities) == 1
    assert entities[0].name == "Real Entity"


def test_parse_llm_response_missing_context_defaults_to_empty() -> None:
    """Entries missing the context field use an empty string."""
    extractor, _ = _make_extractor()
    response = json.dumps([
        {"name": "OpenAI", "type": "company"},  # no context key
    ])
    entities = extractor._parse_llm_response(response)
    assert len(entities) == 1
    assert entities[0].context == ""


# ---------------------------------------------------------------------------
# Test: _truncate_to_tokens
# ---------------------------------------------------------------------------


def test_truncate_to_tokens_short_text_unchanged() -> None:
    """Short text well under the token limit is returned unchanged."""
    text = "This is a short article about machine learning."
    result = EntityExtractor._truncate_to_tokens(text, 4000)
    assert result == text


def test_truncate_to_tokens_truncates_long_text() -> None:
    """Very long text is truncated so the result decodes to fewer tokens."""
    # A long repeating word sequence — each word is ~1 token
    long_text = "transformer " * 5000  # ~5000 tokens
    result = EntityExtractor._truncate_to_tokens(long_text, 4000)
    # Result should be shorter than the original
    assert len(result) < len(long_text)
    # And the result should contain the word "transformer"
    assert "transformer" in result


def test_truncate_to_tokens_exactly_at_limit() -> None:
    """Text with exactly max_tokens tokens is returned as-is."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    # Build text that is exactly 100 tokens
    tokens = enc.encode("hello world " * 50)[:100]
    exact_text = enc.decode(tokens)
    result = EntityExtractor._truncate_to_tokens(exact_text, 100)
    # Should not be truncated further
    result_tokens = enc.encode(result)
    assert len(result_tokens) <= 100


# ---------------------------------------------------------------------------
# Test: extract() — integration with mocked LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_returns_entities_from_llm() -> None:
    """extract() calls LLM and returns parsed entities for a 500-word article."""
    llm_response = json.dumps([
        {"name": "OpenAI", "type": "company", "context": "OpenAI announced GPT-4"},
        {"name": "Sam Altman", "type": "person", "context": "CEO Sam Altman"},
        {"name": "GPT-4", "type": "technology", "context": "the GPT-4 model"},
    ])
    extractor, router = _make_extractor(llm_return=llm_response)

    # A 500-word article about AI
    content = (
        "OpenAI, led by CEO Sam Altman, has released GPT-4. "
        "The model demonstrates remarkable capabilities. " * 30
    )

    entities = await extractor.extract(content)

    assert len(entities) == 3
    assert entities[0].name == "OpenAI"
    assert entities[0].type == "company"
    assert entities[1].name == "Sam Altman"
    assert entities[2].name == "GPT-4"
    router.complete.assert_called_once()
    # Verify the task argument
    call_kwargs = router.complete.call_args.kwargs
    assert call_kwargs.get("task") == "entity_extraction"


@pytest.mark.asyncio
async def test_extract_returns_empty_on_llm_error() -> None:
    """extract() returns [] when the LLM call raises an exception."""
    config = _make_app_config()
    router = MagicMock(spec=LLMRouter)
    router.complete = AsyncMock(side_effect=RuntimeError("API timeout"))
    extractor = EntityExtractor(config=config, llm_router=router)

    entities = await extractor.extract("Some article content about AI.")

    assert entities == []


@pytest.mark.asyncio
async def test_extract_returns_empty_on_parse_error() -> None:
    """extract() returns [] when LLM response is malformed JSON."""
    extractor, _ = _make_extractor(llm_return="Sorry, I cannot extract entities.")

    entities = await extractor.extract("Some content.")

    assert entities == []


@pytest.mark.asyncio
async def test_extract_content_truncated_before_llm_call() -> None:
    """Content is truncated to 4000 tokens before being sent to the LLM."""
    extractor, router = _make_extractor(llm_return="[]")

    # 20,000 word text — far beyond 4000 tokens
    very_long_content = "artificial intelligence " * 20000

    await extractor.extract(very_long_content)

    router.complete.assert_called_once()
    call_kwargs = router.complete.call_args.kwargs
    prompt_used = call_kwargs["prompt"]
    # The prompt includes the template + content; the content should be truncated
    # so the total prompt won't contain all 20,000 repetitions of "artificial intelligence"
    assert prompt_used.count("artificial intelligence") < 20000


# ---------------------------------------------------------------------------
# Test: extract_and_store() — DB integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_and_store_inserts_entities() -> None:
    """extract_and_store() writes entities to the entities table."""
    llm_response = json.dumps([
        {"name": "OpenAI", "type": "company", "context": "OpenAI announced"},
        {"name": "PyTorch", "type": "technology", "context": "using PyTorch"},
    ])
    extractor, _ = _make_extractor(llm_return=llm_response)

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        entities = await extractor.extract_and_store(conn, doc_id, "Article about OpenAI PyTorch.")

        assert len(entities) == 2

        # Verify entities table
        async with conn.execute("SELECT name, entity_type, normalized_name FROM entities") as cur:
            rows = await cur.fetchall()
        names = {row["name"] for row in rows}
        assert "OpenAI" in names
        assert "PyTorch" in names

        # Verify normalized_name dedup keys
        normalized = {row["normalized_name"] for row in rows}
        assert "openai" in normalized
        assert "pytorch" in normalized
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_extract_and_store_links_to_document() -> None:
    """extract_and_store() creates document_entities rows linking doc to entities."""
    llm_response = json.dumps([
        {"name": "Meta", "type": "company", "context": "Meta AI research"},
    ])
    extractor, _ = _make_extractor(llm_return=llm_response)

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        await extractor.extract_and_store(conn, doc_id, "Article about Meta AI research.")

        async with conn.execute(
            "SELECT de.document_id, de.entity_id, de.context FROM document_entities de"
        ) as cur:
            rows = await cur.fetchall()

        assert len(rows) == 1
        assert rows[0]["document_id"] == doc_id
        assert rows[0]["context"] == "Meta AI research"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_extract_and_store_marks_document_extracted() -> None:
    """extract_and_store() sets is_entities_extracted=True on the document."""
    extractor, _ = _make_extractor(llm_return="[]")

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        await extractor.extract_and_store(conn, doc_id, "Content with no entities.")

        async with conn.execute(
            "SELECT is_entities_extracted FROM documents WHERE id = ?", (doc_id,)
        ) as cur:
            row = await cur.fetchone()

        assert row is not None
        assert bool(row["is_entities_extracted"]) is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_extract_and_store_marks_document_even_on_empty_extraction() -> None:
    """Document is marked as processed even when no entities are found."""
    extractor, _ = _make_extractor(llm_return="[]")

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        result = await extractor.extract_and_store(conn, doc_id, "Some generic content.")

        assert result == []  # no entities
        # But document should still be marked
        async with conn.execute(
            "SELECT is_entities_extracted FROM documents WHERE id = ?", (doc_id,)
        ) as cur:
            row = await cur.fetchone()
        assert bool(row["is_entities_extracted"]) is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_extract_and_store_deduplicates_entities() -> None:
    """Extracting the same entity from two documents increments mention_count."""
    llm_response = json.dumps([
        {"name": "OpenAI", "type": "company", "context": "OpenAI GPT-4"},
    ])
    extractor, _ = _make_extractor(llm_return=llm_response)

    conn = await _make_in_memory_db()
    try:
        doc_id_1 = _make_doc_id()
        doc_id_2 = _make_doc_id()
        await _insert_test_document(conn, doc_id_1)
        await _insert_test_document(conn, doc_id_2)

        # Extract same entity from two different documents
        await extractor.extract_and_store(conn, doc_id_1, "Article 1 about OpenAI.")
        await extractor.extract_and_store(conn, doc_id_2, "Article 2 about OpenAI.")

        # There should be only one entity in the entities table
        async with conn.execute("SELECT COUNT(*) as cnt FROM entities") as cur:
            row = await cur.fetchone()
        assert row["cnt"] == 1

        # mention_count should be 2
        async with conn.execute("SELECT mention_count FROM entities WHERE normalized_name='openai'") as cur:
            row = await cur.fetchone()
        assert row["mention_count"] == 2

        # Both documents should be linked
        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM document_entities"
        ) as cur:
            row = await cur.fetchone()
        assert row["cnt"] == 2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_extract_and_store_context_truncated_to_100_chars() -> None:
    """context stored in document_entities is capped at 100 characters."""
    long_context = "x" * 200
    llm_response = json.dumps([
        {"name": "OpenAI", "type": "company", "context": long_context},
    ])
    extractor, _ = _make_extractor(llm_return=llm_response)

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        await extractor.extract_and_store(conn, doc_id, "Article content.")

        async with conn.execute("SELECT context FROM document_entities") as cur:
            row = await cur.fetchone()

        assert len(row["context"]) == 100
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_extract_and_store_returns_entities() -> None:
    """extract_and_store() returns the list of extracted entities."""
    llm_response = json.dumps([
        {"name": "Anthropic", "type": "company", "context": "Anthropic Claude"},
        {"name": "Claude", "type": "product", "context": "Claude model"},
    ])
    extractor, _ = _make_extractor(llm_return=llm_response)

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        result = await extractor.extract_and_store(conn, doc_id, "Article about Anthropic Claude.")

        assert len(result) == 2
        names = {e.name for e in result}
        assert "Anthropic" in names
        assert "Claude" in names
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Test: _load_prompt_template
# ---------------------------------------------------------------------------


def test_load_prompt_template_contains_content_placeholder() -> None:
    """The loaded prompt template contains the {content} placeholder."""
    extractor, _ = _make_extractor()
    assert "{content}" in extractor._prompt_template


def test_load_prompt_template_fallback_on_missing_file() -> None:
    """Falls back to inline prompt when the file cannot be found."""
    config = _make_app_config()
    router = _make_llm_router()

    # Patch _PROMPT_PATH to a non-existent path
    with patch(
        "ai_craftsman_kb.processing.entity_extractor._PROMPT_PATH",
        new=_make_bad_path(),
    ):
        extractor = EntityExtractor(config=config, llm_router=router)

    # Fallback template should still have the placeholder
    assert "{content}" in extractor._prompt_template


def _make_bad_path() -> "Path":
    """Return a Path that definitely does not exist."""
    from pathlib import Path
    return Path("/nonexistent/path/that/cannot/exist/entity_extraction.md")


# ---------------------------------------------------------------------------
# Test: extract() uses task='entity_extraction'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_uses_correct_llm_task() -> None:
    """extract() calls LLMRouter.complete with task='entity_extraction'."""
    extractor, router = _make_extractor(llm_return="[]")

    await extractor.extract("Some content about AI companies.")

    router.complete.assert_called_once()
    call_kwargs = router.complete.call_args.kwargs
    assert call_kwargs["task"] == "entity_extraction"


# ---------------------------------------------------------------------------
# Test: extract() filters invalid types and logs warning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_filters_invalid_entity_types() -> None:
    """extract() discards entities with invalid types."""
    llm_response = json.dumps([
        {"name": "Valid Entity", "type": "technology", "context": "using it"},
        {"name": "Bad Entity", "type": "unknown_type", "context": "bad"},
        {"name": "Another Bad", "type": "organization", "context": "also bad"},
    ])
    extractor, _ = _make_extractor(llm_return=llm_response)

    entities = await extractor.extract("Content with mixed entity types.")

    # Only the technology entity should survive
    assert len(entities) == 1
    assert entities[0].name == "Valid Entity"
    assert entities[0].type == "technology"
