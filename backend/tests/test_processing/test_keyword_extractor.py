"""Unit tests for KeywordExtractor -- LLM-based keyword extraction pipeline.

All LLM calls are mocked via AsyncMock on LLMRouter.complete().
DB operations use an in-memory aiosqlite connection with the full schema.
No network access is required.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
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
from ai_craftsman_kb.processing.keyword_extractor import KeywordExtractor


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_app_config() -> AppConfig:
    """Build a minimal AppConfig for KeywordExtractor tests."""
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


def _make_extractor(llm_return: str = "[]") -> tuple[KeywordExtractor, LLMRouter]:
    """Create a KeywordExtractor and its mock LLMRouter together."""
    config = _make_app_config()
    router = _make_llm_router(llm_return)
    extractor = KeywordExtractor(config=config, llm_router=router)
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
# Test: _normalize_keyword
# ---------------------------------------------------------------------------


def test_normalize_keyword_lowercase() -> None:
    """_normalize_keyword converts to lowercase."""
    extractor, _ = _make_extractor()
    assert extractor._normalize_keyword("Machine Learning") == "machine learning"


def test_normalize_keyword_strip_whitespace() -> None:
    """_normalize_keyword strips leading and trailing whitespace."""
    extractor, _ = _make_extractor()
    assert extractor._normalize_keyword("  API design  ") == "api design"


def test_normalize_keyword_combined() -> None:
    """_normalize_keyword strips whitespace and lowercases simultaneously."""
    extractor, _ = _make_extractor()
    assert extractor._normalize_keyword("  Large Language Models  ") == "large language models"


def test_normalize_keyword_already_normalized() -> None:
    """_normalize_keyword is idempotent on already-normalized strings."""
    extractor, _ = _make_extractor()
    assert extractor._normalize_keyword("deep learning") == "deep learning"


# ---------------------------------------------------------------------------
# Test: _parse_llm_response -- valid JSON
# ---------------------------------------------------------------------------


def test_parse_llm_response_valid_array() -> None:
    """Valid JSON array of strings is parsed correctly."""
    extractor, _ = _make_extractor()
    response = json.dumps(["machine learning", "neural networks", "transformer"])
    keywords = extractor._parse_llm_response(response)
    assert keywords == ["machine learning", "neural networks", "transformer"]


def test_parse_llm_response_normalizes_keywords() -> None:
    """Keywords are normalized to lowercase."""
    extractor, _ = _make_extractor()
    response = json.dumps(["Machine Learning", "NEURAL NETWORKS", "Transformer"])
    keywords = extractor._parse_llm_response(response)
    assert keywords == ["machine learning", "neural networks", "transformer"]


def test_parse_llm_response_deduplicates() -> None:
    """Duplicate keywords after normalization are removed."""
    extractor, _ = _make_extractor()
    response = json.dumps(["Machine Learning", "machine learning", "MACHINE LEARNING"])
    keywords = extractor._parse_llm_response(response)
    assert keywords == ["machine learning"]


def test_parse_llm_response_empty_array() -> None:
    """Empty JSON array returns empty list without error."""
    extractor, _ = _make_extractor()
    keywords = extractor._parse_llm_response("[]")
    assert keywords == []


def test_parse_llm_response_filters_empty_strings() -> None:
    """Empty and whitespace-only strings are filtered out."""
    extractor, _ = _make_extractor()
    response = json.dumps(["valid keyword", "", "  ", "another keyword"])
    keywords = extractor._parse_llm_response(response)
    assert keywords == ["valid keyword", "another keyword"]


def test_parse_llm_response_skips_non_string_entries() -> None:
    """Non-string entries within the array are skipped."""
    extractor, _ = _make_extractor()
    response = json.dumps(["valid keyword", 42, True, None, "another keyword"])
    keywords = extractor._parse_llm_response(response)
    assert keywords == ["valid keyword", "another keyword"]


# ---------------------------------------------------------------------------
# Test: _parse_llm_response -- markdown code fences
# ---------------------------------------------------------------------------


def test_parse_llm_response_strips_json_code_fence() -> None:
    """Markdown ```json code fence is stripped before parsing."""
    extractor, _ = _make_extractor()
    response = '```json\n["deep learning", "attention mechanism"]\n```'
    keywords = extractor._parse_llm_response(response)
    assert keywords == ["deep learning", "attention mechanism"]


def test_parse_llm_response_strips_plain_code_fence() -> None:
    """Markdown ``` (without json language tag) is also stripped."""
    extractor, _ = _make_extractor()
    response = '```\n["api design", "microservices"]\n```'
    keywords = extractor._parse_llm_response(response)
    assert keywords == ["api design", "microservices"]


# ---------------------------------------------------------------------------
# Test: _parse_llm_response -- error handling
# ---------------------------------------------------------------------------


def test_parse_llm_response_invalid_json_returns_empty() -> None:
    """Malformed JSON returns empty list, does not raise."""
    extractor, _ = _make_extractor()
    keywords = extractor._parse_llm_response("not valid json at all")
    assert keywords == []


def test_parse_llm_response_non_array_json_returns_empty() -> None:
    """JSON object (not array) returns empty list with warning."""
    extractor, _ = _make_extractor()
    response = json.dumps({"keywords": ["ml", "ai"]})
    keywords = extractor._parse_llm_response(response)
    assert keywords == []


# ---------------------------------------------------------------------------
# Test: _truncate_to_tokens
# ---------------------------------------------------------------------------


def test_truncate_to_tokens_short_text_unchanged() -> None:
    """Short text well under the token limit is returned unchanged."""
    text = "This is a short article about machine learning."
    result = KeywordExtractor._truncate_to_tokens(text, 4000)
    assert result == text


def test_truncate_to_tokens_truncates_long_text() -> None:
    """Very long text is truncated so the result decodes to fewer tokens."""
    long_text = "transformer " * 5000  # ~5000 tokens
    result = KeywordExtractor._truncate_to_tokens(long_text, 4000)
    assert len(result) < len(long_text)
    assert "transformer" in result


# ---------------------------------------------------------------------------
# Test: extract() -- integration with mocked LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_returns_keywords_from_llm() -> None:
    """extract() calls LLM and returns parsed keywords."""
    llm_response = json.dumps([
        "machine learning",
        "neural networks",
        "deep learning",
        "transformer architecture",
        "attention mechanism",
    ])
    extractor, router = _make_extractor(llm_return=llm_response)

    content = "An article about machine learning and neural networks. " * 20

    keywords = await extractor.extract(content)

    assert len(keywords) == 5
    assert "machine learning" in keywords
    assert "transformer architecture" in keywords
    router.complete.assert_called_once()
    call_kwargs = router.complete.call_args.kwargs
    assert call_kwargs.get("task") == "keyword_extraction"


@pytest.mark.asyncio
async def test_extract_returns_empty_on_llm_error() -> None:
    """extract() returns [] when the LLM call raises an exception."""
    config = _make_app_config()
    router = MagicMock(spec=LLMRouter)
    router.complete = AsyncMock(side_effect=RuntimeError("API timeout"))
    extractor = KeywordExtractor(config=config, llm_router=router)

    keywords = await extractor.extract("Some article content about AI.")

    assert keywords == []


@pytest.mark.asyncio
async def test_extract_returns_empty_on_parse_error() -> None:
    """extract() returns [] when LLM response is malformed JSON."""
    extractor, _ = _make_extractor(llm_return="Sorry, I cannot extract keywords.")

    keywords = await extractor.extract("Some content.")

    assert keywords == []


@pytest.mark.asyncio
async def test_extract_content_truncated_before_llm_call() -> None:
    """Content is truncated to 4000 tokens before being sent to the LLM."""
    extractor, router = _make_extractor(llm_return="[]")

    very_long_content = "artificial intelligence " * 20000

    await extractor.extract(very_long_content)

    router.complete.assert_called_once()
    call_kwargs = router.complete.call_args.kwargs
    prompt_used = call_kwargs["prompt"]
    assert prompt_used.count("artificial intelligence") < 20000


@pytest.mark.asyncio
async def test_extract_uses_correct_llm_task() -> None:
    """extract() calls LLMRouter.complete with task='keyword_extraction'."""
    extractor, router = _make_extractor(llm_return="[]")

    await extractor.extract("Some content about AI companies.")

    router.complete.assert_called_once()
    call_kwargs = router.complete.call_args.kwargs
    assert call_kwargs["task"] == "keyword_extraction"


# ---------------------------------------------------------------------------
# Test: extract_and_store() -- DB integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_and_store_inserts_keywords() -> None:
    """extract_and_store() writes keywords to the document_keywords table."""
    llm_response = json.dumps(["machine learning", "neural networks", "deep learning"])
    extractor, _ = _make_extractor(llm_return=llm_response)

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        keywords = await extractor.extract_and_store(conn, doc_id, "Article about ML.")

        assert len(keywords) == 3

        async with conn.execute(
            "SELECT keyword FROM document_keywords WHERE document_id = ? ORDER BY keyword",
            (doc_id,),
        ) as cur:
            rows = await cur.fetchall()
        stored = [row["keyword"] for row in rows]
        assert "deep learning" in stored
        assert "machine learning" in stored
        assert "neural networks" in stored
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_extract_and_store_marks_document_extracted() -> None:
    """extract_and_store() sets is_keywords_extracted=True on the document."""
    extractor, _ = _make_extractor(llm_return="[]")

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        await extractor.extract_and_store(conn, doc_id, "Content with no keywords.")

        async with conn.execute(
            "SELECT is_keywords_extracted FROM documents WHERE id = ?", (doc_id,)
        ) as cur:
            row = await cur.fetchone()

        assert row is not None
        assert bool(row["is_keywords_extracted"]) is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_extract_and_store_marks_document_even_on_empty_extraction() -> None:
    """Document is marked as processed even when no keywords are found."""
    extractor, _ = _make_extractor(llm_return="[]")

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        result = await extractor.extract_and_store(conn, doc_id, "Some generic content.")

        assert result == []
        async with conn.execute(
            "SELECT is_keywords_extracted FROM documents WHERE id = ?", (doc_id,)
        ) as cur:
            row = await cur.fetchone()
        assert bool(row["is_keywords_extracted"]) is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_extract_and_store_deduplicates_keywords() -> None:
    """Duplicate keywords from LLM are stored only once per document."""
    # LLM returns duplicates after normalization
    llm_response = json.dumps(["Machine Learning", "machine learning", "Deep Learning"])
    extractor, _ = _make_extractor(llm_return=llm_response)

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        keywords = await extractor.extract_and_store(conn, doc_id, "Article about ML.")

        # Should be deduplicated to 2
        assert len(keywords) == 2

        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM document_keywords WHERE document_id = ?",
            (doc_id,),
        ) as cur:
            row = await cur.fetchone()
        assert row["cnt"] == 2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_extract_and_store_returns_keywords() -> None:
    """extract_and_store() returns the list of extracted keywords."""
    llm_response = json.dumps(["api design", "microservices", "rest"])
    extractor, _ = _make_extractor(llm_return=llm_response)

    conn = await _make_in_memory_db()
    try:
        doc_id = _make_doc_id()
        await _insert_test_document(conn, doc_id)

        result = await extractor.extract_and_store(conn, doc_id, "Article about APIs.")

        assert len(result) == 3
        assert "api design" in result
        assert "microservices" in result
        assert "rest" in result
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

    with patch(
        "ai_craftsman_kb.processing.keyword_extractor._PROMPT_PATH",
        new=Path("/nonexistent/path/keyword_extraction.md"),
    ):
        extractor = KeywordExtractor(config=config, llm_router=router)

    assert "{content}" in extractor._prompt_template
