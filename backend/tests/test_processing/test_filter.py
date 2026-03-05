"""Unit tests for ContentFilter — LLM, keyword, and hybrid filtering strategies.

All LLM calls are mocked via AsyncMock on LLMRouter.complete().
No network access is required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_craftsman_kb.config.models import (
    AppConfig,
    FiltersConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    SettingsConfig,
    SourceFilterConfig,
    SourcesConfig,
)
from ai_craftsman_kb.ingestors.base import RawDocument
from ai_craftsman_kb.llm.router import LLMRouter
from ai_craftsman_kb.processing.filter import ContentFilter, FilterResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_filter_cfg(**kwargs) -> SourceFilterConfig:
    """Build a SourceFilterConfig with overrides for test use."""
    return SourceFilterConfig(**kwargs)


def _make_app_config(filters: FiltersConfig | None = None) -> AppConfig:
    """Build a minimal AppConfig suitable for ContentFilter tests."""
    return AppConfig(
        sources=SourcesConfig(),
        settings=SettingsConfig(
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="llama-3-8b"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="llama-3-8b"),
                briefing=LLMTaskConfig(provider="anthropic", model="claude-3-haiku"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="llama-3-8b"),
                keyword_extraction=LLMTaskConfig(provider="openrouter", model="test-model"),
            )
        ),
        filters=filters or FiltersConfig(),
    )


def _make_llm_router(complete_return: str = "7") -> LLMRouter:
    """Build a mock LLMRouter whose complete() returns a fixed string."""
    router = MagicMock(spec=LLMRouter)
    router.complete = AsyncMock(return_value=complete_return)
    return router


def _make_doc(
    title: str = "Test Article",
    raw_content: str = "Some article content about machine learning.",
    source_type: str = "hn",
    metadata: dict | None = None,
) -> RawDocument:
    """Build a minimal RawDocument for testing."""
    return RawDocument(
        url="https://example.com/article",
        title=title,
        raw_content=raw_content,
        source_type=source_type,
        metadata=metadata or {},
    )


def _make_content_filter(
    filters: FiltersConfig | None = None,
    llm_return: str = "7",
) -> tuple[ContentFilter, LLMRouter]:
    """Create a ContentFilter and its mock LLMRouter together."""
    config = _make_app_config(filters=filters)
    router = _make_llm_router(llm_return)
    content_filter = ContentFilter(config=config, llm_router=router)
    return content_filter, router


# ---------------------------------------------------------------------------
# Test: filter disabled source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_disabled_source() -> None:
    """A source with enabled=False always passes without calling LLM."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(enabled=False),
    )
    content_filter, router = _make_content_filter(filters=filters)
    doc = _make_doc(source_type="hn")

    result = await content_filter.filter(doc, "hn")

    assert result.passed is True
    assert result.score is None
    assert result.reason == "filter disabled"
    router.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test: keyword filter — exclude keyword match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_filter_exclude_match() -> None:
    """A document containing an excluded keyword fails keyword filter."""
    filters = FiltersConfig(
        arxiv=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            keywords_exclude=["blockchain", "crypto"],
        )
    )
    content_filter, router = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="Blockchain meets AI",
        raw_content="Using blockchain for decentralized AI",
        source_type="arxiv",
    )

    result = await content_filter.filter(doc, "arxiv")

    assert result.passed is False
    assert result.score == 0.0
    assert "blockchain" in result.reason.lower()
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_keyword_filter_exclude_match_in_content() -> None:
    """Excluded keyword found in content (not title) still fails."""
    filters = FiltersConfig(
        devto=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            keywords_exclude=["NFT"],
        )
    )
    content_filter, _ = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="Cool tech article",
        raw_content="This article discusses NFT markets extensively.",
        source_type="devto",
    )

    result = await content_filter.filter(doc, "devto")

    assert result.passed is False
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Test: keyword filter — no include match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_filter_no_include_match() -> None:
    """A required keyword list is non-empty but none match — fails."""
    filters = FiltersConfig(
        arxiv=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            keywords_include=["transformer", "language model"],
        )
    )
    content_filter, router = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="Advances in computer vision",
        raw_content="This paper is about convolutional networks for image segmentation.",
        source_type="arxiv",
    )

    result = await content_filter.filter(doc, "arxiv")

    assert result.passed is False
    assert result.score == 0.0
    assert "no required keywords" in result.reason
    router.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test: keyword filter passes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_filter_passes() -> None:
    """Document meets all keyword conditions — passes with score=1.0."""
    filters = FiltersConfig(
        arxiv=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            keywords_include=["transformer", "language model"],
            keywords_exclude=["blockchain"],
        )
    )
    content_filter, router = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="Efficient Transformer architectures",
        raw_content="We propose a new language model based on the transformer architecture.",
        source_type="arxiv",
    )

    result = await content_filter.filter(doc, "arxiv")

    assert result.passed is True
    assert result.score == 1.0
    assert result.reason == "keyword filter passed"
    router.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test: keyword filter — empty include list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_filter_empty_include_list() -> None:
    """When keywords_include is empty, all documents pass the include check."""
    filters = FiltersConfig(
        devto=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            keywords_include=[],  # empty — no restriction
            keywords_exclude=[],
        )
    )
    content_filter, router = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="Random cooking article",
        raw_content="How to bake the perfect sourdough bread.",
        source_type="devto",
    )

    result = await content_filter.filter(doc, "devto")

    assert result.passed is True
    assert result.score == 1.0
    router.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test: keyword filter — min_upvotes fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_filter_min_upvotes_fail() -> None:
    """Document with upvotes below min_upvotes threshold fails."""
    filters = FiltersConfig(
        reddit=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            min_upvotes=100,
        )
    )
    content_filter, router = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="New ML paper",
        raw_content="Interesting machine learning research results.",
        source_type="reddit",
        metadata={"upvotes": 42},
    )

    result = await content_filter.filter(doc, "reddit")

    assert result.passed is False
    assert result.score == 0.0
    assert "upvotes" in result.reason
    assert "42" in result.reason
    assert "100" in result.reason
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_keyword_filter_min_upvotes_pass() -> None:
    """Document with upvotes at or above threshold passes the upvotes check."""
    filters = FiltersConfig(
        reddit=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            min_upvotes=10,
        )
    )
    content_filter, _ = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="Trending post",
        raw_content="This post is trending with many upvotes.",
        source_type="reddit",
        metadata={"upvotes": 10},  # exactly at threshold
    )

    result = await content_filter.filter(doc, "reddit")

    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_keyword_filter_min_upvotes_missing_metadata_fails() -> None:
    """Missing upvotes key in metadata defaults to 0, fails if min_upvotes > 0."""
    filters = FiltersConfig(
        reddit=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            min_upvotes=10,
        )
    )
    content_filter, _ = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="Post without upvotes",
        raw_content="No upvotes metadata on this one.",
        source_type="reddit",
        metadata={},  # no upvotes key
    )

    result = await content_filter.filter(doc, "reddit")

    assert result.passed is False
    assert "upvotes 0 < 10" in result.reason


# ---------------------------------------------------------------------------
# Test: keyword filter — min_reactions fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_filter_min_reactions_fail() -> None:
    """Document with reactions below min_reactions threshold fails."""
    filters = FiltersConfig(
        devto=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            min_reactions=20,
        )
    )
    content_filter, router = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="Unpopular DEV.to article",
        raw_content="This article did not get much engagement.",
        source_type="devto",
        metadata={"reactions": 3},
    )

    result = await content_filter.filter(doc, "devto")

    assert result.passed is False
    assert result.score == 0.0
    assert "reactions" in result.reason
    assert "3" in result.reason
    assert "20" in result.reason
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_keyword_filter_min_reactions_missing_metadata_fails() -> None:
    """Missing reactions key in metadata defaults to 0."""
    filters = FiltersConfig(
        devto=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            min_reactions=5,
        )
    )
    content_filter, _ = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="No reactions doc",
        raw_content="Content here.",
        source_type="devto",
        metadata={},
    )

    result = await content_filter.filter(doc, "devto")

    assert result.passed is False
    assert "reactions 0 < 5" in result.reason


# ---------------------------------------------------------------------------
# Test: LLM filter — pass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_filter_pass() -> None:
    """LLM returns score 8, min_score=5 — document passes."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="llm",
            llm_prompt="Rate this article about {title}: {excerpt}",
            min_score=5,
        )
    )
    content_filter, router = _make_content_filter(filters=filters, llm_return="8")
    doc = _make_doc(
        title="Revolutionary AI breakthrough",
        raw_content="Researchers achieved human-level performance on all benchmarks.",
        source_type="hn",
    )

    result = await content_filter.filter(doc, "hn")

    assert result.passed is True
    assert result.score == 8.0
    assert "8" in result.reason
    assert "5" in result.reason
    router.complete.assert_called_once()


@pytest.mark.asyncio
async def test_llm_filter_pass_at_threshold() -> None:
    """LLM returns exactly min_score — document passes (>= comparison)."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="llm",
            llm_prompt="Rate: {title} {excerpt}",
            min_score=7,
        )
    )
    content_filter, _ = _make_content_filter(filters=filters, llm_return="7")
    doc = _make_doc(source_type="hn")

    result = await content_filter.filter(doc, "hn")

    assert result.passed is True
    assert result.score == 7.0


# ---------------------------------------------------------------------------
# Test: LLM filter — fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_filter_fail() -> None:
    """LLM returns score 3, min_score=5 — document fails."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="llm",
            llm_prompt="Rate this: {title} {excerpt}",
            min_score=5,
        )
    )
    content_filter, router = _make_content_filter(filters=filters, llm_return="3")
    doc = _make_doc(
        title="Celebrity gossip",
        raw_content="Famous actor does something unrelated to tech.",
        source_type="hn",
    )

    result = await content_filter.filter(doc, "hn")

    assert result.passed is False
    assert result.score == 3.0
    assert "3" in result.reason
    assert "5" in result.reason
    router.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Test: LLM filter — score format variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response,expected_score",
    [
        ("7", 7),
        ("7/10", 7),
        ("Score: 7", 7),
        ("I rate this a 7 out of 10", 7),
        ("Rating: 10", 10),
        ("The relevance score is 9.", 9),
        ("  8  ", 8),
        ("Based on the content, I give this a 6.", 6),
        ("1", 1),
        ("10/10", 10),
    ],
)
def test_llm_filter_score_formats(response: str, expected_score: int) -> None:
    """_parse_llm_score correctly extracts integer from varied response formats."""
    score = ContentFilter._parse_llm_score(response)
    assert score == expected_score


@pytest.mark.parametrize(
    "response",
    [
        "",
        "no score here",
        "0",           # 0 is not in 1-10 range
        "11",          # 11 is out of range
        "N/A",
        "excellent",
    ],
)
def test_parse_llm_score_no_valid_score(response: str) -> None:
    """_parse_llm_score returns 0 when no valid 1-10 score is found."""
    score = ContentFilter._parse_llm_score(response)
    assert score == 0


# ---------------------------------------------------------------------------
# Test: LLM filter — API error defaults to pass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_filter_api_error() -> None:
    """LLM raises an exception — defaults to passing with error reason."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="llm",
            llm_prompt="Rate: {title} {excerpt}",
            min_score=5,
        )
    )
    config = _make_app_config(filters=filters)
    router = MagicMock(spec=LLMRouter)
    router.complete = AsyncMock(side_effect=RuntimeError("API unavailable"))
    content_filter = ContentFilter(config=config, llm_router=router)
    doc = _make_doc(source_type="hn")

    result = await content_filter.filter(doc, "hn")

    # Should default to passing on LLM error
    assert result.passed is True
    assert result.score is None
    assert "llm error" in result.reason.lower()
    assert "API unavailable" in result.reason


@pytest.mark.asyncio
async def test_llm_filter_no_prompt_configured() -> None:
    """LLM strategy without llm_prompt set defaults to passing."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="llm",
            llm_prompt=None,  # no prompt
            min_score=5,
        )
    )
    content_filter, router = _make_content_filter(filters=filters)
    doc = _make_doc(source_type="hn")

    result = await content_filter.filter(doc, "hn")

    assert result.passed is True
    assert result.score is None
    assert "no prompt configured" in result.reason
    router.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test: LLM filter — prompt interpolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_filter_prompt_interpolation() -> None:
    """LLM prompt is filled with doc title and excerpt before calling router."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="llm",
            llm_prompt="Title: {title}\nExcerpt: {excerpt}\nScore 1-10:",
            min_score=5,
        )
    )
    content_filter, router = _make_content_filter(filters=filters, llm_return="8")
    doc = _make_doc(
        title="AI Research Paper",
        raw_content="This is the content of the research paper about LLMs." * 20,
        source_type="hn",
    )

    await content_filter.filter(doc, "hn")

    # router.complete is called with keyword arguments: task="filtering", prompt=<filled_prompt>
    call_kwargs = router.complete.call_args.kwargs
    assert "prompt" in call_kwargs
    prompt_used = call_kwargs["prompt"]
    assert "AI Research Paper" in prompt_used
    assert "Score 1-10:" in prompt_used


@pytest.mark.asyncio
async def test_llm_filter_excerpt_truncated_to_500_chars() -> None:
    """The excerpt passed to the LLM prompt is at most 500 characters."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="llm",
            llm_prompt="{title} {excerpt}",
            min_score=5,
        )
    )
    content_filter, router = _make_content_filter(filters=filters, llm_return="6")
    long_content = "x" * 2000
    doc = _make_doc(title="Title", raw_content=long_content, source_type="hn")

    await content_filter.filter(doc, "hn")

    call_kwargs = router.complete.call_args.kwargs
    assert "prompt" in call_kwargs
    prompt = call_kwargs["prompt"]
    # Prompt is "{title} {excerpt}" = "Title " + 500 x's
    # Title is "Title" (5 chars) + space + exactly 500 x's
    assert "x" * 500 in prompt
    # The 501st x should NOT appear — excerpt is truncated to exactly 500 chars
    assert prompt.count("x") == 500


# ---------------------------------------------------------------------------
# Test: LLM filter — default min_score when not configured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_filter_default_min_score_is_5() -> None:
    """When min_score is not set, defaults to 5. Score of 5 passes."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="llm",
            llm_prompt="Rate: {title} {excerpt}",
            min_score=None,  # not configured
        )
    )
    content_filter, _ = _make_content_filter(filters=filters, llm_return="5")
    doc = _make_doc(source_type="hn")

    result = await content_filter.filter(doc, "hn")

    assert result.passed is True
    assert result.score == 5.0


@pytest.mark.asyncio
async def test_llm_filter_default_min_score_fail_below_5() -> None:
    """When min_score is None (defaults to 5), score of 4 fails."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="llm",
            llm_prompt="Rate: {title} {excerpt}",
            min_score=None,
        )
    )
    content_filter, _ = _make_content_filter(filters=filters, llm_return="4")
    doc = _make_doc(source_type="hn")

    result = await content_filter.filter(doc, "hn")

    assert result.passed is False
    assert result.score == 4.0


# ---------------------------------------------------------------------------
# Test: hybrid filter — keyword fail skips LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_filter_keyword_fail_skips_llm() -> None:
    """Hybrid: keyword filter fails — LLM is never called."""
    filters = FiltersConfig(
        reddit=SourceFilterConfig(
            enabled=True,
            strategy="hybrid",
            keywords_exclude=["meme"],
            llm_prompt="Rate: {title} {excerpt}",
            min_score=5,
        )
    )
    content_filter, router = _make_content_filter(filters=filters, llm_return="9")
    doc = _make_doc(
        title="The best meme collection",
        raw_content="A collection of internet memes.",
        source_type="reddit",
    )

    result = await content_filter.filter(doc, "reddit")

    assert result.passed is False
    assert result.score == 0.0
    assert "meme" in result.reason.lower()
    router.complete.assert_not_called()  # LLM skipped


# ---------------------------------------------------------------------------
# Test: hybrid filter — both pass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_filter_both_pass() -> None:
    """Hybrid: keyword passes, LLM score meets threshold — document passes."""
    filters = FiltersConfig(
        reddit=SourceFilterConfig(
            enabled=True,
            strategy="hybrid",
            keywords_include=["machine learning"],
            keywords_exclude=["meme"],
            min_upvotes=10,
            llm_prompt="Rate: {title} {excerpt}",
            min_score=5,
        )
    )
    content_filter, router = _make_content_filter(filters=filters, llm_return="8")
    doc = _make_doc(
        title="Machine learning paper discussion",
        raw_content="This paper on machine learning is excellent.",
        source_type="reddit",
        metadata={"upvotes": 50},
    )

    result = await content_filter.filter(doc, "reddit")

    assert result.passed is True
    assert result.score == 8.0
    router.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Test: hybrid filter — LLM fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_filter_llm_fail() -> None:
    """Hybrid: keyword passes but LLM score is below threshold — fails."""
    filters = FiltersConfig(
        reddit=SourceFilterConfig(
            enabled=True,
            strategy="hybrid",
            keywords_include=["python"],
            keywords_exclude=["spam"],
            llm_prompt="Rate: {title} {excerpt}",
            min_score=7,
        )
    )
    content_filter, router = _make_content_filter(filters=filters, llm_return="3")
    doc = _make_doc(
        title="Python tutorial for beginners",
        raw_content="Learn python programming step by step.",
        source_type="reddit",
    )

    result = await content_filter.filter(doc, "reddit")

    assert result.passed is False
    assert result.score == 3.0
    router.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Test: filter_batch respects concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_batch_respects_concurrency() -> None:
    """filter_batch processes all documents and returns (doc, result) pairs."""
    filters = FiltersConfig(
        arxiv=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            keywords_include=["neural"],
        )
    )
    content_filter, router = _make_content_filter(filters=filters)

    docs = [
        _make_doc(
            title=f"Neural network paper {i}",
            raw_content=f"This paper {i} discusses neural architecture search.",
            source_type="arxiv",
        )
        for i in range(10)
    ]

    results = await content_filter.filter_batch(docs, "arxiv", concurrency=3)

    assert len(results) == 10
    for doc, result in results:
        assert isinstance(doc, RawDocument)
        assert isinstance(result, FilterResult)
        assert result.passed is True


@pytest.mark.asyncio
async def test_filter_batch_mixed_results() -> None:
    """filter_batch correctly marks some docs passing and some failing."""
    filters = FiltersConfig(
        arxiv=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            keywords_include=["transformer"],
        )
    )
    content_filter, _ = _make_content_filter(filters=filters)

    matching_doc = _make_doc(
        title="Transformer architecture",
        raw_content="We study the transformer model.",
        source_type="arxiv",
    )
    non_matching_doc = _make_doc(
        title="Computer vision",
        raw_content="CNN for image recognition.",
        source_type="arxiv",
    )

    results = await content_filter.filter_batch(
        [matching_doc, non_matching_doc], "arxiv", concurrency=2
    )

    assert len(results) == 2
    # Find results by checking document URL (matching vs non-matching)
    passing = [r for _, r in results if r.passed]
    failing = [r for _, r in results if not r.passed]
    assert len(passing) == 1
    assert len(failing) == 1


@pytest.mark.asyncio
async def test_filter_batch_empty_list() -> None:
    """filter_batch with empty input returns empty list."""
    content_filter, _ = _make_content_filter()

    results = await content_filter.filter_batch([], "hn", concurrency=5)

    assert results == []


# ---------------------------------------------------------------------------
# Test: _parse_llm_score comprehensive variants
# ---------------------------------------------------------------------------


def test_parse_llm_score_variants() -> None:
    """_parse_llm_score handles a wide range of response formats."""
    # Single digit
    assert ContentFilter._parse_llm_score("5") == 5

    # With /10 suffix
    assert ContentFilter._parse_llm_score("8/10") == 8

    # With "Score:" prefix
    assert ContentFilter._parse_llm_score("Score: 9") == 9

    # Embedded in a sentence
    assert ContentFilter._parse_llm_score("I rate this a 7 out of 10") == 7

    # Only the number 10 (two digits)
    assert ContentFilter._parse_llm_score("10") == 10

    # Extra whitespace
    assert ContentFilter._parse_llm_score("  3  ") == 3

    # Takes the FIRST number found
    assert ContentFilter._parse_llm_score("Score: 7 (previously 4)") == 7

    # No valid score — returns 0
    assert ContentFilter._parse_llm_score("excellent article") == 0
    assert ContentFilter._parse_llm_score("") == 0
    assert ContentFilter._parse_llm_score("11") == 0  # 11 is out of range
    assert ContentFilter._parse_llm_score("0") == 0   # 0 is out of range


def test_parse_llm_score_boundary_values() -> None:
    """_parse_llm_score correctly handles boundary values 1 and 10."""
    assert ContentFilter._parse_llm_score("1") == 1
    assert ContentFilter._parse_llm_score("10") == 10
    # Numbers outside 1-10 range return 0
    assert ContentFilter._parse_llm_score("0") == 0
    assert ContentFilter._parse_llm_score("11") == 0


# ---------------------------------------------------------------------------
# Test: source type mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_type_hn_alias_maps_to_hackernews() -> None:
    """Source type 'hn' maps to the hackernews filter configuration."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            keywords_include=["rust"],
        )
    )
    content_filter, _ = _make_content_filter(filters=filters)
    doc = _make_doc(
        title="Rust programming language update",
        raw_content="The rust ecosystem continues to grow rapidly.",
        source_type="hn",
    )

    result = await content_filter.filter(doc, "hn")

    assert result.passed is True


@pytest.mark.asyncio
async def test_unknown_source_type_defaults_to_pass() -> None:
    """Unknown source type has no filter config — defaults to passing."""
    content_filter, router = _make_content_filter()
    doc = _make_doc(source_type="unknown_source")

    result = await content_filter.filter(doc, "unknown_source")

    assert result.passed is True
    assert result.score is None
    router.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test: FilterResult model
# ---------------------------------------------------------------------------


def test_filter_result_model_defaults() -> None:
    """FilterResult has correct field types and defaults."""
    result = FilterResult(passed=True)
    assert result.passed is True
    assert result.score is None
    assert result.reason is None


def test_filter_result_model_with_all_fields() -> None:
    """FilterResult correctly stores all provided fields."""
    result = FilterResult(passed=False, score=3.0, reason="below threshold")
    assert result.passed is False
    assert result.score == 3.0
    assert result.reason == "below threshold"


# ---------------------------------------------------------------------------
# Test: content filter handles None fields in RawDocument
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_filter_handles_none_title() -> None:
    """Keyword filter handles RawDocument with title=None."""
    filters = FiltersConfig(
        arxiv=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            keywords_include=["neural"],
        )
    )
    content_filter, _ = _make_content_filter(filters=filters)
    doc = RawDocument(
        url="https://example.com",
        title=None,
        raw_content="A paper about neural networks.",
        source_type="arxiv",
    )

    result = await content_filter.filter(doc, "arxiv")

    assert result.passed is True


@pytest.mark.asyncio
async def test_keyword_filter_handles_none_content() -> None:
    """Keyword filter handles RawDocument with raw_content=None."""
    filters = FiltersConfig(
        arxiv=SourceFilterConfig(
            enabled=True,
            strategy="keyword",
            keywords_include=["neural"],
        )
    )
    content_filter, _ = _make_content_filter(filters=filters)
    doc = RawDocument(
        url="https://example.com",
        title="Neural network paper",
        raw_content=None,
        source_type="arxiv",
    )

    result = await content_filter.filter(doc, "arxiv")

    # "neural" is in the title, so should pass
    assert result.passed is True


@pytest.mark.asyncio
async def test_llm_filter_handles_none_title_and_content() -> None:
    """LLM filter handles RawDocument with both title and raw_content None."""
    filters = FiltersConfig(
        hackernews=SourceFilterConfig(
            enabled=True,
            strategy="llm",
            llm_prompt="Title: {title}\nExcerpt: {excerpt}",
            min_score=5,
        )
    )
    content_filter, router = _make_content_filter(filters=filters, llm_return="6")
    doc = RawDocument(
        url="https://example.com",
        title=None,
        raw_content=None,
        source_type="hn",
    )

    result = await content_filter.filter(doc, "hn")

    assert result.passed is True
    # Verify LLM was called — even with None fields, it should not raise
    router.complete.assert_called_once()
    # Verify prompt contains the template labels (filled with empty strings)
    call_kwargs = router.complete.call_args.kwargs
    assert "prompt" in call_kwargs
    prompt = call_kwargs["prompt"]
    assert "Title: " in prompt
    assert "Excerpt: " in prompt
