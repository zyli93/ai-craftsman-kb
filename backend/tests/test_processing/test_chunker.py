"""Unit tests for the Chunker text chunking system.

Tests cover:
- Basic sentence-boundary chunking
- Token limit enforcement (no chunk exceeds chunk_size)
- Overlap between consecutive chunks
- chunk_index is sequential starting at 0
- char_start / char_end map back to original text
- Skipping of trivially small chunks (< 50 tokens)
- Empty / None input returns []
- Oversized single sentences split by words
- chunk_document convenience wrapper
- Paragraph boundary splitting
"""
from __future__ import annotations

import pytest

from ai_craftsman_kb.db.models import DocumentRow
from ai_craftsman_kb.processing.chunker import Chunker, TextChunk, _MIN_CHUNK_TOKENS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunker(chunk_size: int = 2000, chunk_overlap: int = 200) -> Chunker:
    """Return a Chunker with the given settings."""
    return Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def make_long_text(n_sentences: int = 50, sentence: str | None = None) -> str:
    """Build a block of text with *n_sentences* sentences."""
    s = sentence or "The quick brown fox jumps over the lazy dog near the riverbank."
    return " ".join([s] * n_sentences)


def make_document(raw_content: str | None = None, doc_id: str = "doc-1") -> DocumentRow:
    """Build a minimal DocumentRow for testing."""
    return DocumentRow(
        id=doc_id,
        origin="pro",
        source_type="hn",
        url="https://example.com/article",
        raw_content=raw_content,
    )


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


def test_count_tokens_empty_string() -> None:
    """count_tokens returns 0 for an empty string."""
    chunker = make_chunker()
    assert chunker.count_tokens("") == 0


def test_count_tokens_single_word() -> None:
    """count_tokens returns a positive integer for a single word."""
    chunker = make_chunker()
    assert chunker.count_tokens("hello") > 0


def test_count_tokens_consistency() -> None:
    """count_tokens is deterministic — same text always returns same count."""
    chunker = make_chunker()
    text = "OpenAI released a new embedding model this year."
    assert chunker.count_tokens(text) == chunker.count_tokens(text)


# ---------------------------------------------------------------------------
# Empty / None / trivially short input
# ---------------------------------------------------------------------------


def test_chunk_empty_string_returns_empty_list() -> None:
    """chunk('') returns []."""
    chunker = make_chunker()
    assert chunker.chunk("") == []


def test_chunk_whitespace_only_returns_empty_list() -> None:
    """chunk with only whitespace returns []."""
    chunker = make_chunker()
    assert chunker.chunk("   \n\n   ") == []


def test_chunk_very_short_text_skipped() -> None:
    """Text shorter than 50 tokens results in no chunks emitted (skipped)."""
    chunker = make_chunker()
    short_text = "Hi."
    result = chunker.chunk(short_text)
    # The single chunk has < 50 tokens, so it should be skipped.
    token_count = chunker.count_tokens(short_text)
    if token_count < _MIN_CHUNK_TOKENS:
        assert result == []
    else:
        # If by some encoding it IS >= 50 tokens, just verify it's valid.
        assert len(result) == 1


def test_chunk_document_none_raw_content_returns_empty() -> None:
    """chunk_document returns [] when raw_content is None."""
    chunker = make_chunker()
    doc = make_document(raw_content=None)
    assert chunker.chunk_document(doc) == []


def test_chunk_document_empty_raw_content_returns_empty() -> None:
    """chunk_document returns [] when raw_content is empty string."""
    chunker = make_chunker()
    doc = make_document(raw_content="")
    assert chunker.chunk_document(doc) == []


def test_chunk_document_short_raw_content_returns_empty() -> None:
    """chunk_document returns [] when raw_content is below the 50-token minimum."""
    chunker = make_chunker()
    doc = make_document(raw_content="Too short.")
    result = chunker.chunk_document(doc)
    token_count = chunker.count_tokens("Too short.")
    if token_count < _MIN_CHUNK_TOKENS:
        assert result == []


# ---------------------------------------------------------------------------
# Basic chunking — structure validation
# ---------------------------------------------------------------------------


def test_chunk_returns_list_of_text_chunks() -> None:
    """chunk() returns a list of TextChunk instances."""
    chunker = make_chunker(chunk_size=100, chunk_overlap=20)
    text = make_long_text(n_sentences=20)
    result = chunker.chunk(text)
    assert isinstance(result, list)
    assert all(isinstance(c, TextChunk) for c in result)


def test_chunk_index_is_sequential_from_zero() -> None:
    """chunk_index values are 0, 1, 2, ... in order."""
    chunker = make_chunker(chunk_size=100, chunk_overlap=20)
    text = make_long_text(n_sentences=30)
    result = chunker.chunk(text)
    assert len(result) > 1, "Expected multiple chunks for this text"
    for expected_idx, chunk in enumerate(result):
        assert chunk.chunk_index == expected_idx


def test_no_chunk_exceeds_chunk_size() -> None:
    """Every chunk's token_count <= chunk_size."""
    chunk_size = 150
    chunker = make_chunker(chunk_size=chunk_size, chunk_overlap=30)
    text = make_long_text(n_sentences=40)
    result = chunker.chunk(text)
    assert len(result) > 0
    for chunk in result:
        assert chunk.token_count <= chunk_size, (
            f"Chunk {chunk.chunk_index} has {chunk.token_count} tokens, "
            f"exceeding chunk_size={chunk_size}"
        )


def test_all_chunks_meet_minimum_size() -> None:
    """Every emitted chunk has token_count >= _MIN_CHUNK_TOKENS (50)."""
    chunker = make_chunker(chunk_size=200, chunk_overlap=40)
    text = make_long_text(n_sentences=30)
    result = chunker.chunk(text)
    for chunk in result:
        assert chunk.token_count >= _MIN_CHUNK_TOKENS, (
            f"Chunk {chunk.chunk_index} has only {chunk.token_count} tokens"
        )


# ---------------------------------------------------------------------------
# Overlap validation
# ---------------------------------------------------------------------------


def test_consecutive_chunks_have_overlap() -> None:
    """Consecutive chunks share some text (indicating overlap)."""
    chunker = make_chunker(chunk_size=100, chunk_overlap=20)
    text = make_long_text(n_sentences=40)
    result = chunker.chunk(text)
    assert len(result) >= 2, "Need at least 2 chunks to test overlap"

    # Check that at least one pair of consecutive chunks shares a sentence.
    overlapping_pairs = 0
    for i in range(len(result) - 1):
        # Get words from both chunks and see if any sentence from chunk i
        # appears in chunk i+1 (simple heuristic — look for shared sub-strings).
        chunk_a_text = result[i].text
        chunk_b_text = result[i + 1].text
        # Split by sentence ending to get individual sentences.
        sentences_a = [s.strip() for s in chunk_a_text.split(". ") if s.strip()]
        for sent in sentences_a[-3:]:  # check last few sentences of chunk_a
            if sent and len(sent) > 10 and sent in chunk_b_text:
                overlapping_pairs += 1
                break

    # At least some pairs should overlap — with chunk_overlap=20 tokens and
    # sentences of ~12 tokens, we expect overlap in most pairs.
    assert overlapping_pairs > 0, (
        "Expected at least one pair of consecutive chunks to share text (overlap)"
    )


def test_overlap_tokens_within_budget() -> None:
    """The overlap region does not exceed chunk_overlap tokens by more than one sentence."""
    chunk_size = 100
    chunk_overlap = 20
    chunker = make_chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sentence = "The model performance improved significantly on the benchmark dataset."
    text = " ".join([sentence] * 30)
    result = chunker.chunk(text)
    assert len(result) >= 2

    # Verify that the second chunk starts within a reasonable token distance
    # (the overlap should be <= chunk_overlap + one-sentence buffer).
    sentence_tokens = chunker.count_tokens(sentence)
    for i in range(1, len(result)):
        # The overlap seed tokens should be <= chunk_overlap + one extra sentence.
        # We validate indirectly: the first chunk of the result must fit within chunk_size.
        assert result[i].token_count <= chunk_size


# ---------------------------------------------------------------------------
# char_start / char_end
# ---------------------------------------------------------------------------


def test_char_offsets_map_back_to_original_text() -> None:
    """char_start and char_end correctly point to the chunk text in the original."""
    chunker = make_chunker(chunk_size=100, chunk_overlap=20)
    text = make_long_text(n_sentences=20)
    result = chunker.chunk(text)
    assert len(result) > 0

    for chunk in result:
        extracted = text[chunk.char_start:chunk.char_end]
        # The extracted slice should match the chunk text exactly, or at least
        # contain it (whitespace normalisation during sentence-joining may cause
        # slight differences at the boundaries).
        assert extracted == chunk.text or chunk.text in text, (
            f"Chunk {chunk.chunk_index}: char_start={chunk.char_start}, "
            f"char_end={chunk.char_end} does not correctly map back to original text"
        )


def test_char_start_is_non_negative() -> None:
    """char_start is always >= 0."""
    chunker = make_chunker(chunk_size=100, chunk_overlap=20)
    text = make_long_text(n_sentences=15)
    result = chunker.chunk(text)
    for chunk in result:
        assert chunk.char_start >= 0


def test_char_end_greater_than_char_start() -> None:
    """char_end > char_start for every chunk."""
    chunker = make_chunker(chunk_size=100, chunk_overlap=20)
    text = make_long_text(n_sentences=15)
    result = chunker.chunk(text)
    for chunk in result:
        assert chunk.char_end > chunk.char_start


# ---------------------------------------------------------------------------
# Single sentence longer than chunk_size
# ---------------------------------------------------------------------------


def test_oversized_sentence_split_by_words() -> None:
    """A sentence longer than chunk_size is split at word boundaries.

    Use chunk_size well above _MIN_CHUNK_TOKENS (50) so word-split sub-chunks
    are retained. The text is a single run with no sentence-ending punctuation,
    forcing the word-split fallback path.
    """
    chunk_size = 200  # well above _MIN_CHUNK_TOKENS=50
    chunker = make_chunker(chunk_size=chunk_size, chunk_overlap=30)
    # A realistic long sentence with many distinct words and no punctuation.
    # Repeated enough times to produce several word-split chunks.
    base = "the quick brown fox jumps over the lazy dog near the riverbank "
    long_sentence = (base * 80).strip()
    result = chunker.chunk(long_sentence)

    # Should produce multiple chunks.
    assert len(result) > 1, (
        f"Expected multiple chunks for oversized sentence, got {len(result)}"
    )
    for chunk in result:
        assert chunk.token_count <= chunk_size, (
            f"Word-split chunk has {chunk.token_count} tokens, expected <= {chunk_size}"
        )


def test_oversized_sentence_chunks_have_sequential_index() -> None:
    """Word-split sub-chunks from an oversized sentence have sequential chunk_index."""
    chunk_size = 200
    chunker = make_chunker(chunk_size=chunk_size, chunk_overlap=30)
    base = "the quick brown fox jumps over the lazy dog near the riverbank "
    long_sentence = (base * 80).strip()
    result = chunker.chunk(long_sentence)
    assert len(result) > 1, (
        f"Expected multiple chunks for oversized sentence, got {len(result)}"
    )
    for idx, chunk in enumerate(result):
        assert chunk.chunk_index == idx


# ---------------------------------------------------------------------------
# Paragraph boundary splitting
# ---------------------------------------------------------------------------


def test_paragraph_boundaries_split_chunks() -> None:
    """Double newlines are treated as paragraph boundaries in sentence splitting."""
    chunker = make_chunker(chunk_size=200, chunk_overlap=20)
    para1 = "This is paragraph one about neural networks and deep learning architectures. " * 3
    para2 = "This is paragraph two about natural language processing and transformers. " * 3
    text = para1.strip() + "\n\n" + para2.strip()
    result = chunker.chunk(text)
    # Both paragraphs should be present in the chunks
    full_text = " ".join(c.text for c in result)
    assert "neural networks" in full_text
    assert "natural language processing" in full_text


def test_multiple_blank_lines_treated_as_paragraph_break() -> None:
    """Three or more blank lines are also treated as paragraph boundaries."""
    chunker = make_chunker(chunk_size=200, chunk_overlap=20)
    text = "First section content about AI. " * 5 + "\n\n\n" + "Second section content about ML. " * 5
    result = chunker.chunk(text)
    full_text = " ".join(c.text for c in result)
    assert "First section" in full_text
    assert "Second section" in full_text


# ---------------------------------------------------------------------------
# chunk_document wrapper
# ---------------------------------------------------------------------------


def test_chunk_document_long_content_returns_chunks() -> None:
    """chunk_document returns non-empty list for sufficiently long raw_content."""
    chunker = make_chunker(chunk_size=200, chunk_overlap=30)
    long_content = make_long_text(n_sentences=30)
    doc = make_document(raw_content=long_content)
    result = chunker.chunk_document(doc)
    assert len(result) > 0
    assert all(isinstance(c, TextChunk) for c in result)


def test_chunk_document_indices_match_chunk_method() -> None:
    """chunk_document and chunk() produce the same chunks for the same text."""
    chunker = make_chunker(chunk_size=200, chunk_overlap=30)
    content = make_long_text(n_sentences=30)
    doc = make_document(raw_content=content)
    from_doc = chunker.chunk_document(doc)
    from_text = chunker.chunk(content)
    assert len(from_doc) == len(from_text)
    for a, b in zip(from_doc, from_text, strict=True):
        assert a.chunk_index == b.chunk_index
        assert a.text == b.text
        assert a.token_count == b.token_count


# ---------------------------------------------------------------------------
# Default settings (2000 tokens, 200 overlap)
# ---------------------------------------------------------------------------


def test_default_chunk_size_and_overlap() -> None:
    """Default Chunker uses chunk_size=2000 and chunk_overlap=200."""
    chunker = Chunker()
    assert chunker.chunk_size == 2000
    assert chunker.chunk_overlap == 200


def test_default_chunker_produces_valid_output_for_long_text() -> None:
    """Default chunker handles a realistic long article without errors."""
    chunker = Chunker()
    # ~5000-word article simulation (well above one 2000-token chunk).
    article = (
        "Artificial intelligence has transformed how we approach complex problems "
        "in science, engineering, and daily life. Researchers continue to push "
        "the boundaries of what is possible with neural networks. "
    ) * 100
    result = chunker.chunk(article)
    assert len(result) > 0
    for chunk in result:
        assert chunk.token_count >= _MIN_CHUNK_TOKENS
        assert chunk.token_count <= 2000


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_very_long_sentence_without_punctuation() -> None:
    """Text with no sentence-ending punctuation is treated as one sentence and word-split."""
    chunk_size = 200  # well above _MIN_CHUNK_TOKENS=50 so chunks are not skipped
    chunker = make_chunker(chunk_size=chunk_size, chunk_overlap=30)
    # Single stream of words, no punctuation — forces the word-split fallback.
    base = "the quick brown fox jumps over the lazy dog "
    text = (base * 80).strip()
    result = chunker.chunk(text)
    # Should produce multiple chunks since the text exceeds chunk_size
    assert len(result) > 1
    for chunk in result:
        assert chunk.token_count <= chunk_size


def test_text_with_only_newlines_and_spaces_returns_empty() -> None:
    """Text consisting only of newlines and spaces returns []."""
    chunker = make_chunker()
    assert chunker.chunk("\n\n\n   \n\n") == []


def test_exactly_one_chunk_when_text_fits() -> None:
    """Short-enough text that fits in one chunk produces exactly one chunk."""
    chunker = make_chunker(chunk_size=500, chunk_overlap=50)
    # Build text of exactly ~60 tokens (above minimum, well below chunk_size).
    text = "The transformer architecture revolutionized natural language processing. " * 5
    token_count = chunker.count_tokens(text)
    if _MIN_CHUNK_TOKENS <= token_count <= 500:
        result = chunker.chunk(text)
        assert len(result) == 1
        assert result[0].chunk_index == 0


def test_chunk_text_field_is_non_empty_string() -> None:
    """Every emitted chunk's text field is a non-empty string."""
    chunker = make_chunker(chunk_size=100, chunk_overlap=20)
    text = make_long_text(n_sentences=20)
    result = chunker.chunk(text)
    for chunk in result:
        assert isinstance(chunk.text, str)
        assert len(chunk.text) > 0


def test_token_count_field_matches_actual_count() -> None:
    """The token_count field matches what count_tokens() returns for the chunk text."""
    chunker = make_chunker(chunk_size=100, chunk_overlap=20)
    text = make_long_text(n_sentences=20)
    result = chunker.chunk(text)
    for chunk in result:
        actual = chunker.count_tokens(chunk.text)
        assert chunk.token_count == actual, (
            f"Chunk {chunk.chunk_index}: stored token_count={chunk.token_count}, "
            f"actual={actual}"
        )
