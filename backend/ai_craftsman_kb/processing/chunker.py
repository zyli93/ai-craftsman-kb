"""Text chunking system for AI Craftsman KB.

Splits raw document text into overlapping token-bounded chunks suitable for
embedding. Respects sentence and paragraph boundaries where possible, with
fallback to word-level splitting for sentences that exceed the chunk size.
"""
from __future__ import annotations

import logging
import re

import tiktoken
from pydantic import BaseModel

from ai_craftsman_kb.db.models import DocumentRow

logger = logging.getLogger(__name__)

# Minimum number of tokens a chunk must contain to be emitted.
_MIN_CHUNK_TOKENS = 50


class TextChunk(BaseModel):
    """A single chunk of text produced by the Chunker.

    Attributes:
        chunk_index: Zero-based sequential index within the parent document.
        text: The chunk's text content.
        token_count: Number of tokens in this chunk (tiktoken cl100k_base).
        char_start: Character offset of this chunk's start in the original text.
        char_end: Character offset of this chunk's end in the original text (exclusive).
    """

    chunk_index: int
    text: str
    token_count: int
    char_start: int
    char_end: int


class Chunker:
    """Split text into overlapping token-bounded chunks.

    Uses tiktoken's cl100k_base encoder (same as OpenAI embedding models) to
    count tokens accurately. Splits on sentence/paragraph boundaries first,
    falling back to word-level splitting for oversized sentences.

    Args:
        chunk_size: Maximum number of tokens per chunk (default 2000).
        chunk_overlap: Number of tokens to overlap between consecutive chunks
            (default 200). Overlap preserves semantic continuity across boundaries.
    """

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200) -> None:
        """Initialise the chunker with encoding and size parameters.

        Args:
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Tokens of overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # cl100k_base is the tokenizer used by OpenAI text-embedding-* models.
        self._encoder = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens in *text* using the tiktoken cl100k_base encoder.

        Args:
            text: The text to count tokens for.

        Returns:
            Integer token count.
        """
        return len(self._encoder.encode(text))

    def chunk(self, text: str) -> list[TextChunk]:
        """Split *text* into overlapping token-bounded chunks.

        Algorithm:
        1. Split text into sentences on ``. ``, ``! ``, ``? ``, and ``\\n\\n``.
        2. Greedily accumulate sentences into the current chunk until the token
           limit is reached.
        3. When the limit is reached: emit the current chunk, then seed the
           next chunk with enough trailing sentences to cover ``chunk_overlap``
           tokens.
        4. Chunks shorter than ``_MIN_CHUNK_TOKENS`` (50) are skipped.

        Individual sentences that exceed ``chunk_size`` are split by words
        rather than emitted as a single oversized chunk.

        Args:
            text: The input text to chunk.

        Returns:
            List of :class:`TextChunk` with sequential ``chunk_index`` values
            starting at 0. Returns ``[]`` for empty or ``None``-like input.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        raw_chunks: list[str] = []  # plain text of each chunk before metadata

        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            # Handle sentences that are themselves larger than chunk_size.
            if sent_tokens > self.chunk_size:
                # First, emit whatever has accumulated so far.
                if current_sentences:
                    raw_chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                # Break the oversized sentence into word-level sub-chunks.
                word_chunks = self._split_by_words(sentence, self.chunk_size)
                raw_chunks.extend(word_chunks)
                continue

            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                # Emit the current accumulation.
                raw_chunks.append(" ".join(current_sentences))
                # Seed the next chunk with overlap from the end.
                current_sentences = self._trim_to_overlap(
                    current_sentences, self.chunk_overlap
                )
                current_tokens = self.count_tokens(" ".join(current_sentences))

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Emit any remaining sentences as the final chunk.
        if current_sentences:
            raw_chunks.append(" ".join(current_sentences))

        # Build TextChunk objects, skipping trivially short chunks and computing
        # character offsets by searching for each chunk in the original text.
        result: list[TextChunk] = []
        chunk_index = 0
        search_start = 0  # advance search position to avoid re-matching earlier chunks

        for raw_text in raw_chunks:
            token_count = self.count_tokens(raw_text)
            if token_count < _MIN_CHUNK_TOKENS:
                logger.debug(
                    "Skipping chunk with %d tokens (below minimum %d)",
                    token_count,
                    _MIN_CHUNK_TOKENS,
                )
                continue

            # Locate the chunk within the original text.
            char_start, char_end = self._find_char_offsets(
                text, raw_text, search_start
            )
            # Advance search cursor to ensure monotonically increasing offsets.
            # The next chunk starts at or after where this one was found.
            search_start = max(search_start, char_start)

            result.append(
                TextChunk(
                    chunk_index=chunk_index,
                    text=raw_text,
                    token_count=token_count,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            chunk_index += 1

        if result:
            total_tokens = sum(c.token_count for c in result)
            logger.debug(
                "Chunked text into %d chunks (%d total tokens, avg %d tokens/chunk)",
                len(result),
                total_tokens,
                total_tokens // len(result),
            )
        else:
            logger.debug(
                "Chunking produced 0 chunks from %d raw segments (all below %d token minimum)",
                len(raw_chunks),
                _MIN_CHUNK_TOKENS,
            )

        return result

    def chunk_document(self, doc: DocumentRow) -> list[TextChunk]:
        """Convenience wrapper: chunk a :class:`DocumentRow`'s ``raw_content``.

        Args:
            doc: A :class:`DocumentRow` whose ``raw_content`` field will be
                chunked.

        Returns:
            List of :class:`TextChunk`. Returns ``[]`` if ``raw_content`` is
            ``None`` or contains fewer than ``_MIN_CHUNK_TOKENS`` tokens.
        """
        if not doc.raw_content:
            return []
        if self.count_tokens(doc.raw_content) < _MIN_CHUNK_TOKENS:
            logger.debug(
                "Document %s raw_content too short to chunk (< %d tokens)",
                doc.id,
                _MIN_CHUNK_TOKENS,
            )
            return []
        return self.chunk(doc.raw_content)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split *text* into sentence-like segments.

        Splits on:
        - Sentence-ending punctuation followed by whitespace (``[.!?]\\s+``).
        - Paragraph boundaries (``\\n\\n``).

        Empty segments are discarded.

        Args:
            text: Text to split.

        Returns:
            List of non-empty sentence strings.
        """
        # First split on paragraph boundaries.
        paragraphs = re.split(r"\n\n+", text)
        sentences: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Within each paragraph, split on sentence-ending punctuation.
            parts = re.split(r"(?<=[.!?])\s+", para)
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)
        return sentences

    def _trim_to_overlap(self, sentences: list[str], overlap_tokens: int) -> list[str]:
        """Return the trailing sentences whose combined tokens <= *overlap_tokens*.

        Builds the overlap window from the *end* of *sentences*, accumulating
        backwards until the next sentence would exceed the budget.

        Args:
            sentences: The current sentence accumulation.
            overlap_tokens: Token budget for the overlap window.

        Returns:
            A (possibly empty) list of sentences forming the overlap seed.
        """
        overlap: list[str] = []
        tokens_so_far = 0
        for sentence in reversed(sentences):
            sent_tokens = self.count_tokens(sentence)
            if tokens_so_far + sent_tokens > overlap_tokens:
                break
            overlap.insert(0, sentence)
            tokens_so_far += sent_tokens
        return overlap

    def _split_by_words(self, text: str, max_tokens: int) -> list[str]:
        """Split *text* into word-level sub-chunks not exceeding *max_tokens*.

        Used as a fallback when a single sentence is longer than chunk_size.

        Args:
            text: The oversized sentence to split.
            max_tokens: Maximum tokens per sub-chunk.

        Returns:
            List of text sub-chunks.
        """
        words = text.split()
        chunks: list[str] = []
        current_words: list[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = self.count_tokens(word)
            if current_tokens + word_tokens > max_tokens and current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_tokens = 0
            current_words.append(word)
            current_tokens += word_tokens

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    def _find_char_offsets(
        self, original: str, chunk_text: str, search_start: int
    ) -> tuple[int, int]:
        """Find the character start/end of *chunk_text* within *original*.

        Searches starting from *search_start* to handle overlapping chunks
        that share the same prefix text.

        If the chunk text cannot be found (e.g., because whitespace was
        normalised during joining), falls back to a full-string search from
        position 0 and returns ``(0, len(chunk_text))`` as a last resort.

        Args:
            original: The full original text.
            chunk_text: The chunk text to locate.
            search_start: Character index in *original* to start searching from.

        Returns:
            ``(char_start, char_end)`` tuple.
        """
        pos = original.find(chunk_text, search_start)
        if pos != -1:
            return pos, pos + len(chunk_text)

        # Fallback: search from the beginning (handles edge cases with
        # sentence-joining whitespace normalisation).
        pos = original.find(chunk_text)
        if pos != -1:
            return pos, pos + len(chunk_text)

        # Last resort: if the chunk cannot be located in the original text
        # (e.g., due to whitespace collapsing during sentence join), return
        # approximate positions based on search_start.
        logger.warning(
            "Could not locate chunk text in original; using approximate offsets"
        )
        return search_start, search_start + len(chunk_text)
