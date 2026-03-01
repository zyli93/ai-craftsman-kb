# Task 19: Chunking System

## Wave
Wave 7 (parallel with tasks: 18, 22)
Domain: backend

## Objective
Implement the text chunking system that splits `raw_content` into overlapping token-bounded chunks ready for embedding, respecting sentence boundaries where possible.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/processing/chunker.py` — `Chunker` class
- `backend/tests/test_processing/test_chunker.py`

### Key interfaces / implementation details:

**Configuration** (from settings.yaml):
- `embedding.chunk_size`: 2000 tokens (max tokens per chunk)
- `embedding.chunk_overlap`: 200 tokens (overlap between consecutive chunks)

**`Chunker`** (`processing/chunker.py`):
```python
class TextChunk(BaseModel):
    chunk_index: int
    text: str
    token_count: int
    char_start: int    # character offset in original text
    char_end: int

class Chunker:
    """Split text into overlapping token-bounded chunks."""

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = tiktoken.get_encoding('cl100k_base')

    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into chunks.
        Algorithm:
        1. Split text into sentences (on '. ', '! ', '? ', '\n\n')
        2. Greedily add sentences to current chunk until token limit
        3. When limit reached: close chunk, start new chunk overlapping
           by `chunk_overlap` tokens from end of previous chunk
        4. Minimum chunk size: 50 tokens (skip trivially small chunks)
        Returns list of TextChunk with sequential chunk_index."""

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoder."""

    def chunk_document(self, doc: DocumentRow) -> list[TextChunk]:
        """Convenience: chunk a DocumentRow's raw_content.
        Returns [] if raw_content is None or too short (<50 tokens)."""
```

**Chunking algorithm**:
```
sentences = split_into_sentences(text)
chunks = []
current_tokens = 0
current_sentences = []

for sentence in sentences:
    sent_tokens = count_tokens(sentence)
    if current_tokens + sent_tokens > chunk_size and current_sentences:
        # emit chunk
        chunks.append(join(current_sentences))
        # slide window: keep overlap worth of sentences from end
        current_sentences = trim_to_overlap_tokens(current_sentences, chunk_overlap)
        current_tokens = count_tokens(join(current_sentences))
    current_sentences.append(sentence)
    current_tokens += sent_tokens

if current_sentences:
    chunks.append(join(current_sentences))  # final chunk
```

**Sentence splitting**: Use regex `re.split(r'(?<=[.!?])\s+', text)` as baseline. Also split on double newlines (`\n\n`) for paragraph boundaries.

**Edge cases**:
- Single sentence longer than `chunk_size`: split by words instead
- Text shorter than minimum chunk size (50 tokens): return single chunk
- Empty or None text: return `[]`

## Dependencies
- Depends on: task_01 (pyproject.toml for tiktoken)
- Packages needed: `tiktoken` (already in pyproject.toml)

## Acceptance Criteria
- [ ] Chunks never exceed `chunk_size` tokens
- [ ] Consecutive chunks overlap by approximately `chunk_overlap` tokens
- [ ] `chunk_index` is sequential starting at 0
- [ ] `char_start` and `char_end` correctly map back to original text positions
- [ ] Chunks shorter than 50 tokens are not emitted (skipped)
- [ ] Empty/None input returns `[]`
- [ ] Sentences are not split mid-word (only on sentence/paragraph boundaries)
- [ ] Unit tests: known text → verify chunk count, overlap, token counts

## Notes
- Use `tiktoken.get_encoding('cl100k_base')` — same tokenizer as OpenAI embedding models
- The `char_start`/`char_end` fields enable mapping search hits back to document positions
- For YouTube transcripts (already plain text, no punctuation): the paragraph split (`\n\n`) is more useful
- Chunk overlap is important for semantic continuity — a sentence about topic X may span a chunk boundary
- Token counting with tiktoken is fast (< 1ms for typical chunks) — don't optimize prematurely
