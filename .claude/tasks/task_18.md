# Task 18: Embedding Pipeline (OpenAI + Local)

## Wave
Wave 7 (parallel with tasks: 19, 22)
Domain: backend

## Objective
Implement the embedding pipeline that takes text chunks and returns embedding vectors using either OpenAI `text-embedding-3-small` or a local sentence-transformers model, routed by settings.yaml.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/processing/embedder.py` — `Embedder` class
- `backend/tests/test_processing/test_embedder.py`

### Key interfaces / implementation details:

**`Embedder`** (`processing/embedder.py`):
```python
class EmbeddingResult(BaseModel):
    text: str
    vector: list[float]
    token_count: int

class Embedder:
    """Embed text chunks via OpenAI or local sentence-transformers."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embedding_cfg = config.settings.embedding  # provider, model, chunk_size, chunk_overlap
        self._local_model = None  # lazy-loaded if provider == 'local'

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[EmbeddingResult]:
        """Embed a list of texts. Batches to avoid API limits.
        Returns one EmbeddingResult per input text."""

    async def embed_single(self, text: str) -> list[float]:
        """Convenience: embed one text, return vector only."""

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI Embeddings API.
        POST https://api.openai.com/v1/embeddings
        Body: {"model": "text-embedding-3-small", "input": [texts]}
        Response: {"data": [{"embedding": [...], "index": 0}]}"""

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Use sentence_transformers.SentenceTransformer(model).encode(texts).
        Run in executor to avoid blocking event loop."""
```

**OpenAI Embeddings API** (direct httpx, not openai SDK):
```
POST https://api.openai.com/v1/embeddings
Headers: Authorization: Bearer {api_key}
Body: {
  "model": "text-embedding-3-small",
  "input": ["text1", "text2", ...],
  "dimensions": 1536          // optional: reduce for cost savings
}
Response: {
  "data": [{"object":"embedding","embedding":[...],"index":0}, ...],
  "usage": {"prompt_tokens": N, "total_tokens": N}
}
```

**Configuration** (from settings.yaml):
- `embedding.provider`: `'openai'` or `'local'`
- `embedding.model`: `'text-embedding-3-small'` (OpenAI) or `'nomic-embed-text'` (local)
- `embedding.chunk_size`: `2000` tokens — used by chunker (task_19)
- `embedding.chunk_overlap`: `200` tokens — used by chunker (task_19)

**Dimensions**:
- OpenAI `text-embedding-3-small`: 1536
- Local `nomic-embed-text`: 768
- Local `all-MiniLM-L6-v2`: 384

**Token counting**: Use `tiktoken` with `cl100k_base` encoding for OpenAI models; use model's own tokenizer for local.

**Batch processing**: OpenAI allows up to 2048 inputs per request. Default batch_size=100 is conservative and safe.

## Dependencies
- Depends on: task_04 (LLMRouter pattern for API key access), task_02 (AppConfig)
- Packages needed: `openai` (for tiktoken), `sentence-transformers` (add to pyproject.toml as optional dep), `tiktoken`

## Acceptance Criteria
- [ ] OpenAI provider embeds texts and returns correctly shaped vectors (length 1536)
- [ ] Batch processing splits large lists into batches of `batch_size`
- [ ] `embed_single()` returns a flat `list[float]`
- [ ] Token count populated on each `EmbeddingResult`
- [ ] `provider = 'local'` falls back to sentence-transformers (lazy load, run in executor)
- [ ] Missing API key → raises clear error with provider name in message
- [ ] Unit tests mock the OpenAI HTTP endpoint; verify batch splitting

## Notes
- Use `httpx.AsyncClient` for OpenAI calls (same as other providers) — not the `openai` Python SDK
- `tiktoken` is used only for token counting, not for the API call itself
- Local model is loaded once (lazy singleton) — loading sentence-transformers can take 2–5 seconds
- OpenAI cost: `text-embedding-3-small` = $0.02 per million tokens ≈ $0.50/month at estimate
- Retry with exponential backoff on rate limit (HTTP 429) — reuse pattern from LLMRouter (task_04)
