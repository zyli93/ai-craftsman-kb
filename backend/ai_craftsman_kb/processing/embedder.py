"""Embedding pipeline for AI Craftsman KB.

Supports two providers:
- OpenAI ``text-embedding-3-small`` via direct httpx calls (not the openai SDK).
- Local sentence-transformers models (lazy-loaded, run in a thread executor).

Provider selection is controlled by ``settings.embedding.provider`` in settings.yaml.
Batch processing respects OpenAI's 2048-input limit. Token counting uses tiktoken
for OpenAI models and the model's own tokenizer for local models.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx
import tiktoken
from pydantic import BaseModel

from ..llm.retry import with_retry

if TYPE_CHECKING:
    from ..config.models import AppConfig

logger = logging.getLogger(__name__)

# Dimensions for known embedding models
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
}

# OpenAI Embeddings API endpoint
_OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

# Tiktoken encoding used for OpenAI embedding models
_TIKTOKEN_ENCODING = "cl100k_base"


class EmbeddingResult(BaseModel):
    """Result of embedding a single text.

    Attributes:
        text: The original input text.
        vector: The embedding vector as a list of floats.
        token_count: Number of tokens in the input text (estimated for local models).
    """

    text: str
    vector: list[float]
    token_count: int


class Embedder:
    """Embed text chunks via OpenAI or local sentence-transformers.

    Provider selection is driven by ``config.settings.embedding.provider``:

    - ``'openai'``: Calls the OpenAI Embeddings API directly via httpx with
      exponential-backoff retry on rate limits and 5xx errors.
    - ``'local'``: Loads a sentence-transformers model on first use and runs
      inference in a thread executor to avoid blocking the event loop.

    Usage::

        embedder = Embedder(config)
        results = await embedder.embed_texts(["chunk1", "chunk2"])
        vector = await embedder.embed_single("query text")

    Args:
        config: Fully loaded AppConfig. The ``settings.embedding`` section is
            used to select provider, model name, and chunking parameters.
    """

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        self.embedding_cfg = config.settings.embedding
        # Lazy-loaded sentence-transformers model (only when provider == 'local')
        self._local_model: object | None = None
        # Cached tiktoken encoding for token counting
        self._encoding: tiktoken.Encoding | None = None

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[EmbeddingResult]:
        """Embed a list of texts, batching to stay within API limits.

        Splits ``texts`` into batches of at most ``batch_size`` items and
        calls the configured provider for each batch. Results are returned
        in the same order as the input texts.

        Args:
            texts: List of strings to embed.
            batch_size: Maximum number of texts per API call (default 100;
                OpenAI's hard limit is 2048).

        Returns:
            A list of EmbeddingResult objects, one per input text, in input order.

        Raises:
            ValueError: If no texts are provided.
        """
        if not texts:
            return []

        provider = self.embedding_cfg.provider
        all_vectors: list[list[float]] = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(
                "Embedding batch %d-%d of %d texts via '%s'",
                i,
                i + len(batch),
                len(texts),
                provider,
            )

            if provider == "openai":
                vectors = await self._embed_openai(batch)
            elif provider == "local":
                vectors = await asyncio.get_event_loop().run_in_executor(
                    None, self._embed_local, batch
                )
            else:
                raise ValueError(
                    f"Unknown embedding provider: '{provider}'. "
                    "Supported providers: openai, local."
                )

            all_vectors.extend(vectors)

        # Build EmbeddingResult objects with token counts
        results: list[EmbeddingResult] = []
        for text, vector in zip(texts, all_vectors, strict=True):
            token_count = self._count_tokens(text)
            results.append(
                EmbeddingResult(text=text, vector=vector, token_count=token_count)
            )

        return results

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text and return only the vector.

        Convenience wrapper around ``embed_texts`` for use cases where only
        the vector is needed (e.g., query-time search).

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a flat list of floats.
        """
        results = await self.embed_texts([text])
        return results[0].vector

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI Embeddings API via httpx.

        Uses direct HTTP calls (not the openai SDK) consistent with the rest
        of the codebase. Retries automatically on rate limits (HTTP 429) and
        transient server errors via ``with_retry``.

        API contract::

            POST https://api.openai.com/v1/embeddings
            {
                "model": "text-embedding-3-small",
                "input": ["text1", "text2", ...]
            }
            Response: {
                "data": [{"embedding": [...], "index": 0}, ...],
                "usage": {"prompt_tokens": N, "total_tokens": N}
            }

        Args:
            texts: Batch of texts to embed (max 2048 per OpenAI's limits).

        Returns:
            List of embedding vectors in the same order as ``texts``.

        Raises:
            ValueError: If the OpenAI API key is not configured.
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        providers_cfg = self.config.settings.providers
        openai_cfg = providers_cfg.get("openai")
        api_key = openai_cfg.api_key if openai_cfg else None

        if not api_key:
            raise ValueError(
                "OpenAI API key not configured for embedding provider 'openai'. "
                "Set OPENAI_API_KEY or add it to config/settings.yaml."
            )

        model = self.embedding_cfg.model

        async def _call() -> list[list[float]]:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    _OPENAI_EMBEDDINGS_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()
                # Sort by index to guarantee input order (OpenAI does not always return in order)
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]

        return await with_retry(
            _call, operation_name=f"OpenAI embed [{model}]"
        )

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using a local sentence-transformers model.

        The model is loaded lazily on first call and cached as a singleton to
        avoid the 2-5 second startup overhead on subsequent calls.

        This is a synchronous method designed to be called via
        ``asyncio.run_in_executor`` to prevent blocking the event loop during
        model inference.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors as Python lists of floats.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the 'local' embedding provider. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        if self._local_model is None:
            model_name = self.embedding_cfg.model
            logger.info("Loading local embedding model '%s' (may take a few seconds)...", model_name)
            self._local_model = SentenceTransformer(model_name)
            logger.info("Local embedding model '%s' loaded.", model_name)

        model: SentenceTransformer = self._local_model  # type: ignore[assignment]
        # encode() returns a numpy array; convert rows to plain Python lists
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [list(map(float, vec)) for vec in embeddings]

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Uses tiktoken's ``cl100k_base`` encoding for OpenAI models.
        For local models, falls back to a whitespace-based word count
        approximation since the model-specific tokenizers are not
        always available.

        Args:
            text: The text to tokenize.

        Returns:
            Estimated token count.
        """
        provider = self.embedding_cfg.provider

        if provider == "openai":
            if self._encoding is None:
                self._encoding = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
            return len(self._encoding.encode(text))

        # Local model fallback: use the model tokenizer if available,
        # otherwise estimate with word count (roughly 0.75 tokens/word)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            if self._local_model is not None:
                model: SentenceTransformer = self._local_model  # type: ignore[assignment]
                tokenized = model.tokenize([text])
                # SentenceTransformer returns a dict with 'input_ids' tensor
                if "input_ids" in tokenized:
                    return int(tokenized["input_ids"].shape[1])
        except (ImportError, Exception):
            pass

        # Simple approximation: split on whitespace
        return len(text.split())

    def get_embedding_dimension(self) -> int | None:
        """Return the expected embedding dimension for the configured model.

        Looks up the model name in a table of known dimensions. Returns None
        if the model is not recognized (e.g., a custom local model).

        Returns:
            Integer dimension, or None if unknown.
        """
        return _MODEL_DIMENSIONS.get(self.embedding_cfg.model)
