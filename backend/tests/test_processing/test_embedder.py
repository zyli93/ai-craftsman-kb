"""Unit tests for the Embedder embedding pipeline.

All external HTTP calls to OpenAI are intercepted via pytest-httpx.
Local sentence-transformers model loading is mocked entirely.
No real network access or GPU required.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pytest_httpx import HTTPXMock

from ai_craftsman_kb.config.models import (
    AppConfig,
    EmbeddingConfig,
    FiltersConfig,
    LLMRoutingConfig,
    LLMTaskConfig,
    ProviderConfig,
    SettingsConfig,
    SourcesConfig,
)
from ai_craftsman_kb.processing.embedder import (
    _OPENAI_EMBEDDINGS_URL,
    Embedder,
    EmbeddingResult,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_openai_config(
    model: str = "text-embedding-3-small",
    api_key: str = "sk-test-key",
) -> AppConfig:
    """Build an AppConfig configured for the OpenAI embedding provider."""
    return AppConfig(
        sources=SourcesConfig(),
        settings=SettingsConfig(
            data_dir="/tmp/test-kb",
            embedding=EmbeddingConfig(provider="openai", model=model),
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test"),
                briefing=LLMTaskConfig(provider="anthropic", model="test"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test"),
            ),
            providers={"openai": ProviderConfig(api_key=api_key)},
        ),
        filters=FiltersConfig(),
    )


def _make_local_config(model: str = "all-MiniLM-L6-v2") -> AppConfig:
    """Build an AppConfig configured for the local embedding provider."""
    return AppConfig(
        sources=SourcesConfig(),
        settings=SettingsConfig(
            data_dir="/tmp/test-kb",
            embedding=EmbeddingConfig(provider="local", model=model),
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test"),
                briefing=LLMTaskConfig(provider="anthropic", model="test"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test"),
            ),
        ),
        filters=FiltersConfig(),
    )


def _openai_response_body(
    texts: list[str],
    dim: int = 1536,
) -> dict[str, Any]:
    """Build a fake OpenAI embeddings API response body."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1] * dim,
                "index": i,
            }
            for i in range(len(texts))
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 10 * len(texts), "total_tokens": 10 * len(texts)},
    }


# ---------------------------------------------------------------------------
# EmbeddingResult model
# ---------------------------------------------------------------------------


def test_embedding_result_fields() -> None:
    """EmbeddingResult stores text, vector, and token_count correctly."""
    result = EmbeddingResult(text="hello", vector=[0.1, 0.2], token_count=1)
    assert result.text == "hello"
    assert result.vector == [0.1, 0.2]
    assert result.token_count == 1


# ---------------------------------------------------------------------------
# OpenAI provider — embed_texts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_texts_openai_single_batch(httpx_mock: HTTPXMock) -> None:
    """embed_texts returns one EmbeddingResult per text for a small batch."""
    texts = ["alpha", "beta", "gamma"]
    config = _make_openai_config()
    embedder = Embedder(config)

    httpx_mock.add_response(
        url=_OPENAI_EMBEDDINGS_URL,
        method="POST",
        json=_openai_response_body(texts),
    )

    results = await embedder.embed_texts(texts)
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.text == texts[i]
        assert len(result.vector) == 1536
        assert result.token_count > 0


@pytest.mark.asyncio
async def test_embed_texts_openai_batch_splitting(httpx_mock: HTTPXMock) -> None:
    """embed_texts splits large lists into multiple API calls when batch_size is exceeded."""
    texts = [f"text {i}" for i in range(7)]
    config = _make_openai_config()
    embedder = Embedder(config)

    # batch_size=3 means 3 batches: [0-2], [3-5], [6]
    for chunk_start in range(0, len(texts), 3):
        chunk = texts[chunk_start : chunk_start + 3]
        httpx_mock.add_response(
            url=_OPENAI_EMBEDDINGS_URL,
            method="POST",
            json=_openai_response_body(chunk),
        )

    results = await embedder.embed_texts(texts, batch_size=3)

    assert len(results) == 7
    # Verify all original texts are preserved in order
    for i, result in enumerate(results):
        assert result.text == texts[i]


@pytest.mark.asyncio
async def test_embed_texts_empty_list() -> None:
    """embed_texts returns an empty list when given no texts."""
    config = _make_openai_config()
    embedder = Embedder(config)
    results = await embedder.embed_texts([])
    assert results == []


@pytest.mark.asyncio
async def test_embed_texts_sends_correct_request(httpx_mock: HTTPXMock) -> None:
    """embed_texts sends the correct JSON payload to the OpenAI API."""
    texts = ["hello world"]
    config = _make_openai_config(model="text-embedding-3-small", api_key="sk-my-key")
    embedder = Embedder(config)

    httpx_mock.add_response(
        url=_OPENAI_EMBEDDINGS_URL,
        method="POST",
        json=_openai_response_body(texts),
    )

    await embedder.embed_texts(texts)

    # Inspect the request that was sent
    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    req = requests[0]
    assert req.headers["Authorization"] == "Bearer sk-my-key"
    body = req.read()
    import json

    payload = json.loads(body)
    assert payload["model"] == "text-embedding-3-small"
    assert payload["input"] == texts


@pytest.mark.asyncio
async def test_embed_texts_openai_missing_api_key() -> None:
    """embed_texts raises ValueError when OpenAI API key is not configured."""
    config = AppConfig(
        sources=SourcesConfig(),
        settings=SettingsConfig(
            data_dir="/tmp/test-kb",
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test"),
                briefing=LLMTaskConfig(provider="anthropic", model="test"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test"),
            ),
            # No 'openai' entry in providers — no API key
        ),
        filters=FiltersConfig(),
    )
    embedder = Embedder(config)

    with pytest.raises(ValueError, match="openai"):
        await embedder.embed_texts(["test"])


# ---------------------------------------------------------------------------
# embed_single
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_single_returns_flat_vector(httpx_mock: HTTPXMock) -> None:
    """embed_single returns a plain list[float], not a list of EmbeddingResults."""
    config = _make_openai_config()
    embedder = Embedder(config)

    httpx_mock.add_response(
        url=_OPENAI_EMBEDDINGS_URL,
        method="POST",
        json=_openai_response_body(["hello"]),
    )

    vector = await embedder.embed_single("hello")
    assert isinstance(vector, list)
    assert len(vector) == 1536
    assert all(isinstance(v, float) for v in vector)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_count_populated(httpx_mock: HTTPXMock) -> None:
    """token_count is populated on each EmbeddingResult for OpenAI provider."""
    texts = ["The quick brown fox jumps over the lazy dog."]
    config = _make_openai_config()
    embedder = Embedder(config)

    httpx_mock.add_response(
        url=_OPENAI_EMBEDDINGS_URL,
        method="POST",
        json=_openai_response_body(texts),
    )

    results = await embedder.embed_texts(texts)
    # tiktoken encodes this sentence to roughly 9 tokens
    assert results[0].token_count > 0
    assert isinstance(results[0].token_count, int)


# ---------------------------------------------------------------------------
# Local provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_texts_local_provider() -> None:
    """embed_texts uses local sentence-transformers model when provider='local'."""
    texts = ["hello", "world"]
    config = _make_local_config(model="all-MiniLM-L6-v2")
    embedder = Embedder(config)

    # Mock the SentenceTransformer class so we don't actually load a model
    fake_vectors = [[0.5] * 384, [0.3] * 384]

    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array(fake_vectors)

    with patch(
        "ai_craftsman_kb.processing.embedder.Embedder._embed_local",
        return_value=fake_vectors,
    ):
        results = await embedder.embed_texts(texts)

    assert len(results) == 2
    for result in results:
        assert len(result.vector) == 384
        assert result.token_count > 0


@pytest.mark.asyncio
async def test_embed_local_lazy_load() -> None:
    """Local model is loaded lazily on first call and cached as a singleton."""
    import sys
    import types

    import numpy as np

    config = _make_local_config()
    embedder = Embedder(config)

    assert embedder._local_model is None

    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1] * 384])
    fake_model.tokenize.return_value = {}

    # Build a fake sentence_transformers module so the local import inside
    # _embed_local succeeds without the package being installed.
    fake_st_module = types.ModuleType("sentence_transformers")
    mock_cls = MagicMock(return_value=fake_model)
    fake_st_module.SentenceTransformer = mock_cls  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"sentence_transformers": fake_st_module}):
        # Run synchronously (simulating what run_in_executor does)
        embedder._embed_local(["hello"])
        # Model should now be cached
        assert embedder._local_model is not None
        # Call again — should reuse cached model (no second instantiation)
        embedder._embed_local(["world"])
        mock_cls.assert_called_once()


@pytest.mark.asyncio
async def test_embed_local_import_error() -> None:
    """_embed_local raises ImportError with helpful message when sentence-transformers is missing."""
    config = _make_local_config()
    embedder = Embedder(config)

    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            embedder._embed_local(["test"])


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_texts_unknown_provider() -> None:
    """embed_texts raises ValueError for unrecognised provider names."""
    config = AppConfig(
        sources=SourcesConfig(),
        settings=SettingsConfig(
            data_dir="/tmp/test-kb",
            embedding=EmbeddingConfig(provider="unknown_provider", model="some-model"),
            llm=LLMRoutingConfig(
                filtering=LLMTaskConfig(provider="openrouter", model="test"),
                entity_extraction=LLMTaskConfig(provider="openrouter", model="test"),
                briefing=LLMTaskConfig(provider="anthropic", model="test"),
                source_discovery=LLMTaskConfig(provider="openrouter", model="test"),
            ),
        ),
        filters=FiltersConfig(),
    )
    embedder = Embedder(config)

    with pytest.raises(ValueError, match="unknown_provider"):
        await embedder.embed_texts(["test"])


# ---------------------------------------------------------------------------
# Dimension helper
# ---------------------------------------------------------------------------


def test_get_embedding_dimension_known_models() -> None:
    """get_embedding_dimension returns the correct dimension for known models."""
    config = _make_openai_config(model="text-embedding-3-small")
    embedder = Embedder(config)
    assert embedder.get_embedding_dimension() == 1536


def test_get_embedding_dimension_unknown_model() -> None:
    """get_embedding_dimension returns None for unrecognised models."""
    config = _make_openai_config(model="some-custom-model-v99")
    embedder = Embedder(config)
    assert embedder.get_embedding_dimension() is None


def test_get_embedding_dimension_local_minilm() -> None:
    """get_embedding_dimension returns 384 for all-MiniLM-L6-v2."""
    config = _make_local_config(model="all-MiniLM-L6-v2")
    embedder = Embedder(config)
    assert embedder.get_embedding_dimension() == 384


# ---------------------------------------------------------------------------
# Retry on rate limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_openai_retries_on_429(httpx_mock: HTTPXMock) -> None:
    """OpenAI embed retries on HTTP 429 rate limit responses."""
    texts = ["hello"]
    config = _make_openai_config()
    embedder = Embedder(config)

    # First request gets a 429, second succeeds
    httpx_mock.add_response(
        url=_OPENAI_EMBEDDINGS_URL,
        method="POST",
        status_code=429,
        text="rate limited",
    )
    httpx_mock.add_response(
        url=_OPENAI_EMBEDDINGS_URL,
        method="POST",
        json=_openai_response_body(texts),
    )

    # Patch asyncio.sleep to avoid waiting in tests
    with patch("asyncio.sleep"):
        results = await embedder.embed_texts(texts)

    assert len(results) == 1
    assert len(results[0].vector) == 1536


@pytest.mark.asyncio
async def test_embed_openai_raises_after_max_retries(httpx_mock: HTTPXMock) -> None:
    """OpenAI embed raises after exhausting retries on persistent rate limits."""
    texts = ["hello"]
    config = _make_openai_config()
    embedder = Embedder(config)

    # All 3 attempts return 429
    for _ in range(3):
        httpx_mock.add_response(
            url=_OPENAI_EMBEDDINGS_URL,
            method="POST",
            status_code=429,
            text="rate limited",
        )

    with patch("asyncio.sleep"):
        with pytest.raises(Exception):
            await embedder.embed_texts(texts)


# ---------------------------------------------------------------------------
# Order preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_texts_preserves_order_with_out_of_order_response(
    httpx_mock: HTTPXMock,
) -> None:
    """embed_texts preserves input order even when API returns embeddings out of order."""
    texts = ["first", "second", "third"]
    config = _make_openai_config()
    embedder = Embedder(config)

    # Return embeddings in reverse order (index 2, 1, 0) with distinct vectors
    reversed_response = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.3] * 1536, "index": 2},
            {"object": "embedding", "embedding": [0.2] * 1536, "index": 1},
            {"object": "embedding", "embedding": [0.1] * 1536, "index": 0},
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }

    httpx_mock.add_response(
        url=_OPENAI_EMBEDDINGS_URL,
        method="POST",
        json=reversed_response,
    )

    results = await embedder.embed_texts(texts)

    assert len(results) == 3
    # Even though API returned in reverse index order, we should get index 0 first
    assert results[0].vector[0] == pytest.approx(0.1)
    assert results[1].vector[0] == pytest.approx(0.2)
    assert results[2].vector[0] == pytest.approx(0.3)
    assert results[0].text == "first"
    assert results[1].text == "second"
    assert results[2].text == "third"
