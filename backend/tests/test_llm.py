"""Tests for the LLM provider abstraction layer.

All external API calls are mocked; no network access is required.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ai_craftsman_kb.config.models import (
    EndpointConfig,
    LLMGatewayConfig,
    PoolConfig,
    SettingsConfig,
)
from ai_craftsman_kb.llm import (
    AllEndpointsExhausted,
    AsyncRateLimiter,
    CompletionResult,
    EndpointPool,
    LLMProvider,
    LLMRouter,
)
from ai_craftsman_kb.llm.anthropic_provider import AnthropicProvider
from ai_craftsman_kb.llm.gateway import ManagedEndpoint
from ai_craftsman_kb.llm.ollama_provider import OllamaProvider
from ai_craftsman_kb.llm.openai_provider import OpenAIProvider
from ai_craftsman_kb.llm.openrouter_provider import OpenRouterProvider
from ai_craftsman_kb.llm.rate_limiter import DailyLimitExceeded
from ai_craftsman_kb.llm.retry import _is_retryable_error, _parse_retry_after, with_retry


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_app_config(
    *,
    openai_key: str | None = "sk-test",
    openrouter_key: str | None = "or-test",
    anthropic_key: str | None = "ant-test",
    ollama_base_url: str = "http://localhost:11434",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
) -> MagicMock:
    """Build a minimal AppConfig mock suitable for LLMRouter tests."""
    config = MagicMock()

    # Provider configs
    def _provider_cfg(key: str | None, base_url: str | None = None) -> MagicMock:
        m = MagicMock()
        m.api_key = key
        m.base_url = base_url
        return m

    config.settings.providers = {
        "openai": _provider_cfg(openai_key),
        "openrouter": _provider_cfg(openrouter_key),
        "anthropic": _provider_cfg(anthropic_key),
        "ollama": _provider_cfg(None, ollama_base_url),
    }

    # LLM task routing
    def _task_cfg(
        provider: str,
        model: str,
        rate_limit: float | None = None,
        daily_limit: int | None = None,
        max_retries: int = 3,
    ) -> MagicMock:
        m = MagicMock()
        m.provider = provider
        m.model = model
        m.rate_limit = rate_limit
        m.daily_limit = daily_limit
        m.max_retries = max_retries
        return m

    config.settings.llm.filtering = _task_cfg("openrouter", "meta-llama/llama-3.1-8b-instruct")
    config.settings.llm.entity_extraction = _task_cfg(
        "openrouter", "meta-llama/llama-3.1-8b-instruct"
    )
    config.settings.llm.briefing = _task_cfg("anthropic", "claude-haiku-4-5-20251001")
    config.settings.llm.source_discovery = _task_cfg(
        "openrouter", "meta-llama/llama-3.1-8b-instruct"
    )
    config.settings.llm.keyword_extraction = _task_cfg(
        "openrouter", "meta-llama/llama-3.1-8b-instruct"
    )

    # Embedding config
    config.settings.embedding.provider = embedding_provider
    config.settings.embedding.model = embedding_model

    return config


# ---------------------------------------------------------------------------
# Abstract base class tests
# ---------------------------------------------------------------------------


def test_llm_provider_is_abstract() -> None:
    """LLMProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# OpenAI provider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_provider_complete() -> None:
    """OpenAI provider complete() formats messages and returns content."""
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")

    # Build a mock response matching the openai SDK structure
    mock_message = MagicMock()
    mock_message.content = "Hello, world!"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await provider.complete("Say hello", system="You are helpful.")

    assert result.text == "Hello, world!"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.model == "gpt-4o-mini"
    provider._client.chat.completions.create.assert_called_once()
    call_kwargs = provider._client.chat.completions.create.call_args
    messages = call_kwargs.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "user", "content": "Say hello"}


@pytest.mark.asyncio
async def test_openai_provider_complete_no_system() -> None:
    """OpenAI provider complete() without system message sends only user message."""
    provider = OpenAIProvider(api_key="sk-test")

    mock_message = MagicMock()
    mock_message.content = "Response"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 5
    mock_usage.completion_tokens = 3
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

    await provider.complete("Hello")

    call_kwargs = provider._client.chat.completions.create.call_args
    messages = call_kwargs.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


@pytest.mark.asyncio
async def test_openai_provider_embed() -> None:
    """OpenAI provider embed() returns embedding vectors in input order."""
    provider = OpenAIProvider(
        api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )

    vec1 = [0.1, 0.2, 0.3]
    vec2 = [0.4, 0.5, 0.6]

    item1 = MagicMock()
    item1.embedding = vec1
    item2 = MagicMock()
    item2.embedding = vec2
    mock_response = MagicMock()
    mock_response.data = [item1, item2]

    provider._client.embeddings.create = AsyncMock(return_value=mock_response)

    result = await provider.embed(["text one", "text two"])

    assert result == [vec1, vec2]
    provider._client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["text one", "text two"],
    )


# ---------------------------------------------------------------------------
# OpenRouter provider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_provider_complete() -> None:
    """OpenRouter provider complete() calls the correct endpoint and returns content."""
    provider = OpenRouterProvider(
        api_key="or-test",
        model="meta-llama/llama-3.1-8b-instruct",
    )

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "OpenRouter reply"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }
    mock_resp.raise_for_status = MagicMock()

    provider._client.post = AsyncMock(return_value=mock_resp)

    result = await provider.complete("Summarise this", system="Be concise.")

    assert result.text == "OpenRouter reply"
    assert result.input_tokens == 12
    assert result.output_tokens == 8
    provider._client.post.assert_called_once()
    call_args = provider._client.post.call_args
    assert call_args.args[0] == "/chat/completions"
    payload = call_args.kwargs["json"]
    assert payload["model"] == "meta-llama/llama-3.1-8b-instruct"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_openrouter_provider_embed_raises() -> None:
    """OpenRouter provider embed() raises NotImplementedError."""
    provider = OpenRouterProvider(api_key="or-test", model="some-model")
    with pytest.raises(NotImplementedError):
        await provider.embed(["text"])


# ---------------------------------------------------------------------------
# Anthropic provider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_provider_complete() -> None:
    """Anthropic provider complete() calls /v1/messages and returns text."""
    provider = AnthropicProvider(api_key="ant-test", model="claude-haiku-4-5-20251001")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "Anthropic reply"}],
        "usage": {"input_tokens": 15, "output_tokens": 10},
    }
    mock_resp.raise_for_status = MagicMock()

    provider._client.post = AsyncMock(return_value=mock_resp)

    result = await provider.complete("Draft a summary", system="You are an expert.")

    assert result.text == "Anthropic reply"
    assert result.input_tokens == 15
    assert result.output_tokens == 10
    provider._client.post.assert_called_once()
    call_args = provider._client.post.call_args
    assert call_args.args[0] == "/v1/messages"
    payload = call_args.kwargs["json"]
    assert payload["model"] == "claude-haiku-4-5-20251001"
    assert payload["system"] == "You are an expert."
    assert payload["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_anthropic_provider_complete_respects_max_tokens() -> None:
    """Anthropic provider complete() passes max_tokens kwarg through."""
    provider = AnthropicProvider(api_key="ant-test")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    mock_resp.raise_for_status = MagicMock()

    provider._client.post = AsyncMock(return_value=mock_resp)

    await provider.complete("Hello", max_tokens=512)

    payload = provider._client.post.call_args.kwargs["json"]
    assert payload["max_tokens"] == 512


@pytest.mark.asyncio
async def test_anthropic_provider_embed_raises() -> None:
    """Anthropic provider embed() raises NotImplementedError."""
    provider = AnthropicProvider(api_key="ant-test")
    with pytest.raises(NotImplementedError):
        await provider.embed(["text"])


# ---------------------------------------------------------------------------
# Ollama provider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_provider_complete() -> None:
    """Ollama provider complete() calls /api/chat and returns message content."""
    provider = OllamaProvider(model="llama3")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {
        "message": {"content": "Ollama reply"},
        "prompt_eval_count": 7,
        "eval_count": 4,
    }
    mock_resp.raise_for_status = MagicMock()

    provider._client.post = AsyncMock(return_value=mock_resp)

    result = await provider.complete("Hello")

    assert result.text == "Ollama reply"
    assert result.input_tokens == 7
    assert result.output_tokens == 4
    call_args = provider._client.post.call_args
    assert call_args.args[0] == "/api/chat"
    payload = call_args.kwargs["json"]
    assert payload["stream"] is False


@pytest.mark.asyncio
async def test_ollama_provider_embed() -> None:
    """Ollama provider embed() calls /api/embeddings for each text separately."""
    provider = OllamaProvider(model="llama3")

    responses = [
        MagicMock(
            spec=httpx.Response,
            **{
                "json.return_value": {"embedding": [0.1, 0.2]},
                "raise_for_status": MagicMock(),
            },
        ),
        MagicMock(
            spec=httpx.Response,
            **{
                "json.return_value": {"embedding": [0.3, 0.4]},
                "raise_for_status": MagicMock(),
            },
        ),
    ]
    provider._client.post = AsyncMock(side_effect=responses)

    result = await provider.embed(["first", "second"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert provider._client.post.call_count == 2


# ---------------------------------------------------------------------------
# LLMRouter tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_dispatches_filtering_to_openrouter() -> None:
    """LLMRouter routes 'filtering' task to OpenRouterProvider."""
    config = _make_app_config()
    router = LLMRouter(config)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "filtered"}}]
    }
    mock_resp.raise_for_status = MagicMock()

    mock_result = CompletionResult(text="filtered", model="meta-llama/llama-3.1-8b-instruct")
    with patch.object(
        OpenRouterProvider, "complete", new_callable=AsyncMock, return_value=mock_result
    ) as mock_complete:
        result = await router.complete("filtering", prompt="Is this relevant?")

    assert result.text == "filtered"
    mock_complete.assert_called_once_with(
        prompt="Is this relevant?", system=""
    )


@pytest.mark.asyncio
async def test_router_dispatches_briefing_to_anthropic() -> None:
    """LLMRouter routes 'briefing' task to AnthropicProvider."""
    config = _make_app_config()
    router = LLMRouter(config)

    mock_result = CompletionResult(text="briefing text", model="claude-haiku-4-5-20251001")
    with patch.object(
        AnthropicProvider, "complete", new_callable=AsyncMock, return_value=mock_result
    ) as mock_complete:
        result = await router.complete("briefing", prompt="Write a briefing")

    assert result.text == "briefing text"
    mock_complete.assert_called_once()


@pytest.mark.asyncio
async def test_router_embed_uses_embedding_provider() -> None:
    """LLMRouter embed() delegates to the configured embedding provider."""
    config = _make_app_config(embedding_provider="openai", embedding_model="text-embedding-3-small")
    router = LLMRouter(config)

    expected = [[0.1, 0.2, 0.3]]
    with patch.object(
        OpenAIProvider, "embed", new_callable=AsyncMock, return_value=expected
    ) as mock_embed:
        result = await router.embed(["hello world"])

    assert result == expected
    mock_embed.assert_called_once_with(["hello world"])


@pytest.mark.asyncio
async def test_router_raises_on_unknown_task() -> None:
    """LLMRouter raises ValueError for unrecognised task names."""
    config = _make_app_config()
    router = LLMRouter(config)

    with pytest.raises(ValueError, match="Unknown task"):
        await router.complete("invalid_task", prompt="test")


@pytest.mark.asyncio
async def test_router_raises_on_missing_openai_key() -> None:
    """LLMRouter raises ValueError when OpenAI API key is missing."""
    config = _make_app_config(openai_key=None, embedding_provider="openai")
    router = LLMRouter(config)

    with pytest.raises(ValueError, match="OpenAI API key not configured"):
        await router.embed(["test"])


@pytest.mark.asyncio
async def test_router_raises_on_missing_openrouter_key() -> None:
    """LLMRouter raises ValueError when OpenRouter API key is missing."""
    config = _make_app_config(openrouter_key=None)
    router = LLMRouter(config)

    with pytest.raises(ValueError, match="OpenRouter API key not configured"):
        await router.complete("filtering", prompt="test")


@pytest.mark.asyncio
async def test_router_raises_on_missing_anthropic_key() -> None:
    """LLMRouter raises ValueError when Anthropic API key is missing."""
    config = _make_app_config(anthropic_key=None)
    router = LLMRouter(config)

    with pytest.raises(ValueError, match="Anthropic API key not configured"):
        await router.complete("briefing", prompt="test")


@pytest.mark.asyncio
async def test_router_caches_provider_instances() -> None:
    """LLMRouter creates only one provider instance per task across calls."""
    config = _make_app_config()
    router = LLMRouter(config)

    mock_result = CompletionResult(text="ok", model="test")
    with patch.object(
        OpenRouterProvider, "complete", new_callable=AsyncMock, return_value=mock_result
    ):
        await router.complete("filtering", prompt="first call")
        await router.complete("filtering", prompt="second call")

    # Only one provider should have been cached
    assert len(router._task_providers) == 1
    assert "filtering" in router._task_providers


@pytest.mark.asyncio
async def test_router_ollama_provider_no_api_key_required() -> None:
    """LLMRouter can create an Ollama provider without any API key."""
    config = _make_app_config()
    # Manually set filtering task to use ollama
    config.settings.llm.filtering.provider = "ollama"
    config.settings.llm.filtering.model = "llama3"
    router = LLMRouter(config)

    mock_result = CompletionResult(text="local result", model="llama3")
    with patch.object(
        OllamaProvider, "complete", new_callable=AsyncMock, return_value=mock_result
    ) as mock_complete:
        result = await router.complete("filtering", prompt="test")

    assert result.text == "local result"


# ---------------------------------------------------------------------------
# Retry utility tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_on_first_attempt() -> None:
    """with_retry returns immediately if the first call succeeds."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        return "success"

    result = await with_retry(_fn, operation_name="test")
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_retries_on_timeout() -> None:
    """with_retry retries when httpx.TimeoutException is raised."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.TimeoutException("timeout")
        return "eventual success"

    with patch("ai_craftsman_kb.llm.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await with_retry(_fn, max_attempts=3, operation_name="test")

    assert result == "eventual success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_raises_after_max_attempts() -> None:
    """with_retry raises after exhausting all attempts."""
    async def _fn() -> str:
        raise httpx.TimeoutException("always times out")

    with patch("ai_craftsman_kb.llm.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(httpx.TimeoutException):
            await with_retry(_fn, max_attempts=3, operation_name="test")


@pytest.mark.asyncio
async def test_retry_does_not_retry_non_transient_errors() -> None:
    """with_retry raises immediately on non-retryable errors."""
    call_count = 0

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await with_retry(_fn, max_attempts=3, operation_name="test")

    assert call_count == 1  # No retry for non-transient errors


def test_is_retryable_error_timeout() -> None:
    """_is_retryable_error returns True for timeout exceptions."""
    assert _is_retryable_error(httpx.TimeoutException("timeout")) is True


def test_is_retryable_error_rate_limit() -> None:
    """_is_retryable_error returns True for 429 HTTP status errors."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 429
    exc = httpx.HTTPStatusError("rate limit", request=MagicMock(), response=response)
    assert _is_retryable_error(exc) is True


def test_is_retryable_error_server_error() -> None:
    """_is_retryable_error returns True for 5xx HTTP status errors."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 503
    exc = httpx.HTTPStatusError("service unavailable", request=MagicMock(), response=response)
    assert _is_retryable_error(exc) is True


def test_is_retryable_error_client_error() -> None:
    """_is_retryable_error returns False for 4xx errors (except 429)."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 400
    exc = httpx.HTTPStatusError("bad request", request=MagicMock(), response=response)
    assert _is_retryable_error(exc) is False


def test_is_retryable_error_value_error() -> None:
    """_is_retryable_error returns False for non-HTTP errors."""
    assert _is_retryable_error(ValueError("nope")) is False


# ---------------------------------------------------------------------------
# Retry-After header parsing tests
# ---------------------------------------------------------------------------


def _make_429_error(headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    """Build a 429 HTTPStatusError with optional headers."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 429
    response.headers = headers or {}
    return httpx.HTTPStatusError("rate limit", request=MagicMock(), response=response)


def test_parse_retry_after_numeric() -> None:
    """_parse_retry_after extracts numeric seconds from header."""
    exc = _make_429_error({"retry-after": "5"})
    assert _parse_retry_after(exc) == 5.0


def test_parse_retry_after_float() -> None:
    """_parse_retry_after handles float values in the header."""
    exc = _make_429_error({"retry-after": "2.5"})
    assert _parse_retry_after(exc) == 2.5


def test_parse_retry_after_missing() -> None:
    """_parse_retry_after returns None when no header is present."""
    exc = _make_429_error({})
    assert _parse_retry_after(exc) is None


def test_parse_retry_after_unparseable() -> None:
    """_parse_retry_after returns None for garbage header values."""
    exc = _make_429_error({"retry-after": "not-a-number-or-date"})
    assert _parse_retry_after(exc) is None


def test_parse_retry_after_http_date() -> None:
    """_parse_retry_after handles HTTP-date format."""
    from datetime import datetime, timezone
    from email.utils import format_datetime

    future = datetime.now(timezone.utc).replace(microsecond=0)
    from datetime import timedelta

    future = future + timedelta(seconds=10)
    date_str = format_datetime(future)
    exc = _make_429_error({"retry-after": date_str})
    result = _parse_retry_after(exc)
    assert result is not None
    # Should be roughly 10 seconds (allow some tolerance for test execution time)
    assert 8.0 <= result <= 12.0


def test_parse_retry_after_negative_clamped() -> None:
    """_parse_retry_after clamps negative values to 0."""
    exc = _make_429_error({"retry-after": "-5"})
    assert _parse_retry_after(exc) == 0.0


@pytest.mark.asyncio
async def test_retry_uses_retry_after_header_on_429() -> None:
    """with_retry uses Retry-After header delay instead of exponential backoff for 429."""
    call_count = 0
    sleep_delays: list[float] = []

    response = MagicMock(spec=httpx.Response)
    response.status_code = 429
    response.headers = {"retry-after": "7"}

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.HTTPStatusError("rate limit", request=MagicMock(), response=response)
        return "ok"

    async def mock_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    with patch("ai_craftsman_kb.llm.retry.asyncio.sleep", side_effect=mock_sleep):
        result = await with_retry(_fn, max_attempts=3, base_delay=1.0, operation_name="test")

    assert result == "ok"
    assert call_count == 2
    # Should have used the Retry-After value (7) instead of exponential (1.0)
    assert sleep_delays == [7.0]


@pytest.mark.asyncio
async def test_retry_falls_back_to_exponential_on_429_without_header() -> None:
    """with_retry uses exponential backoff for 429 when Retry-After is absent."""
    call_count = 0
    sleep_delays: list[float] = []

    response = MagicMock(spec=httpx.Response)
    response.status_code = 429
    response.headers = {}  # No Retry-After header

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.HTTPStatusError("rate limit", request=MagicMock(), response=response)
        return "ok"

    async def mock_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    with patch("ai_craftsman_kb.llm.retry.asyncio.sleep", side_effect=mock_sleep):
        result = await with_retry(_fn, max_attempts=3, base_delay=1.0, operation_name="test")

    assert result == "ok"
    # Should have used exponential backoff: base_delay * 2^0 = 1.0
    assert sleep_delays == [1.0]


@pytest.mark.asyncio
async def test_retry_exponential_backoff_unchanged_for_5xx() -> None:
    """with_retry still uses exponential backoff for 500-series errors."""
    call_count = 0
    sleep_delays: list[float] = []

    response = MagicMock(spec=httpx.Response)
    response.status_code = 503
    response.headers = {"retry-after": "99"}  # Should be ignored for non-429

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.HTTPStatusError("unavailable", request=MagicMock(), response=response)
        return "ok"

    async def mock_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    with patch("ai_craftsman_kb.llm.retry.asyncio.sleep", side_effect=mock_sleep):
        result = await with_retry(_fn, max_attempts=3, base_delay=1.0, operation_name="test")

    assert result == "ok"
    # Should use exponential backoff: 1.0, 2.0
    assert sleep_delays == [1.0, 2.0]


@pytest.mark.asyncio
async def test_retry_after_capped_by_max_delay() -> None:
    """with_retry caps Retry-After delay at max_delay."""
    call_count = 0
    sleep_delays: list[float] = []

    response = MagicMock(spec=httpx.Response)
    response.status_code = 429
    response.headers = {"retry-after": "120"}

    async def _fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.HTTPStatusError("rate limit", request=MagicMock(), response=response)
        return "ok"

    async def mock_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    with patch("ai_craftsman_kb.llm.retry.asyncio.sleep", side_effect=mock_sleep):
        result = await with_retry(
            _fn, max_attempts=3, base_delay=1.0, max_delay=30.0, operation_name="test"
        )

    assert result == "ok"
    # Retry-After is 120 but max_delay is 30, so should be capped
    assert sleep_delays == [30.0]


# ---------------------------------------------------------------------------
# AsyncRateLimiter tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_first_acquire_immediate() -> None:
    """First acquire() should return immediately without sleeping."""
    limiter = AsyncRateLimiter(rpm=60)
    with patch("ai_craftsman_kb.llm.rate_limiter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await limiter.acquire()
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_rate_limiter_sleeps_when_called_too_fast() -> None:
    """Second acquire() should sleep when called before the interval elapses."""
    limiter = AsyncRateLimiter(rpm=60)  # 1 second interval

    with patch("ai_craftsman_kb.llm.rate_limiter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        # Simulate first call already happened recently
        limiter._last_request = time.monotonic()
        await limiter.acquire()

    mock_sleep.assert_called_once()
    # The sleep duration should be close to 1.0 second
    sleep_duration = mock_sleep.call_args[0][0]
    assert 0.0 < sleep_duration <= 1.0


@pytest.mark.asyncio
async def test_rate_limiter_no_sleep_after_interval() -> None:
    """acquire() should not sleep if enough time has elapsed."""
    limiter = AsyncRateLimiter(rpm=60)  # 1 second interval

    with patch("ai_craftsman_kb.llm.rate_limiter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        # Simulate last request was long ago
        limiter._last_request = time.monotonic() - 5.0
        await limiter.acquire()

    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_rate_limiter_serialises_concurrent_callers() -> None:
    """Multiple concurrent acquire() calls should be serialised by the lock."""
    limiter = AsyncRateLimiter(rpm=600)  # 0.1 second interval
    timestamps: list[float] = []

    async def _record_acquire() -> None:
        await limiter.acquire()
        timestamps.append(time.monotonic())

    # Launch 3 concurrent acquires
    await asyncio.gather(_record_acquire(), _record_acquire(), _record_acquire())

    # All 3 should have completed
    assert len(timestamps) == 3


@pytest.mark.asyncio
async def test_rate_limiter_updates_last_request() -> None:
    """acquire() should update _last_request after completing."""
    limiter = AsyncRateLimiter(rpm=120)
    assert limiter._last_request == 0.0

    with patch("ai_craftsman_kb.llm.rate_limiter.asyncio.sleep", new_callable=AsyncMock):
        await limiter.acquire()

    assert limiter._last_request > 0.0


# ---------------------------------------------------------------------------
# Router wiring tests -- retry, rate limiting, usage tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_complete_wraps_with_retry() -> None:
    """Router complete() wraps provider call in with_retry."""
    config = _make_app_config()
    router = LLMRouter(config)
    call_count = 0

    async def _flaky_complete(*args: Any, **kwargs: Any) -> CompletionResult:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.TimeoutException("timeout")
        return CompletionResult(text="ok", model="test", input_tokens=5, output_tokens=3)

    with (
        patch.object(OpenRouterProvider, "complete", side_effect=_flaky_complete),
        patch("ai_craftsman_kb.llm.retry.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await router.complete("filtering", prompt="test")

    assert result.text == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_router_complete_uses_task_max_retries() -> None:
    """Router complete() respects the max_retries from task config."""
    config = _make_app_config()
    # Set max_retries to 1 so no retries happen
    config.settings.llm.filtering.max_retries = 1
    router = LLMRouter(config)

    with patch.object(
        OpenRouterProvider,
        "complete",
        new_callable=AsyncMock,
        side_effect=httpx.TimeoutException("timeout"),
    ):
        with pytest.raises(httpx.TimeoutException):
            await router.complete("filtering", prompt="test")


@pytest.mark.asyncio
async def test_router_complete_acquires_rate_limiter() -> None:
    """Router complete() calls rate limiter acquire() when rate_limit is set."""
    config = _make_app_config()
    config.settings.llm.filtering.rate_limit = 30.0
    router = LLMRouter(config)

    mock_result = CompletionResult(text="ok", model="test")
    with patch.object(
        OpenRouterProvider, "complete", new_callable=AsyncMock, return_value=mock_result
    ):
        await router.complete("filtering", prompt="first")
        await router.complete("filtering", prompt="second")

    # A rate limiter should have been created and cached for the task
    assert "filtering" in router._rate_limiters
    assert router._rate_limiters["filtering"]._min_interval == 60.0 / 30.0


@pytest.mark.asyncio
async def test_router_complete_no_rate_limiter_when_none() -> None:
    """Router complete() does not create a rate limiter when rate_limit is None."""
    config = _make_app_config()
    # rate_limit defaults to None in _make_app_config
    router = LLMRouter(config)

    mock_result = CompletionResult(text="ok", model="test")
    with patch.object(
        OpenRouterProvider, "complete", new_callable=AsyncMock, return_value=mock_result
    ):
        await router.complete("filtering", prompt="test")

    assert "filtering" not in router._rate_limiters


@pytest.mark.asyncio
async def test_router_complete_records_usage_on_success() -> None:
    """Router complete() records usage via tracker on successful call."""
    config = _make_app_config()
    tracker = AsyncMock()
    router = LLMRouter(config, usage_tracker=tracker)

    mock_result = CompletionResult(
        text="ok", model="test-model", input_tokens=10, output_tokens=5
    )
    with patch.object(
        OpenRouterProvider, "complete", new_callable=AsyncMock, return_value=mock_result
    ):
        await router.complete("filtering", prompt="test")

    tracker.record.assert_called_once()
    call_kwargs = tracker.record.call_args.kwargs
    assert call_kwargs["provider"] == "openrouter"
    assert call_kwargs["model"] == "meta-llama/llama-3.1-8b-instruct"
    assert call_kwargs["task"] == "filtering"
    assert call_kwargs["input_tokens"] == 10
    assert call_kwargs["output_tokens"] == 5
    assert call_kwargs["success"] is True
    assert isinstance(call_kwargs["duration_ms"], int)
    assert call_kwargs["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_router_complete_records_usage_on_failure() -> None:
    """Router complete() records usage with success=False when the call fails."""
    config = _make_app_config()
    config.settings.llm.filtering.max_retries = 1
    tracker = AsyncMock()
    router = LLMRouter(config, usage_tracker=tracker)

    with patch.object(
        OpenRouterProvider,
        "complete",
        new_callable=AsyncMock,
        side_effect=ValueError("bad"),
    ):
        with pytest.raises(ValueError, match="bad"):
            await router.complete("filtering", prompt="test")

    tracker.record.assert_called_once()
    call_kwargs = tracker.record.call_args.kwargs
    assert call_kwargs["success"] is False
    assert call_kwargs["task"] == "filtering"


@pytest.mark.asyncio
async def test_router_complete_no_tracker_no_error() -> None:
    """Router complete() works fine without a usage tracker."""
    config = _make_app_config()
    router = LLMRouter(config)  # No tracker

    mock_result = CompletionResult(text="ok", model="test")
    with patch.object(
        OpenRouterProvider, "complete", new_callable=AsyncMock, return_value=mock_result
    ):
        result = await router.complete("filtering", prompt="test")

    assert result.text == "ok"


@pytest.mark.asyncio
async def test_router_embed_records_usage_on_success() -> None:
    """Router embed() records usage via tracker on successful call."""
    config = _make_app_config()
    tracker = AsyncMock()
    router = LLMRouter(config, usage_tracker=tracker)

    expected = [[0.1, 0.2, 0.3]]
    with patch.object(
        OpenAIProvider, "embed", new_callable=AsyncMock, return_value=expected
    ):
        result = await router.embed(["hello"])

    assert result == expected
    tracker.record.assert_called_once()
    call_kwargs = tracker.record.call_args.kwargs
    assert call_kwargs["provider"] == "openai"
    assert call_kwargs["model"] == "text-embedding-3-small"
    assert call_kwargs["task"] == "embedding"
    assert call_kwargs["success"] is True
    assert isinstance(call_kwargs["duration_ms"], int)


@pytest.mark.asyncio
async def test_router_embed_records_usage_on_failure() -> None:
    """Router embed() records usage with success=False when the call fails."""
    config = _make_app_config()
    tracker = AsyncMock()
    router = LLMRouter(config, usage_tracker=tracker)

    with patch.object(
        OpenAIProvider,
        "embed",
        new_callable=AsyncMock,
        side_effect=RuntimeError("embed failed"),
    ):
        with pytest.raises(RuntimeError, match="embed failed"):
            await router.embed(["hello"])

    tracker.record.assert_called_once()
    call_kwargs = tracker.record.call_args.kwargs
    assert call_kwargs["success"] is False
    assert call_kwargs["task"] == "embedding"


# ---------------------------------------------------------------------------
# Gateway / EndpointPool tests
# ---------------------------------------------------------------------------


def _make_managed_endpoint(
    name: str,
    daily_limit: int | None = None,
    daily_count: int = 0,
    rpm: float = 60.0,
) -> ManagedEndpoint:
    """Build a ManagedEndpoint with a mock provider."""
    limiter = AsyncRateLimiter(rpm=rpm, daily_limit=daily_limit)
    limiter._daily_count = daily_count
    if daily_limit is not None:
        # Ensure the day counter is active
        import time
        limiter._day_start = time.monotonic()
    provider = AsyncMock(spec=LLMProvider)
    provider.complete = AsyncMock(
        return_value=CompletionResult(text=f"from-{name}", model=f"model-{name}")
    )
    return ManagedEndpoint(
        name=name,
        provider_name=f"provider-{name}",
        model=f"model-{name}",
        provider=provider,
        rate_limiter=limiter,
    )


@pytest.mark.asyncio
async def test_pool_selects_endpoint_with_most_remaining() -> None:
    """EndpointPool picks the endpoint with the most daily quota remaining."""
    ep_low = _make_managed_endpoint("low", daily_limit=100, daily_count=90)
    ep_high = _make_managed_endpoint("high", daily_limit=100, daily_count=10)

    pool = EndpointPool("test", [ep_low, ep_high])
    result, selected = await pool.complete("hello")

    assert selected.name == "high"
    assert result.text == "from-high"


@pytest.mark.asyncio
async def test_pool_prefers_unlimited_endpoints() -> None:
    """EndpointPool puts unlimited endpoints (daily_limit=None) first."""
    ep_limited = _make_managed_endpoint("limited", daily_limit=100, daily_count=0)
    ep_unlimited = _make_managed_endpoint("unlimited", daily_limit=None)

    pool = EndpointPool("test", [ep_limited, ep_unlimited])
    result, selected = await pool.complete("hello")

    assert selected.name == "unlimited"


@pytest.mark.asyncio
async def test_pool_failover_on_daily_limit_exceeded() -> None:
    """EndpointPool skips exhausted endpoints and falls through to the next."""
    ep_exhausted = _make_managed_endpoint("exhausted", daily_limit=10, daily_count=10)
    ep_available = _make_managed_endpoint("available", daily_limit=100, daily_count=0)

    pool = EndpointPool("test", [ep_exhausted, ep_available])
    result, selected = await pool.complete("hello")

    assert selected.name == "available"
    assert result.text == "from-available"


@pytest.mark.asyncio
async def test_pool_all_exhausted_raises() -> None:
    """EndpointPool raises AllEndpointsExhausted when every endpoint is spent."""
    ep1 = _make_managed_endpoint("a", daily_limit=10, daily_count=10)
    ep2 = _make_managed_endpoint("b", daily_limit=10, daily_count=10)

    pool = EndpointPool("test", [ep1, ep2])
    with pytest.raises(AllEndpointsExhausted, match="All endpoints"):
        await pool.complete("hello")


@pytest.mark.asyncio
async def test_pool_failover_on_provider_error() -> None:
    """EndpointPool tries next endpoint when current one raises a non-retryable error."""
    ep_broken = _make_managed_endpoint("broken", daily_limit=100, daily_count=0)
    ep_broken.provider.complete = AsyncMock(side_effect=ValueError("broken provider"))
    ep_ok = _make_managed_endpoint("ok", daily_limit=100, daily_count=50)

    pool = EndpointPool("test", [ep_broken, ep_ok], max_retries=1)

    with patch("ai_craftsman_kb.llm.retry.asyncio.sleep", new_callable=AsyncMock):
        result, selected = await pool.complete("hello")

    assert selected.name == "ok"


# ---------------------------------------------------------------------------
# Gateway config parsing tests
# ---------------------------------------------------------------------------


def test_settings_config_parses_gateway_format() -> None:
    """SettingsConfig with 'endpoints' key parses as LLMGatewayConfig."""
    data = {
        "llm": {
            "endpoints": {
                "ep1": {"provider": "openrouter", "model": "test-model", "rate_limit": 20, "daily_limit": 100},
            },
            "pools": {
                "pool1": {"endpoints": ["ep1"]},
            },
            "tasks": {
                "filtering": "pool1",
            },
        }
    }
    cfg = SettingsConfig(**data)
    assert isinstance(cfg.llm, LLMGatewayConfig)
    assert "ep1" in cfg.llm.endpoints
    assert cfg.llm.tasks["filtering"] == "pool1"


def test_settings_config_parses_legacy_format() -> None:
    """SettingsConfig with 'filtering' key parses as LLMRoutingConfig (legacy)."""
    data = {
        "llm": {
            "filtering": {"provider": "openrouter", "model": "test"},
            "entity_extraction": {"provider": "openrouter", "model": "test"},
            "briefing": {"provider": "openrouter", "model": "test"},
            "source_discovery": {"provider": "openrouter", "model": "test"},
            "keyword_extraction": {"provider": "openrouter", "model": "test"},
        }
    }
    cfg = SettingsConfig(**data)
    from ai_craftsman_kb.config.models import LLMRoutingConfig
    assert isinstance(cfg.llm, LLMRoutingConfig)


# ---------------------------------------------------------------------------
# Router gateway-mode tests
# ---------------------------------------------------------------------------


def _make_gateway_config(
    *,
    ep_keys: list[str] | None = None,
) -> MagicMock:
    """Build a minimal AppConfig mock with gateway LLM config."""
    config = MagicMock()

    def _provider_cfg(key: str | None, base_url: str | None = None) -> MagicMock:
        m = MagicMock()
        m.api_key = key
        m.base_url = base_url
        return m

    config.settings.providers = {
        "openrouter": _provider_cfg("or-test"),
        "groq": _provider_cfg("groq-test"),
        "cerebras": _provider_cfg("cerebras-test"),
        "openai": _provider_cfg("sk-test"),
    }

    ep_keys = ep_keys or ["openrouter-llama70b", "groq-llama70b"]
    endpoints = {}
    if "openrouter-llama70b" in ep_keys:
        endpoints["openrouter-llama70b"] = EndpointConfig(
            provider="openrouter", model="llama-70b", rate_limit=20, daily_limit=100,
        )
    if "groq-llama70b" in ep_keys:
        endpoints["groq-llama70b"] = EndpointConfig(
            provider="groq", model="llama-70b", rate_limit=30, daily_limit=200,
        )

    gw = LLMGatewayConfig(
        endpoints=endpoints,
        pools={"free": PoolConfig(endpoints=list(endpoints.keys()), max_retries=2)},
        tasks={"filtering": "free", "entity_extraction": "free", "briefing": "free",
               "source_discovery": "free", "keyword_extraction": "free"},
    )
    config.settings.llm = gw

    config.settings.embedding.provider = "openai"
    config.settings.embedding.model = "text-embedding-3-small"

    return config


@pytest.mark.asyncio
async def test_router_gateway_mode_routes_through_pool() -> None:
    """Router in gateway mode delegates to EndpointPool."""
    config = _make_gateway_config()
    router = LLMRouter(config)

    mock_result = CompletionResult(text="gateway-ok", model="llama-70b")

    # Mock both OpenRouter and OpenAI (groq uses OpenAIProvider)
    with (
        patch.object(OpenRouterProvider, "complete", new_callable=AsyncMock, return_value=mock_result),
        patch.object(OpenAIProvider, "complete", new_callable=AsyncMock, return_value=mock_result),
    ):
        result = await router.complete("filtering", prompt="test")

    assert result.text == "gateway-ok"


@pytest.mark.asyncio
async def test_router_gateway_shared_rate_limiter() -> None:
    """Gateway mode: two tasks using the same pool share endpoint rate limiters."""
    config = _make_gateway_config(ep_keys=["openrouter-llama70b"])
    router = LLMRouter(config)

    # The pool for "filtering" and "entity_extraction" should share endpoints
    pool_f = router._task_pools["filtering"]
    pool_e = router._task_pools["entity_extraction"]
    # Same pool object since they map to the same pool name
    assert pool_f is pool_e


@pytest.mark.asyncio
async def test_router_gateway_unknown_task_raises() -> None:
    """Router in gateway mode raises ValueError for unknown tasks."""
    config = _make_gateway_config()
    router = LLMRouter(config)

    with pytest.raises(ValueError, match="Unknown task"):
        await router.complete("nonexistent_task", prompt="test")
