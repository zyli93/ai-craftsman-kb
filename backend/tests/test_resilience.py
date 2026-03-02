"""Tests for the resilience module — retry decorator and error classes."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from ai_craftsman_kb.resilience import (
    APIError,
    AppError,
    ConfigError,
    QuotaExceededError,
    retry_async,
)

# ---------------------------------------------------------------------------
# Error class tests
# ---------------------------------------------------------------------------


def test_app_error_recoverable_by_default() -> None:
    """AppError should be recoverable by default."""
    err = AppError("something went wrong")
    assert err.recoverable is True
    assert str(err) == "something went wrong"


def test_app_error_non_recoverable() -> None:
    """AppError can be marked as non-recoverable."""
    err = AppError("fatal error", recoverable=False)
    assert err.recoverable is False


def test_api_error_contains_provider_info() -> None:
    """APIError should embed provider and status code in message."""
    err = APIError(provider="openai", status_code=429, message="rate limited")
    assert "openai" in str(err)
    assert "429" in str(err)
    assert err.provider == "openai"
    assert err.status_code == 429


def test_api_error_5xx_is_recoverable() -> None:
    """5xx status codes should be recoverable."""
    err = APIError(provider="anthropic", status_code=503, message="service unavailable")
    assert err.recoverable is True


def test_api_error_4xx_is_not_recoverable() -> None:
    """4xx status codes (except 429) should not be recoverable."""
    err = APIError(provider="openai", status_code=401, message="unauthorized")
    assert err.recoverable is False


def test_config_error_is_not_recoverable() -> None:
    """ConfigError should always be non-recoverable."""
    err = ConfigError(field="providers.openai.api_key", message="API key not set")
    assert err.recoverable is False
    assert "providers.openai.api_key" in str(err)
    assert err.field == "providers.openai.api_key"


def test_quota_exceeded_error_is_api_error() -> None:
    """QuotaExceededError should be a subclass of APIError."""
    err = QuotaExceededError(provider="openai", status_code=429, message="quota exceeded")
    assert isinstance(err, APIError)
    assert isinstance(err, AppError)


# ---------------------------------------------------------------------------
# retry_async decorator tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_on_first_attempt() -> None:
    """retry_async should return immediately when the function succeeds."""
    call_count = 0

    @retry_async(max_attempts=3)
    async def succeed() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await succeed()
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_retries_on_api_error() -> None:
    """retry_async should retry when an APIError is raised."""
    call_count = 0

    @retry_async(max_attempts=3, backoff_base=0.001)
    async def fail_twice() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise APIError(provider="test", status_code=500, message="server error")
        return "ok"

    result = await fail_twice()
    assert result == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_raises_after_max_attempts() -> None:
    """retry_async should raise the last exception after all attempts fail."""
    call_count = 0

    @retry_async(max_attempts=3, backoff_base=0.001)
    async def always_fail() -> str:
        nonlocal call_count
        call_count += 1
        raise APIError(provider="test", status_code=500, message="always fails")

    with pytest.raises(APIError):
        await always_fail()

    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_skips_on_quota_exceeded() -> None:
    """retry_async should not retry QuotaExceededError."""
    call_count = 0

    @retry_async(max_attempts=3, backoff_base=0.001)
    async def quota_fail() -> str:
        nonlocal call_count
        call_count += 1
        raise QuotaExceededError(provider="openai", status_code=429, message="quota exceeded")

    with pytest.raises(QuotaExceededError):
        await quota_fail()

    assert call_count == 1  # No retries for QuotaExceededError


@pytest.mark.asyncio
async def test_retry_skips_on_config_error() -> None:
    """retry_async should not retry ConfigError."""
    call_count = 0

    @retry_async(max_attempts=3, backoff_base=0.001)
    async def config_fail() -> str:
        nonlocal call_count
        call_count += 1
        raise ConfigError(field="api_key", message="API key not set")

    with pytest.raises(ConfigError):
        await config_fail()

    assert call_count == 1  # No retries for ConfigError


@pytest.mark.asyncio
async def test_retry_raises_non_matching_exception_immediately() -> None:
    """Non-listed exceptions should be raised immediately without retry."""
    call_count = 0

    @retry_async(max_attempts=3, backoff_base=0.001, retry_on=(APIError,))
    async def value_error_fail() -> str:
        nonlocal call_count
        call_count += 1
        raise ValueError("not an API error")

    with pytest.raises(ValueError):
        await value_error_fail()

    assert call_count == 1  # No retries for ValueError (not in retry_on)


@pytest.mark.asyncio
async def test_retry_backoff_capped_at_backoff_max() -> None:
    """Backoff delay should not exceed backoff_max."""
    delays: list[float] = []

    async def mock_sleep(delay: float) -> None:
        delays.append(delay)

    with patch("asyncio.sleep", side_effect=mock_sleep):
        call_count = 0

        @retry_async(max_attempts=4, backoff_base=10.0, backoff_max=15.0)
        async def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise APIError(provider="test", status_code=500, message="fail")

        with pytest.raises(APIError):
            await always_fail()

    # All delays should be <= backoff_max=15.0
    assert all(d <= 15.0 for d in delays)
    # With base=10.0: attempt1→10, attempt2→15 (capped), attempt3→15 (capped)
    assert len(delays) == 3  # 4 attempts → 3 sleeps


@pytest.mark.asyncio
async def test_retry_custom_retry_on() -> None:
    """Custom retry_on tuple should control which exceptions are retried."""
    call_count = 0

    @retry_async(max_attempts=3, backoff_base=0.001, retry_on=(ValueError,))
    async def value_error_twice() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("try again")
        return "ok"

    result = await value_error_twice()
    assert result == "ok"
    assert call_count == 3


# ---------------------------------------------------------------------------
# Doctor check function tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_api_key_present(minimal_config) -> None:
    """_check_api_key returns 'ok' when key is present."""
    from ai_craftsman_kb.cli import _check_api_key
    from ai_craftsman_kb.config.models import ProviderConfig

    # Add a provider with a key
    config = minimal_config.model_copy(deep=True)
    config.settings.providers["openai"] = ProviderConfig(api_key="sk-test-key-12345678")

    status, message = await _check_api_key(config, "openai")
    assert status == "ok"
    assert "sk-tes" in message  # masked key prefix


@pytest.mark.asyncio
async def test_check_api_key_missing(minimal_config) -> None:
    """_check_api_key returns 'warn' when key is absent."""
    from ai_craftsman_kb.cli import _check_api_key

    status, message = await _check_api_key(minimal_config, "openai")
    assert status == "warn"
    assert "OPENAI_API_KEY" in message


@pytest.mark.asyncio
async def test_check_config_returns_ok(minimal_config) -> None:
    """_check_config always returns 'ok' with data_dir in message."""
    from ai_craftsman_kb.cli import _check_config

    status, message = await _check_config(minimal_config)
    assert status == "ok"
    assert "data_dir=" in message


@pytest.mark.asyncio
async def test_check_youtube_key_missing(minimal_config) -> None:
    """_check_youtube_key returns 'warn' when key is not configured."""
    from ai_craftsman_kb.cli import _check_youtube_key

    status, message = await _check_youtube_key(minimal_config)
    assert status == "warn"
    assert "YOUTUBE_API_KEY" in message


@pytest.mark.asyncio
async def test_check_reddit_credentials_missing(minimal_config) -> None:
    """_check_reddit_credentials returns 'warn' when credentials absent."""
    from ai_craftsman_kb.cli import _check_reddit_credentials

    status, message = await _check_reddit_credentials(minimal_config)
    assert status == "warn"
    assert "REDDIT_CLIENT_ID" in message


@pytest.mark.asyncio
async def test_check_connectivity_ok() -> None:
    """_check_connectivity returns 'ok' for a 200 response."""
    import httpx
    from ai_craftsman_kb.cli import _check_connectivity

    mock_response = httpx.Response(200)

    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url: str) -> httpx.Response:
            return mock_response

    with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
        status, message = await _check_connectivity("https://example.com", "test")

    assert status == "ok"
    assert "200" in message


@pytest.mark.asyncio
async def test_check_connectivity_error() -> None:
    """_check_connectivity returns 'error' on connection failure."""
    from ai_craftsman_kb.cli import _check_connectivity

    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url: str) -> None:
            raise ConnectionError("timeout")

    with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
        status, message = await _check_connectivity("https://example.com", "test")

    assert status == "error"
    assert "Unreachable" in message


@pytest.mark.asyncio
async def test_check_data_dir_ok(tmp_path, minimal_config) -> None:
    """_check_data_dir returns 'ok' when directory is writable."""
    from ai_craftsman_kb.cli import _check_data_dir

    # Override data_dir to use tmp_path via model_copy(update=...)
    new_settings = minimal_config.settings.model_copy(update={"data_dir": str(tmp_path)})
    config = minimal_config.model_copy(update={"settings": new_settings})

    status, message = await _check_data_dir(config)
    assert status == "ok"
    assert "GB free" in message
