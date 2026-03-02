"""Retry decorator and structured error classes for AI Craftsman KB.

This module provides:
- A hierarchy of typed application errors (AppError, APIError, ConfigError,
  QuotaExceededError) used throughout the ingest pipeline and LLM providers.
- ``retry_async`` — a decorator that wraps async functions with exponential
  backoff retry logic, skipping non-retryable error types immediately.

Usage::

    from ai_craftsman_kb.resilience import retry_async, APIError

    @retry_async(max_attempts=3, retry_on=(APIError,))
    async def fetch_data(url: str) -> dict:
        ...
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Base class for all structured application errors.

    All domain-specific errors in AI Craftsman KB derive from this class so
    that catch-all handling can distinguish application errors from unexpected
    Python exceptions.

    Args:
        message: Human-readable description of the error.
        recoverable: Whether the error is considered transient and may resolve
            on retry. Non-recoverable errors (e.g. bad config) should be
            raised immediately without retrying.
    """

    def __init__(self, message: str, recoverable: bool = True) -> None:
        super().__init__(message)
        self.recoverable = recoverable


class APIError(AppError):
    """Raised when an external API call fails.

    Carries the provider name and HTTP status code alongside the message so
    that retry logic and error reporting can make informed decisions.

    Args:
        provider: Name of the external provider (e.g. 'openai', 'hn').
        status_code: HTTP status code returned by the API, or 0 for
            non-HTTP failures (connection errors, timeouts).
        message: Human-readable description of the failure.
    """

    def __init__(self, provider: str, status_code: int, message: str) -> None:
        super().__init__(
            f"[{provider}] HTTP {status_code}: {message}",
            recoverable=status_code not in (400, 401, 403, 404, 422),
        )
        self.provider = provider
        self.status_code = status_code


class ConfigError(AppError):
    """Raised when application configuration is invalid or missing.

    Configuration errors are *not* recoverable — they require user action
    (editing a config file or setting an environment variable) and should
    never be retried.

    Args:
        field: The configuration field that is invalid or missing (e.g.
            ``settings.providers.openai.api_key``).
        message: Human-readable description of the problem, ideally including
            a suggested fix.
    """

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"Config error [{field}]: {message}", recoverable=False)
        self.field = field


class QuotaExceededError(APIError):
    """Raised when an API quota or rate-limit is exhausted.

    Quota errors are a sub-type of :class:`APIError` but are treated as
    non-retryable within the same request — the quota will not reset quickly
    enough for a simple retry to help. Callers should surface these to the
    user instead.

    The ``retry_async`` decorator skips retries for this error type by
    default.
    """


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def retry_async(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    retry_on: tuple[type[Exception], ...] = (APIError,),
    skip_on: tuple[type[Exception], ...] = (QuotaExceededError, ConfigError),
) -> Callable[[Any], Any]:
    """Decorator that retries a failing async function with exponential backoff.

    Retries are only performed for exceptions whose type is listed in
    ``retry_on`` *and* not listed in ``skip_on``.  Non-matching exceptions
    are re-raised immediately without consuming any retry attempts.

    Backoff delay formula::

        delay = min(backoff_base * 2 ** (attempt - 1), backoff_max)

    So with the defaults (base=1.0, max=30.0) the delays are:
      - Attempt 1 fails → wait 1 s before attempt 2
      - Attempt 2 fails → wait 2 s before attempt 3
      - Attempt 3 fails → re-raise (max_attempts exhausted)

    Args:
        max_attempts: Total number of attempts, including the first.
            Must be >= 1. Default: 3.
        backoff_base: Initial backoff in seconds. Default: 1.0.
        backoff_max: Upper cap on the backoff delay in seconds. Default: 30.0.
        retry_on: Tuple of exception types that trigger a retry.
            Default: ``(APIError,)``.
        skip_on: Tuple of exception types that are *never* retried, even if
            they are subclasses of a type in ``retry_on``.
            Default: ``(QuotaExceededError, ConfigError)``.

    Returns:
        A decorator that wraps an async function with retry logic.

    Example::

        @retry_async(max_attempts=4, backoff_base=0.5)
        async def call_api(url: str) -> dict:
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except skip_on:
                    # Non-retryable — raise immediately regardless of attempt count
                    raise
                except retry_on as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__qualname__,
                            max_attempts,
                            exc,
                        )
                        raise
                    delay = min(backoff_base * 2 ** (attempt - 1), backoff_max)
                    logger.warning(
                        "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                        func.__qualname__,
                        attempt,
                        max_attempts,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        return wrapper

    return decorator
