"""Retry utilities with exponential backoff for LLM provider calls."""
import asyncio
import logging
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
from typing import TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes considered transient/retryable
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _parse_retry_after(exc: httpx.HTTPStatusError) -> float | None:
    """Extract a delay in seconds from a Retry-After response header.

    Supports both numeric seconds (e.g. ``Retry-After: 5``) and HTTP-date
    format (e.g. ``Retry-After: Thu, 01 Jan 2026 00:00:05 GMT``).

    Args:
        exc: An HTTPStatusError whose response may contain Retry-After.

    Returns:
        The number of seconds to wait, or ``None`` if the header is missing
        or unparseable.
    """
    header = exc.response.headers.get("retry-after")
    if header is None:
        return None

    # Try numeric seconds first
    try:
        return max(float(header), 0.0)
    except ValueError:
        pass

    # Try HTTP-date format
    try:
        from datetime import datetime, timezone

        target = parsedate_to_datetime(header)
        # Ensure the parsed datetime is timezone-aware
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        delta = (target - datetime.now(timezone.utc)).total_seconds()
        return max(delta, 0.0)
    except Exception:  # noqa: BLE001
        return None


def _is_retryable_error(exc: Exception) -> bool:
    """Determine if an exception is transient and worth retrying.

    Args:
        exc: The exception to inspect.

    Returns:
        True if the exception is considered transient and retryable.
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    # openai RateLimitError / APIStatusError — check by name to avoid hard import
    exc_type = type(exc).__name__
    if exc_type in ("RateLimitError", "APITimeoutError", "APIConnectionError"):
        return True
    return False


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    operation_name: str = "operation",
) -> T:
    """Run an async callable with exponential backoff retry logic.

    Retries on transient errors (timeouts, rate limits, 5xx responses).
    Raises immediately on non-retryable errors.

    Args:
        fn: Async callable (no arguments) to execute.
        max_attempts: Maximum number of total attempts (default 3).
        base_delay: Initial delay in seconds between retries (default 1.0).
        max_delay: Maximum delay cap in seconds (default 30.0).
        operation_name: Human-readable label for log messages.

    Returns:
        The result of the successful callable invocation.

    Raises:
        Exception: Re-raises the last exception if all attempts are exhausted
            or if the error is non-retryable.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_error(exc):
                # Non-transient error — raise immediately without retrying
                raise
            if attempt == max_attempts:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation_name,
                    max_attempts,
                    exc,
                )
                raise

            # For 429 responses, prefer the server-provided Retry-After delay
            retry_after_delay: float | None = None
            if (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code == 429
            ):
                retry_after_delay = _parse_retry_after(exc)

            if retry_after_delay is not None:
                delay = min(retry_after_delay, max_delay)
            else:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

            logger.warning(
                "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                operation_name,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    # This point is unreachable but satisfies the type checker
    raise RuntimeError("Unexpected exit from retry loop") from last_exc
