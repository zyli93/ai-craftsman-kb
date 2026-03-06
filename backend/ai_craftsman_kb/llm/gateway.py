"""LLM Gateway — multi-provider endpoint pool with automatic failover."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .base import CompletionResult, LLMProvider
from .rate_limiter import AsyncRateLimiter, DailyLimitExceeded
from .retry import with_retry

logger = logging.getLogger(__name__)


class AllEndpointsExhausted(DailyLimitExceeded):
    """Raised when every endpoint in a pool has exceeded its daily limit."""


@dataclass
class ManagedEndpoint:
    """An endpoint bundled with its provider instance and rate limiter."""

    name: str
    provider_name: str
    model: str
    provider: LLMProvider
    rate_limiter: AsyncRateLimiter


class EndpointPool:
    """Routes requests across a pool of interchangeable endpoints.

    Endpoints are tried in order of most daily quota remaining.  On
    ``DailyLimitExceeded``, the pool skips to the next endpoint.  On
    retryable errors (timeouts, 5xx), ``with_retry`` is used on the
    same endpoint before moving on.

    Args:
        name: Human-readable pool name (for logging).
        endpoints: Ordered list of managed endpoints in the pool.
        max_retries: Passed to ``with_retry`` for each endpoint attempt.
    """

    def __init__(
        self,
        name: str,
        endpoints: list[ManagedEndpoint],
        max_retries: int = 3,
    ) -> None:
        self.name = name
        self._endpoints = endpoints
        self._max_retries = max_retries

    def _sorted_endpoints(self) -> list[ManagedEndpoint]:
        """Sort limited endpoints by daily_remaining descending, then unlimited in config order."""
        limited = [ep for ep in self._endpoints if ep.rate_limiter.daily_remaining is not None]
        unlimited = [ep for ep in self._endpoints if ep.rate_limiter.daily_remaining is None]
        limited.sort(key=lambda ep: ep.rate_limiter.daily_remaining, reverse=True)  # type: ignore[arg-type]
        return limited + unlimited

    async def complete(
        self,
        prompt: str,
        system: str = "",
        **kwargs: object,
    ) -> tuple[CompletionResult, ManagedEndpoint]:
        """Try each endpoint in the pool until one succeeds.

        Returns:
            Tuple of (CompletionResult, the ManagedEndpoint that handled it).

        Raises:
            AllEndpointsExhausted: If every endpoint's daily limit is hit.
        """
        sorted_eps = self._sorted_endpoints()
        last_exc: Exception | None = None

        for ep in sorted_eps:
            try:
                await ep.rate_limiter.acquire()
            except DailyLimitExceeded:
                logger.warning(
                    "Pool '%s': endpoint '%s' daily limit reached, trying next",
                    self.name,
                    ep.name,
                )
                last_exc = DailyLimitExceeded(f"{ep.name} daily limit reached")
                continue

            try:
                result = await with_retry(
                    lambda _ep=ep: _ep.provider.complete(prompt=prompt, system=system, **kwargs),
                    max_attempts=self._max_retries,
                    operation_name=f"pool.{self.name}.{ep.name}",
                )
                return result, ep
            except Exception as exc:
                logger.warning(
                    "Pool '%s': endpoint '%s' failed: %s",
                    self.name,
                    ep.name,
                    exc,
                )
                last_exc = exc
                continue

        raise AllEndpointsExhausted(
            f"All endpoints in pool '{self.name}' exhausted"
        ) from last_exc
