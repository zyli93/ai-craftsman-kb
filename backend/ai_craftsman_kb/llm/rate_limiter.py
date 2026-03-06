"""Async rate limiter for controlling LLM request rates."""
import asyncio
import time


class AsyncRateLimiter:
    """Token-bucket rate limiter using asyncio.

    Ensures that concurrent callers are serialised through a lock and
    spaced at least ``60 / rpm`` seconds apart.  When multiple coroutines
    call :meth:`acquire` simultaneously, they queue behind the lock and
    each waits only for its own remaining interval.

    Args:
        rpm: Maximum requests per minute.
    """

    def __init__(self, rpm: float) -> None:
        self._min_interval: float = 60.0 / rpm  # seconds between requests
        self._last_request: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a request slot is available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request = time.monotonic()
