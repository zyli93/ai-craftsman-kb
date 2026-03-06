"""Async rate limiter for controlling LLM request rates."""
import asyncio
import time


class DailyLimitExceeded(Exception):
    """Raised when the daily request limit has been reached."""


class AsyncRateLimiter:
    """Token-bucket rate limiter using asyncio.

    Ensures that concurrent callers are serialised through a lock and
    spaced at least ``60 / rpm`` seconds apart.  When multiple coroutines
    call :meth:`acquire` simultaneously, they queue behind the lock and
    each waits only for its own remaining interval.

    Optionally enforces a daily request cap.  The daily counter resets
    automatically when the calendar day rolls over (based on monotonic
    time tracking of the first request each day).

    Args:
        rpm: Maximum requests per minute.
        daily_limit: Maximum requests per day.  None = unlimited.
    """

    def __init__(self, rpm: float, daily_limit: int | None = None) -> None:
        self._min_interval: float = 60.0 / rpm  # seconds between requests
        self._last_request: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()
        self._daily_limit = daily_limit
        self._daily_count: int = 0
        self._day_start: float = 0.0  # monotonic timestamp of first request today

    def _check_day_reset(self) -> None:
        """Reset the daily counter if 24 hours have elapsed since day_start."""
        now = time.monotonic()
        if self._day_start == 0.0 or (now - self._day_start) >= 86400:
            self._daily_count = 0
            self._day_start = now

    @property
    def daily_remaining(self) -> int | None:
        """Return remaining daily requests, or None if no daily limit."""
        if self._daily_limit is None:
            return None
        self._check_day_reset()
        return max(0, self._daily_limit - self._daily_count)

    async def acquire(self) -> None:
        """Block until a request slot is available.

        Raises:
            DailyLimitExceeded: If the daily request cap has been reached.
        """
        async with self._lock:
            self._check_day_reset()

            if self._daily_limit is not None and self._daily_count >= self._daily_limit:
                raise DailyLimitExceeded(
                    f"Daily limit of {self._daily_limit} requests reached. "
                    f"Resets in {int(86400 - (time.monotonic() - self._day_start))}s."
                )

            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request = time.monotonic()
            self._daily_count += 1
