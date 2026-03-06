"""LLM usage tracking -- records token consumption to SQLite.

Provides the UsageTracker class that writes usage rows to the ``llm_usage``
table and exposes query helpers for the dashboard API.
"""
import logging
from datetime import datetime
from pathlib import Path

from ..db.sqlite import get_db

logger = logging.getLogger(__name__)


class UsageTracker:
    """Records and queries LLM token usage stored in the ``llm_usage`` table.

    Designed to be injected into components that make LLM calls (e.g. the
    LLMRouter). Each instance is bound to a ``data_dir`` used to locate the
    SQLite database.

    Args:
        data_dir: Directory containing ``craftsman.db``.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    async def record(
        self,
        *,
        provider: str,
        model: str,
        task: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        duration_ms: int | None = None,
        success: bool = True,
    ) -> None:
        """Insert a single usage record into the ``llm_usage`` table.

        Args:
            provider: LLM provider name (e.g. ``"openai"``, ``"anthropic"``).
            model: Model identifier used for the request.
            task: Logical task name (e.g. ``"filtering"``, ``"briefing"``).
            input_tokens: Number of prompt/input tokens consumed.
            output_tokens: Number of completion/output tokens produced.
            duration_ms: Wall-clock duration of the request in milliseconds.
            success: Whether the request completed successfully.
        """
        async with get_db(self._data_dir) as conn:
            await conn.execute(
                """
                INSERT INTO llm_usage
                    (provider, model, task, input_tokens, output_tokens, duration_ms, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (provider, model, task, input_tokens, output_tokens, duration_ms, success),
            )
            await conn.commit()

    async def get_summary(self, since: datetime) -> list[dict]:
        """Return aggregated usage grouped by provider, model, and task.

        Args:
            since: Only include records with a timestamp at or after this value.

        Returns:
            A list of dicts, each containing ``provider``, ``model``, ``task``,
            ``total_input_tokens``, ``total_output_tokens``, and
            ``request_count``.
        """
        async with get_db(self._data_dir) as conn:
            cursor = await conn.execute(
                """
                SELECT
                    provider,
                    model,
                    task,
                    SUM(input_tokens)  AS total_input_tokens,
                    SUM(output_tokens) AS total_output_tokens,
                    COUNT(*)           AS request_count
                FROM llm_usage
                WHERE timestamp >= ?
                GROUP BY provider, model, task
                ORDER BY request_count DESC
                """,
                (since.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "provider": row["provider"],
                    "model": row["model"],
                    "task": row["task"],
                    "total_input_tokens": row["total_input_tokens"],
                    "total_output_tokens": row["total_output_tokens"],
                    "request_count": row["request_count"],
                }
                for row in rows
            ]

    async def get_recent(self, limit: int = 50) -> list[dict]:
        """Return the most recent usage records.

        Args:
            limit: Maximum number of records to return (default 50).

        Returns:
            A list of dicts with all columns from the ``llm_usage`` table,
            ordered by timestamp descending.
        """
        async with get_db(self._data_dir) as conn:
            cursor = await conn.execute(
                """
                SELECT
                    id, timestamp, provider, model, task,
                    input_tokens, output_tokens, duration_ms, success
                FROM llm_usage
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "task": row["task"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "duration_ms": row["duration_ms"],
                    "success": row["success"],
                }
                for row in rows
            ]
