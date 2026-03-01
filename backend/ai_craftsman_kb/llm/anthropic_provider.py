"""Anthropic provider via httpx Messages API."""
import logging

import httpx

from .base import LLMProvider
from .retry import with_retry

logger = logging.getLogger(__name__)

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    """Anthropic provider for Claude models via the Messages API.

    Does not support embeddings — use OpenAIProvider for that.

    Args:
        api_key: Anthropic API key.
        model: Claude model identifier (e.g. 'claude-haiku-4-5-20251001').
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=ANTHROPIC_BASE_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
                "content-type": "application/json",
            },
            timeout=120.0,
        )

    async def complete(self, prompt: str, system: str = "", **kwargs: object) -> str:
        """Call the Anthropic Messages API.

        Args:
            prompt: The user message.
            system: Optional system prompt.
            **kwargs: Optional parameters forwarded to the API.
                Supported: max_tokens (default 4096), temperature.

        Returns:
            The assistant's text response.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        max_tokens = int(kwargs.get("max_tokens", DEFAULT_MAX_TOKENS))  # type: ignore[arg-type]

        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]

        async def _call() -> str:
            response = await self._client.post("/v1/messages", json=payload)
            response.raise_for_status()
            data = response.json()
            # Content is a list of content blocks; the first text block is the response
            return data["content"][0]["text"]

        return await with_retry(
            _call, operation_name=f"Anthropic complete [{self.model}]"
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Not supported by Anthropic.

        Raises:
            NotImplementedError: Always. Use OpenAIProvider for embeddings.
        """
        raise NotImplementedError(
            "Anthropic does not support embeddings — use OpenAIProvider instead."
        )
