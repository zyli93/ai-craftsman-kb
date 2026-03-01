"""OpenRouter provider — OpenAI-compatible API via httpx."""
import logging

import httpx

from .base import LLMProvider
from .retry import with_retry

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Only these kwargs are forwarded to the OpenRouter API to avoid errors
_ALLOWED_KWARGS = {"temperature", "max_tokens", "top_p", "stop"}


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider for chat completions.

    Uses the OpenAI-compatible API endpoint at openrouter.ai.
    Does not support embeddings — use OpenAIProvider for that.

    Args:
        api_key: OpenRouter API key.
        model: Model identifier (e.g. 'meta-llama/llama-3.1-8b-instruct').
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                # OpenRouter recommends including these headers
                "HTTP-Referer": "https://github.com/ai-craftsman-kb",
                "X-Title": "AI Craftsman KB",
            },
            timeout=60.0,
        )

    async def complete(self, prompt: str, system: str = "", **kwargs: object) -> str:
        """Call OpenRouter chat completions (OpenAI-compatible).

        Args:
            prompt: The user message.
            system: Optional system message.
            **kwargs: Optional parameters: temperature, max_tokens, top_p, stop.

        Returns:
            The model's text response.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if k in _ALLOWED_KWARGS},
        }

        async def _call() -> str:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"] or ""

        return await with_retry(
            _call, operation_name=f"OpenRouter complete [{self.model}]"
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Not supported by OpenRouter.

        Raises:
            NotImplementedError: Always. Use OpenAIProvider for embeddings.
        """
        raise NotImplementedError(
            "OpenRouter does not support embeddings — use OpenAIProvider instead."
        )
