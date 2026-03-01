"""OpenAI provider for completions and embeddings."""
import logging

from openai import AsyncOpenAI

from .base import LLMProvider
from .retry import with_retry

logger = logging.getLogger(__name__)

# Default embedding model when used specifically for embeddings
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIProvider(LLMProvider):
    """OpenAI provider supporting GPT models and text-embedding-3-small/large.

    This provider uses the official openai SDK which handles connection
    pooling and batching efficiently.

    Args:
        api_key: OpenAI API key.
        model: Chat completion model name (e.g. 'gpt-4o-mini').
        embedding_model: Embedding model name (e.g. 'text-embedding-3-small').
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            **({"base_url": base_url} if base_url else {}),
        )
        self.model = model
        self.embedding_model = embedding_model

    async def complete(self, prompt: str, system: str = "", **kwargs: object) -> str:
        """Call OpenAI chat completions API.

        Args:
            prompt: The user message.
            system: Optional system message prepended to the conversation.
            **kwargs: Extra options forwarded to the API (e.g. temperature,
                max_tokens).

        Returns:
            The assistant's text response.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Filter to known safe kwargs to avoid API errors from unknown params
        safe_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("temperature", "max_tokens", "top_p", "stop", "stream")
        }

        async def _call() -> str:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                **safe_kwargs,
            )
            return response.choices[0].message.content or ""

        return await with_retry(_call, operation_name=f"OpenAI complete [{self.model}]")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts using the OpenAI embeddings API.

        The openai SDK handles batching internally for large lists of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string, in the same order.
        """

        async def _call() -> list[list[float]]:
            response = await self._client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            # Results are returned sorted by index, matching input order
            return [item.embedding for item in response.data]

        return await with_retry(
            _call, operation_name=f"OpenAI embed [{self.embedding_model}]"
        )
