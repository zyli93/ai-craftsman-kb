"""Ollama provider for local model inference."""
import logging

import httpx

from .base import CompletionResult, LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """Ollama provider for running local models.

    Uses Ollama's REST API which must be running locally.
    Supports both chat completions and embeddings.

    Args:
        base_url: Ollama server base URL (default 'http://localhost:11434').
        model: Model name as registered in Ollama (e.g. 'llama3', 'mistral').
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = "llama3",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=120.0,  # Local models can be slow on first token
        )

    async def complete(
        self, prompt: str, system: str = "", **kwargs: object
    ) -> CompletionResult:
        """Call the Ollama /api/chat endpoint.

        Args:
            prompt: The user message.
            system: Optional system message.
            **kwargs: Unused; included for interface compatibility.

        Returns:
            A CompletionResult with the response text and token usage.

        Raises:
            httpx.HTTPStatusError: On API errors.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        text = data["message"]["content"]
        return CompletionResult(
            text=text,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            model=self.model,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Ollama's /api/embeddings endpoint.

        Ollama does not support batch embedding, so texts are processed
        one at a time. This may be slow for large lists.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string, in the same order.

        Raises:
            httpx.HTTPStatusError: On API errors.
        """
        embeddings: list[list[float]] = []
        for text in texts:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])

        return embeddings
