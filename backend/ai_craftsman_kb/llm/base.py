"""Abstract base class for LLM providers."""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base for all LLM providers.

    All providers must implement async complete() and embed() methods.
    Providers that do not support embeddings should raise NotImplementedError
    in their embed() implementation.
    """

    @abstractmethod
    async def complete(self, prompt: str, system: str = "", **kwargs: object) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user message/prompt.
            system: Optional system message.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.).

        Returns:
            The generated text response.
        """
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...
