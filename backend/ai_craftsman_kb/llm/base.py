"""Abstract base class for LLM providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Structured result from an LLM completion call.

    Attributes:
        text: The generated text response.
        input_tokens: Number of prompt/input tokens consumed, or None if
            the provider did not report usage.
        output_tokens: Number of completion/output tokens generated, or None
            if the provider did not report usage.
        model: The model identifier that produced the response.
    """

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None


class LLMProvider(ABC):
    """Abstract base for all LLM providers.

    All providers must implement async complete() and embed() methods.
    Providers that do not support embeddings should raise NotImplementedError
    in their embed() implementation.
    """

    @abstractmethod
    async def complete(
        self, prompt: str, system: str = "", **kwargs: object
    ) -> CompletionResult:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user message/prompt.
            system: Optional system message.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.).

        Returns:
            A CompletionResult containing the generated text and token usage.
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
