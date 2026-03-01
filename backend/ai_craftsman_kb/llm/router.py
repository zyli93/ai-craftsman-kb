"""LLM Router — dispatches tasks to configured providers."""
import logging
from typing import TYPE_CHECKING

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider

if TYPE_CHECKING:
    from ..config.models import AppConfig

logger = logging.getLogger(__name__)

# Valid task names that map to llm routing config
TASK_NAMES = ("filtering", "entity_extraction", "briefing", "source_discovery")


class LLMRouter:
    """Routes LLM tasks to providers based on settings.yaml configuration.

    The router lazily instantiates providers on first use and caches them
    for subsequent calls within the same session.

    Usage::

        router = LLMRouter(config)
        result = await router.complete("filtering", prompt="...")
        embeddings = await router.embed(["text1", "text2"])

    Args:
        config: Fully loaded AppConfig from the config system.
    """

    def __init__(self, config: "AppConfig") -> None:
        self._config = config
        # Cache of lazily-instantiated task providers (keyed by task name)
        self._task_providers: dict[str, LLMProvider] = {}
        # Lazily-instantiated embedding provider
        self._embedding_provider: LLMProvider | None = None

    def _build_provider(self, provider_name: str, model: str) -> LLMProvider:
        """Instantiate a provider by name using configuration from AppConfig.

        Args:
            provider_name: One of 'openai', 'openrouter', 'anthropic',
                'ollama', 'fireworks'.
            model: The model identifier to pass to the provider.

        Returns:
            An instantiated LLMProvider.

        Raises:
            ValueError: If the provider is unknown or required API key is missing.
        """
        providers_cfg = self._config.settings.providers
        cfg = providers_cfg.get(provider_name)
        api_key = cfg.api_key if cfg else None

        if provider_name == "openai":
            if not api_key:
                raise ValueError(
                    "OpenAI API key not configured. "
                    "Set OPENAI_API_KEY or add it to config/settings.yaml."
                )
            return OpenAIProvider(api_key=api_key, model=model)

        if provider_name == "openrouter":
            if not api_key:
                raise ValueError(
                    "OpenRouter API key not configured. "
                    "Set OPENROUTER_API_KEY or add it to config/settings.yaml."
                )
            return OpenRouterProvider(api_key=api_key, model=model)

        if provider_name == "anthropic":
            if not api_key:
                raise ValueError(
                    "Anthropic API key not configured. "
                    "Set ANTHROPIC_API_KEY or add it to config/settings.yaml."
                )
            return AnthropicProvider(api_key=api_key, model=model)

        if provider_name == "ollama":
            base_url = (cfg.base_url if cfg else None) or "http://localhost:11434"
            return OllamaProvider(base_url=base_url, model=model)

        if provider_name == "fireworks":
            # Fireworks uses an OpenAI-compatible API
            if not api_key:
                raise ValueError(
                    "Fireworks API key not configured. "
                    "Set FIREWORKS_API_KEY or add it to config/settings.yaml."
                )
            return OpenAIProvider(
                api_key=api_key,
                model=model,
                base_url="https://api.fireworks.ai/inference/v1",
            )

        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Supported providers: openai, openrouter, anthropic, ollama, fireworks."
        )

    def _get_task_provider(self, task: str) -> LLMProvider:
        """Get or create the cached provider for a specific task.

        Args:
            task: Task name (must be in TASK_NAMES).

        Returns:
            The configured LLMProvider for the task.
        """
        if task not in self._task_providers:
            # Access the per-task config: config.settings.llm.<task>
            task_cfg = getattr(self._config.settings.llm, task)
            self._task_providers[task] = self._build_provider(
                task_cfg.provider, task_cfg.model
            )
        return self._task_providers[task]

    def _get_embedding_provider(self) -> LLMProvider:
        """Get or create the cached embedding provider.

        Returns:
            The configured embedding LLMProvider.
        """
        if self._embedding_provider is None:
            emb_cfg = self._config.settings.embedding
            self._embedding_provider = self._build_provider(
                emb_cfg.provider, emb_cfg.model
            )
        return self._embedding_provider

    async def complete(
        self,
        task: str,
        prompt: str,
        system: str = "",
        **kwargs: object,
    ) -> str:
        """Route a completion request to the configured provider for the task.

        Args:
            task: One of 'filtering', 'entity_extraction', 'briefing',
                'source_discovery'.
            prompt: The user message.
            system: Optional system message.
            **kwargs: Additional provider-specific options (e.g. temperature,
                max_tokens).

        Returns:
            The generated text response.

        Raises:
            ValueError: If the task name is not recognised.
            Exception: Re-raises provider errors after logging.
        """
        if task not in TASK_NAMES:
            raise ValueError(
                f"Unknown task: '{task}'. Must be one of {TASK_NAMES}."
            )
        provider = self._get_task_provider(task)
        try:
            return await provider.complete(prompt=prompt, system=system, **kwargs)
        except Exception:
            logger.exception("LLM completion failed for task '%s'", task)
            raise

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the configured embedding provider.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string.

        Raises:
            Exception: Re-raises provider errors after logging.
        """
        provider = self._get_embedding_provider()
        try:
            return await provider.embed(texts)
        except Exception:
            logger.exception("Embedding failed")
            raise
