# Task 04: LLM Provider Abstraction

## Objective
Create an abstract LLM provider system that routes different tasks
(embedding, filtering, entity extraction, briefing) to different providers.

## Scope
### Files to create:
- `backend/ai_craftsman_kb/llm/base.py` — Abstract base class
- `backend/ai_craftsman_kb/llm/openai_provider.py`
- `backend/ai_craftsman_kb/llm/openrouter_provider.py`
- `backend/ai_craftsman_kb/llm/anthropic_provider.py`
- `backend/ai_craftsman_kb/llm/ollama_provider.py`
- `backend/ai_craftsman_kb/llm/router.py` — Routes tasks to configured providers

### Interface:
```python
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str = "", **kwargs) -> str: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class LLMRouter:
    """Routes tasks to providers based on settings.yaml config."""
    async def complete(self, task: str, prompt: str, ...) -> str: ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

## Dependencies
- Depends on: task_01 (project exists)
- Uses: config from task_02 (but can use hardcoded defaults for now)

## Acceptance Criteria
- [ ] Abstract base class with complete() and embed() methods
- [ ] OpenAI provider works for both completion and embedding
- [ ] OpenRouter provider works for completion
- [ ] Router reads settings.yaml and dispatches to correct provider per task
- [ ] Graceful error handling (API key missing, rate limit, timeout)
- [ ] Unit tests with mocked API calls

## Notes
- Use httpx for HTTP calls, not provider-specific SDKs where possible
  (except openai SDK for embeddings — it handles batching well)
- Every provider method should be async
- Include retry logic with exponential backoff
