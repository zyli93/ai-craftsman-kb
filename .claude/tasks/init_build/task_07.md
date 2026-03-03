# Task 07: Content Filter (LLM + Keyword + Hybrid)

## Wave
Wave 3 (parallel with tasks: 06, 08)
Domain: backend

## Objective
Implement a content filtering module that scores and passes/rejects `RawDocument` items based on three strategies: LLM scoring, keyword matching, or a hybrid of both. Filter configuration is per source type (from filters.yaml).

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/processing/filter.py` — `ContentFilter` class with all three strategies
- `backend/tests/test_processing/test_filter.py` — Unit tests with mocked LLM

### Key interfaces / implementation details:

**`ContentFilter`** (`processing/filter.py`):
```python
class FilterResult(BaseModel):
    passed: bool
    score: float | None       # 1–10 for LLM, 0–1 for keyword, None if not scored
    reason: str | None        # human-readable explanation

class ContentFilter:
    """Filter RawDocuments based on per-source filter config from filters.yaml."""

    def __init__(self, config: AppConfig, llm_router: LLMRouter) -> None:
        self.config = config
        self.llm_router = llm_router

    async def filter(
        self,
        doc: RawDocument,
        source_type: str,
    ) -> FilterResult:
        """Apply configured filter strategy for this source type."""
        filter_cfg = self._get_source_filter(source_type)
        if not filter_cfg.enabled:
            return FilterResult(passed=True, score=None, reason='filter disabled')
        if filter_cfg.strategy == 'llm':
            return await self._llm_filter(doc, filter_cfg)
        elif filter_cfg.strategy == 'keyword':
            return self._keyword_filter(doc, filter_cfg)
        else:  # hybrid
            return await self._hybrid_filter(doc, filter_cfg)

    async def filter_batch(
        self,
        docs: list[RawDocument],
        source_type: str,
        concurrency: int = 5,
    ) -> list[tuple[RawDocument, FilterResult]]:
        """Filter a list concurrently with bounded concurrency."""

    async def _llm_filter(self, doc: RawDocument, cfg: SourceFilterConfig) -> FilterResult:
        """Use LLM to score relevance 1–10. Prompt from cfg.llm_prompt.
        Substitutes {title} and {excerpt} (first 500 chars of raw_content).
        min_score threshold from cfg.min_score."""

    def _keyword_filter(self, doc: RawDocument, cfg: SourceFilterConfig) -> FilterResult:
        """Pass if:
        - any cfg.keywords_include word found in title+content (if list non-empty)
        - no cfg.keywords_exclude word found in title+content
        - cfg.min_upvotes / min_reactions satisfied (from doc.metadata)
        Score = 0.0 (fail) or 1.0 (pass)."""

    async def _hybrid_filter(self, doc: RawDocument, cfg: SourceFilterConfig) -> FilterResult:
        """Run keyword filter first. If it passes, run LLM filter.
        Both must pass for hybrid to pass."""

    def _get_source_filter(self, source_type: str) -> SourceFilterConfig:
        """Look up filter config by source_type key from FiltersConfig."""
```

**LLM prompt flow** (for LLM strategy):
- Template: `cfg.llm_prompt` with `{title}` and `{excerpt}` substituted
- Excerpt = first 500 chars of `raw_content`
- LLM call: `llm_router.complete(task='filtering', prompt=filled_prompt)`
- Parse response: extract integer 1–10 from response string
- Pass if score >= `cfg.min_score` (default 5)

**Keyword logic**:
- Check `keywords_exclude` against `(title + ' ' + raw_content[:2000]).lower()`
- Check `keywords_include` — if list non-empty, at least one must match
- Check `min_upvotes` from `doc.metadata.get('upvotes', 0)` (Reddit)
- Check `min_reactions` from `doc.metadata.get('reactions', 0)` (DEV.to)

## Dependencies
- Depends on: task_04 (LLMRouter for LLM strategy), task_05 (RawDocument, BaseIngestor types)
- Packages needed: none new

## Acceptance Criteria
- [ ] All three strategies (llm, keyword, hybrid) implemented and tested
- [ ] `filter_batch()` respects concurrency limit via `asyncio.Semaphore`
- [ ] LLM filter correctly parses integer score from varied LLM response formats (e.g. "7", "7/10", "Score: 7")
- [ ] Keyword filter handles empty `keywords_include` (all pass) and `keywords_exclude` (none blocked)
- [ ] `FilterResult.passed` and `filter_score` correctly set for each strategy
- [ ] Source types with `enabled: false` always return `passed=True` without LLM call
- [ ] Unit tests mock `LLMRouter.complete()` to return known strings; test all edge cases

## Notes
- Filtering happens AFTER fetching from source, BEFORE storing to DB
- The `filter_score` and `filter_passed` fields on `DocumentRow` are populated from `FilterResult`
- Keep the `_llm_filter` timeout-safe — LLM calls are wrapped in the router's retry logic (task_04)
- Hybrid: keyword pass → LLM score, keyword fail → short-circuit to fail without LLM call (saves cost)
- Use `asyncio.gather` with a semaphore for batch filtering
