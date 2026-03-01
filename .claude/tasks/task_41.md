# Task 41: Briefing Generator Engine

## Wave
Wave 15 (parallel with tasks 31, 40; depends on tasks 21, 26)
Domain: backend

## Objective
Implement the briefing generator that searches indexed content (and optionally runs radar + ingest), assembles a context window of relevant documents, and sends it to an LLM to produce a structured briefing with themes, insights, and content ideas.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/briefing/generator.py` — `BriefingGenerator`
- `config/prompts/briefing.md` — Briefing prompt template
- `backend/ai_craftsman_kb/api/briefings.py` — Implement `POST /api/briefings` (extends task_30 stub)
- `backend/tests/test_briefing/test_generator.py`

### Key interfaces / implementation details:

**Briefing prompt template** (`config/prompts/briefing.md`):
```
You are a research assistant helping a technical writer understand what's happening in their field.

Below are {doc_count} documents from sources they follow, all related to: "{topic}"

---
{document_summaries}
---

Based on these documents, provide a structured briefing with:

## Key Themes
List 3-5 major themes or narratives emerging from this content.

## Unique Angles
What perspectives or stories are NOT being widely covered that might be worth exploring?
What's missing from the conversation?

## Content Ideas
Suggest 3-5 specific article or video ideas based on gaps and opportunities you see.
Format: "Title idea" → one-sentence explanation of the angle.

## Notable Entities
List key people, companies, and technologies mentioned across multiple sources.

Keep the briefing concise and actionable. Focus on insights that would help a creator develop original content.
```

**`BriefingGenerator`** (`briefing/generator.py`):
```python
class BriefingGenerator:
    """Generate content briefings via LLM using hybrid search + optional radar."""

    def __init__(
        self,
        config: AppConfig,
        llm_router: LLMRouter,
        hybrid_search: HybridSearch,
        radar_engine: RadarEngine,
        ingest_runner: IngestRunner,
    ) -> None: ...

    async def generate(
        self,
        conn: aiosqlite.Connection,
        topic: str,
        run_radar: bool = True,
        run_ingest: bool = True,
        limit: int = 20,
    ) -> BriefingRow:
        """
        Pipeline:
        1. If run_ingest: trigger pro ingest for all sources (fire-and-forget or await)
        2. If run_radar: run RadarEngine.search(topic) to pull fresh open-web content
        3. HybridSearch.search(topic, limit=limit) → top relevant documents
        4. Assemble document_summaries: title + excerpt for each doc (max 500 chars each)
        5. Build prompt from briefing.md template
        6. LLMRouter.complete(task='briefing', prompt=...) → raw briefing text
        7. Save to briefings table → return BriefingRow
        """

    def _assemble_context(self, docs: list[SearchResult], topic: str) -> str:
        """Format documents into the {document_summaries} block.
        Each doc: '### {title}\nSource: {source_type} | {published_at}\n{excerpt}\n'
        Truncate total to stay within LLM context limit (~12k tokens)."""

    def _extract_title(self, content: str, topic: str) -> str:
        """Extract or generate briefing title from LLM output or use 'Briefing: {topic}'."""
```

**Context assembly** — budget per document:
- Max documents: 20
- Max chars per document: 800 (title + excerpt)
- Total context budget: ~16,000 chars → ~4,000 tokens (well within context window)

**LLM routing**: `task='briefing'` → routes to `anthropic/claude-sonnet-4-20250514` by default (from settings.yaml).

**Storage**: After generation, insert into `briefings` table with:
- `title`: extracted from LLM output or `f"Briefing: {topic}"`
- `query`: the `topic` string
- `content`: full LLM output markdown
- `source_document_ids`: JSON list of document IDs used
- `format`: `'markdown'`

**API implementation** (`api/briefings.py`):
```python
@router.post('/briefings', response_model=BriefingOut)
async def create_briefing(body: CreateBriefingRequest, request: Request):
    generator = BriefingGenerator(
        config=request.app.state.config,
        llm_router=request.app.state.llm_router,
        hybrid_search=...,
        radar_engine=...,
        ingest_runner=...,
    )
    async with get_db(db_path) as conn:
        briefing = await generator.generate(conn, body.query, body.run_radar, body.run_ingest)
    return briefing_to_out(briefing)
```

## Dependencies
- Depends on: task_21 (HybridSearch), task_26 (RadarEngine), task_04 (LLMRouter)
- Packages needed: none new

## Acceptance Criteria
- [ ] `POST /api/briefings {"query": "LLM inference"}` returns a `BriefingOut` with non-empty `content`
- [ ] Briefing content is structured markdown with Key Themes, Unique Angles, Content Ideas sections
- [ ] `run_radar=true` triggers `RadarEngine.search()` before generating
- [ ] `run_ingest=true` triggers a pro ingest before generating
- [ ] Source document IDs stored in `briefings.source_document_ids`
- [ ] Briefing saved to DB and retrievable via `GET /api/briefings/{id}`
- [ ] Context assembly truncates at ~4000 tokens to stay within LLM limits
- [ ] Unit tests mock LLMRouter and HybridSearch; verify DB insert

## Notes
- `run_ingest=true` in a web request context means awaiting the full ingest — this can take 30–60 seconds. Consider making it truly async (background task) vs synchronous. For now: synchronous with a long timeout.
- The briefing prompt is loaded from `config/prompts/briefing.md` at runtime — users can customize it
- LLM output is stored verbatim — no post-processing needed (markdown renders fine in dashboard)
- Estimated cost: ~$0.01–0.05 per briefing with Claude Sonnet (4k token context)
