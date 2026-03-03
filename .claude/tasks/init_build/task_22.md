# Task 22: Entity Extraction Pipeline

## Wave
Wave 7 (parallel with tasks: 18, 19)
Domain: backend

## Objective
Implement the entity extraction pipeline that uses an LLM to extract named entities (people, companies, technologies, events, books, papers, products) from document content and stores them in the `entities` and `document_entities` tables.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/processing/entity_extractor.py` — `EntityExtractor`
- `config/prompts/entity_extraction.md` — LLM prompt template
- `backend/tests/test_processing/test_entity_extractor.py`

### Key interfaces / implementation details:

**Entity types** (from plan.md):
- `person` — Named individuals
- `company` — Organizations, companies, institutions
- `technology` — Tools, frameworks, algorithms, models
- `event` — Named conferences, product releases, events
- `book` — Books and publications
- `paper` — Research papers and academic works
- `product` — Commercial products, services, platforms

**LLM prompt template** (`config/prompts/entity_extraction.md`):
```
Extract named entities from the following text.

For each entity found, provide:
- name: the canonical/most common name
- type: one of [person, company, technology, event, book, paper, product]
- context: a short phrase showing how it's mentioned (max 100 chars)

Rules:
- Only include clearly and explicitly mentioned entities
- Use canonical names (e.g. "Meta" not "Facebook, Inc.")
- Prefer specific names over generic terms
- If no entities found, return an empty array

Return ONLY a JSON array with no other text:
[{"name": "...", "type": "...", "context": "..."}, ...]

Text:
{content}
```

**`EntityExtractor`** (`processing/entity_extractor.py`):
```python
class ExtractedEntity(BaseModel):
    name: str
    type: str     # must be one of the 7 valid types
    context: str

class EntityExtractor:
    """Extract and store named entities from documents using LLM."""

    def __init__(self, config: AppConfig, llm_router: LLMRouter) -> None:
        self.config = config
        self.llm_router = llm_router
        self._prompt_template = self._load_prompt_template()

    async def extract(self, content: str) -> list[ExtractedEntity]:
        """Run LLM extraction on content (up to first 4000 tokens).
        Parse JSON response → validate entity types → return list."""

    async def extract_and_store(
        self,
        conn: aiosqlite.Connection,
        document_id: str,
        content: str,
    ) -> list[ExtractedEntity]:
        """Extract entities, upsert to entities table, link to document.
        Updates document.is_entities_extracted = True when done."""

    def _normalize_name(self, name: str) -> str:
        """Lowercase + strip for dedup: 'OpenAI' → 'openai'"""

    def _parse_llm_response(self, response: str) -> list[ExtractedEntity]:
        """Parse JSON from LLM response. Strip markdown code fences if present.
        Filter to valid entity types. Log and skip invalid entries."""

    def _load_prompt_template(self) -> str:
        """Load prompt from config/prompts/entity_extraction.md."""
```

**Dedup strategy** (via DB unique constraint):
- `normalize_name()`: lowercase + strip whitespace
- Insert via `upsert_entity(name, type, normalized_name)` — `UNIQUE(normalized_name, entity_type)` in DB
- On conflict: increment `mention_count`, update `description` if better

**Content truncation**: Extract from first 4000 tokens of `raw_content` (don't send entire docs to LLM — cost control). Use tiktoken to truncate.

**LLM routing**: uses `task='entity_extraction'` → routes to `openrouter/meta-llama/llama-3.1-8b-instruct` by default.

## Dependencies
- Depends on: task_04 (LLMRouter), task_03 (DB queries: upsert_entity, link_document_entity)
- Packages needed: none new

## Acceptance Criteria
- [ ] Extracts entities from a 500-word article (mocked LLM response)
- [ ] All 7 entity types parsed and validated; invalid types discarded with warning
- [ ] `normalized_name` is lowercase + stripped for dedup
- [ ] `extract_and_store()` inserts to `entities` table and `document_entities` table
- [ ] `document.is_entities_extracted` set to `True` after successful extraction
- [ ] JSON parse errors logged; extraction returns `[]` on failure (don't crash pipeline)
- [ ] Content truncated to 4000 tokens before sending to LLM
- [ ] Unit tests mock LLMRouter, verify DB writes with in-memory SQLite

## Notes
- LLM may return varied JSON formats (with/without markdown fences) — handle both
- Extraction runs once per document and is idempotent — skip docs where `is_entities_extracted = True`
- `mention_count` on entity is incremented per new document that mentions it (via `document_entities` insert)
- The `context` field in `document_entities` is the snippet from the LLM response showing how entity is mentioned
