"""Entity extraction pipeline using LLM to identify named entities in documents.

Extracts people, companies, technologies, events, books, papers, and products
from document content and persists them to the entities and document_entities tables.

Deduplication is handled by the DB UNIQUE(normalized_name, entity_type) constraint.
Content is truncated to 4000 tokens before LLM submission to control costs.
"""
import json
import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
import tiktoken
from pydantic import BaseModel

from ..db.models import EntityRow, utcnow_iso
from ..db.queries import link_document_entity, update_document_flags, upsert_entity

if TYPE_CHECKING:
    from ..config.models import AppConfig
    from ..llm.router import LLMRouter

logger = logging.getLogger(__name__)

# Valid entity types as defined in the data schema and task spec
VALID_ENTITY_TYPES = frozenset(
    {"person", "company", "technology", "event", "book", "paper", "product"}
)

# Maximum number of tokens to send to the LLM for extraction (cost control)
_MAX_TOKENS = 4000

# Default tiktoken encoding to use when counting tokens
_TIKTOKEN_ENCODING = "cl100k_base"

# Path to the prompt template relative to the project config directory
_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "config" / "prompts" / "entity_extraction.md"


class ExtractedEntity(BaseModel):
    """A single named entity extracted from document content by the LLM.

    Attributes:
        name: The canonical/most common form of the entity name.
        type: Entity category — must be one of the 7 valid types.
        context: Short excerpt (<=100 chars) showing how the entity is mentioned.
    """

    name: str
    type: str  # validated against VALID_ENTITY_TYPES after parsing
    context: str


class EntityExtractor:
    """Extract and store named entities from documents using an LLM.

    Uses the LLMRouter with task='entity_extraction' to run the prompt and
    parse the JSON response. Entities are upserted into the entities table
    using (normalized_name, entity_type) for deduplication, then linked to
    the source document via the document_entities join table.

    The extraction is idempotent — callers should skip documents where
    is_entities_extracted is already True.

    Usage::

        extractor = EntityExtractor(config, llm_router)
        entities = await extractor.extract_and_store(conn, doc_id, content)

    Args:
        config: Fully loaded AppConfig.
        llm_router: Configured LLM router for making completion requests.
    """

    def __init__(self, config: "AppConfig", llm_router: "LLMRouter") -> None:
        self.config = config
        self.llm_router = llm_router
        self._prompt_template = self._load_prompt_template()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, content: str) -> list[ExtractedEntity]:
        """Run LLM extraction on content truncated to 4000 tokens.

        Sends the content (truncated to avoid excessive LLM costs) to the
        entity_extraction LLM task and parses the JSON response. Filters
        entities whose type is not in the 7 valid types. Returns an empty
        list on any LLM or parse error so the pipeline does not crash.

        Args:
            content: Raw text content to extract entities from.

        Returns:
            List of ExtractedEntity objects parsed from the LLM response.
            Returns [] on error (error is logged, not raised).
        """
        truncated = self._truncate_to_tokens(content, _MAX_TOKENS)
        prompt = self._prompt_template.replace("{content}", truncated)

        try:
            response = await self.llm_router.complete(
                task="entity_extraction",
                prompt=prompt,
            )
        except Exception as exc:
            logger.error(
                "LLM call failed during entity extraction: %s — returning empty list",
                exc,
            )
            return []

        return self._parse_llm_response(response)

    async def extract_and_store(
        self,
        conn: aiosqlite.Connection,
        document_id: str,
        content: str,
    ) -> list[ExtractedEntity]:
        """Extract entities, upsert to the entities table, and link to document.

        Steps:
        1. Calls extract() to get entities from the LLM.
        2. For each entity, upserts an EntityRow (incrementing mention_count on
           conflict via the DB UNIQUE constraint).
        3. Creates a document_entities link with the LLM context snippet.
        4. Marks the document's is_entities_extracted flag as True.

        If extraction returns an empty list (LLM error or no entities found),
        the document is still marked as processed so it is not retried endlessly.

        Args:
            conn: Open aiosqlite connection with row_factory configured.
            document_id: UUID of the document being processed.
            content: Raw text content to extract entities from.

        Returns:
            The list of ExtractedEntity objects that were stored.
        """
        entities = await self.extract(content)

        for entity in entities:
            normalized = self._normalize_name(entity.name)
            # Check whether entity already exists in DB to preserve mention_count
            existing_id, existing_count = await self._get_existing_entity(
                conn, normalized, entity.type
            )

            if existing_id is not None:
                # Entity already exists — increment mention_count
                entity_id = existing_id
                await self._increment_mention_count(conn, entity_id, existing_count)
            else:
                # New entity — insert fresh row
                entity_id = str(uuid.uuid4())
                entity_row = EntityRow(
                    id=entity_id,
                    name=entity.name,
                    entity_type=entity.type,
                    normalized_name=normalized,
                    description=None,
                    first_seen_at=utcnow_iso(),
                    mention_count=1,
                    metadata={},
                )
                await upsert_entity(conn, entity_row)

            # Link document to entity with the context snippet from LLM
            await link_document_entity(
                conn,
                document_id=document_id,
                entity_id=entity_id,
                context=entity.context[:100],  # enforce 100-char limit
            )
            logger.debug(
                "Linked entity '%s' (%s) to document %s",
                entity.name,
                entity.type,
                document_id,
            )

        # Mark document as processed regardless of how many entities were found
        await update_document_flags(
            conn,
            doc_id=document_id,
            is_entities_extracted=True,
        )
        logger.info(
            "Entity extraction complete for document %s — %d entities stored",
            document_id,
            len(entities),
        )
        return entities

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_name(self, name: str) -> str:
        """Normalise an entity name for deduplication purposes.

        Converts to lowercase and strips leading/trailing whitespace.
        'OpenAI' -> 'openai', '  Meta  ' -> 'meta'.

        Args:
            name: The raw entity name string.

        Returns:
            The normalised name suitable for DB deduplication.
        """
        return name.strip().lower()

    def _parse_llm_response(self, response: str) -> list[ExtractedEntity]:
        """Parse a JSON entity list from the LLM response string.

        Handles responses wrapped in markdown code fences (```json ... ```)
        as well as bare JSON arrays. Filters entries whose type field is not
        in VALID_ENTITY_TYPES. Logs and skips malformed individual entries.

        On a top-level JSON parse error, logs the error and returns an empty
        list so the pipeline continues without crashing.

        Args:
            response: Raw string returned by the LLM completion.

        Returns:
            A list of valid ExtractedEntity objects. May be empty on failure.
        """
        # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
        cleaned = response.strip()
        fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse LLM entity extraction response as JSON: %s\n"
                "Response snippet: %.200s",
                exc,
                response,
            )
            return []

        if not isinstance(parsed, list):
            logger.warning(
                "LLM entity extraction response is not a JSON array (got %s) — skipping",
                type(parsed).__name__,
            )
            return []

        entities: list[ExtractedEntity] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                logger.warning("Entity entry %d is not a dict — skipping: %r", i, item)
                continue

            # Validate required fields
            name = item.get("name", "").strip()
            entity_type = item.get("type", "").strip()
            context = item.get("context", "").strip()

            if not name:
                logger.debug("Entity entry %d has empty name — skipping", i)
                continue

            if entity_type not in VALID_ENTITY_TYPES:
                logger.warning(
                    "Entity '%s' has invalid type '%s' — valid types are %s — skipping",
                    name,
                    entity_type,
                    sorted(VALID_ENTITY_TYPES),
                )
                continue

            entities.append(ExtractedEntity(name=name, type=entity_type, context=context))

        return entities

    def _load_prompt_template(self) -> str:
        """Load the entity extraction prompt template from disk.

        Reads config/prompts/entity_extraction.md relative to the project root.
        Falls back to a minimal inline template if the file cannot be read,
        so the system degrades gracefully in test or misconfigured environments.

        Returns:
            The prompt template string with a {content} placeholder.
        """
        # Try the canonical path computed at module load time
        candidates = [
            _PROMPT_PATH,
            # Also try relative to cwd for environments where the module location differs
            Path("config") / "prompts" / "entity_extraction.md",
        ]
        for path in candidates:
            try:
                template = path.read_text(encoding="utf-8")
                logger.debug("Loaded entity extraction prompt from %s", path)
                return template
            except OSError:
                continue

        # Fallback inline prompt
        logger.warning(
            "Could not load entity_extraction.md prompt template — using inline fallback"
        )
        return (
            "Extract named entities from the following text.\n\n"
            "For each entity found, provide:\n"
            "- name: the canonical/most common name\n"
            "- type: one of [person, company, technology, event, book, paper, product]\n"
            "- context: a short phrase showing how it's mentioned (max 100 chars)\n\n"
            "Rules:\n"
            "- Only include clearly and explicitly mentioned entities\n"
            "- Use canonical names (e.g. \"Meta\" not \"Facebook, Inc.\")\n"
            "- Prefer specific names over generic terms\n"
            "- If no entities found, return an empty array\n\n"
            "Return ONLY a JSON array with no other text:\n"
            '[{"name": "...", "type": "...", "context": "..."}, ...]\n\n'
            "Text:\n{content}"
        )

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to at most max_tokens tokens using tiktoken.

        Uses the cl100k_base encoding (compatible with GPT-3.5/4 and most
        other modern LLMs for approximate token counting). If tiktoken fails
        for any reason, falls back to a conservative character-based truncation
        (4 chars ≈ 1 token) to avoid sending oversized prompts.

        Args:
            text: The text to truncate.
            max_tokens: Maximum number of tokens to retain.

        Returns:
            The (possibly truncated) text string.
        """
        try:
            enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            # Decode only the first max_tokens tokens back to a string
            return enc.decode(tokens[:max_tokens])
        except Exception as exc:
            logger.warning(
                "tiktoken truncation failed (%s) — falling back to char-based truncation",
                exc,
            )
            # Rough fallback: 4 characters per token
            char_limit = max_tokens * 4
            return text[:char_limit]

    async def _get_existing_entity(
        self,
        conn: aiosqlite.Connection,
        normalized_name: str,
        entity_type: str,
    ) -> tuple[str | None, int]:
        """Look up an entity by (normalized_name, entity_type).

        Args:
            conn: Open aiosqlite connection.
            normalized_name: The lowercase-stripped entity name.
            entity_type: The entity category string.

        Returns:
            A tuple of (entity_id, mention_count) if found, or (None, 0).
        """
        async with conn.execute(
            "SELECT id, mention_count FROM entities "
            "WHERE normalized_name = ? AND entity_type = ?",
            (normalized_name, entity_type),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None, 0
        return row["id"], row["mention_count"]

    async def _increment_mention_count(
        self,
        conn: aiosqlite.Connection,
        entity_id: str,
        current_count: int,
    ) -> None:
        """Increment the mention_count on an existing entity row.

        Args:
            conn: Open aiosqlite connection.
            entity_id: UUID of the entity to update.
            current_count: The current mention_count value (used for optimistic update).
        """
        await conn.execute(
            "UPDATE entities SET mention_count = ? WHERE id = ?",
            (current_count + 1, entity_id),
        )
        await conn.commit()
