"""Keyword extraction pipeline using LLM to identify key topics in documents.

Extracts 5-15 keywords/phrases from document content and persists them to the
document_keywords table. Keywords are normalized (lowercase, stripped, deduplicated)
before storage. Content is truncated to 4000 tokens before LLM submission to
control costs.
"""
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
import tiktoken

from ..db.queries import update_document_flags

if TYPE_CHECKING:
    from ..config.models import AppConfig
    from ..llm.router import LLMRouter

logger = logging.getLogger(__name__)

# Maximum number of tokens to send to the LLM for extraction (cost control)
_MAX_TOKENS = 4000

# Default tiktoken encoding to use when counting tokens
_TIKTOKEN_ENCODING = "cl100k_base"

# Path to the prompt template relative to the project config directory
_PROMPT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "config"
    / "prompts"
    / "keyword_extraction.md"
)


class KeywordExtractor:
    """Extract and store keywords from documents using an LLM.

    Uses the LLMRouter with task='keyword_extraction' to run the prompt and
    parse the JSON response. Keywords are normalized to lowercase, stripped of
    whitespace, and deduplicated before insertion into the document_keywords
    table.

    The extraction is idempotent -- callers should skip documents where
    is_keywords_extracted is already True.

    Usage::

        extractor = KeywordExtractor(config, llm_router)
        keywords = await extractor.extract_and_store(conn, doc_id, content)

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

    async def extract(self, content: str) -> list[str]:
        """Run LLM extraction on content truncated to 4000 tokens.

        Sends the content (truncated to avoid excessive LLM costs) to the
        keyword_extraction LLM task and parses the JSON response. Returns
        normalized, deduplicated keywords. Returns an empty list on any LLM
        or parse error so the pipeline does not crash.

        Args:
            content: Raw text content to extract keywords from.

        Returns:
            List of normalized keyword strings parsed from the LLM response.
            Returns [] on error (error is logged, not raised).
        """
        truncated = self._truncate_to_tokens(content, _MAX_TOKENS)
        prompt = self._prompt_template.replace("{content}", truncated)

        try:
            response = await self.llm_router.complete(
                task="keyword_extraction",
                prompt=prompt,
            )
        except Exception as exc:
            logger.error(
                "LLM call failed during keyword extraction: %s -- returning empty list",
                exc,
            )
            return []

        return self._parse_llm_response(response)

    async def extract_and_store(
        self,
        conn: aiosqlite.Connection,
        document_id: str,
        content: str,
    ) -> list[str]:
        """Extract keywords, insert into document_keywords, and flag document.

        Steps:
        1. Calls extract() to get keywords from the LLM.
        2. Inserts each keyword into document_keywords (skipping duplicates
           via INSERT OR IGNORE on the UNIQUE(document_id, keyword) constraint).
        3. Marks the document's is_keywords_extracted flag as True.

        If extraction returns an empty list (LLM error or no keywords found),
        the document is still marked as processed so it is not retried endlessly.

        Args:
            conn: Open aiosqlite connection with row_factory configured.
            document_id: UUID of the document being processed.
            content: Raw text content to extract keywords from.

        Returns:
            The list of normalized keyword strings that were stored.
        """
        keywords = await self.extract(content)

        for keyword in keywords:
            await conn.execute(
                "INSERT OR IGNORE INTO document_keywords (document_id, keyword) VALUES (?, ?)",
                (document_id, keyword),
            )
        await conn.commit()

        logger.debug(
            "Stored %d keywords for document %s",
            len(keywords),
            document_id,
        )

        # Mark document as processed regardless of how many keywords were found
        await update_document_flags(
            conn,
            doc_id=document_id,
            is_keywords_extracted=True,
        )
        logger.info(
            "Keyword extraction complete for document %s -- %d keywords stored",
            document_id,
            len(keywords),
        )
        return keywords

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize a keyword for storage and deduplication.

        Converts to lowercase and strips leading/trailing whitespace.
        'Machine Learning' -> 'machine learning', '  API  ' -> 'api'.

        Args:
            keyword: The raw keyword string.

        Returns:
            The normalized keyword string.
        """
        return keyword.strip().lower()

    def _parse_llm_response(self, response: str) -> list[str]:
        """Parse a JSON keyword list from the LLM response string.

        Handles responses wrapped in markdown code fences (```json ... ```)
        as well as bare JSON arrays. Filters out empty strings and duplicates
        after normalization. Logs and returns an empty list on parse errors.

        Args:
            response: Raw string returned by the LLM completion.

        Returns:
            A list of normalized, deduplicated keyword strings. May be empty.
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
                "Failed to parse LLM keyword extraction response as JSON: %s\n"
                "Response snippet: %.200s",
                exc,
                response,
            )
            return []

        if not isinstance(parsed, list):
            logger.warning(
                "LLM keyword extraction response is not a JSON array (got %s) -- skipping",
                type(parsed).__name__,
            )
            return []

        # Normalize, filter empties, and deduplicate while preserving order
        seen: set[str] = set()
        keywords: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                logger.debug("Keyword entry is not a string (%r) -- skipping", item)
                continue
            normalized = self._normalize_keyword(item)
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            keywords.append(normalized)

        return keywords

    def _load_prompt_template(self) -> str:
        """Load the keyword extraction prompt template from disk.

        Reads config/prompts/keyword_extraction.md relative to the project root.
        Falls back to a minimal inline template if the file cannot be read,
        so the system degrades gracefully in test or misconfigured environments.

        Returns:
            The prompt template string with a {content} placeholder.
        """
        candidates = [
            _PROMPT_PATH,
            Path("config") / "prompts" / "keyword_extraction.md",
        ]
        for path in candidates:
            try:
                template = path.read_text(encoding="utf-8")
                logger.debug("Loaded keyword extraction prompt from %s", path)
                return template
            except OSError:
                continue

        # Fallback inline prompt
        logger.warning(
            "Could not load keyword_extraction.md prompt template -- using inline fallback"
        )
        return (
            "Extract 5 to 15 keywords or key phrases from the following text.\n\n"
            "Rules:\n"
            "- Focus on the most important topics, concepts, technologies, and themes\n"
            "- Include both single words and short phrases (2-3 words max)\n"
            "- Prefer specific terms over generic ones\n"
            "- If no meaningful keywords can be extracted, return an empty array\n\n"
            'Return ONLY a JSON array of strings with no other text:\n'
            '["keyword1", "keyword2", ...]\n\n'
            "Text:\n{content}"
        )

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to at most max_tokens tokens using tiktoken.

        Uses the cl100k_base encoding (compatible with GPT-3.5/4 and most
        other modern LLMs for approximate token counting). If tiktoken fails
        for any reason, falls back to a conservative character-based truncation
        (4 chars ~ 1 token) to avoid sending oversized prompts.

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
            return enc.decode(tokens[:max_tokens])
        except Exception as exc:
            logger.warning(
                "tiktoken truncation failed (%s) -- falling back to char-based truncation",
                exc,
            )
            char_limit = max_tokens * 4
            return text[:char_limit]
