"""Briefing generator engine.

Searches indexed content (optionally running radar + ingest first),
assembles a context window of relevant documents, and sends it to an LLM
to produce a structured briefing with themes, insights, and content ideas.
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

from ..db.models import BriefingRow
from ..db.queries import insert_briefing

if TYPE_CHECKING:
    from ..config.models import AppConfig
    from ..ingestors.runner import IngestRunner
    from ..llm.router import LLMRouter
    from ..radar.engine import RadarEngine
    from ..search.hybrid import HybridSearch, SearchResult

logger = logging.getLogger(__name__)

# Path to the briefing prompt template, relative to the project config directory.
# Resolved at runtime so users can customize it.
_PROMPT_TEMPLATE_PATHS = [
    # Production: config/prompts/briefing.md relative to the package root
    Path(__file__).parent.parent.parent.parent / "config" / "prompts" / "briefing.md",
    # Fallback for installed packages
    Path(__file__).parent.parent.parent / "config" / "prompts" / "briefing.md",
]

# Context budget constants
_MAX_CHARS_PER_DOC = 800   # title + excerpt per document
_MAX_TOTAL_CHARS = 16_000  # ~4,000 tokens — stays well within LLM context window


def _load_prompt_template() -> str:
    """Load the briefing prompt template from disk.

    Tries multiple candidate paths to support both development and installed
    package layouts. Falls back to a minimal inline template if no file is found.

    Returns:
        The raw template string with {topic}, {doc_count}, {document_summaries}
        placeholders.
    """
    for path in _PROMPT_TEMPLATE_PATHS:
        if path.exists():
            logger.debug("Loaded briefing prompt template from %s", path)
            return path.read_text(encoding="utf-8")

    logger.warning(
        "Briefing prompt template not found at any candidate path. Using fallback."
    )
    return (
        "You are a research assistant. Below are {doc_count} documents about: \"{topic}\"\n\n"
        "---\n{document_summaries}\n---\n\n"
        "Provide a structured briefing with Key Themes, Unique Angles, Content Ideas, "
        "and Notable Entities."
    )


class BriefingGenerator:
    """Generate content briefings via LLM using hybrid search + optional radar.

    The generation pipeline:
    1. Optionally trigger pro ingest (fire-and-forget background refresh).
    2. Optionally run RadarEngine.search(topic) to pull fresh open-web content.
    3. HybridSearch.search(topic, limit=limit) to find the top relevant documents.
    4. Assemble a context window from document title + excerpt (max 800 chars each).
    5. Render the briefing.md prompt template.
    6. Call LLMRouter.complete(task='briefing', prompt=...) for the LLM response.
    7. Insert the result into the briefings table and return a BriefingRow.

    Args:
        config: Fully loaded AppConfig.
        llm_router: Configured LLMRouter for LLM completions.
        hybrid_search: HybridSearch instance for document retrieval.
        radar_engine: RadarEngine for on-demand topic search.
        ingest_runner: IngestRunner for triggering pro ingest.
    """

    def __init__(
        self,
        config: "AppConfig",
        llm_router: "LLMRouter",
        hybrid_search: "HybridSearch",
        radar_engine: "RadarEngine",
        ingest_runner: "IngestRunner",
    ) -> None:
        self._config = config
        self._llm_router = llm_router
        self._hybrid_search = hybrid_search
        self._radar_engine = radar_engine
        self._ingest_runner = ingest_runner
        self._prompt_template = _load_prompt_template()

    async def generate(
        self,
        conn: aiosqlite.Connection,
        topic: str,
        run_radar: bool = True,
        run_ingest: bool = True,
        limit: int = 20,
    ) -> BriefingRow:
        """Generate a briefing for the given topic.

        Pipeline:
        1. If run_ingest: trigger pro ingest for all sources (synchronous, awaited).
        2. If run_radar: run RadarEngine.search(topic) to pull fresh open-web content.
        3. HybridSearch.search(topic, limit=limit) to get the top relevant documents.
        4. Assemble document_summaries: title + excerpt for each doc (max 500 chars each).
        5. Build prompt from briefing.md template.
        6. LLMRouter.complete(task='briefing', prompt=...) -> raw briefing text.
        7. Save to briefings table -> return BriefingRow.

        Args:
            conn: An open aiosqlite connection with row_factory=aiosqlite.Row.
            topic: The topic string to generate a briefing for.
            run_radar: If True, run RadarEngine.search(topic) before generating.
            run_ingest: If True, run pro ingest for all sources before generating.
                        Note: This can take 30-60 seconds.
            limit: Maximum number of source documents to include (default 20).

        Returns:
            A BriefingRow saved to the database.
        """
        # Step 1: Trigger pro ingest (synchronous — can take 30-60 seconds)
        if run_ingest:
            logger.info("BriefingGenerator: running pro ingest before generating...")
            try:
                await self._ingest_runner.run_all()
                logger.info("BriefingGenerator: pro ingest complete.")
            except Exception as exc:
                # Ingest failure should not block briefing generation
                logger.warning(
                    "BriefingGenerator: pro ingest failed (continuing): %s", exc
                )

        # Step 2: Run radar search to pull fresh content for this topic
        if run_radar:
            logger.info("BriefingGenerator: running radar search for %r...", topic)
            try:
                radar_report = await self._radar_engine.search(
                    conn, topic, limit_per_source=10
                )
                logger.info(
                    "BriefingGenerator: radar found %d docs (%d new) for %r",
                    radar_report.total_found,
                    radar_report.new_documents,
                    topic,
                )
            except Exception as exc:
                logger.warning(
                    "BriefingGenerator: radar search failed (continuing): %s", exc
                )

        # Step 3: Hybrid search to retrieve the most relevant documents
        logger.info(
            "BriefingGenerator: searching for top %d documents for %r...", limit, topic
        )
        try:
            docs = await self._hybrid_search.search(conn, topic, limit=limit)
        except Exception as exc:
            logger.error(
                "BriefingGenerator: hybrid search failed for %r: %s", topic, exc
            )
            docs = []

        logger.info("BriefingGenerator: retrieved %d documents for %r", len(docs), topic)

        # Steps 4 & 5: Assemble context and build prompt
        context = self._assemble_context(docs, topic)
        prompt = self._prompt_template.format(
            topic=topic,
            doc_count=len(docs),
            document_summaries=context,
        )

        # Step 6: LLM completion
        logger.info("BriefingGenerator: calling LLM for topic %r...", topic)
        try:
            result = await self._llm_router.complete(
                task="briefing",
                prompt=prompt,
            )
            raw_content = result.text
        except Exception as exc:
            logger.error(
                "BriefingGenerator: LLM completion failed for %r: %s", topic, exc
            )
            raise

        # Step 7: Build and save the BriefingRow
        title = self._extract_title(raw_content, topic)
        source_ids = [doc.document_id for doc in docs]

        briefing = BriefingRow(
            id=str(uuid.uuid4()),
            title=title,
            query=topic,
            content=raw_content,
            source_document_ids=source_ids,
            format="markdown",
        )

        await insert_briefing(conn, briefing)
        logger.info(
            "BriefingGenerator: saved briefing %s for topic %r", briefing.id, topic
        )

        return briefing

    def _assemble_context(
        self, docs: "list[SearchResult]", topic: str
    ) -> str:
        """Format documents into the {document_summaries} block.

        Each document is formatted as:
            ### {title}
            Source: {source_type} | {published_at}
            {excerpt}

        The total context is capped at ``_MAX_TOTAL_CHARS`` (~4,000 tokens) to
        stay within LLM context limits. Individual documents are truncated to
        ``_MAX_CHARS_PER_DOC`` characters before the total budget check.

        Args:
            docs: List of SearchResult objects from hybrid search.
            topic: The topic string (used for logging purposes only).

        Returns:
            A formatted string containing all document summaries, ready for
            insertion into the prompt template as {document_summaries}.
        """
        if not docs:
            return f"No documents found for topic: {topic}"

        parts: list[str] = []
        total_chars = 0

        for i, doc in enumerate(docs, start=1):
            title = doc.title or "Untitled"
            published = doc.published_at or "unknown date"
            excerpt = doc.excerpt or ""

            # Build the per-document block
            block = (
                f"### [{i}] {title}\n"
                f"Source: {doc.source_type} | {published}\n"
                f"{excerpt}"
            )

            # Truncate individual document to the per-doc budget
            if len(block) > _MAX_CHARS_PER_DOC:
                block = block[:_MAX_CHARS_PER_DOC] + "..."

            # Check total context budget — stop adding docs if over limit
            if total_chars + len(block) > _MAX_TOTAL_CHARS:
                logger.debug(
                    "BriefingGenerator: context budget reached at %d/%d docs "
                    "(%d chars). Truncating.",
                    i - 1,
                    len(docs),
                    total_chars,
                )
                break

            parts.append(block)
            total_chars += len(block)

        return "\n\n".join(parts)

    def _extract_title(self, content: str, topic: str) -> str:
        """Extract or generate a briefing title from LLM output.

        Looks for the first markdown H1 heading (``# Title``) in the LLM
        response. If none is found, falls back to ``"Briefing: {topic}"``.

        Args:
            content: Raw LLM output text.
            topic: The topic string used as a fallback.

        Returns:
            A title string for the briefing.
        """
        # Look for a markdown H1 heading at the start of a line
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # No H1 found — use a generic title
        return f"Briefing: {topic}"
