"""Content filtering with LLM, keyword, and hybrid strategies.

Filtering happens after fetching from source, before storing to DB.
The FilterResult.passed and score fields are used to populate
filter_passed and filter_score on DocumentRow.
"""
import asyncio
import logging
import re

from pydantic import BaseModel

from ..config.models import AppConfig, SourceFilterConfig
from ..ingestors.base import RawDocument
from ..llm.router import LLMRouter

logger = logging.getLogger(__name__)


class FilterResult(BaseModel):
    """Result of applying a content filter to a document.

    Attributes:
        passed: Whether the document passed the filter.
        score: Numeric relevance score. 1-10 for LLM strategy,
            0.0 (fail) or 1.0 (pass) for keyword strategy,
            None if the filter is disabled.
        reason: Human-readable explanation of the filter decision.
    """

    passed: bool
    score: float | None = None
    reason: str | None = None


class ContentFilter:
    """Filter RawDocuments based on per-source filter config from filters.yaml.

    Supports three strategies:

    - ``keyword``: Fast string match on title + content. Checks exclude
      keywords, include keywords, min_upvotes, and min_reactions thresholds.
    - ``llm``: LLM scores relevance 1-10 using a configurable prompt template.
      Passes if score >= cfg.min_score (default 5).
    - ``hybrid``: Runs keyword filter first; if it passes, runs LLM filter.
      Both must pass. Short-circuits on keyword fail to avoid LLM cost.

    Usage::

        filter = ContentFilter(config, llm_router)
        result = await filter.filter(doc, source_type="hn")
        if result.passed:
            await db.save(doc)

    Args:
        config: Fully loaded AppConfig with filters configuration.
        llm_router: Configured LLM router for making completion requests.
    """

    def __init__(self, config: AppConfig, llm_router: LLMRouter) -> None:
        self.config = config
        self.llm_router = llm_router

    async def filter(self, doc: RawDocument, source_type: str) -> FilterResult:
        """Apply the configured filter strategy for the given source type.

        Looks up the SourceFilterConfig for the source type, then dispatches
        to the appropriate strategy method. If filtering is disabled for the
        source, always returns passed=True without any scoring.

        Args:
            doc: The raw document to evaluate.
            source_type: The source type string (e.g. 'hn', 'reddit', 'arxiv').

        Returns:
            A FilterResult with passed, score, and reason populated.
        """
        filter_cfg = self._get_source_filter(source_type)
        if not filter_cfg.enabled:
            return FilterResult(passed=True, score=None, reason="filter disabled")

        if filter_cfg.strategy == "llm":
            return await self._llm_filter(doc, filter_cfg)
        elif filter_cfg.strategy == "keyword":
            return self._keyword_filter(doc, filter_cfg)
        else:  # hybrid
            return await self._hybrid_filter(doc, filter_cfg)

    async def filter_batch(
        self,
        docs: list[RawDocument],
        source_type: str,
        concurrency: int = 5,
    ) -> list[tuple[RawDocument, FilterResult]]:
        """Filter a list of documents concurrently with bounded concurrency.

        Uses asyncio.Semaphore to cap the number of simultaneous LLM calls,
        preventing rate limit exhaustion when processing large batches.

        Args:
            docs: List of raw documents to filter.
            source_type: The source type string for all documents in the batch.
            concurrency: Maximum number of concurrent filter operations.

        Returns:
            List of (doc, FilterResult) tuples in the same order as input docs.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _filter_one(doc: RawDocument) -> tuple[RawDocument, FilterResult]:
            async with semaphore:
                result = await self.filter(doc, source_type)
                return doc, result

        return list(await asyncio.gather(*[_filter_one(d) for d in docs]))

    async def _llm_filter(self, doc: RawDocument, cfg: SourceFilterConfig) -> FilterResult:
        """Use LLM to score document relevance on a 1-10 scale.

        Fills {title} and {excerpt} (first 500 chars of raw_content) into
        cfg.llm_prompt, calls the LLM router with task='filtering', then
        parses the integer score from the response. Passes if score >= min_score.

        On LLM error, logs the exception and defaults to passing the document
        so that transient API failures do not silently discard content.

        Args:
            doc: The document to evaluate.
            cfg: The source filter configuration containing the prompt template.

        Returns:
            A FilterResult with the LLM score and pass/fail decision.
        """
        if not cfg.llm_prompt:
            logger.warning(
                "LLM filter configured but no llm_prompt set; defaulting to pass"
            )
            return FilterResult(passed=True, score=None, reason="no prompt configured")

        title = doc.title or ""
        excerpt = (doc.raw_content or "")[:500]
        prompt = cfg.llm_prompt.format(title=title, excerpt=excerpt)

        try:
            response = await self.llm_router.complete(task="filtering", prompt=prompt)
            score = self._parse_llm_score(response)
        except Exception as e:
            logger.error("LLM filter failed: %s; defaulting to pass", e)
            return FilterResult(passed=True, score=None, reason=f"llm error: {e}")

        min_score = cfg.min_score if cfg.min_score is not None else 5
        passed = score >= min_score
        return FilterResult(
            passed=passed,
            score=float(score),
            reason=f"llm score {score}/{min_score} threshold",
        )

    def _keyword_filter(self, doc: RawDocument, cfg: SourceFilterConfig) -> FilterResult:
        """Filter based on keyword presence, upvote count, and reaction count.

        Checks in order:
        1. If any keyword in cfg.keywords_exclude is found in title + content,
           fails immediately (score=0.0).
        2. If cfg.keywords_include is non-empty, at least one keyword must be
           found in title + content; otherwise fails (score=0.0).
        3. If cfg.min_upvotes is set, doc.metadata['upvotes'] must meet the
           threshold; otherwise fails (score=0.0).
        4. If cfg.min_reactions is set, doc.metadata['reactions'] must meet the
           threshold; otherwise fails (score=0.0).

        Text matching is case-insensitive. Content is truncated to 2000 chars
        to avoid excessive processing on very long documents.

        Args:
            doc: The document to evaluate.
            cfg: The source filter configuration with keyword lists and thresholds.

        Returns:
            A FilterResult with score=1.0 on pass or score=0.0 on fail.
        """
        text = f"{doc.title or ''} {(doc.raw_content or '')[:2000]}".lower()

        # Check exclude keywords — any match is an immediate fail
        for kw in cfg.keywords_exclude:
            if kw.lower() in text:
                return FilterResult(
                    passed=False,
                    score=0.0,
                    reason=f"excluded keyword: {kw}",
                )

        # Check include keywords — if list is non-empty, at least one must match
        if cfg.keywords_include:
            matched = next(
                (kw for kw in cfg.keywords_include if kw.lower() in text), None
            )
            if not matched:
                return FilterResult(
                    passed=False,
                    score=0.0,
                    reason="no required keywords found",
                )

        # Check min_upvotes threshold (Reddit metadata)
        if cfg.min_upvotes is not None:
            upvotes = doc.metadata.get("upvotes", 0)
            if upvotes < cfg.min_upvotes:
                return FilterResult(
                    passed=False,
                    score=0.0,
                    reason=f"upvotes {upvotes} < {cfg.min_upvotes}",
                )

        # Check min_reactions threshold (DEV.to metadata)
        if cfg.min_reactions is not None:
            reactions = doc.metadata.get("reactions", 0)
            if reactions < cfg.min_reactions:
                return FilterResult(
                    passed=False,
                    score=0.0,
                    reason=f"reactions {reactions} < {cfg.min_reactions}",
                )

        return FilterResult(passed=True, score=1.0, reason="keyword filter passed")

    async def _hybrid_filter(self, doc: RawDocument, cfg: SourceFilterConfig) -> FilterResult:
        """Run keyword filter first, then LLM filter if keyword passes.

        Both the keyword filter and the LLM filter must pass for the document
        to be accepted. If the keyword filter fails, the LLM is never called,
        saving token costs on clear rejects.

        Args:
            doc: The document to evaluate.
            cfg: The source filter configuration with both keyword and LLM settings.

        Returns:
            The FilterResult from the failing filter, or the LLM result if both pass.
        """
        keyword_result = self._keyword_filter(doc, cfg)
        if not keyword_result.passed:
            # Short-circuit: do not spend LLM tokens on clear rejects
            return keyword_result
        return await self._llm_filter(doc, cfg)

    def _get_source_filter(self, source_type: str) -> SourceFilterConfig:
        """Look up filter configuration by source type string.

        Maps common source_type aliases (e.g. 'hn' -> 'hackernews') to
        the corresponding field on FiltersConfig. If no config is found
        for the given source_type, returns a disabled SourceFilterConfig
        so the document passes through by default.

        Args:
            source_type: The source type identifier from RawDocument.source_type.

        Returns:
            The SourceFilterConfig for this source, or a disabled default.
        """
        filters = self.config.filters
        # Map ingestor source_type strings to FiltersConfig field names
        mapping: dict[str, str] = {
            "hn": "hackernews",
            "hackernews": "hackernews",
            "substack": "substack",
            "youtube": "youtube",
            "reddit": "reddit",
            "arxiv": "arxiv",
            "rss": "rss",
            "devto": "devto",
        }
        attr = mapping.get(source_type, source_type)
        cfg = getattr(filters, attr, None)
        if cfg is None:
            logger.debug(
                "No filter config for source_type %s; defaulting to pass", source_type
            )
            return SourceFilterConfig(enabled=False)
        return cfg

    @staticmethod
    def _parse_llm_score(response: str) -> int:
        """Extract an integer score 1-10 from varied LLM response formats.

        Handles formats like:
        - "7"
        - "7/10"
        - "Score: 7"
        - "I rate this a 7 out of 10"
        - "Rating: 10"

        Returns the first valid integer in range [1, 10] found in the response.
        Returns 0 if no valid score is found, which will cause the document
        to fail LLM filtering (score 0 < any reasonable min_score).

        Args:
            response: The raw string response from the LLM completion.

        Returns:
            An integer from 1 to 10, or 0 if no valid score is parseable.
        """
        # Match integers 1-9 or the literal 10, as whole words (not substrings)
        numbers = re.findall(r"\b(10|[1-9])\b", response)
        if numbers:
            return int(numbers[0])
        return 0
