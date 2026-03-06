"""Source discovery engine for analyzing ingested documents.

Discovers new sources to follow by parsing outbound links, YouTube channel
handles, and generating LLM-based suggestions. Results are stored in the
``discovered_sources`` table for user review.

Discovery methods:
    - outbound_link: Extract URLs classified by domain pattern
    - citation: ArXiv paper references
    - mention: YouTube channel handle mentions
    - llm_suggestion: LLM-generated source recommendations
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import aiosqlite

from ..db.models import DiscoveredSourceRow, DocumentRow, SourceRow
from ..db.queries import list_discovered_sources, list_documents, list_sources, upsert_discovered_source

if TYPE_CHECKING:
    from ..config.models import AppConfig
    from ..llm.router import LLMRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL classification patterns
# Each pattern captures a meaningful identifier (slug, handle, subreddit, etc.)
# ---------------------------------------------------------------------------

PATTERNS: dict[str, str] = {
    "substack": r"https?://([a-z0-9-]+)\.substack\.com",
    "youtube": r"(?:youtube\.com/(?:@([a-zA-Z0-9_-]+)|c/([a-zA-Z0-9_-]+))|youtu\.be/)",
    "reddit": r"reddit\.com/r/([a-zA-Z0-9_]+)",
    "arxiv": r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d+)",
}

# Compiled versions for performance
_COMPILED_PATTERNS: dict[str, re.Pattern[str]] = {
    source_type: re.compile(pattern) for source_type, pattern in PATTERNS.items()
}

# YouTube handle pattern for @mention extraction (in plain text)
_YT_HANDLE_PATTERN: re.Pattern[str] = re.compile(r"@([a-zA-Z0-9_-]{3,})")

# Confidence scores based on mention frequency across documents
_CONFIDENCE_SINGLE = 0.4    # Found in 1 document
_CONFIDENCE_DOUBLE = 0.7    # Found in 2 documents
_CONFIDENCE_MULTI = 0.9     # Found in 3+ documents
_CONFIDENCE_LLM = 0.6       # LLM suggestion


def _compute_confidence(mention_count: int) -> float:
    """Compute confidence score based on how many documents mention a source.

    Args:
        mention_count: Number of distinct documents that mention the source.

    Returns:
        A float confidence score between 0.0 and 1.0.
    """
    if mention_count >= 3:
        return _CONFIDENCE_MULTI
    if mention_count == 2:
        return _CONFIDENCE_DOUBLE
    return _CONFIDENCE_SINGLE


class SourceDiscoverer:
    """Analyze documents to suggest new sources to follow.

    Runs multiple discovery strategies on batches of ingested documents:
    - Extracts outbound URLs and classifies them by domain pattern
    - Finds YouTube channel handle mentions (@handle)
    - Generates LLM suggestions based on recent reading patterns

    Usage::

        discoverer = SourceDiscoverer(config, llm_router)
        async with get_db(data_dir) as conn:
            new_sources = await discoverer.discover_from_documents(conn, docs)

    Args:
        config: Fully loaded AppConfig with LLM routing configuration.
        llm_router: Configured LLM router for LLM-based discovery.
    """

    def __init__(self, config: "AppConfig", llm_router: "LLMRouter") -> None:
        self._config = config
        self._llm_router = llm_router

    async def discover_from_documents(
        self,
        conn: aiosqlite.Connection,
        documents: list[DocumentRow],
    ) -> list[DiscoveredSourceRow]:
        """Run all discovery methods on a batch of documents.

        Combines results from outbound link extraction and YouTube handle
        mentions. Deduplicates against existing sources and already-discovered
        sources. Persists new suggestions to the database.

        Args:
            conn: An open aiosqlite connection.
            documents: Batch of DocumentRows to analyze.

        Returns:
            List of newly persisted DiscoveredSourceRow objects (not previously
            in the sources or discovered_sources tables).
        """
        if not documents:
            return []

        # Collect all raw candidates keyed by (source_type, identifier)
        # mapping to list of document IDs that mentioned them
        candidates: dict[tuple[str, str], list[str]] = defaultdict(list)
        candidate_meta: dict[tuple[str, str], dict] = {}

        for doc in documents:
            if not doc.raw_content:
                continue

            # Outbound links
            for row in self._extract_outbound_links(doc):
                key = (row.source_type, row.identifier)
                candidates[key].append(doc.id)
                if key not in candidate_meta:
                    candidate_meta[key] = {
                        "display_name": row.display_name,
                        "discovery_method": row.discovery_method,
                        "first_doc_id": doc.id,
                    }

            # YouTube handles
            for row in self._extract_youtube_handles(doc):
                key = (row.source_type, row.identifier)
                candidates[key].append(doc.id)
                if key not in candidate_meta:
                    candidate_meta[key] = {
                        "display_name": row.display_name,
                        "discovery_method": row.discovery_method,
                        "first_doc_id": doc.id,
                    }

        # Load existing sources and discovered sources to filter duplicates
        existing_sources = await list_sources(conn)
        existing_keys = {(s.source_type, s.identifier) for s in existing_sources}

        existing_discovered = await list_discovered_sources(conn, status="suggested")
        existing_discovered += await list_discovered_sources(conn, status="added")
        existing_discovered += await list_discovered_sources(conn, status="dismissed")
        existing_discovered_keys = {(d.source_type, d.identifier) for d in existing_discovered}

        already_known = existing_keys | existing_discovered_keys

        new_sources: list[DiscoveredSourceRow] = []
        for key, doc_ids in candidates.items():
            if key in already_known:
                continue

            meta = candidate_meta[key]
            confidence = _compute_confidence(len(set(doc_ids)))
            row = DiscoveredSourceRow(
                id=str(uuid.uuid4()),
                source_type=key[0],
                identifier=key[1],
                display_name=meta.get("display_name"),
                discovered_from_document_id=meta.get("first_doc_id"),
                discovery_method=meta.get("discovery_method", "outbound_link"),
                confidence=confidence,
                status="suggested",
            )
            try:
                await upsert_discovered_source(conn, row)
                new_sources.append(row)
                already_known.add(key)  # prevent adding again in same run
            except Exception:
                logger.exception(
                    "Failed to persist discovered source %s/%s", key[0], key[1]
                )

        logger.info(
            "Discovery: found %d new source suggestions from %d documents",
            len(new_sources),
            len(documents),
        )
        return new_sources

    def _extract_outbound_links(self, doc: DocumentRow) -> list[DiscoveredSourceRow]:
        """Parse URLs from raw_content using regex and classify by domain.

        Classifies by the following rules:
        - ``*.substack.com`` → source_type='substack', identifier=slug
        - ``youtube.com/c/*`` or ``youtube.com/@*`` → source_type='youtube', identifier=handle
        - ``reddit.com/r/*`` → source_type='reddit', identifier=subreddit_name
        - ``arxiv.org/abs/*`` or ``arxiv.org/pdf/*`` → source_type='arxiv', identifier=paper_id

        Args:
            doc: The document to scan for outbound links.

        Returns:
            A list of DiscoveredSourceRow objects (not yet persisted).
        """
        results: list[DiscoveredSourceRow] = []
        content = doc.raw_content or ""

        # Substack
        for match in _COMPILED_PATTERNS["substack"].finditer(content):
            slug = match.group(1)
            if slug:
                results.append(
                    DiscoveredSourceRow(
                        id=str(uuid.uuid4()),
                        source_type="substack",
                        identifier=slug,
                        display_name=slug,
                        discovered_from_document_id=doc.id,
                        discovery_method="outbound_link",
                        confidence=_CONFIDENCE_SINGLE,
                    )
                )

        # YouTube (from URLs — handles and /c/ channels)
        for match in _COMPILED_PATTERNS["youtube"].finditer(content):
            # Group 1 is @handle, group 2 is /c/<name>
            handle = match.group(1) or match.group(2)
            if handle:
                results.append(
                    DiscoveredSourceRow(
                        id=str(uuid.uuid4()),
                        source_type="youtube",
                        identifier=f"@{handle}" if not handle.startswith("@") else handle,
                        display_name=handle,
                        discovered_from_document_id=doc.id,
                        discovery_method="outbound_link",
                        confidence=_CONFIDENCE_SINGLE,
                    )
                )

        # Reddit
        for match in _COMPILED_PATTERNS["reddit"].finditer(content):
            subreddit = match.group(1)
            if subreddit:
                results.append(
                    DiscoveredSourceRow(
                        id=str(uuid.uuid4()),
                        source_type="reddit",
                        identifier=subreddit,
                        display_name=f"r/{subreddit}",
                        discovered_from_document_id=doc.id,
                        discovery_method="outbound_link",
                        confidence=_CONFIDENCE_SINGLE,
                    )
                )

        # ArXiv — discovery_method='citation' for academic paper cross-references
        for match in _COMPILED_PATTERNS["arxiv"].finditer(content):
            paper_id = match.group(1)
            if paper_id:
                results.append(
                    DiscoveredSourceRow(
                        id=str(uuid.uuid4()),
                        source_type="arxiv",
                        identifier=paper_id,
                        display_name=f"arxiv:{paper_id}",
                        discovered_from_document_id=doc.id,
                        discovery_method="citation",
                        confidence=_CONFIDENCE_SINGLE,
                    )
                )

        return results

    def _extract_youtube_handles(self, doc: DocumentRow) -> list[DiscoveredSourceRow]:
        """Find @handle patterns in document text (YouTube channel mentions).

        Searches for ``@handle`` patterns in raw_content. Useful for YouTube
        video descriptions that mention other channels, or articles that
        reference specific creators.

        Args:
            doc: The document to scan for YouTube handles.

        Returns:
            A list of DiscoveredSourceRow objects (not yet persisted).
        """
        results: list[DiscoveredSourceRow] = []
        content = doc.raw_content or ""

        for match in _YT_HANDLE_PATTERN.finditer(content):
            handle = match.group(1)
            # Filter out handles that look like email addresses or very short/generic ones
            if len(handle) < 3:
                continue
            results.append(
                DiscoveredSourceRow(
                    id=str(uuid.uuid4()),
                    source_type="youtube",
                    identifier=f"@{handle}",
                    display_name=handle,
                    discovered_from_document_id=doc.id,
                    discovery_method="mention",
                    confidence=_CONFIDENCE_SINGLE,
                )
            )

        return results

    async def _llm_suggestions(
        self,
        conn: aiosqlite.Connection,
        limit: int = 5,
    ) -> list[DiscoveredSourceRow]:
        """Fetch recent documents and ask the LLM to suggest new sources.

        Retrieves the 20 most recent documents, formats a prompt, calls the
        LLM, and parses the JSON response into DiscoveredSourceRow objects.
        Invalid or unparseable responses are logged and skipped.

        Args:
            conn: An open aiosqlite connection.
            limit: Maximum number of LLM suggestions to return.

        Returns:
            A list of DiscoveredSourceRow objects from LLM suggestions.
        """
        # Fetch 20 recent non-deleted documents for context
        docs = await list_documents(conn, limit=20)
        if not docs:
            return []

        # Build the article list for the prompt
        article_lines: list[str] = []
        for doc in docs:
            title = doc.title or "(no title)"
            url = doc.url
            source_type = doc.source_type
            article_lines.append(f"- [{source_type}] {title} ({url})")

        article_list = "\n".join(article_lines)

        prompt = (
            "You are helping someone discover new content sources to follow in their "
            "RSS/subscription reader.\n\n"
            "Here are 20 recent articles they've been reading:\n\n"
            f"{article_list}\n\n"
            "Based on this reading pattern, suggest new sources they should follow.\n"
            "For each suggestion:\n"
            "- source_type: one of [substack, youtube, reddit, rss, arxiv, devto]\n"
            "- identifier: the specific slug/handle/subreddit/URL\n"
            "- display_name: friendly name\n"
            "- reason: one sentence why this source would be valuable\n\n"
            "Return ONLY a JSON array with no other text:\n"
            '[{"source_type": "...", "identifier": "...", "display_name": "...", "reason": "..."}, ...]'
        )

        try:
            result = await self._llm_router.complete(
                task="source_discovery",
                prompt=prompt,
            )
            response = result.text
        except Exception:
            logger.exception("LLM source discovery call failed; skipping")
            return []

        return self._parse_llm_response(response, limit=limit)

    def _parse_llm_response(
        self,
        response: str,
        limit: int = 5,
    ) -> list[DiscoveredSourceRow]:
        """Parse JSON array from LLM source discovery response.

        Extracts a JSON array from the response text, validates each entry
        has required fields (source_type, identifier), and converts to
        DiscoveredSourceRow objects. Malformed entries are skipped with
        a warning log.

        Args:
            response: Raw LLM response text (expected to contain a JSON array).
            limit: Maximum number of suggestions to return.

        Returns:
            A list of DiscoveredSourceRow objects (up to ``limit`` items).
        """
        # Find the JSON array in the response (it may have surrounding text)
        response = response.strip()

        # Try to extract a JSON array even if there's surrounding text
        start = response.find("[")
        end = response.rfind("]")
        if start == -1 or end == -1:
            logger.warning("LLM response did not contain a JSON array; skipping")
            return []

        json_str = response[start : end + 1]
        try:
            items = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM JSON response: %s", e)
            return []

        if not isinstance(items, list):
            logger.warning("LLM response JSON is not a list; skipping")
            return []

        results: list[DiscoveredSourceRow] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            source_type = item.get("source_type", "").strip()
            identifier = item.get("identifier", "").strip()
            if not source_type or not identifier:
                logger.debug("LLM suggestion missing source_type or identifier; skipping: %s", item)
                continue

            display_name = item.get("display_name") or identifier
            results.append(
                DiscoveredSourceRow(
                    id=str(uuid.uuid4()),
                    source_type=source_type,
                    identifier=identifier,
                    display_name=display_name,
                    discovered_from_document_id=None,
                    discovery_method="llm_suggestion",
                    confidence=_CONFIDENCE_LLM,
                    status="suggested",
                )
            )

        return results

    async def run_periodic_discovery(
        self,
        conn: aiosqlite.Connection,
    ) -> int:
        """Run all discovery methods on recent documents (last 7 days).

        Fetches documents from the last 7 days and passes them through the
        full discovery pipeline including link extraction and LLM suggestions.
        This is intended to be called periodically (e.g. once per day) rather
        than on every ingest run.

        Args:
            conn: An open aiosqlite connection.

        Returns:
            Count of new suggestions added to the discovered_sources table.
        """
        # Fetch recent documents (last 7 days) — use a large limit and filter by date
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        # list_documents doesn't support date filtering directly; fetch with large limit
        # and filter in Python (acceptable for periodic discovery runs)
        all_recent = await list_documents(conn, limit=500)
        recent_docs = [
            doc for doc in all_recent
            if doc.published_at and doc.published_at >= cutoff
            or doc.fetched_at and doc.fetched_at >= cutoff
        ]

        logger.info(
            "run_periodic_discovery: processing %d documents from the last 7 days",
            len(recent_docs),
        )

        new_from_links = await self.discover_from_documents(conn, recent_docs)
        new_count = len(new_from_links)

        # LLM suggestions (at most once per day is enforced by the caller;
        # here we always attempt it during periodic discovery)
        try:
            llm_rows = await self._llm_suggestions(conn, limit=5)
        except Exception:
            logger.exception("LLM suggestions failed during periodic discovery; skipping")
            llm_rows = []

        # Load existing keys to deduplicate LLM suggestions
        existing_sources = await list_sources(conn)
        existing_keys = {(s.source_type, s.identifier) for s in existing_sources}
        existing_discovered = await list_discovered_sources(conn, status="suggested")
        existing_discovered += await list_discovered_sources(conn, status="added")
        existing_discovered += await list_discovered_sources(conn, status="dismissed")
        existing_discovered_keys = {(d.source_type, d.identifier) for d in existing_discovered}
        already_known = existing_keys | existing_discovered_keys

        for row in llm_rows:
            key = (row.source_type, row.identifier)
            if key in already_known:
                continue
            try:
                await upsert_discovered_source(conn, row)
                new_count += 1
                already_known.add(key)
            except Exception:
                logger.exception(
                    "Failed to persist LLM-suggested source %s/%s", row.source_type, row.identifier
                )

        logger.info("run_periodic_discovery: added %d new source suggestions", new_count)
        return new_count
