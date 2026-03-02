"""Export formatters for search results, documents, entities, and briefings.

Supports three output formats:
- Markdown: human-readable documents with headers and metadata
- JSON: machine-readable structured data
- CSV: tabular data for entities

All functions are pure (no I/O) — they accept data models and return strings
or dicts. File I/O is handled by the callers (CLI commands, API routes).

File naming convention:
- Search: ``search-{sanitized_query}-{date}.md``
- Briefing: ``briefing-{sanitized_topic}-{date}.md``
- Documents: ``documents-export-{date}.json``
- Entities: ``entities-{date}.csv``
"""
from __future__ import annotations

import csv
import io
import json
import re
from enum import Enum
from typing import Any

from .db.models import BriefingRow, DocumentRow, EntityRow
from .search.hybrid import SearchResult


class ExportFormat(str, Enum):
    """Supported export formats."""

    markdown = "markdown"
    json = "json"
    csv = "csv"


def sanitize_filename(text: str) -> str:
    """Sanitize a string for safe use as a filename component.

    Replaces whitespace with underscores and removes any character that is not
    alphanumeric, hyphen, or underscore. Collapses consecutive underscores.

    Args:
        text: The raw string to sanitize (e.g. a query or topic).

    Returns:
        A safe filename-compatible string, lowercased and truncated to 50
        characters.

    Examples:
        >>> sanitize_filename("LLM agents & tools!")
        'llm_agents_tools'
        >>> sanitize_filename("  hello world  ")
        'hello_world'
    """
    # Lowercase first for consistent naming
    text = text.lower().strip()
    # Replace whitespace sequences with underscore
    text = re.sub(r"\s+", "_", text)
    # Remove any character that is not alphanumeric, underscore, or hyphen
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    # Collapse multiple consecutive underscores
    text = re.sub(r"_+", "_", text)
    # Trim leading/trailing underscores
    text = text.strip("_")
    # Truncate to keep filenames manageable
    return text[:50]


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------


def search_results_to_markdown(
    results: list[SearchResult],
    query: str,
    generated_at: str,
) -> str:
    """Format search results as a standalone Markdown document.

    Produces a document with a title block showing the query and generation
    time, followed by one section per result with metadata and an excerpt.

    Format::

        # Search Results: "{query}"
        Generated: {generated_at}

        ---

        ## 1. {title}
        **Source**: {source_type} | **Date**: {published_at} | **Score**: {score:.2f}
        **URL**: {url}
        **Author**: {author}

        {excerpt}

        ---

    Args:
        results: Ordered list of :class:`~ai_craftsman_kb.search.hybrid.SearchResult`
            objects (descending score order).
        query: The original search query string.
        generated_at: ISO 8601 timestamp of when the export was generated.

    Returns:
        A Markdown string representing the full export document.
    """
    lines: list[str] = []
    lines.append(f'# Search Results: "{query}"')
    lines.append(f"Generated: {generated_at}")
    lines.append("")

    if not results:
        lines.append("*No results found.*")
        return "\n".join(lines)

    for i, result in enumerate(results, start=1):
        lines.append("---")
        lines.append("")
        title = result.title or "(no title)"
        lines.append(f"## {i}. {title}")

        date_str = result.published_at or "unknown"
        lines.append(
            f"**Source**: {result.source_type} | "
            f"**Date**: {date_str} | "
            f"**Score**: {result.score:.2f}"
        )
        lines.append(f"**URL**: {result.url}")
        if result.author:
            lines.append(f"**Author**: {result.author}")
        lines.append("")

        if result.excerpt:
            lines.append(result.excerpt)
            lines.append("")

    lines.append("---")
    return "\n".join(lines)


def search_results_to_json(results: list[SearchResult]) -> str:
    """Serialize search results as a JSON array string.

    Each element in the array contains the same fields as the API response:
    ``document_id``, ``score``, ``title``, ``url``, ``source_type``,
    ``author``, ``published_at``, ``excerpt``, and ``origin``.

    Args:
        results: List of :class:`~ai_craftsman_kb.search.hybrid.SearchResult`
            objects to serialize.

    Returns:
        A valid JSON string containing an array of result dicts.
    """
    items: list[dict[str, Any]] = [r.model_dump() for r in results]
    return json.dumps(items, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def documents_to_markdown(docs: list[DocumentRow]) -> str:
    """Export a list of documents as concatenated Markdown sections.

    Each document is rendered as a top-level section with YAML-like metadata
    and the full ``raw_content`` body. Documents are separated by horizontal
    rules so the output reads as a single composite document.

    Args:
        docs: List of :class:`~ai_craftsman_kb.db.models.DocumentRow` objects.

    Returns:
        A Markdown string with all documents concatenated.
    """
    lines: list[str] = []
    lines.append("# Document Export")
    lines.append("")

    if not docs:
        lines.append("*No documents to export.*")
        return "\n".join(lines)

    for doc in docs:
        lines.append("---")
        lines.append("")
        title = doc.title or "(no title)"
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- **ID**: {doc.id}")
        lines.append(f"- **Source**: {doc.source_type}")
        lines.append(f"- **URL**: {doc.url}")
        if doc.author:
            lines.append(f"- **Author**: {doc.author}")
        if doc.published_at:
            lines.append(f"- **Published**: {doc.published_at}")
        lines.append(f"- **Origin**: {doc.origin}")
        lines.append("")

        if doc.raw_content:
            lines.append(doc.raw_content)
            lines.append("")

    lines.append("---")
    return "\n".join(lines)


def documents_to_json(docs: list[DocumentRow]) -> str:
    """Serialize a list of documents as a JSON array string.

    Args:
        docs: List of :class:`~ai_craftsman_kb.db.models.DocumentRow` objects.

    Returns:
        A valid JSON string containing an array of document dicts.
    """
    items: list[dict[str, Any]] = [d.model_dump() for d in docs]
    return json.dumps(items, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


def entities_to_csv(entities: list[EntityRow]) -> str:
    """Format entities as a CSV string with a header row.

    Header columns: ``id,name,entity_type,mention_count,first_seen_at``

    Args:
        entities: List of :class:`~ai_craftsman_kb.db.models.EntityRow` objects.

    Returns:
        A CSV string with a header row followed by one row per entity.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    # Write header
    writer.writerow(["id", "name", "entity_type", "mention_count", "first_seen_at"])
    for entity in entities:
        writer.writerow([
            entity.id,
            entity.name,
            entity.entity_type,
            entity.mention_count,
            entity.first_seen_at or "",
        ])
    return output.getvalue()


def entities_to_json(entities: list[EntityRow]) -> str:
    """Serialize entities as a JSON array string.

    Args:
        entities: List of :class:`~ai_craftsman_kb.db.models.EntityRow` objects.

    Returns:
        A valid JSON string containing an array of entity dicts.
    """
    items: list[dict[str, Any]] = [e.model_dump() for e in entities]
    return json.dumps(items, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Briefings
# ---------------------------------------------------------------------------


def briefing_to_markdown(briefing: BriefingRow) -> str:
    """Format a briefing as a Markdown document with YAML frontmatter.

    Prepends a YAML frontmatter block containing briefing metadata (id, title,
    query, created_at) before the briefing's Markdown content body.

    Args:
        briefing: A :class:`~ai_craftsman_kb.db.models.BriefingRow` object
            whose ``content`` field is already Markdown.

    Returns:
        A Markdown string with YAML frontmatter followed by the briefing body.
    """
    lines: list[str] = []
    # YAML frontmatter block
    lines.append("---")
    lines.append(f"id: {briefing.id}")
    lines.append(f'title: "{briefing.title}"')
    if briefing.query:
        lines.append(f'query: "{briefing.query}"')
    lines.append(f"created_at: {briefing.created_at}")
    lines.append(f"format: {briefing.format}")
    lines.append("---")
    lines.append("")
    # Briefing body (already Markdown)
    lines.append(briefing.content)
    return "\n".join(lines)


def briefing_to_json(briefing: BriefingRow) -> dict[str, Any]:
    """Serialize a briefing as a dict with all fields.

    Args:
        briefing: A :class:`~ai_craftsman_kb.db.models.BriefingRow` object.

    Returns:
        A dict with all briefing fields suitable for JSON serialization.
    """
    return briefing.model_dump()
