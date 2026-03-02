"""Tests for the export module.

Covers all export formatters:
- search_results_to_markdown
- search_results_to_json
- documents_to_markdown
- documents_to_json
- entities_to_csv
- entities_to_json
- briefing_to_markdown
- briefing_to_json
- sanitize_filename
"""
from __future__ import annotations

import csv
import io
import json

import pytest

from ai_craftsman_kb.db.models import BriefingRow, DocumentRow, EntityRow
from ai_craftsman_kb.export import (
    ExportFormat,
    briefing_to_json,
    briefing_to_markdown,
    documents_to_json,
    documents_to_markdown,
    entities_to_csv,
    entities_to_json,
    sanitize_filename,
    search_results_to_json,
    search_results_to_markdown,
)
from ai_craftsman_kb.search.hybrid import SearchResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_search_result(
    document_id: str = "doc-1",
    score: float = 0.85,
    title: str | None = "Test Title",
    url: str = "https://example.com/test",
    source_type: str = "hn",
    author: str | None = "Test Author",
    published_at: str | None = "2025-01-15T12:00:00",
    excerpt: str | None = "This is a test excerpt with some content.",
    origin: str = "pro",
) -> SearchResult:
    """Create a SearchResult with sensible defaults."""
    return SearchResult(
        document_id=document_id,
        score=score,
        title=title,
        url=url,
        source_type=source_type,
        author=author,
        published_at=published_at,
        excerpt=excerpt,
        origin=origin,
    )


def _make_document(
    doc_id: str = "doc-1",
    title: str | None = "Document Title",
    url: str = "https://example.com/doc",
    source_type: str = "hn",
    author: str | None = "Author",
    published_at: str | None = "2025-01-01T00:00:00",
    raw_content: str | None = "Document body text.",
    origin: str = "pro",
) -> DocumentRow:
    """Create a DocumentRow with sensible defaults."""
    return DocumentRow(
        id=doc_id,
        origin=origin,
        source_type=source_type,
        url=url,
        title=title,
        author=author,
        published_at=published_at,
        raw_content=raw_content,
    )


def _make_entity(
    entity_id: str = "ent-1",
    name: str = "Python",
    entity_type: str = "technology",
    mention_count: int = 42,
    first_seen_at: str | None = "2025-01-01T00:00:00",
) -> EntityRow:
    """Create an EntityRow with sensible defaults."""
    return EntityRow(
        id=entity_id,
        name=name,
        entity_type=entity_type,
        normalized_name=name.lower(),
        mention_count=mention_count,
        first_seen_at=first_seen_at,
    )


def _make_briefing(
    briefing_id: str = "brief-1",
    title: str = "AI Trends Briefing",
    query: str | None = "AI trends",
    content: str = "## Overview\nAI is growing rapidly.\n\n## Key Points\n- LLMs are powerful.",
    created_at: str = "2025-01-15T10:00:00",
) -> BriefingRow:
    """Create a BriefingRow with sensible defaults."""
    return BriefingRow(
        id=briefing_id,
        title=title,
        query=query,
        content=content,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    """Tests for the sanitize_filename utility."""

    def test_spaces_replaced_by_underscores(self) -> None:
        """Spaces in the input are converted to underscores."""
        result = sanitize_filename("hello world")
        assert result == "hello_world"

    def test_special_chars_removed(self) -> None:
        """Special characters (not alphanumeric, hyphen, underscore) are removed."""
        result = sanitize_filename("LLM agents & tools!")
        assert result == "llm_agents_tools"

    def test_lowercased(self) -> None:
        """Output is always lowercased."""
        result = sanitize_filename("MixedCase")
        assert result == "mixedcase"

    def test_leading_trailing_underscores_stripped(self) -> None:
        """Leading and trailing underscores are removed."""
        result = sanitize_filename("  hello world  ")
        assert result == "hello_world"

    def test_consecutive_underscores_collapsed(self) -> None:
        """Multiple consecutive underscores are collapsed to one."""
        result = sanitize_filename("a   b   c")
        assert result == "a_b_c"

    def test_truncated_to_50_chars(self) -> None:
        """Result is at most 50 characters."""
        long_text = "a" * 100
        result = sanitize_filename(long_text)
        assert len(result) <= 50

    def test_empty_string(self) -> None:
        """Empty string input returns empty string."""
        result = sanitize_filename("")
        assert result == ""

    def test_hyphen_preserved(self) -> None:
        """Hyphens are preserved in the output."""
        result = sanitize_filename("well-known topic")
        assert "-" in result or "well" in result  # hyphen kept or spaces removed

    def test_numbers_preserved(self) -> None:
        """Numbers are preserved in the output."""
        result = sanitize_filename("GPT-4 model")
        assert "4" in result
        assert "gpt" in result


# ---------------------------------------------------------------------------
# search_results_to_markdown
# ---------------------------------------------------------------------------


class TestSearchResultsToMarkdown:
    """Tests for search_results_to_markdown."""

    def test_empty_results(self) -> None:
        """Empty result list produces a header with 'No results found' notice."""
        output = search_results_to_markdown([], "LLM agents", "2025-01-15T10:00:00")
        assert "# Search Results:" in output
        assert "LLM agents" in output
        assert "No results found" in output

    def test_header_contains_query(self) -> None:
        """Header includes the original query string."""
        output = search_results_to_markdown([], "my query", "2025-01-15T10:00:00")
        assert '"my query"' in output

    def test_header_contains_generated_at(self) -> None:
        """Header includes the generated_at timestamp."""
        output = search_results_to_markdown([], "q", "2025-01-15T10:00:00+00:00")
        assert "2025-01-15T10:00:00+00:00" in output

    def test_single_result_numbered(self) -> None:
        """Single result is rendered with number prefix '## 1.'."""
        result = _make_search_result()
        output = search_results_to_markdown([result], "test", "2025-01-15")
        assert "## 1. Test Title" in output

    def test_multiple_results_numbered_sequentially(self) -> None:
        """Multiple results are numbered sequentially."""
        results = [
            _make_search_result(document_id="a", url="https://a.com"),
            _make_search_result(document_id="b", title="Second", url="https://b.com"),
            _make_search_result(document_id="c", title="Third", url="https://c.com"),
        ]
        output = search_results_to_markdown(results, "q", "2025-01-15")
        assert "## 1." in output
        assert "## 2." in output
        assert "## 3." in output

    def test_score_formatted_two_decimals(self) -> None:
        """Score is formatted with two decimal places."""
        result = _make_search_result(score=0.85123)
        output = search_results_to_markdown([result], "q", "2025-01-15")
        assert "0.85" in output

    def test_source_metadata_present(self) -> None:
        """Source type, date, and score metadata line is present."""
        result = _make_search_result(source_type="arxiv", published_at="2025-01-10")
        output = search_results_to_markdown([result], "q", "2025-01-15")
        assert "**Source**: arxiv" in output
        assert "2025-01-10" in output

    def test_url_included(self) -> None:
        """Result URL is included in the output."""
        result = _make_search_result(url="https://example.com/article")
        output = search_results_to_markdown([result], "q", "2025-01-15")
        assert "https://example.com/article" in output

    def test_author_included_when_present(self) -> None:
        """Author field is included when not None."""
        result = _make_search_result(author="Jane Doe")
        output = search_results_to_markdown([result], "q", "2025-01-15")
        assert "Jane Doe" in output

    def test_author_omitted_when_none(self) -> None:
        """Author field is omitted when None."""
        result = _make_search_result(author=None)
        output = search_results_to_markdown([result], "q", "2025-01-15")
        assert "**Author**" not in output

    def test_excerpt_included(self) -> None:
        """Excerpt text appears in the output."""
        result = _make_search_result(excerpt="This is the excerpt text.")
        output = search_results_to_markdown([result], "q", "2025-01-15")
        assert "This is the excerpt text." in output

    def test_no_title_uses_placeholder(self) -> None:
        """When title is None, a placeholder is shown."""
        result = _make_search_result(title=None)
        output = search_results_to_markdown([result], "q", "2025-01-15")
        assert "(no title)" in output

    def test_output_is_string(self) -> None:
        """Return type is always str."""
        output = search_results_to_markdown([], "q", "2025-01-15")
        assert isinstance(output, str)

    def test_horizontal_rules_present(self) -> None:
        """Results are separated by horizontal rules ('---')."""
        result = _make_search_result()
        output = search_results_to_markdown([result], "q", "2025-01-15")
        assert "---" in output


# ---------------------------------------------------------------------------
# search_results_to_json
# ---------------------------------------------------------------------------


class TestSearchResultsToJson:
    """Tests for search_results_to_json."""

    def test_empty_list_returns_valid_json_array(self) -> None:
        """Empty input produces a valid empty JSON array."""
        output = search_results_to_json([])
        parsed = json.loads(output)
        assert parsed == []

    def test_single_result_serialized(self) -> None:
        """Single result is serialized with expected fields."""
        result = _make_search_result(document_id="doc-1", score=0.9)
        output = search_results_to_json([result])
        parsed = json.loads(output)
        assert len(parsed) == 1
        assert parsed[0]["document_id"] == "doc-1"
        assert parsed[0]["score"] == 0.9

    def test_all_fields_present(self) -> None:
        """All SearchResult fields are present in the JSON output."""
        result = _make_search_result()
        output = search_results_to_json([result])
        parsed = json.loads(output)
        expected_fields = {
            "document_id", "score", "title", "url", "source_type",
            "author", "published_at", "excerpt", "origin",
        }
        assert expected_fields.issubset(set(parsed[0].keys()))

    def test_multiple_results_ordered(self) -> None:
        """Multiple results are serialized in the same order as input."""
        results = [
            _make_search_result(document_id="a", url="https://a.com"),
            _make_search_result(document_id="b", url="https://b.com"),
        ]
        output = search_results_to_json(results)
        parsed = json.loads(output)
        assert parsed[0]["document_id"] == "a"
        assert parsed[1]["document_id"] == "b"

    def test_none_fields_preserved(self) -> None:
        """None values are preserved as JSON null."""
        result = _make_search_result(title=None, author=None, published_at=None)
        output = search_results_to_json([result])
        parsed = json.loads(output)
        assert parsed[0]["title"] is None
        assert parsed[0]["author"] is None

    def test_valid_json_parseable(self) -> None:
        """Output is valid JSON parseable by json.loads."""
        results = [_make_search_result() for _ in range(3)]
        output = search_results_to_json(results)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 3


# ---------------------------------------------------------------------------
# documents_to_markdown
# ---------------------------------------------------------------------------


class TestDocumentsToMarkdown:
    """Tests for documents_to_markdown."""

    def test_empty_list(self) -> None:
        """Empty document list produces a header with no-documents notice."""
        output = documents_to_markdown([])
        assert "# Document Export" in output
        assert "No documents" in output

    def test_single_document(self) -> None:
        """Single document renders with title and metadata."""
        doc = _make_document(title="My Article")
        output = documents_to_markdown([doc])
        assert "## My Article" in output
        assert "https://example.com/doc" in output

    def test_document_metadata_included(self) -> None:
        """Document source, URL, author, published date are included."""
        doc = _make_document(
            source_type="arxiv",
            author="Jane Doe",
            published_at="2025-01-10",
        )
        output = documents_to_markdown([doc])
        assert "arxiv" in output
        assert "Jane Doe" in output
        assert "2025-01-10" in output

    def test_raw_content_included(self) -> None:
        """raw_content is included in the output."""
        doc = _make_document(raw_content="Full article text here.")
        output = documents_to_markdown([doc])
        assert "Full article text here." in output

    def test_no_title_uses_placeholder(self) -> None:
        """Documents with no title show a placeholder."""
        doc = _make_document(title=None)
        output = documents_to_markdown([doc])
        assert "(no title)" in output

    def test_multiple_documents_separated_by_rules(self) -> None:
        """Multiple documents are separated by horizontal rules."""
        docs = [
            _make_document(doc_id="d1", url="https://a.com"),
            _make_document(doc_id="d2", url="https://b.com"),
        ]
        output = documents_to_markdown(docs)
        assert output.count("---") >= 2


# ---------------------------------------------------------------------------
# documents_to_json
# ---------------------------------------------------------------------------


class TestDocumentsToJson:
    """Tests for documents_to_json."""

    def test_empty_list(self) -> None:
        """Empty list produces a valid empty JSON array."""
        output = documents_to_json([])
        assert json.loads(output) == []

    def test_single_document_serialized(self) -> None:
        """Single document is serialized with expected fields."""
        doc = _make_document(doc_id="doc-42")
        output = documents_to_json([doc])
        parsed = json.loads(output)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "doc-42"

    def test_valid_json(self) -> None:
        """Output is valid JSON."""
        docs = [_make_document(doc_id=f"d{i}", url=f"https://example.com/{i}") for i in range(5)]
        output = documents_to_json(docs)
        parsed = json.loads(output)
        assert len(parsed) == 5


# ---------------------------------------------------------------------------
# entities_to_csv
# ---------------------------------------------------------------------------


class TestEntitiesToCsv:
    """Tests for entities_to_csv."""

    def test_header_row_present(self) -> None:
        """Output always has a header row with the expected columns."""
        output = entities_to_csv([])
        reader = csv.reader(io.StringIO(output))
        header = next(reader)
        assert header == ["id", "name", "entity_type", "mention_count", "first_seen_at"]

    def test_empty_entities_only_header(self) -> None:
        """Empty entity list produces only the header row."""
        output = entities_to_csv([])
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        # Only header row
        assert len(rows) == 1

    def test_single_entity_row(self) -> None:
        """Single entity is serialized as one data row after the header."""
        entity = _make_entity(
            entity_id="ent-1",
            name="Python",
            entity_type="technology",
            mention_count=42,
            first_seen_at="2025-01-01T00:00:00",
        )
        output = entities_to_csv([entity])
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        assert len(rows) == 2  # header + 1 data row
        data_row = rows[1]
        assert data_row[0] == "ent-1"
        assert data_row[1] == "Python"
        assert data_row[2] == "technology"
        assert data_row[3] == "42"
        assert data_row[4] == "2025-01-01T00:00:00"

    def test_multiple_entities(self) -> None:
        """Multiple entities produce one row per entity after the header."""
        entities = [
            _make_entity(entity_id=f"ent-{i}", name=f"Entity {i}", entity_type="technology")
            for i in range(5)
        ]
        output = entities_to_csv(entities)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        assert len(rows) == 6  # header + 5 data rows

    def test_none_first_seen_at_is_empty_string(self) -> None:
        """first_seen_at=None is serialized as an empty string in the CSV."""
        entity = _make_entity(first_seen_at=None)
        output = entities_to_csv([entity])
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        # first_seen_at column is index 4
        assert rows[1][4] == ""

    def test_output_is_string(self) -> None:
        """Return type is always str."""
        output = entities_to_csv([])
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# entities_to_json
# ---------------------------------------------------------------------------


class TestEntitiesToJson:
    """Tests for entities_to_json."""

    def test_empty_list(self) -> None:
        """Empty list produces a valid empty JSON array."""
        output = entities_to_json([])
        assert json.loads(output) == []

    def test_single_entity(self) -> None:
        """Single entity is serialized with expected fields."""
        entity = _make_entity(entity_id="ent-1", name="OpenAI")
        output = entities_to_json([entity])
        parsed = json.loads(output)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "ent-1"
        assert parsed[0]["name"] == "OpenAI"

    def test_valid_json(self) -> None:
        """Output is valid JSON parseable by json.loads."""
        entities = [_make_entity(entity_id=f"e{i}", name=f"Entity {i}") for i in range(3)]
        output = entities_to_json(entities)
        parsed = json.loads(output)
        assert len(parsed) == 3


# ---------------------------------------------------------------------------
# briefing_to_markdown
# ---------------------------------------------------------------------------


class TestBriefingToMarkdown:
    """Tests for briefing_to_markdown."""

    def test_yaml_frontmatter_present(self) -> None:
        """Output includes a YAML frontmatter block delimited by '---'."""
        briefing = _make_briefing()
        output = briefing_to_markdown(briefing)
        # YAML frontmatter: starts with ---, another --- after metadata
        lines = output.split("\n")
        assert lines[0] == "---"
        # Find closing ---
        assert "---" in lines[1:]

    def test_frontmatter_contains_id(self) -> None:
        """Frontmatter includes the briefing id."""
        briefing = _make_briefing(briefing_id="brief-42")
        output = briefing_to_markdown(briefing)
        assert "id: brief-42" in output

    def test_frontmatter_contains_title(self) -> None:
        """Frontmatter includes the briefing title."""
        briefing = _make_briefing(title="Weekly AI Digest")
        output = briefing_to_markdown(briefing)
        assert "Weekly AI Digest" in output

    def test_frontmatter_contains_query_when_present(self) -> None:
        """Frontmatter includes the query when it is not None."""
        briefing = _make_briefing(query="LLM trends")
        output = briefing_to_markdown(briefing)
        assert "LLM trends" in output

    def test_frontmatter_no_query_when_none(self) -> None:
        """Frontmatter omits the query line when query is None."""
        briefing = _make_briefing(query=None)
        output = briefing_to_markdown(briefing)
        # 'query:' should not appear
        assert "query:" not in output

    def test_content_body_preserved(self) -> None:
        """The briefing content (already Markdown) is preserved in the output."""
        content = "## Heading\nSome paragraph text here."
        briefing = _make_briefing(content=content)
        output = briefing_to_markdown(briefing)
        assert "## Heading" in output
        assert "Some paragraph text here." in output

    def test_frontmatter_contains_created_at(self) -> None:
        """Frontmatter includes the created_at timestamp."""
        briefing = _make_briefing(created_at="2025-01-15T10:00:00")
        output = briefing_to_markdown(briefing)
        assert "2025-01-15T10:00:00" in output

    def test_output_is_string(self) -> None:
        """Return type is always str."""
        briefing = _make_briefing()
        output = briefing_to_markdown(briefing)
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# briefing_to_json
# ---------------------------------------------------------------------------


class TestBriefingToJson:
    """Tests for briefing_to_json."""

    def test_returns_dict(self) -> None:
        """briefing_to_json returns a dict."""
        briefing = _make_briefing()
        result = briefing_to_json(briefing)
        assert isinstance(result, dict)

    def test_dict_contains_all_fields(self) -> None:
        """Returned dict contains all BriefingRow fields."""
        briefing = _make_briefing(
            briefing_id="brief-1",
            title="Test",
            query="ai",
            content="Some content",
        )
        result = briefing_to_json(briefing)
        assert result["id"] == "brief-1"
        assert result["title"] == "Test"
        assert result["query"] == "ai"
        assert result["content"] == "Some content"

    def test_dict_json_serializable(self) -> None:
        """Returned dict is JSON-serializable."""
        briefing = _make_briefing()
        result = briefing_to_json(briefing)
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert parsed["id"] == briefing.id


# ---------------------------------------------------------------------------
# ExportFormat enum
# ---------------------------------------------------------------------------


class TestExportFormat:
    """Tests for the ExportFormat enum."""

    def test_markdown_value(self) -> None:
        """ExportFormat.markdown has value 'markdown'."""
        assert ExportFormat.markdown == "markdown"

    def test_json_value(self) -> None:
        """ExportFormat.json has value 'json'."""
        assert ExportFormat.json == "json"

    def test_csv_value(self) -> None:
        """ExportFormat.csv has value 'csv'."""
        assert ExportFormat.csv == "csv"

    def test_str_subtype(self) -> None:
        """ExportFormat is a subtype of str."""
        assert isinstance(ExportFormat.markdown, str)
