"""Tests for cli_output rich formatting helpers.

These tests use rich's Console(record=True) to capture rendered output
and assert on the textual content without caring about ANSI escape codes.
"""
from __future__ import annotations

from rich.console import Console

from ai_craftsman_kb.cli_output import (
    SOURCE_COLORS,
    ENTITY_TYPE_COLORS,
    _source_badge,
    make_ingest_progress,
    print_entities,
    print_error,
    print_ingest_report,
    print_radar_results,
    print_search_results,
    print_stats,
    print_success,
    print_warning,
)
from ai_craftsman_kb.db.models import DocumentRow, EntityRow
from ai_craftsman_kb.ingestors.runner import IngestReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture(func, *args, **kwargs) -> str:
    """Run *func* with a recording Console and return plain text output."""
    record_console = Console(record=True, highlight=False, markup=True, width=120)
    # Temporarily monkey-patch the module-level console used inside helpers
    import ai_craftsman_kb.cli_output as co
    original = co.console
    co.console = record_console
    try:
        func(*args, **kwargs)
    finally:
        co.console = original
    return record_console.export_text()


def _make_doc(**kwargs) -> DocumentRow:
    """Create a minimal DocumentRow for testing."""
    defaults = {
        "id": "doc-001",
        "origin": "pro",
        "source_type": "hn",
        "url": "https://news.ycombinator.com/item?id=1",
        "title": "Test Article",
        "published_at": "2025-01-15T10:00:00Z",
    }
    defaults.update(kwargs)
    return DocumentRow(**defaults)


def _make_entity(**kwargs) -> EntityRow:
    """Create a minimal EntityRow for testing."""
    defaults = {
        "id": "ent-001",
        "name": "OpenAI",
        "entity_type": "company",
        "normalized_name": "openai",
        "mention_count": 42,
        "first_seen_at": "2025-01-01T00:00:00Z",
    }
    defaults.update(kwargs)
    return EntityRow(**defaults)


def _make_report(**kwargs) -> IngestReport:
    """Create a minimal IngestReport for testing."""
    defaults = {
        "source_type": "hn",
        "fetched": 10,
        "passed_filter": 8,
        "stored": 5,
        "skipped_duplicate": 3,
        "errors": [],
    }
    defaults.update(kwargs)
    return IngestReport(**defaults)


# ---------------------------------------------------------------------------
# source badge
# ---------------------------------------------------------------------------


def test_source_badge_hn() -> None:
    """_source_badge returns a Text object for known source type."""
    badge = _source_badge("hn")
    assert "hn" in badge.plain


def test_source_badge_unknown() -> None:
    """_source_badge returns plain white Text for unknown source type."""
    badge = _source_badge("unknown_source")
    assert "unknown_source" in badge.plain


def test_source_colors_contains_expected_types() -> None:
    """SOURCE_COLORS should include the main source types."""
    expected = {"hn", "arxiv", "youtube", "reddit", "substack", "rss", "devto"}
    assert expected.issubset(SOURCE_COLORS.keys())


def test_entity_type_colors_contains_expected_types() -> None:
    """ENTITY_TYPE_COLORS should include common entity types."""
    expected = {"person", "company", "technology"}
    assert expected.issubset(ENTITY_TYPE_COLORS.keys())


# ---------------------------------------------------------------------------
# print_search_results
# ---------------------------------------------------------------------------


def test_print_search_results_empty() -> None:
    """Empty results list prints a 'No results' notice."""
    output = _capture(print_search_results, [])
    assert "No results" in output


def test_print_search_results_plain_docs() -> None:
    """Plain DocumentRow objects are rendered with title and source."""
    docs = [_make_doc(title="My Great Article", source_type="arxiv")]
    output = _capture(print_search_results, docs)
    assert "My Great Article" in output
    assert "arxiv" in output


def test_print_search_results_with_score() -> None:
    """SearchResult-like objects show score column."""

    class FakeResult:
        def __init__(self, doc: DocumentRow, score: float) -> None:
            self.document = doc
            self.score = score

    doc = _make_doc(title="Vector Search", source_type="hn")
    results = [FakeResult(doc, 0.95)]
    output = _capture(print_search_results, results)
    assert "Vector Search" in output
    assert "0.95" in output


def test_print_search_results_no_title() -> None:
    """Documents without a title show '(no title)' placeholder."""
    docs = [_make_doc(title=None)]
    output = _capture(print_search_results, docs)
    assert "(no title)" in output


def test_print_search_results_no_date() -> None:
    """Documents without published_at show dash placeholder."""
    docs = [_make_doc(published_at=None)]
    output = _capture(print_search_results, docs)
    assert "—" in output


def test_print_search_results_multiple_docs() -> None:
    """Multiple results are all rendered."""
    docs = [_make_doc(id=f"doc-{i}", title=f"Article {i}") for i in range(3)]
    output = _capture(print_search_results, docs)
    for i in range(3):
        assert f"Article {i}" in output


# ---------------------------------------------------------------------------
# print_entities
# ---------------------------------------------------------------------------


def test_print_entities_empty() -> None:
    """Empty entity list prints a 'No entities' notice."""
    output = _capture(print_entities, [])
    assert "No entities" in output


def test_print_entities_shows_name_and_type() -> None:
    """Entity name, type, mention count are visible in output."""
    entities = [_make_entity(name="OpenAI", entity_type="company", mention_count=42)]
    output = _capture(print_entities, entities)
    assert "OpenAI" in output
    assert "company" in output
    assert "42" in output


def test_print_entities_no_date() -> None:
    """Entities without first_seen_at show a dash."""
    entities = [_make_entity(first_seen_at=None)]
    output = _capture(print_entities, entities)
    assert "—" in output


# ---------------------------------------------------------------------------
# print_stats
# ---------------------------------------------------------------------------


def test_print_stats_shows_counts() -> None:
    """Stats panel shows document and entity counts."""
    stats_dict = {
        "total_documents": 1234,
        "total_entities": 567,
        "total_sources": 8,
        "total_briefings": 3,
        "embedded_documents": 1000,
        "unembedded_documents": 234,
    }
    output = _capture(print_stats, stats_dict)
    assert "1,234" in output
    assert "567" in output


def test_print_stats_embed_percentage() -> None:
    """Embedded percentage is computed correctly."""
    stats_dict = {
        "total_documents": 100,
        "embedded_documents": 75,
        "total_entities": 0,
        "total_sources": 0,
        "total_briefings": 0,
        "unembedded_documents": 25,
    }
    output = _capture(print_stats, stats_dict)
    assert "75%" in output


def test_print_stats_zero_docs() -> None:
    """Zero documents prints a dash instead of computing percentage."""
    stats_dict = {
        "total_documents": 0,
        "embedded_documents": 0,
        "total_entities": 0,
        "total_sources": 0,
        "total_briefings": 0,
        "unembedded_documents": 0,
    }
    output = _capture(print_stats, stats_dict)
    assert "—" in output


def test_print_stats_with_qdrant_info() -> None:
    """Qdrant info dict is displayed when provided."""
    stats_dict = {
        "total_documents": 10,
        "embedded_documents": 10,
        "total_entities": 5,
        "total_sources": 2,
        "total_briefings": 1,
        "unembedded_documents": 0,
    }
    qdrant = {"vectors_count": 50000, "disk_data_size": 50 * 1024 * 1024}
    output = _capture(print_stats, stats_dict, qdrant)
    assert "50,000" in output or "50000" in output
    assert "50 MB" in output


# ---------------------------------------------------------------------------
# print_ingest_report
# ---------------------------------------------------------------------------


def test_print_ingest_report_empty() -> None:
    """Empty report list prints a notice."""
    output = _capture(print_ingest_report, [])
    assert "No ingest reports" in output


def test_print_ingest_report_shows_source_and_counts() -> None:
    """Report table shows source type, fetched, stored counts."""
    reports = [_make_report(source_type="hn", fetched=10, stored=5)]
    output = _capture(print_ingest_report, reports)
    assert "hn" in output
    assert "10" in output
    assert "5" in output


def test_print_ingest_report_shows_errors() -> None:
    """Error messages are printed below the table."""
    reports = [_make_report(source_type="arxiv", errors=["timeout fetching item 123"])]
    output = _capture(print_ingest_report, reports)
    assert "timeout fetching item 123" in output


def test_print_ingest_report_multiple_sources() -> None:
    """Multiple source reports all appear in output."""
    reports = [
        _make_report(source_type="hn"),
        _make_report(source_type="reddit"),
    ]
    output = _capture(print_ingest_report, reports)
    assert "hn" in output
    assert "reddit" in output


# ---------------------------------------------------------------------------
# print_radar_results
# ---------------------------------------------------------------------------


def test_print_radar_results_empty() -> None:
    """Empty radar results print a notice."""
    output = _capture(print_radar_results, [])
    assert "No radar results" in output


def test_print_radar_results_shows_titles() -> None:
    """Radar docs appear in output with title and source."""
    docs = [_make_doc(title="LLM Survey", source_type="arxiv")]
    output = _capture(print_radar_results, docs, "llm agents")
    assert "LLM Survey" in output
    assert "arxiv" in output


def test_print_radar_results_grouped_by_source() -> None:
    """Results from different sources appear in separate sections."""
    docs = [
        _make_doc(id="d1", title="HN Post", source_type="hn"),
        _make_doc(id="d2", title="ArXiv Paper", source_type="arxiv"),
    ]
    output = _capture(print_radar_results, docs, "AI")
    assert "HN Post" in output
    assert "ArXiv Paper" in output


def test_print_radar_results_topic_in_title() -> None:
    """Topic string appears in panel title."""
    docs = [_make_doc(title="Article")]
    output = _capture(print_radar_results, docs, "machine learning")
    assert "machine learning" in output


# ---------------------------------------------------------------------------
# make_ingest_progress
# ---------------------------------------------------------------------------


def test_make_ingest_progress_returns_progress() -> None:
    """make_ingest_progress returns a rich Progress instance."""
    from rich.progress import Progress

    p = make_ingest_progress()
    assert isinstance(p, Progress)


def test_make_ingest_progress_usable_as_context_manager() -> None:
    """Progress can be started and stopped without error."""
    p = make_ingest_progress()
    with p:
        task_id = p.add_task("Test task", total=10)
        p.update(task_id, completed=5)


# ---------------------------------------------------------------------------
# Error / warning / success helpers
# ---------------------------------------------------------------------------


def test_print_error_contains_message() -> None:
    """print_error renders the error message."""
    output = _capture(print_error, "Something went wrong")
    assert "Something went wrong" in output


def test_print_error_with_hint() -> None:
    """print_error renders hint text when provided."""
    output = _capture(print_error, "Bad input", hint="Check the docs")
    assert "Check the docs" in output


def test_print_warning_contains_message() -> None:
    """print_warning renders the warning text."""
    output = _capture(print_warning, "Low disk space")
    assert "Low disk space" in output


def test_print_success_contains_message() -> None:
    """print_success renders the success message."""
    output = _capture(print_success, "All good!")
    assert "All good!" in output
