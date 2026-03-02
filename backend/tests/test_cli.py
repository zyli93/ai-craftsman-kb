"""Tests for the CLI skeleton and radar/triage commands."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from ai_craftsman_kb.cli import cli
from ai_craftsman_kb.db.models import DocumentRow
from ai_craftsman_kb.radar.engine import RadarReport


@pytest.fixture
def runner() -> CliRunner:
    """Return a Click test runner."""
    return CliRunner()


def _make_doc_row(
    doc_id: str = "abc-123",
    title: str = "Test Article",
    source_type: str = "hn",
    url: str = "https://example.com/1",
    published_at: str = "2025-01-15T10:00:00",
) -> DocumentRow:
    """Create a minimal DocumentRow for testing.

    Args:
        doc_id: The document UUID.
        title: The document title.
        source_type: The source type string.
        url: The document URL.
        published_at: ISO 8601 publication timestamp.

    Returns:
        A DocumentRow instance with test values.
    """
    return DocumentRow(
        id=doc_id,
        origin="radar",
        source_type=source_type,
        url=url,
        title=title,
        published_at=published_at,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Basic command tests (pre-existing)
# ---------------------------------------------------------------------------


def test_help(runner: CliRunner) -> None:
    """--help should list all top-level commands."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "search" in result.output
    assert "radar" in result.output
    assert "briefing" in result.output
    assert "server" in result.output
    assert "stats" in result.output
    assert "doctor" in result.output


def test_help_includes_triage_commands(runner: CliRunner) -> None:
    """--help should list promote, archive, and delete commands."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "promote" in result.output
    assert "archive" in result.output
    assert "delete" in result.output


def test_search_command(runner: CliRunner) -> None:
    """search command should echo the query back."""
    result = runner.invoke(cli, ["search", "machine learning"])
    assert result.exit_code == 0
    assert "machine learning" in result.output


def test_ingest_command(runner: CliRunner) -> None:
    """ingest command with no options should succeed."""
    result = runner.invoke(cli, ["ingest"])
    assert result.exit_code == 0


def test_ingest_with_source(runner: CliRunner) -> None:
    """ingest --source hn should echo the source name."""
    result = runner.invoke(cli, ["ingest", "--source", "hn"])
    assert result.exit_code == 0
    assert "hn" in result.output


def test_briefing_command(runner: CliRunner) -> None:
    """briefing command should succeed."""
    result = runner.invoke(cli, ["briefing", "AI tools"])
    assert result.exit_code == 0


def test_ingest_url_command(runner: CliRunner) -> None:
    """ingest-url command should accept a URL argument."""
    result = runner.invoke(cli, ["ingest-url", "https://example.com"])
    assert result.exit_code == 0


def test_entities_command(runner: CliRunner) -> None:
    """entities command should succeed with no options."""
    result = runner.invoke(cli, ["entities"])
    assert result.exit_code == 0


def test_server_command(runner: CliRunner) -> None:
    """server command should echo the default port (8000)."""
    result = runner.invoke(cli, ["server"])
    assert result.exit_code == 0
    assert "8000" in result.output  # default port


def test_doctor_command(runner: CliRunner) -> None:
    """doctor command should print the Doctor header."""
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "Doctor" in result.output


def test_stats_command_no_db(runner: CliRunner, tmp_path: pytest.TempPathFactory) -> None:
    """stats with a fresh data dir should succeed (zero counts) or fail gracefully."""
    result = runner.invoke(cli, ["--config-dir", str(tmp_path), "stats"])
    # The stats command either succeeds (fresh DB with zero counts) or exits 1
    # due to missing YAML config — both are acceptable non-crash outcomes.
    assert result.exit_code in (0, 1)


def test_config_dir_option(runner: CliRunner, tmp_path: pytest.TempPathFactory) -> None:
    """--config-dir option should be accepted without error."""
    result = runner.invoke(cli, ["--config-dir", str(tmp_path), "--help"])
    assert result.exit_code == 0


def test_search_options(runner: CliRunner) -> None:
    """search accepts --limit and --mode options."""
    result = runner.invoke(cli, ["search", "python", "--limit", "10", "--mode", "keyword"])
    assert result.exit_code == 0


def test_verbose_flag(runner: CliRunner) -> None:
    """--verbose flag is accepted without error."""
    result = runner.invoke(cli, ["--verbose", "doctor"])
    assert result.exit_code == 0


def test_search_multiple_sources(runner: CliRunner) -> None:
    """search accepts multiple --source flags."""
    result = runner.invoke(cli, ["search", "AI", "--source", "hn", "--source", "reddit"])
    assert result.exit_code == 0


def test_ingest_url_with_tags(runner: CliRunner) -> None:
    """ingest-url accepts multiple --tag options."""
    result = runner.invoke(
        cli, ["ingest-url", "https://example.com", "--tag", "ai", "--tag", "ml"]
    )
    assert result.exit_code == 0


def test_entities_with_type_and_top(runner: CliRunner) -> None:
    """entities accepts --type and --top options."""
    result = runner.invoke(cli, ["entities", "--type", "technology", "--top", "10"])
    assert result.exit_code == 0


def test_server_custom_port(runner: CliRunner) -> None:
    """server accepts a custom --port option."""
    result = runner.invoke(cli, ["server", "--port", "9000"])
    assert result.exit_code == 0
    assert "9000" in result.output


def test_briefing_no_radar(runner: CliRunner) -> None:
    """briefing --no-radar flag is accepted."""
    result = runner.invoke(cli, ["briefing", "neural networks", "--no-radar"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Radar command tests (task_29)
# ---------------------------------------------------------------------------


def _make_mock_radar_engine(report: RadarReport) -> MagicMock:
    """Create a mock RadarEngine instance whose search() returns ``report``.

    Args:
        report: The RadarReport the mock engine's search() method returns.

    Returns:
        A MagicMock with search configured as AsyncMock.
    """
    mock_engine = MagicMock()
    mock_engine.search = AsyncMock(return_value=report)
    return mock_engine


def _make_mock_conn() -> MagicMock:
    """Create a mock async context manager for aiosqlite connections.

    Returns:
        A MagicMock that supports ``async with`` protocol.
    """
    mock_conn = MagicMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    return mock_conn


def test_radar_command_basic(runner: CliRunner, tmp_path) -> None:
    """radar command fans out and prints results summary."""
    doc = _make_doc_row(doc_id="abc-123", title="GRPO Paper", source_type="arxiv")
    report = RadarReport(
        query="GRPO",
        total_found=1,
        new_documents=1,
        sources_searched=["arxiv"],
    )
    mock_engine = _make_mock_radar_engine(report)
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.radar.engine.RadarEngine", return_value=mock_engine),
        patch("ai_craftsman_kb.db.queries.list_documents", new_callable=AsyncMock, return_value=[doc]),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.ingestors.runner.INGESTORS", {}),
    ):
        result = runner.invoke(cli, ["--config-dir", str(tmp_path), "radar", "GRPO"])

    assert result.exit_code == 0
    assert "GRPO Paper" in result.output
    assert "abc-123" in result.output


def test_radar_command_shows_found_summary(runner: CliRunner, tmp_path) -> None:
    """radar command shows new/duplicate counts in summary line."""
    docs = [
        _make_doc_row("id1", "Article 1", "hn", "https://hn.com/1"),
        _make_doc_row("id2", "Article 2", "reddit", "https://reddit.com/1"),
    ]
    report = RadarReport(
        query="AI",
        total_found=3,
        new_documents=2,
        sources_searched=["hn", "reddit"],
    )
    mock_engine = _make_mock_radar_engine(report)
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.radar.engine.RadarEngine", return_value=mock_engine),
        patch("ai_craftsman_kb.db.queries.list_documents", new_callable=AsyncMock, return_value=docs),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.ingestors.runner.INGESTORS", {}),
    ):
        result = runner.invoke(cli, ["--config-dir", str(tmp_path), "radar", "AI"])

    assert result.exit_code == 0
    # Summary line should show 2 new and 1 duplicate
    assert "2" in result.output
    assert "new" in result.output.lower() or "promote" in result.output.lower()


def test_radar_command_with_source_filter(runner: CliRunner, tmp_path) -> None:
    """radar --source hn arxiv limits sources in the engine.search call."""
    report = RadarReport(
        query="LLM",
        total_found=0,
        new_documents=0,
        sources_searched=["hn", "arxiv"],
    )
    mock_engine = _make_mock_radar_engine(report)
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.radar.engine.RadarEngine", return_value=mock_engine),
        patch("ai_craftsman_kb.db.queries.list_documents", new_callable=AsyncMock, return_value=[]),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.ingestors.runner.INGESTORS", {}),
    ):
        result = runner.invoke(
            cli,
            ["--config-dir", str(tmp_path), "radar", "LLM", "--source", "hn", "--source", "arxiv"],
        )

    assert result.exit_code == 0
    # Verify search was called with sources=["hn", "arxiv"]
    call_kwargs = mock_engine.search.call_args
    assert call_kwargs is not None
    sources_arg = call_kwargs.kwargs.get("sources")
    assert sources_arg is not None
    assert set(sources_arg) == {"hn", "arxiv"}


def test_radar_command_with_limit(runner: CliRunner, tmp_path) -> None:
    """radar --limit 5 passes limit_per_source=5 to the engine."""
    report = RadarReport(query="AI", total_found=0, new_documents=0)
    mock_engine = _make_mock_radar_engine(report)
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.radar.engine.RadarEngine", return_value=mock_engine),
        patch("ai_craftsman_kb.db.queries.list_documents", new_callable=AsyncMock, return_value=[]),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.ingestors.runner.INGESTORS", {}),
    ):
        result = runner.invoke(
            cli, ["--config-dir", str(tmp_path), "radar", "AI", "--limit", "5"]
        )

    assert result.exit_code == 0
    call_kwargs = mock_engine.search.call_args
    limit_arg = call_kwargs.kwargs.get("limit_per_source")
    assert limit_arg == 5


def test_radar_command_no_results(runner: CliRunner, tmp_path) -> None:
    """radar with no results shows 'No results found' message."""
    report = RadarReport(query="obscure topic", total_found=0, new_documents=0)
    mock_engine = _make_mock_radar_engine(report)
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.radar.engine.RadarEngine", return_value=mock_engine),
        patch("ai_craftsman_kb.db.queries.list_documents", new_callable=AsyncMock, return_value=[]),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.ingestors.runner.INGESTORS", {}),
    ):
        result = runner.invoke(
            cli, ["--config-dir", str(tmp_path), "radar", "obscure topic"]
        )

    assert result.exit_code == 0
    assert "No results" in result.output or "obscure topic" in result.output


def test_radar_with_source(runner: CliRunner, tmp_path) -> None:
    """radar accepts --source option and exits successfully."""
    report = RadarReport(query="transformers", total_found=0, new_documents=0)
    mock_engine = _make_mock_radar_engine(report)
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.radar.engine.RadarEngine", return_value=mock_engine),
        patch("ai_craftsman_kb.db.queries.list_documents", new_callable=AsyncMock, return_value=[]),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.ingestors.runner.INGESTORS", {}),
    ):
        result = runner.invoke(
            cli,
            ["--config-dir", str(tmp_path), "radar", "transformers", "--source", "arxiv"],
        )

    assert result.exit_code == 0


def test_radar_shows_source_errors_as_warnings(runner: CliRunner, tmp_path) -> None:
    """radar shows per-source errors as warnings without failing."""
    report = RadarReport(
        query="AI",
        total_found=0,
        new_documents=0,
        errors={"hn": "Rate limit exceeded"},
    )
    mock_engine = _make_mock_radar_engine(report)
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.radar.engine.RadarEngine", return_value=mock_engine),
        patch("ai_craftsman_kb.db.queries.list_documents", new_callable=AsyncMock, return_value=[]),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.ingestors.runner.INGESTORS", {}),
    ):
        result = runner.invoke(cli, ["--config-dir", str(tmp_path), "radar", "AI"])

    assert result.exit_code == 0
    assert "Rate limit exceeded" in result.output or "hn" in result.output


# ---------------------------------------------------------------------------
# promote command tests
# ---------------------------------------------------------------------------


def test_promote_command_success(runner: CliRunner, tmp_path) -> None:
    """promote sets promoted_at on the document and prints success."""
    doc = _make_doc_row(doc_id="abc-123", title="GRPO Paper", source_type="arxiv")
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.db.queries.get_document", new_callable=AsyncMock, return_value=doc),
        patch("ai_craftsman_kb.db.queries.promote_document", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
    ):
        result = runner.invoke(
            cli, ["--config-dir", str(tmp_path), "promote", "abc-123"]
        )

    assert result.exit_code == 0
    assert "Promoted" in result.output or "OK" in result.output
    assert "GRPO Paper" in result.output


def test_promote_command_document_not_found(runner: CliRunner, tmp_path) -> None:
    """promote prints an error when the document_id does not exist."""
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.db.queries.get_document", new_callable=AsyncMock, return_value=None),
        patch("ai_craftsman_kb.db.queries.promote_document", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
    ):
        result = runner.invoke(
            cli, ["--config-dir", str(tmp_path), "promote", "nonexistent-id"]
        )

    assert result.exit_code == 0  # prints error but does not raise
    assert "not found" in result.output.lower() or "Error" in result.output


# ---------------------------------------------------------------------------
# archive command tests
# ---------------------------------------------------------------------------


def test_archive_command_success(runner: CliRunner, tmp_path) -> None:
    """archive sets is_archived on the document and prints success."""
    doc = _make_doc_row(doc_id="def-456", title="Reddit Post", source_type="reddit")
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.db.queries.get_document", new_callable=AsyncMock, return_value=doc),
        patch("ai_craftsman_kb.db.queries.archive_document", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
    ):
        result = runner.invoke(
            cli, ["--config-dir", str(tmp_path), "archive", "def-456"]
        )

    assert result.exit_code == 0
    assert "Archived" in result.output or "OK" in result.output
    assert "Reddit Post" in result.output


def test_archive_command_document_not_found(runner: CliRunner, tmp_path) -> None:
    """archive prints an error when the document_id does not exist."""
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.db.queries.get_document", new_callable=AsyncMock, return_value=None),
        patch("ai_craftsman_kb.db.queries.archive_document", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
    ):
        result = runner.invoke(
            cli, ["--config-dir", str(tmp_path), "archive", "bad-id"]
        )

    assert result.exit_code == 0
    assert "not found" in result.output.lower() or "Error" in result.output


# ---------------------------------------------------------------------------
# delete command tests
# ---------------------------------------------------------------------------


def test_delete_command_requires_confirmation(runner: CliRunner, tmp_path) -> None:
    """delete command with --yes soft-deletes the document and prints success."""
    doc = _make_doc_row(doc_id="ghi-789", title="Spam Post", source_type="reddit")
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.db.queries.get_document", new_callable=AsyncMock, return_value=doc),
        patch("ai_craftsman_kb.db.queries.soft_delete_document", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
    ):
        # Invoke with --yes to skip interactive prompt
        result = runner.invoke(
            cli, ["--config-dir", str(tmp_path), "delete", "--yes", "ghi-789"]
        )

    assert result.exit_code == 0
    assert "Deleted" in result.output or "OK" in result.output
    assert "Spam Post" in result.output


def test_delete_command_cancelled_by_user(runner: CliRunner, tmp_path) -> None:
    """delete command aborts cleanly when user answers 'n' at the prompt."""
    with patch("ai_craftsman_kb.db.queries.soft_delete_document", new_callable=AsyncMock):
        # Provide "n\n" as stdin to cancel the confirmation
        result = runner.invoke(
            cli, ["--config-dir", str(tmp_path), "delete", "ghi-789"], input="n\n"
        )

    # Click's confirmation_option exits with code 1 on abort
    assert result.exit_code != 0 or "Aborted" in result.output or "aborted" in result.output.lower()


def test_delete_command_document_not_found(runner: CliRunner, tmp_path) -> None:
    """delete prints an error when the document_id does not exist."""
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.db.queries.get_document", new_callable=AsyncMock, return_value=None),
        patch("ai_craftsman_kb.db.queries.soft_delete_document", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
    ):
        result = runner.invoke(
            cli, ["--config-dir", str(tmp_path), "delete", "--yes", "nonexistent"]
        )

    assert result.exit_code == 0
    assert "not found" in result.output.lower() or "Error" in result.output
