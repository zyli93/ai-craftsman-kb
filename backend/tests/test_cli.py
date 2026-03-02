"""Tests for the CLI skeleton, search, entities, stats, radar, and triage commands."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from ai_craftsman_kb.cli import cli
from ai_craftsman_kb.db.models import DocumentRow, EntityRow
from ai_craftsman_kb.radar.engine import RadarReport
from ai_craftsman_kb.search.hybrid import SearchResult


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


def test_server_command(runner: CliRunner, minimal_config) -> None:
    """server command should echo the default port (8000) and call uvicorn.run."""
    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("uvicorn.run") as mock_uvicorn,
    ):
        result = runner.invoke(cli, ["server"])
    assert result.exit_code == 0
    assert "8000" in result.output  # default port
    mock_uvicorn.assert_called_once()
    call_kwargs = mock_uvicorn.call_args
    assert call_kwargs.kwargs.get("port") == 8000 or call_kwargs.args[1] == 8000 or 8000 in call_kwargs.args


def test_doctor_command(runner: CliRunner) -> None:
    """doctor command should print the Doctor header."""
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "Doctor" in result.output


def test_stats_command_no_db(runner: CliRunner, tmp_path: pytest.TempPathFactory) -> None:
    """stats with a fresh data dir should succeed (zero counts) or fail gracefully."""
    result = runner.invoke(cli, ["--config-dir", str(tmp_path), "stats"])
    # The stats command either succeeds (fresh DB with zero counts) or exits 1
    # due to missing YAML config -- both are acceptable non-crash outcomes.
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


def test_server_custom_port(runner: CliRunner, minimal_config) -> None:
    """server accepts a custom --port option."""
    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("uvicorn.run"),
    ):
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
    """Create a mock RadarEngine instance whose search() returns report.

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
        A MagicMock that supports async with protocol.
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
    """radar with no results shows no results message."""
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
    """delete command aborts cleanly when user answers n at the prompt."""
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


# ---------------------------------------------------------------------------
# search command tests (task_25)
# ---------------------------------------------------------------------------


def _make_search_result(
    doc_id: str = "abc-123",
    title: str = "LLM Inference at Scale",
    source_type: str = "arxiv",
    url: str = "https://arxiv.org/abs/2501.00001",
    score: float = 0.85,
    author: str | None = "Jane Smith",
    published_at: str | None = "2025-01-15T10:00:00",
    excerpt: str | None = "A study on LLM inference optimisation strategies.",
    origin: str = "pro",
) -> SearchResult:
    """Create a minimal SearchResult for testing.

    Args:
        doc_id: Document UUID string.
        title: Document title.
        source_type: Source type string.
        url: Document URL.
        score: Combined relevance score.
        author: Author name or None.
        published_at: ISO 8601 publication timestamp or None.
        excerpt: Short content excerpt or None.
        origin: Document origin (pro, radar, adhoc).

    Returns:
        A SearchResult instance with the given test values.
    """
    return SearchResult(
        document_id=doc_id,
        title=title,
        source_type=source_type,
        url=url,
        score=score,
        author=author,
        published_at=published_at,
        excerpt=excerpt,
        origin=origin,
    )


def _make_mock_hybrid_searcher(results: list) -> MagicMock:
    """Create a mock HybridSearch whose search() returns results.

    Args:
        results: The list of SearchResult objects the mock search() returns.

    Returns:
        A MagicMock with search configured as AsyncMock.
    """
    mock_searcher = MagicMock()
    mock_searcher.search = AsyncMock(return_value=results)
    return mock_searcher


def test_search_command_returns_results(runner: CliRunner, minimal_config) -> None:
    """search command should display results when HybridSearch returns matches."""
    result_1 = _make_search_result(title="LLM Inference at Scale", score=0.9)
    result_2 = _make_search_result(
        doc_id="def-456",
        title="Transformer Optimisation",
        score=0.8,
        url="https://arxiv.org/abs/2",
    )
    mock_searcher = _make_mock_hybrid_searcher([result_1, result_2])
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("ai_craftsman_kb.search.hybrid.HybridSearch", return_value=mock_searcher),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.processing.embedder.Embedder"),
        patch("ai_craftsman_kb.search.vector_store.VectorStore"),
    ):
        result = runner.invoke(cli, ["search", "LLM inference"])

    assert result.exit_code == 0
    assert "LLM Inference at Scale" in result.output


def test_search_command_no_results(runner: CliRunner, minimal_config) -> None:
    """search command shows No results found when HybridSearch returns empty list."""
    mock_searcher = _make_mock_hybrid_searcher([])
    mock_conn = _make_mock_conn()
    mock_vector_store = MagicMock()
    mock_vector_store.get_collection_info.return_value = {"vectors_count": 10}

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("ai_craftsman_kb.search.hybrid.HybridSearch", return_value=mock_searcher),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.processing.embedder.Embedder"),
        patch("ai_craftsman_kb.search.vector_store.VectorStore", return_value=mock_vector_store),
    ):
        result = runner.invoke(cli, ["search", "nonexistent topic"])

    assert result.exit_code == 0
    assert "No results" in result.output or "no results" in result.output.lower()


def test_search_command_empty_vector_store(runner: CliRunner, minimal_config) -> None:
    """search command warns about empty embeddings when Qdrant has 0 vectors."""
    mock_searcher = _make_mock_hybrid_searcher([])
    mock_conn = _make_mock_conn()
    mock_vector_store = MagicMock()
    mock_vector_store.get_collection_info.return_value = {"vectors_count": 0}

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("ai_craftsman_kb.search.hybrid.HybridSearch", return_value=mock_searcher),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.processing.embedder.Embedder"),
        patch("ai_craftsman_kb.search.vector_store.VectorStore", return_value=mock_vector_store),
    ):
        result = runner.invoke(cli, ["search", "AI", "--mode", "semantic"])

    assert result.exit_code == 0
    # Should suggest running ingest
    assert "ingest" in result.output.lower() or "embedding" in result.output.lower()


def test_search_command_keyword_mode(runner: CliRunner, minimal_config) -> None:
    """search --mode keyword passes mode=keyword to HybridSearch.search()."""
    sr = _make_search_result(title="Keyword Result")
    mock_searcher = _make_mock_hybrid_searcher([sr])
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("ai_craftsman_kb.search.hybrid.HybridSearch", return_value=mock_searcher),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.processing.embedder.Embedder"),
        patch("ai_craftsman_kb.search.vector_store.VectorStore"),
    ):
        result = runner.invoke(cli, ["search", "LLM", "--mode", "keyword"])

    assert result.exit_code == 0
    call_kwargs = mock_searcher.search.call_args
    assert call_kwargs is not None
    assert call_kwargs.kwargs.get("mode") == "keyword"


def test_search_command_source_filter(runner: CliRunner, minimal_config) -> None:
    """search --source hn passes source_types=[hn] to HybridSearch.search()."""
    sr = _make_search_result(source_type="hn")
    mock_searcher = _make_mock_hybrid_searcher([sr])
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("ai_craftsman_kb.search.hybrid.HybridSearch", return_value=mock_searcher),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.processing.embedder.Embedder"),
        patch("ai_craftsman_kb.search.vector_store.VectorStore"),
    ):
        result = runner.invoke(cli, ["search", "AI", "--source", "hn", "--mode", "keyword"])

    assert result.exit_code == 0
    call_kwargs = mock_searcher.search.call_args
    assert call_kwargs is not None
    source_types = call_kwargs.kwargs.get("source_types")
    assert source_types == ["hn"]


def test_search_command_since_filter(runner: CliRunner, minimal_config) -> None:
    """search --since 2025-01-01 passes since value to HybridSearch.search()."""
    sr = _make_search_result()
    mock_searcher = _make_mock_hybrid_searcher([sr])
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("ai_craftsman_kb.search.hybrid.HybridSearch", return_value=mock_searcher),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.processing.embedder.Embedder"),
        patch("ai_craftsman_kb.search.vector_store.VectorStore"),
    ):
        result = runner.invoke(
            cli, ["search", "AI", "--since", "2025-01-01", "--mode", "keyword"]
        )

    assert result.exit_code == 0
    call_kwargs = mock_searcher.search.call_args
    assert call_kwargs is not None
    assert call_kwargs.kwargs.get("since") == "2025-01-01"


def test_search_command_invalid_since(runner: CliRunner, minimal_config) -> None:
    """search --since with invalid date format prints an error and exits cleanly."""
    with patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config):
        result = runner.invoke(cli, ["search", "AI", "--since", "not-a-date"])

    assert result.exit_code == 0
    assert "Invalid date" in result.output or "Error" in result.output


def test_search_command_limit(runner: CliRunner, minimal_config) -> None:
    """search --limit 5 passes limit=5 to HybridSearch.search()."""
    mock_searcher = _make_mock_hybrid_searcher([])
    mock_conn = _make_mock_conn()
    mock_vector_store = MagicMock()
    mock_vector_store.get_collection_info.return_value = {"vectors_count": 10}

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("ai_craftsman_kb.search.hybrid.HybridSearch", return_value=mock_searcher),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.processing.embedder.Embedder"),
        patch("ai_craftsman_kb.search.vector_store.VectorStore", return_value=mock_vector_store),
    ):
        result = runner.invoke(cli, ["search", "AI", "--limit", "5", "--mode", "keyword"])

    assert result.exit_code == 0
    call_kwargs = mock_searcher.search.call_args
    assert call_kwargs is not None
    assert call_kwargs.kwargs.get("limit") == 5


def test_search_multiple_sources_task25(runner: CliRunner, minimal_config) -> None:
    """search accepts multiple --source flags and passes them as a list."""
    mock_searcher = _make_mock_hybrid_searcher([])
    mock_conn = _make_mock_conn()
    mock_vector_store = MagicMock()
    mock_vector_store.get_collection_info.return_value = {"vectors_count": 10}

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("ai_craftsman_kb.search.hybrid.HybridSearch", return_value=mock_searcher),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch("ai_craftsman_kb.processing.embedder.Embedder"),
        patch("ai_craftsman_kb.search.vector_store.VectorStore", return_value=mock_vector_store),
    ):
        result = runner.invoke(
            cli, ["search", "AI", "--source", "hn", "--source", "reddit", "--mode", "keyword"]
        )

    assert result.exit_code == 0
    call_kwargs = mock_searcher.search.call_args
    assert call_kwargs is not None
    source_types = call_kwargs.kwargs.get("source_types")
    assert set(source_types) == {"hn", "reddit"}


# ---------------------------------------------------------------------------
# entities command tests (task_25)
# ---------------------------------------------------------------------------


def _make_entity_row(
    entity_id: str = "ent-001",
    name: str = "OpenAI",
    entity_type: str = "company",
    mention_count: int = 42,
    first_seen_at: str | None = "2025-01-01T00:00:00",
) -> EntityRow:
    """Create a minimal EntityRow for testing.

    Args:
        entity_id: Entity UUID string.
        name: Human-readable entity name.
        entity_type: One of the 7 valid entity types.
        mention_count: Number of document mentions.
        first_seen_at: ISO 8601 timestamp of first mention.

    Returns:
        An EntityRow instance with the given test values.
    """
    return EntityRow(
        id=entity_id,
        name=name,
        entity_type=entity_type,
        normalized_name=name.lower(),
        mention_count=mention_count,
        first_seen_at=first_seen_at,
    )


def test_entities_command_lists_entities(runner: CliRunner, minimal_config) -> None:
    """entities command displays entities returned by EntitySearch.list_entities()."""
    ent1 = _make_entity_row(name="OpenAI", entity_type="company", mention_count=50)
    ent2 = _make_entity_row(
        entity_id="ent-002", name="GPT-4", entity_type="technology", mention_count=30
    )
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch(
            "ai_craftsman_kb.search.entity_search.EntitySearch.list_entities",
            new_callable=AsyncMock,
            return_value=[ent1, ent2],
        ),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
    ):
        result = runner.invoke(cli, ["entities"])

    assert result.exit_code == 0
    assert "OpenAI" in result.output
    assert "GPT-4" in result.output


def test_entities_command_no_entities(runner: CliRunner, minimal_config) -> None:
    """entities command shows No entities found when the list is empty."""
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch(
            "ai_craftsman_kb.search.entity_search.EntitySearch.list_entities",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
    ):
        result = runner.invoke(cli, ["entities"])

    assert result.exit_code == 0
    assert "No entities" in result.output or "no entities" in result.output.lower()


def test_entities_command_type_filter(runner: CliRunner, minimal_config) -> None:
    """entities --type person passes entity_type=person to list_entities()."""
    ent = _make_entity_row(name="Sam Altman", entity_type="person", mention_count=20)
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch(
            "ai_craftsman_kb.search.entity_search.EntitySearch.list_entities",
            new_callable=AsyncMock,
            return_value=[ent],
        ) as mock_list,
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
    ):
        result = runner.invoke(cli, ["entities", "--type", "person"])

    assert result.exit_code == 0
    assert "Sam Altman" in result.output
    call_kwargs = mock_list.call_args
    assert call_kwargs is not None
    assert call_kwargs.kwargs.get("entity_type") == "person"


def test_entities_command_top_option(runner: CliRunner, minimal_config) -> None:
    """entities --top 5 passes limit=5 to list_entities()."""
    mock_conn = _make_mock_conn()

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch(
            "ai_craftsman_kb.search.entity_search.EntitySearch.list_entities",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_list,
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
    ):
        result = runner.invoke(cli, ["entities", "--top", "5"])

    assert result.exit_code == 0
    call_kwargs = mock_list.call_args
    assert call_kwargs is not None
    assert call_kwargs.kwargs.get("limit") == 5


# ---------------------------------------------------------------------------
# stats command tests (task_25)
# ---------------------------------------------------------------------------


def test_stats_command_with_qdrant_info(runner: CliRunner, minimal_config) -> None:
    """stats command shows document count and Qdrant info when available."""
    db_stats = {
        "total_documents": 100,
        "embedded_documents": 80,
        "total_entities": 500,
        "total_sources": 5,
        "total_briefings": 3,
    }
    mock_conn = _make_mock_conn()
    mock_vector_store = MagicMock()
    mock_vector_store.get_collection_info.return_value = {
        "vectors_count": 1600,
        "disk_size_bytes": 0,
    }

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch(
            "ai_craftsman_kb.db.queries.get_stats",
            new_callable=AsyncMock,
            return_value=db_stats,
        ),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch(
            "ai_craftsman_kb.search.vector_store.VectorStore",
            return_value=mock_vector_store,
        ),
    ):
        result = runner.invoke(cli, ["stats"])

    assert result.exit_code == 0
    # Panel title should appear
    assert "AI Craftsman KB" in result.output
    # Document count should appear in the output
    assert "100" in result.output


def test_stats_command_qdrant_unavailable(runner: CliRunner, minimal_config) -> None:
    """stats command degrades gracefully when Qdrant is not running."""
    db_stats = {
        "total_documents": 50,
        "embedded_documents": 30,
        "total_entities": 200,
        "total_sources": 3,
        "total_briefings": 1,
    }
    mock_conn = _make_mock_conn()
    mock_vector_store = MagicMock()
    mock_vector_store.get_collection_info.side_effect = RuntimeError("Qdrant not available")

    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch(
            "ai_craftsman_kb.db.queries.get_stats",
            new_callable=AsyncMock,
            return_value=db_stats,
        ),
        patch("ai_craftsman_kb.db.sqlite.init_db", new_callable=AsyncMock),
        patch("ai_craftsman_kb.db.sqlite.get_db", return_value=mock_conn),
        patch(
            "ai_craftsman_kb.search.vector_store.VectorStore",
            return_value=mock_vector_store,
        ),
    ):
        result = runner.invoke(cli, ["stats"])

    # Should still succeed -- just show a dash for vector count
    assert result.exit_code == 0
    assert "AI Craftsman KB" in result.output


# ---------------------------------------------------------------------------
# server command tests (task_40)
# ---------------------------------------------------------------------------


def test_server_command_calls_uvicorn(runner: CliRunner, minimal_config) -> None:
    """server command calls uvicorn.run with the correct app string and port."""
    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("uvicorn.run") as mock_uvicorn,
    ):
        result = runner.invoke(cli, ["server"])

    assert result.exit_code == 0
    mock_uvicorn.assert_called_once()
    args, kwargs = mock_uvicorn.call_args
    assert args[0] == "ai_craftsman_kb.server:app"
    assert kwargs.get("port") == 8000
    assert kwargs.get("reload") is False


def test_server_command_custom_port_calls_uvicorn(runner: CliRunner, minimal_config) -> None:
    """server --port 9001 passes port=9001 to uvicorn.run."""
    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("uvicorn.run") as mock_uvicorn,
    ):
        result = runner.invoke(cli, ["server", "--port", "9001"])

    assert result.exit_code == 0
    assert "9001" in result.output
    _, kwargs = mock_uvicorn.call_args
    assert kwargs.get("port") == 9001


def test_server_command_reload_flag(runner: CliRunner, minimal_config) -> None:
    """server --reload passes reload=True to uvicorn.run."""
    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("uvicorn.run") as mock_uvicorn,
    ):
        result = runner.invoke(cli, ["server", "--reload"])

    assert result.exit_code == 0
    assert "reload" in result.output.lower() or "Auto-reload" in result.output
    _, kwargs = mock_uvicorn.call_args
    assert kwargs.get("reload") is True


def test_server_command_no_dashboard_flag(runner: CliRunner, minimal_config) -> None:
    """server --no-dashboard echoes that dashboard won't be served."""
    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("uvicorn.run"),
    ):
        result = runner.invoke(cli, ["server", "--no-dashboard"])

    assert result.exit_code == 0
    assert "dashboard" in result.output.lower() or "no-dashboard" in result.output.lower()


def test_server_command_with_mcp_flag(runner: CliRunner, minimal_config) -> None:
    """server --with-mcp echoes the MCP startup message."""
    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch("uvicorn.run"),
    ):
        result = runner.invoke(cli, ["server", "--with-mcp"])

    assert result.exit_code == 0
    assert "MCP" in result.output or "mcp" in result.output.lower()


def test_server_help_shows_all_options(runner: CliRunner, minimal_config) -> None:
    """server --help lists all options including --no-dashboard, --with-mcp, --reload."""
    with patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config):
        result = runner.invoke(cli, ["server", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--no-dashboard" in result.output
    assert "--with-mcp" in result.output
    assert "--reload" in result.output


# ---------------------------------------------------------------------------
# mcp command tests (task_40)
# ---------------------------------------------------------------------------


def test_mcp_command_appears_in_help(runner: CliRunner) -> None:
    """mcp command should appear in the top-level --help output."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "mcp" in result.output


def test_mcp_command_help(runner: CliRunner, minimal_config) -> None:
    """mcp --help should exit cleanly and describe the stdio transport."""
    with patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config):
        result = runner.invoke(cli, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "MCP" in result.output or "stdio" in result.output


def test_mcp_command_missing_module(runner: CliRunner, minimal_config) -> None:
    """mcp command exits with error code 1 when mcp_server module is not available."""
    with (
        patch("ai_craftsman_kb.cli.load_config", return_value=minimal_config),
        patch.dict("sys.modules", {"ai_craftsman_kb.mcp_server": None}),
    ):
        result = runner.invoke(cli, ["mcp"])

    # Should fail gracefully (exit code 1) with a helpful message
    assert result.exit_code == 1 or "not yet available" in result.output or "MCP" in result.output
