"""Tests for the CLI skeleton."""
import pytest
from click.testing import CliRunner

from ai_craftsman_kb.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    """Return a Click test runner."""
    return CliRunner()


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


def test_radar_command(runner: CliRunner) -> None:
    """radar command should echo the query."""
    result = runner.invoke(cli, ["radar", "LLM agents"])
    assert result.exit_code == 0
    assert "LLM agents" in result.output


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


def test_radar_with_source(runner: CliRunner) -> None:
    """radar accepts --source option."""
    result = runner.invoke(cli, ["radar", "transformers", "--source", "arxiv"])
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
