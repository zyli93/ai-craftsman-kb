"""AI Craftsman KB CLI entry point.

Entry point: cr = "ai_craftsman_kb.cli:cli"
Commands are implemented in later tasks (02–45).
"""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="ai-craftsman-kb")
def cli() -> None:
    """AI Craftsman KB — aggregate, index, and search AI content locally."""
