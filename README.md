# AI Craftsman KB

A local-first CLI + web dashboard for aggregating, indexing, and semantically searching
AI/ML content from Hacker News, Substack, YouTube, Reddit, ArXiv, RSS, and DEV.to.

## Features

- **Pro mode**: Subscribe to sources and pull new content automatically
- **Radar mode**: On-demand search across sources by topic
- **Semantic search**: Hybrid FTS + vector search with Reciprocal Rank Fusion
- **Entity extraction**: Automatic extraction of people, companies, technologies
- **Briefings**: AI-generated summaries on any topic
- **Dashboard**: Local web UI for browsing and managing content
- **MCP server**: Use your KB as a tool in Claude and other MCP-compatible AI assistants

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [pnpm](https://pnpm.io/) (for the dashboard, optional)

## Installation

```bash
# Clone the repo
git clone https://github.com/zyli93/ai-craftsman-kb.git
cd ai-craftsman-kb

# Install dependencies
uv sync

# Verify installation
uv run cr --version
```

## Quick Start

```bash
# Check system health
uv run cr doctor

# Configure your sources (edit config/sources.yaml)
# Then run first ingest
uv run cr ingest pro

# Search your knowledge base
uv run cr search "retrieval augmented generation"

# Run radar search on a topic
uv run cr radar "LLM agents 2025"

# Start the web dashboard
uv run cr server
```

## Configuration

Copy and edit the config files in `config/`:

- `sources.yaml` — which sources to ingest and their settings
- `settings.yaml` — LLM provider, embedding model, data directory
- `filters.yaml` — content filtering rules per source

Set API keys via environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export YOUTUBE_API_KEY=AIza...
export REDDIT_CLIENT_ID=...
export REDDIT_CLIENT_SECRET=...
```

## CLI Commands

```
cr ingest pro          # Run Pro mode ingestion for all enabled sources
cr ingest radar        # Run Radar mode search
cr search <query>      # Semantic search across all content
cr radar <topic>       # On-demand topic search
cr entities            # Browse extracted entities
cr stats               # Show KB statistics
cr doctor              # Check system health
cr server              # Start web dashboard + API
cr mcp                 # Start MCP server (stdio)
```

## Architecture

- **Backend**: Python 3.12+, FastAPI, SQLite (FTS5), Qdrant (local), Click CLI
- **Dashboard**: TypeScript, React, Vite, Tailwind CSS v4, shadcn/ui
- **MCP Server**: Python MCP SDK, stdio transport

See [doc/manual.md](doc/manual.md) for full operational guide and CLI reference.

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint
uv run ruff check backend/
uv run mypy backend/
```

## License

MIT
