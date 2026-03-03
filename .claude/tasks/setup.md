# Setup Guide — AI Craftsman KB

Step-by-step from a fresh clone to a fully running system.
All steps verified against the actual codebase (v0.1.0, 45 tasks merged).

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12+ | `brew install python@3.12` or `pyenv` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js | 18+ | `brew install node` or `nvm` |
| pnpm | latest | `npm install -g pnpm` |
| Qdrant | local binary | see step 3 |

---

## Step 1 — Clone & Install Backend

```bash
git clone <repo-url> ai-craftsman-kb
cd ai-craftsman-kb

# Install all Python deps (including dev extras)
uv sync --extra dev

# Verify the `cr` CLI is available
uv run cr --help
```

Expected: you see the cr command group (ingest, search, radar, server, doctor, …).

---

## Step 2 — Set Up Credentials

See `credentials.md` for what each key is and where to get it.
See `vault.md` for how to store them securely.

**Minimum to get started (embeddings + filtering):**
```
OPENAI_API_KEY      — required for vector embeddings
OPENROUTER_API_KEY  — required for LLM filtering + entity extraction
```

The remaining keys unlock optional sources (YouTube, Reddit) and features
(briefing generation). The system degrades gracefully without them.

---

## Step 3 — Start Qdrant (local vector store)

Qdrant runs as a separate process. The backend connects to it on `localhost:6333`.

**Option A — Docker (recommended):**
```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Option B — Native binary:**
```bash
# Download from https://github.com/qdrant/qdrant/releases
# macOS arm64:
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-aarch64-apple-darwin.tar.gz | tar xz
./qdrant  # runs on :6333 by default
```

Qdrant data persists in `./qdrant_storage` (Docker) or `./storage` (native).
Move it somewhere stable, e.g. `~/.ai-craftsman-kb/qdrant/`.

---

## Step 4 — Build the Dashboard

```bash
cd dashboard
pnpm install
pnpm build   # outputs to dashboard/dist/
cd ..
```

The backend serves `dashboard/dist/` as static files at `/`.
You only need to rebuild after UI code changes.

For development with hot-reload, run `pnpm dev` in a separate terminal
and access the dashboard at `http://localhost:5173` instead.

---

## Step 5 — Initialize the Database

The first run of any command auto-creates the SQLite schema:

```bash
uv run cr stats
```

Expected: `0 documents`, Qdrant shows `0 vectors`. That's correct — no data yet.

---

## Step 6 — Health Check

```bash
uv run cr doctor
```

Expected output for a clean setup:
```
  ✓ Config file          data_dir=~/.ai-craftsman-kb/data
  ✓ SQLite DB            0 documents
  ✓ Qdrant               0 vectors
  ✓ OpenAI API key       key set (sk-pro…)
  ✓ OpenRouter API key   key set (sk-or-…)
  ⚠ Anthropic API key   Not configured — set ANTHROPIC_API_KEY (optional)
  ⚠ YouTube API key     Not configured — set YOUTUBE_API_KEY (optional)
  ⚠ Reddit credentials  Not configured — set REDDIT_CLIENT_ID / SECRET (optional)
  ✓ HN connectivity      HTTP 200
  ✓ ArXiv connectivity   HTTP 200
  ✓ Data directory       ~/.ai-craftsman-kb/data (120.4 GB free)
```

Warnings (⚠) are fine — they mean optional sources are disabled.
Errors (✗) must be fixed before proceeding.

---

## Step 7 — First Ingest

Start with sources that need no credentials:

```bash
# HN + ArXiv + RSS + DEV.to (no API keys required)
uv run cr ingest --source hn
uv run cr ingest --source arxiv
uv run cr ingest --source rss
uv run cr ingest --source devto
```

Or run everything at once (skips sources with missing keys):
```bash
uv run cr ingest
```

Verify it worked:
```bash
uv run cr stats
uv run cr search "large language models"
```

---

## Step 8 — Launch the Web Dashboard

```bash
uv run cr server
# → http://localhost:8000       (dashboard)
# → http://localhost:8000/docs  (API docs / Swagger UI)
```

Or with live-reload during development:
```bash
uv run cr server --reload --no-dashboard
# In a second terminal:
cd dashboard && pnpm dev
```

---

## Step 9 (Optional) — MCP Server for Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-craftsman-kb": {
      "command": "uv",
      "args": ["--project", "/path/to/ai-craftsman-kb", "run", "cr", "mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

Restart Claude Desktop. The KB tools will appear in Claude's tool list.

---

## Quick Verification Checklist

- [ ] `uv run cr --help` shows all commands
- [ ] `uv run cr doctor` shows no ✗ errors
- [ ] `uv run cr ingest --source hn` completes without errors
- [ ] `uv run cr search "AI"` returns results
- [ ] `http://localhost:8000` shows the dashboard
- [ ] `http://localhost:8000/docs` shows the API

---

## Data Locations

| Item | Path |
|------|------|
| SQLite database | `~/.ai-craftsman-kb/data/craftsman.db` |
| Qdrant vectors | wherever you mounted `qdrant_storage` |
| Config (settings) | `~/.ai-craftsman-kb/settings.yaml` |
| Config (sources) | `~/.ai-craftsman-kb/sources.yaml` |
| Credentials | `~/.ai-craftsman-kb/.env` (see vault.md) |

On first run, if `~/.ai-craftsman-kb/` doesn't exist, the system falls back
to the repo's `config/` directory for defaults.
