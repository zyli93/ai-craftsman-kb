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

### How env vars are loaded

Credentials are stored **outside the repo** and loaded automatically via
[direnv](https://direnv.net/).

**1. Install direnv and hook it into your shell:**

```bash
brew install direnv

# Add to ~/.zshrc (or ~/.bashrc):
eval "$(direnv hook zsh)"
```

Restart your shell after adding the hook.

**2. Store your keys in `~/.ai-craftsman-kb/.env`:**

```bash
mkdir -p ~/.ai-craftsman-kb
cat > ~/.ai-craftsman-kb/.env << 'EOF'
export OPENROUTER_API_KEY='sk-or-...'
export YOUTUBE_API_KEY='...'
export FIREWORKS_API_KEY='...'
export GROQ_API_KEY='...'
export CEREBRAS_API_KEY='...'
EOF
chmod 600 ~/.ai-craftsman-kb/.env
```

The file uses `export KEY=value` syntax. Add or remove keys as needed.

**3. The project `.envrc` sources that file:**

```bash
# .envrc (already in the repo root, gitignored)
source ~/.ai-craftsman-kb/.env
```

**4. Allow direnv to load it:**

```bash
cd /path/to/ai-craftsman-kb
direnv allow
```

After this, every time you `cd` into the project, direnv automatically
exports the keys into your shell. `uv run cr doctor` will confirm they're
picked up.

### How the config reads env vars

`config/settings.yaml` references env vars with `${VAR_NAME}` placeholders:

```yaml
providers:
  groq:
    api_key: ${GROQ_API_KEY}
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
```

The config loader (`config/loader.py`) interpolates these at load time by
reading `os.environ`. As long as direnv has exported the vars into your
shell, the CLI and server will find them.

### Important notes

- **`.envrc` is gitignored** — it will never be committed.
- **Keys live in `~/.ai-craftsman-kb/.env`**, not in the repo.
- Use `export` syntax in the `.env` file — direnv's `dotenv` command does
  NOT support `export`, so the `.envrc` uses `source` instead.
- After editing `~/.ai-craftsman-kb/.env`, run `direnv allow` again (or
  re-enter the directory) to pick up changes.
- Run `uv run cr doctor` to verify all keys are detected.

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

## Step 5 — Configure Your Sources

Edit `config/sources.yaml` to choose what content you want to index.
This is the file that controls which feeds, channels, and topics get ingested.

### Hacker News (no API key needed)

```yaml
hackernews:
  mode: top        # top | new | best
  limit: 30        # number of stories per ingest
```

### ArXiv (no API key needed)

```yaml
arxiv:
  queries:
    - "cat:cs.CL AND abs:large language model"
    - "cat:cs.AI AND abs:reinforcement learning"
  max_results: 20  # per query
```

Use [ArXiv search syntax](https://info.arxiv.org/help/api/user-manual.html#query_details):
`cat:` for category, `abs:` for abstract, `ti:` for title, `au:` for author.

### RSS / Atom Feeds (no API key needed)

```yaml
rss:
  - url: https://lobste.rs/rss
    name: "Lobste.rs"
  - url: https://openai.com/blog/rss.xml
    name: "OpenAI Blog"
```

Any valid RSS or Atom feed URL works. Add as many as you like.

### DEV.to (no API key needed)

```yaml
devto:
  tags:
    - ai
    - machinelearning
    - llm
  limit: 20        # articles per tag
```

### Substack (no API key needed)

```yaml
substack:
  - slug: karpathy          # from the URL: karpathy.substack.com
    name: "Andrej Karpathy"
  - slug: simonwillison
    name: "Simon Willison"
```

The `slug` is the subdomain from the Substack URL (e.g. `karpathy` from `karpathy.substack.com`).

### YouTube (requires `YOUTUBE_API_KEY`)

```yaml
youtube_channels:
  - handle: "@AndrejKarpathy"   # the @handle from the channel URL
    name: "Andrej Karpathy"
  - handle: "@YannicKilcher"
    name: "Yannic Kilcher"
```

### Reddit (requires `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET`)

```yaml
subreddits:
  - name: LocalLLaMA
    sort: hot              # hot | new | top | rising
    limit: 25
  - name: MachineLearning
    sort: hot
    limit: 25
```

### Minimal Example

If you just want to get started quickly with no API keys at all:

```yaml
hackernews:
  mode: top
  limit: 20

rss:
  - url: https://lobste.rs/rss
    name: "Lobste.rs"

devto:
  tags:
    - ai
  limit: 10
```

Remove or comment out any source blocks you don't want.
Sources with missing API keys are automatically skipped during ingest.

---

## Step 6 — Initialize the Database

The first run of any command auto-creates the SQLite schema:

```bash
uv run cr stats
```

Expected: `0 documents`, Qdrant shows `0 vectors`. That's correct — no data yet.

---

## Step 7 — Health Check

```bash
uv run cr doctor
```

Expected output for a clean setup:
```
Configuration
  Config dir:  /path/to/ai-craftsman-kb/config
  Data dir:    /path/to/ai-craftsman-kb/data
  ✓ .../config/settings.yaml    LLM routing, providers, embedding, server settings (1049 bytes)
  ✓ .../config/sources.yaml     Content source definitions (HN, Reddit, YouTube, etc.) (710 bytes)
  ✓ .../config/filters.yaml     Per-source content filtering rules (855 bytes)

Health Checks
  ✓ LLM routing                 4/4 tasks configured
  ✓ SQLite DB                   0 documents
  ✓ Qdrant                      0 vectors
  ✓ Openrouter API key          key set (sk-or-…)
  ⚠ Anthropic API key           Not configured — set ANTHROPIC_API_KEY (optional)
  ⚠ YouTube API key             Not configured — set YOUTUBE_API_KEY (optional)
  ✓ Ollama API key              local provider (http://localhost:11434)
  ✓ HN connectivity             HTTP 200
  ✓ ArXiv connectivity          HTTP 200
  ✓ Data directory              /path/to/ai-craftsman-kb/data (120.4 GB free)
```

Warnings (⚠) are fine — they mean optional sources/providers are not configured.
Errors (✗) must be fixed before proceeding.

---

## Step 8 — First Ingest

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

## Step 9 — Launch the Web Dashboard

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

## Step 10 (Optional) — MCP Server for Claude Desktop

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
| SQLite database | `./data/craftsman.db` |
| Qdrant vectors | wherever you mounted `qdrant_storage` |
| Config (settings) | `config/settings.yaml` |
| Config (sources) | `config/sources.yaml` |
| Config (filters) | `config/filters.yaml` |
| Credentials | `~/.ai-craftsman-kb/.env` (loaded by direnv via `.envrc`) |

All paths are relative to the project root. Run `uv run cr doctor` to see
absolute paths for your setup.
