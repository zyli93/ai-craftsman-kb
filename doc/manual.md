# Manual — AI Craftsman KB

Everything you need to operate, maintain, and troubleshoot the system.

---

## Mental Model

**Two modes, one tool:**

- **Pro mode** — scheduled/manual pull of curated sources you define.
  Content is always indexed, always searchable. Run `cr ingest` daily or via cron.

- **Radar mode** — on-demand topic search across the open web.
  Fans out to HN, Reddit, ArXiv, DEV.to, YouTube simultaneously.
  New results are stored with `origin=radar`. You triage them and promote
  what's worth keeping.

Both modes feed the same search index (SQLite FTS + Qdrant vectors).

---

## CLI Reference

All commands are invoked as `uv run cr <command>` (or just `cr` if installed).

### Global Options

```
--config-dir PATH   Config directory (default: ~/.ai-craftsman-kb/)
-v, --verbose       Enable verbose/debug logging
--help              Show help
```

### `cr ingest`

Pull latest content from all enabled pro-tier sources.

```bash
cr ingest                 # all configured sources
cr ingest --source hn     # only Hacker News
cr ingest --source arxiv  # only ArXiv
```

Sources with missing API keys are automatically skipped.

### `cr ingest-url`

Ingest a single URL into the index.

```bash
cr ingest-url "https://example.com/article"
cr ingest-url "https://example.com/article" --tag ai --tag llm
```

### `cr search`

Search indexed content.

```bash
cr search "transformer architecture"
cr search "rust async" --source hn --source reddit --since 2024-01-01
cr search "agent orchestration" --mode semantic
cr search "GPT-4o" --mode keyword
cr search "query" --format json -o results.json
cr search "query" --format markdown -o results.md
```

| Option | Default | Description |
|--------|---------|-------------|
| `--source TEXT` | all | Filter by source type (repeatable) |
| `--since TEXT` | — | Only results after this date (YYYY-MM-DD) |
| `--limit INT` | 20 | Max results |
| `--mode` | hybrid | `hybrid`, `semantic`, or `keyword` |
| `--format` | terminal | `markdown` or `json` |
| `-o, --output PATH` | stdout | Write to file |

### `cr radar`

Search the open web on-demand for a topic. Results are stored with `origin=radar`.

```bash
cr radar "agentic AI 2025"
cr radar "rust embedded" --source hn --source reddit --limit 20
cr radar "LLM inference" --since 2025-01-01
```

| Option | Default | Description |
|--------|---------|-------------|
| `--source TEXT` | all | Limit to these source types (repeatable) |
| `--since TEXT` | — | Only results after this date |
| `--limit INT` | 10 | Max results per source |

### `cr promote`

Promote a radar result to pro tier.

```bash
cr promote <document-id>
```

After promoting, the document appears alongside pro-tier documents in search.

### `cr archive`

Archive a document (hide from default views). Not deleted — recoverable.

```bash
cr archive <document-id>
```

### `cr delete`

Soft-delete a document (sets `deleted_at` timestamp). Excluded from all views
but NOT physically removed from the database.

```bash
cr delete <document-id>          # interactive confirmation
cr delete <document-id> --yes    # skip confirmation (for scripts)
```

### `cr reset`

Delete all data (SQLite DB + Qdrant vectors) so ingestion starts fresh.
Shows what will be deleted and asks for confirmation.

```bash
cr reset                        # interactive confirmation
cr reset -y                     # skip confirmation (for scripts)
```

### `cr entities`

List top entities by mention count.

```bash
cr entities                      # top 20 of all types
cr entities --type person --top 30
```

### `cr briefing`

Generate a content briefing on a topic using LLM.

```bash
cr briefing "AI agents"
cr briefing "AI agents" --no-radar --no-ingest
cr briefing "AI agents" -o briefing.md
```

| Option | Default | Description |
|--------|---------|-------------|
| `--run-radar / --no-radar` | on | Run radar search before generating |
| `--run-ingest / --no-ingest` | on | Run ingest before generating |
| `-o, --output PATH` | stdout | Write briefing to Markdown file |

### `cr stats`

Show system statistics (documents, entities, sources, briefings).

```bash
cr stats
```

### `cr doctor`

Check system health and configuration. Output is grouped into:

1. **Configuration** — config file locations and presence
2. **API Keys** — credential status (available first, then missing)
3. **Services** — database, Qdrant, local LLM servers, backend/frontend, connectivity
4. **Config** — LLM routing, keyword extraction
5. **Quick Start** — hints for services that are down

```bash
cr doctor
```

Exit code 0 if all checks pass, 1 if any errors (⚠ warnings don't fail).

### `cr server`

Start the FastAPI backend (+ dashboard).

```bash
cr server                                    # http://localhost:8000
cr server --host 0.0.0.0 --port 9000        # custom bind
cr server --reload --no-dashboard            # dev mode (use with pnpm dev)
cr server --with-mcp                         # also start MCP server
```

### `cr mcp`

Start the MCP server (stdio transport for Claude Desktop). See the MCP section below.

```bash
cr mcp
```

---

## Config Files

The system looks for config in this order:
1. `~/.ai-craftsman-kb/settings.yaml` (user config — override defaults here)
2. `~/.ai-craftsman-kb/sources.yaml` (your sources list)
3. Falls back to `config/settings.yaml` and `config/sources.yaml` in the repo

**Recommended: copy the repo defaults to your home dir and edit there:**
```bash
mkdir -p ~/.ai-craftsman-kb
cp config/settings.yaml ~/.ai-craftsman-kb/settings.yaml
cp config/sources.yaml ~/.ai-craftsman-kb/sources.yaml
```

This way your personal config never gets overwritten by git pulls.

### Customizing Sources (`sources.yaml`)

```yaml
# Hacker News (no API key)
hackernews:
  mode: top        # top | new | best
  limit: 30

# ArXiv (no API key)
arxiv:
  queries:
    - "cat:cs.CL AND abs:large language model"
  max_results: 20

# RSS / Atom (no API key)
rss:
  - url: https://simonwillison.net/atom/everything/
    name: "Simon Willison"

# DEV.to (no API key)
devto:
  tags: [ai, machinelearning, llm]
  limit: 20

# Substack (no API key)
substack:
  - slug: thezvi
    name: "Zvi Mowshowitz"

# YouTube (requires YOUTUBE_API_KEY)
youtube_channels:
  - handle: "@lexfridman"
    name: "Lex Fridman"

# Reddit (requires REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET)
subreddits:
  - name: singularity
    sort: hot
    limit: 20
```

### Disabling Sources

Skip certain sources during bulk `cr ingest` without removing their config:
```yaml
disabled:
  - reddit
  - youtube
```

Valid source keys: `hn`, `substack`, `youtube`, `reddit`, `arxiv`, `rss`, `devto`.

---

## Environment Variables & Credentials

Keys are stored **outside the repo** at `~/.ai-craftsman-kb/.env` and loaded
automatically via direnv. See `setup.md` Step 2 for full details.

Quick reference:
```bash
# ~/.ai-craftsman-kb/.env (uses export syntax)
export OPENROUTER_API_KEY='sk-or-...'
export GROQ_API_KEY='gsk_...'
export CEREBRAS_API_KEY='csk-...'
export FIREWORKS_API_KEY='fw_...'
export YOUTUBE_API_KEY='AIza...'

# .envrc (project root, gitignored)
source ~/.ai-craftsman-kb/.env
```

After editing `.env`, re-enter the directory or run `direnv allow`.
Run `cr doctor` to verify all keys are detected.

---

## LLM Gateway (Multi-Provider with Failover)

### How it works

1. **Endpoints** — named provider+model combos with optional rate/daily limits
2. **Pools** — groups of interchangeable endpoints, tried in order of remaining quota
3. **Tasks** — map each LLM task to a pool

When an endpoint hits its daily limit or returns errors, the gateway
automatically routes to the next endpoint in the pool.

### Config structure (`settings.yaml`)

```yaml
llm:
  endpoints:
    cerebras-gpt-oss-120b:
      provider: cerebras
      model: gpt-oss-120b
      rate_limit: 1000
      daily_limit: 1440000

    groq-llama70b:
      provider: groq
      model: llama-3.3-70b-versatile
      rate_limit: 30
      daily_limit: 1000

    ollama-gpt-oss-20b:
      provider: ollama
      model: gpt-oss-20b
      # no limits = unlimited

  pools:
    gpt-oss-120b:
      endpoints: [cerebras-gpt-oss-120b, groq-gpt-oss-120b, openrouter-gpt-oss-120b]
      max_retries: 3

  tasks:
    filtering: gpt-oss-120b
    entity_extraction: gpt-oss-120b
    briefing: gpt-oss-120b
    source_discovery: gpt-oss-120b
    keyword_extraction: gpt-oss-120b
```

### Supported providers

| Provider | Env var | Notes |
|----------|---------|-------|
| `openrouter` | `OPENROUTER_API_KEY` | Many free models, rate-limited |
| `groq` | `GROQ_API_KEY` | Fast inference, generous free tier |
| `cerebras` | `CEREBRAS_API_KEY` | Very fast, high daily limits |
| `fireworks` | `FIREWORKS_API_KEY` | Open models, paid |
| `openai` | `OPENAI_API_KEY` | GPT-4o etc., paid |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude models, paid |
| `ollama` | — (base_url) | Local, free, no API key needed |
| `llamacpp` | — (base_url) | Local, free, no API key needed |

Provider credentials in `settings.yaml`:
```yaml
providers:
  groq:
    api_key: ${GROQ_API_KEY}
  cerebras:
    api_key: ${CEREBRAS_API_KEY}
  ollama:
    base_url: http://localhost:11434
  llamacpp:
    base_url: http://localhost:9990
```

### Legacy config (still supported)

```yaml
llm:
  filtering:
    provider: ollama
    model: llama3.1:8b
  entity_extraction:
    provider: ollama
    model: llama3.1:8b
```

### Running fully local (zero API cost)

1. Run Qdrant: `docker run -d -p 6333:6333 qdrant/qdrant`
2. Run Ollama: `ollama serve && ollama pull gpt-oss-20b`
3. Run llama.cpp for embeddings: `llama-server -m v5-small-retrieval-Q8_0.gguf --port 9990 --embedding -c 8192 -np 1 -b 2048 -ub 2048`
4. Configure all pools to use ollama endpoints

---

## Embeddings

| Provider | Model | Dimensions | Notes |
|----------|-------|-----------|-------|
| `llamacpp` | `v5-small-retrieval-Q8_0.gguf` | 1024 | Local, free |
| `openai` | `text-embedding-3-small` | 1536 | Paid, high quality |

Configure in `settings.yaml`:
```yaml
embedding:
  provider: llamacpp
  model: v5-small-retrieval-Q8_0.gguf
  chunk_size: 512
  chunk_overlap: 64
```

For llamacpp, the embedding server must be running:
```bash
llama-server -m /path/to/v5-small-retrieval-Q8_0.gguf --port 9990 --embedding -c 8192 -np 1 -b 2048 -ub 2048
```

---

## Search Modes

| Mode | How it works | When to use |
|------|-------------|-------------|
| `hybrid` (default) | FTS + vector + optional keyword tags, merged via RRF | Most queries |
| `semantic` | Vector similarity only | Conceptual/fuzzy queries |
| `keyword` | SQLite FTS only | Exact term lookup, fast |

### Search tuning (`settings.yaml`)

```yaml
search:
  default_limit: 20
  hybrid_weight_semantic: 0.6
  hybrid_weight_keyword: 0.4
  hybrid_weight_keyword_tags: 0.0  # >0 to include keyword tag search in RRF
```

### Keyword tag search

The processing pipeline extracts keywords from documents using the LLM.
When `hybrid_weight_keyword_tags > 0`, these keywords contribute a third
ranked list to hybrid search via RRF, improving recall for specific terms.

---

## Document Lifecycle

```
ingest/radar --> stored (origin=pro or radar)
                    |
              promote (radar -> pro tier)
              archive (hide from views)
              favorite (mark for quick access)
              tag (add custom tags)
              delete  (soft-delete, recoverable)
```

Archived and deleted documents are excluded from all searches by default
but are NOT erased from the database. The system is append-only at the
storage layer.

---

## Processing Pipeline

When content is ingested, it flows through:

1. **Content extraction** — full text pulled from source
2. **Filtering** — LLM or keyword-based relevance scoring
3. **Chunking** — split into overlapping segments for embedding
4. **Embedding** — vector representations stored in Qdrant
5. **Entity extraction** — people, orgs, concepts identified via LLM
6. **Keyword extraction** — topic keywords extracted via LLM

Steps 3-6 run automatically as a post-ingest hook.

---

## Services

### Starting services

The system depends on several local services. `cr doctor` checks all of
them and shows Quick Start hints for anything that's down.

| Service | Command | Port |
|---------|---------|------|
| Qdrant (vectors) | `docker run -d -p 6333:6333 qdrant/qdrant` | 6333 |
| llamacpp (embeddings) | `llama-server -m <model>.gguf --port 9990 --embedding -c 8192 -np 1 -b 2048 -ub 2048` | 9990 |
| Ollama (LLM inference) | `ollama serve` | 11434 |
| Backend + dashboard | `uv run cr server` | 8000 |
| Frontend dev server | `cd dashboard && pnpm dev` | 3000/5173 |

Qdrant runs as a server on port 6333. Start it with Docker:

```bash
docker run -d -p 6333:6333 -v $(pwd)/data/qdrant_storage:/qdrant/storage qdrant/qdrant
```

The collection `ai_craftsman_kb` is created automatically on first ingest.
Dashboard: http://localhost:6333/dashboard

### LLM Usage Tracking

All LLM calls are tracked in an `llm_usage` table with token counts per
provider/model/task. View consumption in the dashboard Usage page or via API:

```bash
curl http://localhost:8000/api/usage/daily
curl http://localhost:8000/api/usage/by-provider
curl http://localhost:8000/api/usage/by-task
```

---

## Dashboard Pages

| URL | What it shows |
|-----|--------------|
| `/` | Overview: stats, health status, recent documents |
| `/sources` | Source editor: add/edit/delete sources, trigger ingest |
| `/search` | Search interface + Radar search |
| `/entities` | Entity explorer (people, orgs, concepts) |
| `/documents` | Document manager (list, filter, archive, favorite, tag, delete) |
| `/adhoc` | Ingest a single URL, discover related sources |
| `/briefings` | Briefing builder |
| `/usage` | LLM usage dashboard (token consumption by provider/task/day) |

---

## MCP Server for Claude Desktop

The MCP server exposes your KB as tools usable from Claude Desktop.
Available tools:
- `search` — search your indexed content
- `radar` — search the open web on-demand
- `get_document` — fetch full document by ID
- `list_sources` — list configured sources
- `get_stats` — database statistics

Setup: see `setup.md` Step 10.

---

## Scheduled Ingest

### macOS launchd (recommended)

```bash
cat > ~/Library/LaunchAgents/com.aicraftsman.ingest.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.aicraftsman.ingest</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/uv</string>
    <string>--project</string>
    <string>/path/to/ai-craftsman-kb</string>
    <string>run</string>
    <string>cr</string>
    <string>ingest</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>OPENROUTER_API_KEY</key>
    <string>sk-or-...</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>8</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>/tmp/aicraftsman-ingest.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/aicraftsman-ingest.err</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.aicraftsman.ingest.plist
```

### Simple cron

```bash
crontab -e
# Run every day at 8am:
0 8 * * * source ~/.ai-craftsman-kb/.env && cd /path/to/ai-craftsman-kb && uv run cr ingest >> /tmp/cr-ingest.log 2>&1
```

---

## Maintenance

### Health check

```bash
cr doctor    # grouped by API Keys, Services, Config — available items first
```

### Cleaning up indexed data

Use `cr reset` to delete all data and start fresh:

```bash
cr reset          # interactive confirmation
cr reset -y       # skip confirmation
```

This removes `craftsman.db` (SQLite) and deletes the Qdrant collection.
The next `cr ingest` recreates everything from scratch.

### Backup

| Data | Backup how |
|------|-----------|
| `./data/craftsman.db` | `sqlite3 craftsman.db ".backup backup.db"` |
| Qdrant collection | Qdrant snapshots API or Docker volume backup |
| Config (`settings.yaml` + `sources.yaml`) | Version control or copy |

Qdrant vectors can be rebuilt by re-running embeddings on existing documents
— costly but possible if the vector store is lost.

### Updating the system

```bash
git pull origin main
uv sync                  # update Python deps
cd dashboard && pnpm install && pnpm build && cd ..
cr doctor                # verify everything still works
```

Database migrations run automatically on next startup — no manual steps needed.

---

## Troubleshooting

**`cr search` returns nothing after ingest:**
- Embeddings may not be built yet. Check `cr stats` → `vectors` count.
- If 0 vectors: the embedding pipeline may have failed. Run
  `cr ingest --source hn -v` to see verbose output.

**Qdrant connection refused:**
- Qdrant server must be running on port 6333.
- Start with: `docker run -d -p 6333:6333 qdrant/qdrant`
- Dashboard: http://localhost:6333/dashboard

**All endpoints exhausted (LLM gateway):**
- Every endpoint in the pool has hit its daily limit.
- Wait for limits to reset (midnight UTC), add more endpoints to the pool,
  or add an Ollama endpoint (no rate limits) as a fallback.

**Reddit 403 errors:**
- Check that `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` are set correctly.
- Verify the app type in Reddit prefs is "script".

**YouTube quota exceeded:**
- The YouTube Data API v3 has a 10,000 unit/day free quota.
- Each channel search ~ 100 units. Reduce channel count or add a second project.

**llamacpp embedding server not responding:**
- Check: `curl http://localhost:9990/health`
- Start: `llama-server -m v5-small-retrieval-Q8_0.gguf --port 9990 --embedding -c 8192 -np 1 -b 2048 -ub 2048`
- `cr doctor` shows this under Services and includes a Quick Start hint.

**direnv not loading env vars:**
- Ensure your shell hook is set up: `eval "$(direnv hook zsh)"` in `~/.zshrc`
- Run `direnv allow` after editing `.envrc` or `.env`
- The `.env` file must use `export KEY=value` syntax (not bare `KEY=value`)
