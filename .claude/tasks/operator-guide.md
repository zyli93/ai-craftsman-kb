# Operator Guide — AI Craftsman KB

Everything you need to know as the sole user of this system.

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

## Daily Driver Commands

```bash
# Pull new content from all configured sources
cr ingest

# Search everything you've indexed
cr search "transformer architecture"
cr search "rust async" --source hn reddit --since 2024-01-01

# Search the open web on-demand
cr radar "agentic AI 2025" --limit 20

# Promote a radar result to your permanent index
cr promote <document-id>

# Archive something you don't want showing up
cr archive <document-id>

# System health check
cr doctor

# Stats overview
cr stats

# Launch the web dashboard
cr server
```

---

## Config Files

The system looks for config in this order:
1. `~/.ai-craftsman-kb/settings.yaml` (user config — override defaults here)
2. `~/.ai-craftsman-kb/sources.yaml` (your sources list)
3. Falls back to `config/settings.yaml` and `config/sources.yaml` in the repo

**You should copy the repo defaults to your home dir and edit there:**
```bash
mkdir -p ~/.ai-craftsman-kb
cp config/settings.yaml ~/.ai-craftsman-kb/settings.yaml
cp config/sources.yaml ~/.ai-craftsman-kb/sources.yaml
```

This way your personal config never gets overwritten by git pulls.

### Customizing Sources (`sources.yaml`)

Add any Substack newsletter:
```yaml
substack:
  - slug: thezvi       # from substack.com/p/@thezvi
    name: "Zvi Mowshowitz"
```

Add a YouTube channel:
```yaml
youtube_channels:
  - handle: "@lexfridman"
    name: "Lex Fridman"
```

Add an RSS feed:
```yaml
rss:
  - url: https://simonwillison.net/atom/everything/
    name: "Simon Willison"
```

Add a subreddit:
```yaml
subreddits:
  - name: singularity
    sort: hot
    limit: 20
```

### Switching LLM Provider to Ollama (free, local)

Edit `~/.ai-craftsman-kb/settings.yaml`:
```yaml
llm:
  filtering:
    provider: ollama
    model: llama3.1:8b
  entity_extraction:
    provider: ollama
    model: llama3.1:8b
```

Then run Ollama: `ollama serve` and `ollama pull llama3.1:8b`.
Eliminates the OpenRouter cost entirely. Slower but zero API cost.

---

## Search Modes

| Mode | How it works | When to use |
|------|-------------|-------------|
| `hybrid` (default) | FTS + vector, merged via RRF | Most queries |
| `semantic` | Vector similarity only | Conceptual/fuzzy queries |
| `keyword` | SQLite FTS only | Exact term lookup, fast |

```bash
cr search "agent orchestration" --mode semantic
cr search "GPT-4o" --mode keyword
```

---

## Document Lifecycle

```
ingest/radar → stored (origin=pro or radar)
                    ↓
              promote (radar → pro tier)
              archive (hide from views)
              delete  (soft-delete, recoverable)
```

Archived and deleted documents are excluded from all searches by default
but are NOT erased from the database. The system is append-only at the
storage layer.

---

## Running Ingest on a Schedule

**macOS launchd** (recommended — runs even without a terminal open):

```bash
# Create ~/Library/LaunchAgents/com.aicraftsman.ingest.plist
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
    <key>OPENAI_API_KEY</key>
    <string>sk-proj-...</string>
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

**Simple cron** (if you prefer):
```bash
crontab -e
# Run every day at 8am:
0 8 * * * source ~/.ai-craftsman-kb/.env && cd /path/to/ai-craftsman-kb && uv run cr ingest >> /tmp/cr-ingest.log 2>&1
```

---

## MCP Server for Claude Desktop

The MCP server exposes your KB as tools usable from Claude Desktop.
Available tools (as of v0.1.0):
- `search` — search your indexed content
- `radar` — search the open web on-demand
- `get_document` — fetch full document by ID
- `list_sources` — list configured sources
- `get_stats` — database statistics

Setup: see `setup.md` step 9.

---

## Dashboard Pages

| URL | What it shows |
|-----|--------------|
| `/` | Overview: stats, health status, recent documents |
| `/sources` | Source editor: add/edit/delete sources, trigger ingest |
| `/search` | Search interface + Radar search |
| `/entities` | Entity explorer (people, orgs, concepts) |
| `/documents` | Document manager (list, filter, archive, delete) |
| `/adhoc` | Ingest a single URL, discover related sources |
| `/briefings` | Briefing builder (requires Anthropic key) |

---

## Common Issues

**`cr search` returns nothing after ingest:**
- Embeddings may not be built yet. Check: `cr stats` → `vectors` count
- If 0 vectors: the embedding pipeline may have failed silently. Run
  `cr ingest --source hn -v` to see verbose output.

**Qdrant connection refused:**
- Make sure Qdrant is running: `docker ps | grep qdrant` or `ps aux | grep qdrant`
- Restart: `docker restart qdrant`

**Reddit 403 errors:**
- Check that `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` are set correctly
- Verify the app type in Reddit prefs is "script"

**YouTube quota exceeded:**
- The YouTube Data API v3 has a 10,000 unit/day free quota
- Each channel search ≈ 100 units. Reduce channel count or add a second project

**`cr briefing` not implemented:**
- The briefing generator stub is in `cli.py:539` — it prints a placeholder.
  The engine (`task_41`) was merged but the CLI hookup is incomplete.
  Use the dashboard briefings page instead, or call the API directly:
  `POST http://localhost:8000/api/briefings`

---

## Backup Strategy

| Data | Backup how |
|------|-----------|
| `~/.ai-craftsman-kb/data/craftsman.db` | Copy or `sqlite3 craftsman.db ".backup backup.db"` |
| Qdrant storage | Copy `qdrant_storage/` directory |
| `~/.ai-craftsman-kb/settings.yaml` + `sources.yaml` | Version control or copy |

Qdrant vectors can be rebuilt from scratch by re-running embeddings on
existing documents — costly but possible if the vector store is lost.

---

## Updating the System

```bash
git pull origin main
uv sync          # update Python deps
cd dashboard && pnpm install && pnpm build && cd ..
uv run cr doctor # verify everything still works
```

Database migrations run automatically on next startup — no manual steps needed.
