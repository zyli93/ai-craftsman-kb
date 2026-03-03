# Credentials Reference — AI Craftsman KB

Which API keys you need, why, and where to get them.

---

## Required (system won't work without these)

### `OPENAI_API_KEY`
- **Why**: All vector embeddings use `text-embedding-3-small`. This is the
  core search capability — without it, semantic and hybrid search won't work.
- **Where**: https://platform.openai.com/api-keys
- **Cost estimate**: ~$0.02 per 1M tokens. With typical daily ingestion
  (~200 docs), expect < $1/month.
- **Model used**: `text-embedding-3-small` (1536 dims)

### `OPENROUTER_API_KEY`
- **Why**: Used for LLM-based content filtering and entity extraction
  (`meta-llama/llama-3.1-8b-instruct` by default — cheap and fast).
- **Where**: https://openrouter.ai/keys
- **Cost estimate**: Llama 3.1 8B via OpenRouter is ~$0.05/1M tokens.
  Very cheap for filtering/extraction.
- **Can substitute**: Change `settings.yaml` to use `ollama` with a local
  model instead (set `provider: ollama`, `model: llama3.1:8b`).
  See `operator-guide.md` for Ollama setup.

---

## Strongly Recommended (unlock most features)

### `ANTHROPIC_API_KEY`
- **Why**: Used for the briefing generator (`claude-sonnet-4-20250514`).
  Without it, `cr briefing` won't work.
- **Where**: https://console.anthropic.com/settings/keys
- **Cost estimate**: Sonnet 4 is ~$3/1M input, $15/1M output.
  A briefing = ~5-15k tokens. Expect $0.05–0.20 per briefing.

### `YOUTUBE_API_KEY`
- **Why**: Fetches video metadata and enables YouTube channel ingestion.
  Without it, the YouTube ingestor is disabled.
- **Where**: https://console.cloud.google.com/apis/credentials
  → Create Project → Enable "YouTube Data API v3" → Create API Key
- **Cost**: Free tier is 10,000 units/day, which is plenty for personal use.
  Each video metadata fetch ≈ 1-3 units.

### `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET`
- **Why**: Authenticates to Reddit's API for subreddit ingestion.
- **Where**: https://www.reddit.com/prefs/apps → "Create App"
  → Type: **script** → redirect URI: `http://localhost:8080`
- **Cost**: Free. Personal use scripts get 100 requests/min.
- **Note**: Reddit's API Terms changed in 2023. Personal/non-commercial use
  at this scale is fine.

---

## Optional (alternative providers)

### `FIREWORKS_API_KEY`
- **Why**: Alternative LLM provider (faster inference for some models).
  Not used by default — change `provider: fireworks` in settings to enable.
- **Where**: https://fireworks.ai/account/api-keys

---

## Not Needed (no key required)

| Source | Why no key |
|--------|-----------|
| HN (Algolia) | Public API, no auth |
| ArXiv | Public API, no auth |
| DEV.to | Public API, no auth |
| RSS feeds | Just HTTP |
| Substack | RSS-based scraping |

---

## Where These Get Set

The system reads credentials from environment variables. The `settings.yaml`
uses `${VAR_NAME}` interpolation — it does NOT read `.env` files itself.

You need to export these variables before running any `cr` command.
See `vault.md` for how to set this up cleanly.

---

## Summary Table

| Variable | Required | Source | Notes |
|----------|----------|--------|-------|
| `OPENAI_API_KEY` | ✅ Yes | platform.openai.com | Embeddings |
| `OPENROUTER_API_KEY` | ✅ Yes | openrouter.ai | LLM filter/extract |
| `ANTHROPIC_API_KEY` | Recommended | console.anthropic.com | Briefings |
| `YOUTUBE_API_KEY` | Recommended | Google Cloud Console | YouTube ingestor |
| `REDDIT_CLIENT_ID` | Recommended | reddit.com/prefs/apps | Reddit ingestor |
| `REDDIT_CLIENT_SECRET` | Recommended | reddit.com/prefs/apps | Reddit ingestor |
| `FIREWORKS_API_KEY` | Optional | fireworks.ai | Alt LLM provider |
