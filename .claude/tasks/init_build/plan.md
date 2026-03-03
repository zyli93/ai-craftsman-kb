# AI Craftsman KB — Execution Plan v2

## Project Overview

A local-first content aggregation, indexing, and semantic search system with a web dashboard and MCP server. Two operating modes:

- **Pro Tier**: Subscription-based ingestion of curated sources (always indexed, always searchable)
- **Radar Tier**: On-demand topic search as a capability — callable by you, your agents, or any MCP client

Runs locally on your laptop. Python backend + TypeScript/React dashboard. SQLite + Qdrant local. No cloud dependency.

---

## Feedback Responses & Key Decisions

### 1. Storage Estimation

Here's a realistic breakdown for 6 months of daily use:

**Raw content storage (SQLite text fields):**

| Source | Items/day | Avg size/item | Monthly | 6 months |
|--------|-----------|---------------|---------|----------|
| HN articles (full text) | 30 | 8 KB | 7.2 MB | 43 MB |
| Substack posts | 10 | 15 KB | 4.5 MB | 27 MB |
| YouTube transcripts | 10 | 25 KB (10-min avg) | 7.5 MB | 45 MB |
| Reddit posts + comments | 50 | 5 KB | 7.5 MB | 45 MB |
| ArXiv abstracts | 20 | 3 KB | 1.8 MB | 11 MB |
| RSS articles | 20 | 8 KB | 4.8 MB | 29 MB |
| DEV.to articles | 10 | 10 KB | 3.0 MB | 18 MB |
| **Subtotal raw text** | | | | **~218 MB** |

**Vector embeddings (Qdrant):**

| Model | Dims | Bytes/vector | Chunks (est 50k) | Total |
|-------|------|-------------|-------------------|-------|
| OpenAI text-embedding-3-small | 1536 | 6,144 | 50,000 | ~307 MB |
| nomic-embed-text (local) | 768 | 3,072 | 50,000 | ~154 MB |

**Entity index (SQLite FTS + entity tables):**
- ~50 MB for FTS index on 50k documents
- ~20 MB for entity extraction results

**Total estimate for 6 months:**
- Conservative: **~600 MB** (with OpenAI embeddings)
- Heavy use: **~1.2 GB** (more sources, longer transcripts, more radar searches)
- 1 year aggressive use: **~2-3 GB**

**Verdict**: Easily fits on any laptop. Even a 256GB MacBook Air has plenty of room. You'd need years of heavy use to even notice it. **Local storage is the right call.**

If you ever outgrow local, the migration path is: SQLite → Turso (SQLite-compatible, dirt cheap at $8/mo for 9GB) + Qdrant local → Qdrant Cloud free tier (1GB vectors free). Both are drop-in replacements. But you won't need this for a long time.

### 2. Radar as a Capability, Not a Log

You're right — radar shouldn't be a search history table. It should be a **search function** that any caller can invoke. Removing `radar_searches` and `radar_search_results` tables. Radar is just an API endpoint / MCP tool:

```
radar.search(query, sources?, since?, max_results?) → Document[]
```

Results still get stored in the `documents` table (with `tier='radar'`), so they're searchable later. But there's no separate tracking of "searches performed." The caller (you, an agent, MCP client) decides what to do with results.

### 3. Entity Extraction + Keyword Search

Adding a proper entity layer. The pipeline:

1. Content comes in → store raw text
2. Run entity extraction via a cheap/free LLM (Llama 3.1 8B via Ollama, or Qwen 2.5 via OpenRouter at ~$0.001/article)
3. Store extracted entities: people, companies, technologies, events, books, papers
4. SQLite FTS5 gives you fuzzy matching for free (with `porter` tokenizer)
5. Entity co-occurrence lets you build a knowledge graph later if you want

Schema:
```sql
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,    -- 'person', 'company', 'technology', 'event', 'book', 'paper'
    normalized_name TEXT,         -- lowercase, deduped version
    first_seen_at TIMESTAMP,
    mention_count INTEGER DEFAULT 1
);

CREATE TABLE document_entities (
    document_id TEXT REFERENCES documents(id),
    entity_id TEXT REFERENCES entities(id),
    context TEXT,                 -- surrounding sentence for quick reference
    PRIMARY KEY (document_id, entity_id)
);

-- FTS on entities for fuzzy search
CREATE VIRTUAL TABLE entities_fts USING fts5(
    name, normalized_name, entity_type,
    tokenize='porter unicode61'
);
```

This gives you queries like:
```
cr search --entity "Karpathy"              # everything mentioning Karpathy
cr search --entity-type technology "GRPO"   # tech mentions only
cr entities --type person --top 20          # most mentioned people this month
```

### 4. Centralized Config

All human-editable config lives in one place: `~/.ai-craftsman-kb/config/`

```
~/.ai-craftsman-kb/
├── config/
│   ├── sources.yaml        # all source subscriptions
│   ├── settings.yaml       # global settings (models, paths, API keys)
│   ├── filters.yaml        # LLM filtering rules per source
│   └── prompts/            # customizable prompt templates
│       ├── briefing.md
│       ├── entity_extraction.md
│       └── content_filter.md
├── data/
│   ├── ai_craftsman_kb.db    # SQLite
│   └── vectors/            # Qdrant local storage
└── logs/
```

The dashboard (Phase 5) will provide a web editor for `sources.yaml` and `filters.yaml`, so you can add/remove sources visually. But they're always plain YAML files you can edit directly too.

### 5. Podcasts = YouTube

Agreed. Dropping the separate podcast ingestion pipeline. Most tech podcasts you'd care about are on YouTube. YouTube transcripts cover this. If you ever find a podcast-only source, you can add its YouTube mirror to your channel list.

### 6. Local-First, Migrate Later

Decision: **Local SQLite + local Qdrant. Period.**

Migration path if you ever need it:
- SQLite → Turso ($8/mo, SQLite wire protocol, drop-in) or Supabase Postgres ($25/mo — you're right, it's expensive for what you'd use)
- Qdrant local → Qdrant Cloud (1GB free tier, then $9/mo for 4GB)
- Cheapest cloud path: Turso + Qdrant Cloud free = **$8/mo total**

But realistically you won't need this. Your use case is single-user, read-heavy, write-infrequent. SQLite is literally perfect for this.

The "crawl once, index locally" model is exactly right. You're not building a real-time system. You pull content when you want it, it sits in your local DB forever, and you search it whenever.

### 7. Language Choice: Python + TypeScript Hybrid

**Python for the backend/CLI/ingestion/search.** No contest — every library you need (feedparser, youtube-transcript-api, arxiv, readability, sentence-transformers, whisper) is Python-first. The async story is good with httpx + asyncio.

**TypeScript/React for the dashboard.** You want a proper web UI with a source editor, stats dashboard, and search interface. React + Vite is fast to build and you know it.

**The bridge: MCP server in Python.** The MCP SDK has a Python implementation. Your Python backend exposes MCP tools, the dashboard talks to the same backend via REST API.

Architecture:
```
┌──────────────────────┐     ┌──────────────────────┐
│  TypeScript/React     │     │  Claude / Agents      │
│  Dashboard (Vite)     │     │  (MCP clients)        │
└──────────┬───────────┘     └──────────┬───────────┘
           │ REST API                    │ MCP Protocol
           │                             │
┌──────────▼─────────────────────────────▼───────────┐
│              Python Backend (FastAPI)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Ingestors│ │ Search   │ │ MCP Server       │   │
│  │ (all src)│ │ Engine   │ │ (tools endpoint) │   │
│  └────┬─────┘ └────┬─────┘ └────────┬─────────┘   │
│       │             │                │              │
│  ┌────▼─────────────▼────────────────▼──────────┐  │
│  │     SQLite + Qdrant Local + Entity Index      │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 8. LLM-Based Content Filtering

Adding a filtering layer between ingestion and storage. Configurable per source in `filters.yaml`:

```yaml
filters:
  hackernews:
    enabled: true
    strategy: llm          # 'llm', 'keyword', or 'hybrid'
    prompt: |
      Score this article 1-10 for relevance to: AI/ML engineering,
      software architecture, developer tools, LLM applications.
      Return just the number.
    min_score: 5
    fallback: keyword       # if LLM fails, use keyword filter
    keywords_include: ["AI", "LLM", "ML", "transformer", "agent", "inference"]
    keywords_exclude: ["crypto", "NFT", "blockchain"]

  reddit:
    enabled: true
    strategy: keyword       # cheaper, good enough for pre-filtered subreddits
    min_upvotes: 10

  substack:
    enabled: false          # you curated these, no filter needed

  youtube:
    enabled: false          # channel-based, already curated
```

The LLM filter runs on the cheap model (same one used for entity extraction). At ~$0.001 per article, filtering 30 HN articles/day costs basically nothing.

### 9. Abstract LLM Provider

```python
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str = "", **kwargs) -> str: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class OpenAIProvider(LLMProvider): ...
class OpenRouterProvider(LLMProvider): ...   # access to Llama, Qwen, Mistral, etc.
class FireworksProvider(LLMProvider): ...
class OllamaProvider(LLMProvider): ...       # fully local, free
class AnthropicProvider(LLMProvider): ...
```

Config in `settings.yaml`:
```yaml
llm:
  # Which provider to use for each task
  embedding:
    provider: openai
    model: text-embedding-3-small
  filtering:
    provider: openrouter
    model: meta-llama/llama-3.1-8b-instruct  # cheap
  entity_extraction:
    provider: openrouter
    model: meta-llama/llama-3.1-8b-instruct
  briefing:
    provider: anthropic
    model: claude-sonnet-4-20250514          # quality matters here
  
  # Provider credentials
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
    openrouter:
      api_key: ${OPENROUTER_API_KEY}
    fireworks:
      api_key: ${FIREWORKS_API_KEY}
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
    ollama:
      base_url: http://localhost:11434
```

### 10 + 11 + 14. Dashboard + MCP Server + Source Management

This is now a major feature. Promoting it from a Phase 6 afterthought to a core Phase 5. Full design below in the architecture section.

### 12. Web App vs Electron

**Web app. No question.**

Reasons:
- You're already running a Python backend server — the dashboard is just a frontend on `localhost:3000`
- Electron adds 200MB+ of overhead for literally no benefit when you have a browser
- Web app is accessible from your phone on the same WiFi if you want
- MCP server runs alongside the web server — one process, two interfaces
- No build/packaging headache
- If you ever want to move to a VPS, the web app just works remotely

The UX: you run `cr server` and it starts both the FastAPI backend (port 8000) and the Vite dev server (port 3000, or serves built assets in production). One command, open browser, done.

### 13. Source Discovery

Adding a "discover" capability that suggests new sources based on your existing content:

- Extract outbound links from articles you've ingested (authors link to other authors)
- Extract YouTube "recommended" channels from video descriptions
- Extract cited papers from ArXiv (citation graph traversal)
- Extract mentioned Substacks, blogs, tools
- LLM-powered: "Given these 20 articles the user reads, suggest 5 new sources they'd like"

This becomes a dashboard feature: "Suggested Sources" panel that shows discovered sources with a one-click "Add to Pro" button.

---

## Updated Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard (React + Vite)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Source    │ │ Search   │ │ Stats &  │ │ Source        │  │
│  │ Editor   │ │ & Radar  │ │ Health   │ │ Discovery     │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Content  │ │ Entity   │ │ Briefing │ │ Adhoc URL     │  │
│  │ Browser  │ │ Explorer │ │ Builder  │ │ Ingest        │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API (localhost:8000)
┌──────────────────────────▼──────────────────────────────────┐
│                 Python Backend (FastAPI)                      │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    REST API Layer                     │    │
│  │  /api/sources     - CRUD sources                     │    │
│  │  /api/ingest      - trigger ingestion                │    │
│  │  /api/search      - hybrid search                    │    │
│  │  /api/radar       - on-demand topic search           │    │
│  │  /api/documents   - browse, tag, delete, promote     │    │
│  │  /api/entities    - entity search and browse         │    │
│  │  /api/briefing    - generate content brief           │    │
│  │  /api/stats       - dashboard stats                  │    │
│  │  /api/discover    - source suggestions               │    │
│  │  /api/adhoc       - ingest arbitrary URL             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   MCP Server Layer                    │    │
│  │  Tools exposed:                                       │    │
│  │  - search(query, filters)                             │    │
│  │  - radar(query, sources, since)                       │    │
│  │  - ingest(source?)                                    │    │
│  │  - briefing(topic)                                    │    │
│  │  - get_entities(type?, query?)                        │    │
│  │  - add_source(type, identifier)                       │    │
│  │  - ingest_url(url)                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐     │
│  │  Ingestors   │ │ Processing   │ │  LLM Provider    │     │
│  │  (per source)│ │ Pipeline     │ │  Abstraction     │     │
│  │             │ │              │ │                  │     │
│  │  - HN       │ │ - Extract    │ │ - OpenAI         │     │
│  │  - Substack │ │ - Clean      │ │ - OpenRouter     │     │
│  │  - YouTube  │ │ - Filter     │ │ - Fireworks      │     │
│  │  - Reddit   │ │ - Chunk      │ │ - Ollama         │     │
│  │  - RSS      │ │ - Embed      │ │ - Anthropic      │     │
│  │  - ArXiv    │ │ - Entities   │ │                  │     │
│  │  - DEV.to   │ │ - Store      │ │                  │     │
│  │  - Adhoc URL│ │              │ │                  │     │
│  └──────┬──────┘ └──────┬───────┘ └────────┬─────────┘     │
│         │               │                   │               │
│  ┌──────▼───────────────▼───────────────────▼────────────┐  │
│  │                   Data Layer                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │  │
│  │  │   SQLite      │  │ Qdrant Local │  │ Config      │ │  │
│  │  │ - documents   │  │ - vectors    │  │ (YAML)      │ │  │
│  │  │ - entities    │  │ - payloads   │  │ - sources   │ │  │
│  │  │ - FTS index   │  │              │  │ - settings  │ │  │
│  │  │ - source meta │  │              │  │ - filters   │ │  │
│  │  └──────────────┘  └──────────────┘  └─────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Updated Data Model

### SQLite Schema

```sql
-- ============================================================
-- SOURCES
-- ============================================================

-- Managed via dashboard or YAML. Dashboard writes to YAML too.
CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,       -- 'substack', 'youtube_channel', 'subreddit',
                                     -- 'rss', 'hackernews', 'arxiv', 'devto'
    identifier TEXT NOT NULL,        -- slug, channel ID, subreddit name, feed URL
    display_name TEXT,
    tier TEXT NOT NULL DEFAULT 'pro', -- 'pro' only (radar is a capability, not a source)
    enabled BOOLEAN DEFAULT TRUE,
    last_fetched_at TIMESTAMP,
    fetch_error TEXT,                -- last error message if any
    config JSON,                     -- source-specific settings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- DOCUMENTS
-- ============================================================

CREATE TABLE documents (
    id TEXT PRIMARY KEY,             -- deterministic hash of url
    source_id TEXT REFERENCES sources(id) ON DELETE SET NULL,
    origin TEXT NOT NULL,            -- 'pro', 'radar', or 'adhoc'
    source_type TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    author TEXT,
    published_at TIMESTAMP,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_type TEXT,               -- 'article', 'transcript', 'comment_thread', 'paper'
    raw_content TEXT,                -- full text / transcript
    word_count INTEGER,
    metadata JSON,                   -- engagement metrics, tags, extra data
    -- Processing state
    is_embedded BOOLEAN DEFAULT FALSE,
    is_entities_extracted BOOLEAN DEFAULT FALSE,
    filter_score REAL,               -- LLM relevance score (null if unfiltered)
    filter_passed BOOLEAN,
    -- User actions
    is_favorited BOOLEAN DEFAULT FALSE,
    is_archived BOOLEAN DEFAULT FALSE,
    user_tags JSON DEFAULT '[]',
    user_notes TEXT,
    promoted_at TIMESTAMP,           -- radar/adhoc → promoted
    -- Cleanup
    deleted_at TIMESTAMP             -- soft delete
);

CREATE INDEX idx_documents_source ON documents(source_type, origin);
CREATE INDEX idx_documents_date ON documents(published_at DESC);
CREATE INDEX idx_documents_processing ON documents(is_embedded, is_entities_extracted);

-- Full-text search with porter stemming
CREATE VIRTUAL TABLE documents_fts USING fts5(
    title, raw_content, author,
    content='documents',
    tokenize='porter unicode61'
);

-- ============================================================
-- ENTITIES (extracted by LLM)
-- ============================================================

CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,        -- 'person', 'company', 'technology',
                                      -- 'event', 'book', 'paper', 'product'
    normalized_name TEXT NOT NULL,     -- lowercase, trimmed for dedup
    description TEXT,                  -- brief LLM-generated description
    first_seen_at TIMESTAMP,
    mention_count INTEGER DEFAULT 1,
    metadata JSON,                     -- links, aliases, etc.
    UNIQUE(normalized_name, entity_type)
);

CREATE TABLE document_entities (
    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
    entity_id TEXT REFERENCES entities(id) ON DELETE CASCADE,
    context TEXT,                      -- surrounding sentence
    relevance TEXT,                    -- 'primary', 'mentioned', 'tangential'
    PRIMARY KEY (document_id, entity_id)
);

-- FTS on entities for fuzzy keyword search
CREATE VIRTUAL TABLE entities_fts USING fts5(
    name, normalized_name, description,
    content='entities',
    tokenize='porter unicode61'
);

-- ============================================================
-- SOURCE DISCOVERY
-- ============================================================

CREATE TABLE discovered_sources (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    identifier TEXT NOT NULL,
    display_name TEXT,
    discovered_from_document_id TEXT REFERENCES documents(id),
    discovery_method TEXT,            -- 'outbound_link', 'citation', 'llm_suggestion'
    confidence REAL,
    status TEXT DEFAULT 'suggested',  -- 'suggested', 'accepted', 'dismissed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_type, identifier)
);
```

### Qdrant Collection

```
Collection: "ai_craftsman_kb"
Vector size: 1536 (OpenAI) or 768 (local)
Distance: Cosine

Payload schema:
{
    document_id: string,
    source_type: string,
    origin: string,          // pro, radar, adhoc
    title: string,
    author: string,
    published_at: string,    // ISO datetime for filtering
    chunk_index: int,        // 0 for single-chunk docs
    total_chunks: int
}
```

### Config Directory: `~/.ai-craftsman-kb/config/`

#### sources.yaml
```yaml
# ============================================================
# PRO TIER SOURCES
# All sources listed here are ingested on every `cr ingest pro`
# Edit this file directly or use the dashboard source editor
# ============================================================

substack:
  - slug: karpathy
    name: "Andrej Karpathy"
  - slug: simonwillison
    name: "Simon Willison"
  - slug: oneusefulthing
    name: "One Useful Thing (Ethan Mollick)"

youtube_channels:
  - handle: "@AndrejKarpathy"
    name: "Andrej Karpathy"
  - handle: "@YannicKilcher"
    name: "Yannic Kilcher"
  - handle: "@3blue1brown"
    name: "3Blue1Brown"

subreddits:
  - name: LocalLLaMA
    sort: hot
    limit: 25
  - name: MachineLearning
    sort: hot
    limit: 25

rss:
  - url: https://lobste.rs/rss
    name: "Lobste.rs"
  - url: https://deepmind.google/blog/rss.xml
    name: "DeepMind Blog"
  - url: https://openai.com/blog/rss.xml
    name: "OpenAI Blog"

hackernews:
  mode: top
  limit: 30
  # Filtering handled in filters.yaml

arxiv:
  queries:
    - "cat:cs.CL AND abs:large language model"
    - "cat:cs.AI AND abs:reinforcement learning"
  max_results: 20

devto:
  tags:
    - ai
    - machinelearning
    - llm
  limit: 20
```

#### settings.yaml
```yaml
# ============================================================
# GLOBAL SETTINGS
# ============================================================

# Data storage
data_dir: ~/.ai-craftsman-kb/data

# Embedding
embedding:
  provider: openai               # or 'local'
  model: text-embedding-3-small
  chunk_size: 2000               # tokens
  chunk_overlap: 200

# LLM routing — assign providers per task
llm:
  filtering:
    provider: openrouter
    model: meta-llama/llama-3.1-8b-instruct
  entity_extraction:
    provider: openrouter
    model: meta-llama/llama-3.1-8b-instruct
  briefing:
    provider: anthropic
    model: claude-sonnet-4-20250514
  source_discovery:
    provider: openrouter
    model: meta-llama/llama-3.1-8b-instruct

# Provider credentials (or use env vars)
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
  fireworks:
    api_key: ${FIREWORKS_API_KEY}
  ollama:
    base_url: http://localhost:11434

# YouTube
youtube:
  api_key: ${YOUTUBE_API_KEY}
  transcript_langs: ["en"]

# Reddit
reddit:
  client_id: ${REDDIT_CLIENT_ID}
  client_secret: ${REDDIT_CLIENT_SECRET}

# Server
server:
  backend_port: 8000
  dashboard_port: 3000

# Search defaults
search:
  default_limit: 20
  hybrid_weight_semantic: 0.6    # vs 0.4 for keyword
  hybrid_weight_keyword: 0.4
```

#### filters.yaml
```yaml
# ============================================================
# CONTENT FILTERING RULES
# Controls what gets indexed vs discarded during ingestion
# ============================================================

hackernews:
  enabled: true
  strategy: llm                  # 'llm', 'keyword', or 'hybrid'
  llm_prompt: |
    Rate this article 1-10 for relevance to these topics:
    - AI/ML engineering and research
    - Software architecture and systems design
    - Developer tools and productivity
    - LLM applications and agents
    - Tech industry strategy and business

    Article title: {title}
    Article excerpt: {excerpt}

    Respond with ONLY a number 1-10.
  min_score: 5
  # Keyword fallback if LLM fails or for pre-filtering
  keywords_include: []           # empty = no keyword pre-filter
  keywords_exclude:
    - crypto
    - NFT
    - blockchain
    - hiring
    - "Show HN: My weekend project"

reddit:
  enabled: true
  strategy: hybrid
  min_upvotes: 10
  keywords_exclude:
    - meme
    - shitpost

substack:
  enabled: false                 # curated sources, trust them

youtube:
  enabled: false                 # channel-based, already curated

arxiv:
  enabled: true
  strategy: keyword
  min_citations: 0               # new papers won't have citations
  keywords_include:
    - transformer
    - language model
    - reinforcement learning
    - agent

rss:
  enabled: false
devto:
  enabled: true
  strategy: keyword
  min_reactions: 5
```

---

## Dashboard Design

### Pages

#### 1. Overview / Home
```
┌─────────────────────────────────────────────────────────┐
│  AI Craftsman KB Dashboard                    [Ingest Now] │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ 2,847        │ │ 14,203       │ │ 1,847        │    │
│  │ Documents    │ │ Entities     │ │ Embedded     │    │
│  │ ↑ 142 today  │ │ ↑ 89 today   │ │ 100%         │    │
│  └──────────────┘ └──────────────┘ └──────────────┘    │
│                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ 623 MB       │ │ 307 MB       │ │ 12           │    │
│  │ SQLite Size  │ │ Vector Store │ │ Active Srcs  │    │
│  └──────────────┘ └──────────────┘ └──────────────┘    │
│                                                          │
│  Source Health                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │ ● Substack (3)      Last pull: 2h ago    ✓ OK    │  │
│  │ ● YouTube (3)       Last pull: 2h ago    ✓ OK    │  │
│  │ ● Reddit (2)        Last pull: 2h ago    ✓ OK    │  │
│  │ ● Hacker News       Last pull: 2h ago    ✓ OK    │  │
│  │ ● ArXiv             Last pull: 1d ago    ⚠ Stale │  │
│  │ ● RSS (3)           Last pull: 2h ago    ✓ OK    │  │
│  │ ● DEV.to            Last pull: 3d ago    ✗ Error │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  Recent Documents                          [View All →] │
│  ┌───────────────────────────────────────────────────┐  │
│  │ ★ "Scaling LLM Inference..."  │ Substack │ 2h ago│  │
│  │   "New GRPO Paper Drops..."   │ ArXiv    │ 5h ago│  │
│  │   "Show HN: LocalAI v2..."    │ HN       │ 6h ago│  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 2. Source Editor
```
┌─────────────────────────────────────────────────────────┐
│  Sources                        [+ Add Source] [Import] │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Filter: [All Types ▾]  [Enabled Only ☑]  [Search...]  │
│                                                          │
│  Substack                                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ ● Andrej Karpathy    │ karpathy    │ ✓ │ [Edit] │  │
│  │ ● Simon Willison     │ simonw...   │ ✓ │ [Edit] │  │
│  │ ● One Useful Thing   │ oneuse...   │ ✓ │ [Edit] │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  YouTube Channels                                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │ ● Andrej Karpathy    │ @Andrej..   │ ✓ │ [Edit] │  │
│  │ ● Yannic Kilcher     │ @Yanni...   │ ✓ │ [Edit] │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  + Add Source Dialog:                                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Type: [Substack ▾]                                │  │
│  │ Identifier: [________________]                    │  │
│  │ Display Name: [________________]                  │  │
│  │                              [Cancel] [Add]       │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  Discovered Sources (5 suggestions)      [Review All →] │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 💡 @laborjack (YouTube) - found in 3 articles    │  │
│  │    [Add to Pro] [Dismiss]                         │  │
│  │ 💡 interconnects.substack.com - cited 5 times     │  │
│  │    [Add to Pro] [Dismiss]                         │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 3. Search & Radar
```
┌─────────────────────────────────────────────────────────┐
│  Search                                                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 🔍 GRPO reinforcement learning LLM               │  │
│  └───────────────────────────────────────────────────┘  │
│  Mode: (•) Hybrid  ( ) Semantic  ( ) Keyword             │
│  Sources: [All ▾]  Since: [Any time ▾]  [Search]        │
│                                                          │
│  ┌─ Tabs ──────────────────────────────────────────┐    │
│  │ [Pro Results (23)] [Radar Search] [Adhoc URL]   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  Results:                                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 📄 "DeepSeek's GRPO: A New Approach to..."       │  │
│  │    ArXiv · 3 days ago · Score: 0.94               │  │
│  │    ...proposes Group Relative Policy Optimization  │  │
│  │    as an alternative to PPO for language model...  │  │
│  │    [★ Favorite] [🏷 Tag] [📋 Copy] [🔗 Open]     │  │
│  │                                                    │  │
│  │ 🎥 "GRPO Explained - Why DeepSeek Changed..."     │  │
│  │    YouTube · @YannicKilcher · 1 week ago · 0.91   │  │
│  │    ...transcript: at its core GRPO removes the    │  │
│  │    need for a separate value model by using...     │  │
│  │    [★ Favorite] [🏷 Tag] [📋 Copy] [🔗 Open]     │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  Radar Tab:                                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Search the open web for: [GRPO reinforcement   ]  │  │
│  │ Sources: ☑ YouTube ☑ Reddit ☑ HN ☑ ArXiv ☐ DEV  │  │
│  │ Since: [Last 30 days ▾]   [🔍 Radar Search]      │  │
│  │                                                    │  │
│  │ Searching... YouTube ✓  Reddit ✓  HN ⏳  ArXiv ⏳ │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  Adhoc URL Tab:                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Paste any URL to ingest:                          │  │
│  │ [https://youtube.com/watch?v=...              ]   │  │
│  │ Detected type: YouTube Video                      │  │
│  │ [Ingest & Index]                                  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 4. Entity Explorer
```
┌─────────────────────────────────────────────────────────┐
│  Entities                                                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Filter: [All Types ▾]  Sort: [Most Mentioned ▾]        │
│  Search: [________________]                              │
│                                                          │
│  People (847)                                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Andrej Karpathy     │ 142 mentions │ [View Docs →]│  │
│  │ Ilya Sutskever      │ 89 mentions  │ [View Docs →]│  │
│  │ Dario Amodei        │ 67 mentions  │ [View Docs →]│  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  Technologies (423)                                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │ GRPO                │ 34 mentions  │ [View Docs →]│  │
│  │ Mixture of Experts  │ 28 mentions  │ [View Docs →]│  │
│  │ FlashAttention      │ 21 mentions  │ [View Docs →]│  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  Entity Detail: "GRPO"                                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Type: Technology                                   │  │
│  │ First seen: 2025-01-15                            │  │
│  │ 34 documents mention this entity                   │  │
│  │                                                    │  │
│  │ Related entities:                                  │  │
│  │  DeepSeek (15 co-occurrences)                     │  │
│  │  PPO (12 co-occurrences)                          │  │
│  │  RLHF (9 co-occurrences)                          │  │
│  │                                                    │  │
│  │ Documents:                                         │  │
│  │  📄 "DeepSeek's GRPO Paper..."  ArXiv  Jan 15    │  │
│  │  🎥 "GRPO Explained..."         YouTube Jan 18   │  │
│  │  💬 "r/LocalLLaMA: GRPO..."     Reddit  Jan 16   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 5. Content Briefing Builder
```
┌─────────────────────────────────────────────────────────┐
│  Briefing Builder                                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Topic: [LLM inference optimization                  ]   │
│                                                          │
│  Options:                                                │
│  ☑ Ingest fresh pro content   ☑ Run radar search        │
│  Sources for radar: ☑ All                                │
│  LLM: [Claude Sonnet ▾]                                 │
│                                                          │
│  [Generate Briefing]                                     │
│                                                          │
│  ─── Generated Briefing ────────────────────────────── │
│                                                          │
│  Key Themes:                                             │
│  1. Speculative decoding is becoming mainstream...       │
│  2. KV cache optimization is the new bottleneck...      │
│                                                          │
│  Unique Angles:                                          │
│  - Nobody is writing about the cost implications of...  │
│  - The gap between academic benchmarks and real...      │
│                                                          │
│  Content Ideas:                                          │
│  1. "Why Your LLM Inference Stack is 10x Too..." →     │
│  2. "The Hidden Cost of Long Context Windows" →         │
│                                                          │
│  Sources Used: (18 documents)                            │
│  [📄 Source 1] [🎥 Source 2] [📄 Source 3] ...          │
│                                                          │
│  [📋 Copy as Markdown] [💾 Export] [🔄 Regenerate]      │
└─────────────────────────────────────────────────────────┘
```

#### 6. Document Management
```
┌─────────────────────────────────────────────────────────┐
│  Documents                                               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Filter: [All Sources ▾] [All Origins ▾] [Date Range]   │
│  Tags: [content-idea ×] [+]                              │
│  ☑ Show archived  ☐ Show deleted                        │
│                                                          │
│  Bulk Actions: [Select All] [Archive] [Delete] [Tag]    │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ ☐ 📄 "Scaling Laws for Neural..."                 │  │
│  │   ArXiv · Kaplan et al · 2024-12-01               │  │
│  │   Tags: #scaling #research                        │  │
│  │   Entities: OpenAI, Scaling Laws, Chinchilla      │  │
│  │   [Archive] [Delete] [View Full] [Open Source]    │  │
│  │                                                    │  │
│  │ ☐ 🎥 "I spent a mass on GRPO..."                  │  │
│  │   YouTube · @YannicKilcher · 2025-01-20           │  │
│  │   Origin: radar (not promoted)                    │  │
│  │   [Promote to Pro] [Delete] [View Transcript]     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## MCP Server Tools

The MCP server runs alongside the REST API and exposes these tools:

```python
@mcp.tool()
async def search(
    query: str,
    mode: str = "hybrid",           # hybrid, semantic, keyword
    sources: list[str] | None = None,
    since: str | None = None,        # "7d", "30d", "2025-01-01"
    limit: int = 20,
    entity_type: str | None = None,
) -> list[SearchResult]:
    """Search indexed content across all sources."""

@mcp.tool()
async def radar(
    query: str,
    sources: list[str] | None = None,  # youtube, reddit, hn, arxiv, devto
    since: str | None = None,
    max_results_per_source: int = 10,
) -> list[SearchResult]:
    """Search the open web for a topic, ingest and index results."""

@mcp.tool()
async def ingest(
    source_type: str | None = None,  # null = all pro sources
) -> IngestReport:
    """Pull latest content from pro-tier sources."""

@mcp.tool()
async def ingest_url(
    url: str,
    tags: list[str] | None = None,
) -> Document:
    """Ingest a single URL (article, video, etc.) into the index."""

@mcp.tool()
async def briefing(
    topic: str,
    run_radar: bool = True,
    run_ingest: bool = True,
) -> Briefing:
    """Generate a content briefing on a topic."""

@mcp.tool()
async def get_entities(
    query: str | None = None,
    entity_type: str | None = None,
    limit: int = 20,
) -> list[Entity]:
    """Search and browse extracted entities."""

@mcp.tool()
async def get_stats() -> SystemStats:
    """Get system stats: document counts, storage, source health."""

@mcp.tool()
async def manage_source(
    action: str,                     # 'add', 'remove', 'disable', 'enable'
    source_type: str,
    identifier: str,
    display_name: str | None = None,
) -> Source:
    """Add, remove, or modify a source subscription."""

@mcp.tool()
async def tag_document(
    document_id: str,
    tags: list[str],
    action: str = "add",             # 'add', 'remove', 'set'
) -> Document:
    """Tag or untag a document."""

@mcp.tool()
async def discover_sources(
    based_on: str = "recent",        # 'recent', 'favorites', 'topic:XYZ'
    limit: int = 5,
) -> list[DiscoveredSource]:
    """Suggest new sources based on existing content."""
```

---

## Updated Project Structure

```
ai-craftsman-kb/
├── pyproject.toml
├── README.md
│
├── config/                              # Default config (copied to ~/.ai-craftsman-kb/config/ on init)
│   ├── sources.yaml
│   ├── settings.yaml
│   ├── filters.yaml
│   └── prompts/
│       ├── briefing.md
│       ├── entity_extraction.md
│       ├── content_filter.md
│       └── source_discovery.md
│
├── backend/                             # Python
│   ├── ai_craftsman_kb/
│   │   ├── __init__.py
│   │   ├── cli.py                       # Click CLI (cr command)
│   │   ├── server.py                    # FastAPI app
│   │   ├── mcp_server.py               # MCP tool definitions
│   │   │
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py               # YAML config loader with env var expansion
│   │   │   └── models.py               # Pydantic config models
│   │   │
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── sqlite.py               # SQLite connection + migrations
│   │   │   ├── queries.py              # Query helpers
│   │   │   └── models.py               # Pydantic DB models (Document, Entity, Source)
│   │   │
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # Abstract LLMProvider
│   │   │   ├── openai_provider.py
│   │   │   ├── openrouter_provider.py
│   │   │   ├── fireworks_provider.py
│   │   │   ├── anthropic_provider.py
│   │   │   ├── ollama_provider.py
│   │   │   └── router.py              # Routes tasks to configured providers
│   │   │
│   │   ├── ingestors/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # Abstract BaseIngestor
│   │   │   ├── hackernews.py
│   │   │   ├── substack.py
│   │   │   ├── youtube.py
│   │   │   ├── reddit.py
│   │   │   ├── rss.py
│   │   │   ├── arxiv.py
│   │   │   ├── devto.py
│   │   │   └── adhoc.py                # Universal URL ingestor
│   │   │
│   │   ├── processing/
│   │   │   ├── __init__.py
│   │   │   ├── extractor.py            # URL → clean text (readability)
│   │   │   ├── chunker.py              # Text → chunks with overlap
│   │   │   ├── embedder.py             # Chunks → vectors
│   │   │   ├── entity_extractor.py     # Text → entities via LLM
│   │   │   ├── filter.py               # LLM/keyword content filter
│   │   │   └── discoverer.py           # Outbound link / source discovery
│   │   │
│   │   ├── search/
│   │   │   ├── __init__.py
│   │   │   ├── vector_store.py         # Qdrant local wrapper
│   │   │   ├── keyword.py              # SQLite FTS wrapper
│   │   │   └── hybrid.py              # RRF fusion of vector + keyword
│   │   │
│   │   ├── radar/
│   │   │   ├── __init__.py
│   │   │   └── engine.py              # Fan-out search orchestrator
│   │   │
│   │   └── briefing/
│   │       ├── __init__.py
│   │       └── generator.py           # LLM briefing pipeline
│   │
│   └── tests/
│       ├── test_ingestors/
│       ├── test_search/
│       └── test_processing/
│
├── dashboard/                           # TypeScript/React
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── api/
│   │   │   └── client.ts              # Typed API client for backend
│   │   ├── pages/
│   │   │   ├── Overview.tsx
│   │   │   ├── Sources.tsx
│   │   │   ├── Search.tsx
│   │   │   ├── Entities.tsx
│   │   │   ├── Briefing.tsx
│   │   │   ├── Documents.tsx
│   │   │   └── Settings.tsx
│   │   ├── components/
│   │   │   ├── SourceEditor.tsx
│   │   │   ├── SearchBar.tsx
│   │   │   ├── DocumentCard.tsx
│   │   │   ├── EntityGraph.tsx
│   │   │   ├── StatsCards.tsx
│   │   │   ├── SourceHealth.tsx
│   │   │   ├── AdhocIngest.tsx
│   │   │   └── DiscoveredSources.tsx
│   │   └── hooks/
│   │       ├── useSearch.ts
│   │       ├── useSources.ts
│   │       └── useStats.ts
│   └── public/
│
└── scripts/
    ├── setup.sh                        # First-time setup
    └── dev.sh                          # Start backend + dashboard in dev mode
```

---

## Updated Phase Plan

### Phase 1: Foundation (Week 1)

**Goal**: CLI skeleton, SQLite, config loading, LLM provider abstraction, one working source (HN).

Tasks:
1. Project scaffolding (Python backend with `uv`, config directory)
2. Config loader — reads YAML, expands `${ENV_VARS}`, validates with Pydantic
3. SQLite setup — all tables, migrations, query helpers
4. LLM provider abstraction — base class + OpenAI + OpenRouter implementations
5. Base ingestor interface with `fetch_pro()` and `search_radar()` methods
6. HN ingestor — pro mode (top stories) + radar mode (Algolia search)
7. Content extractor — URL → clean text via readability-lxml
8. Content filter — LLM-based scoring for HN articles per filters.yaml
9. CLI: `cr ingest pro --source hn`, `cr search "query"` (FTS only), `cr stats`

Exit criteria:
- `cr ingest pro --source hn` pulls top HN stories, filters them via LLM, stores in SQLite
- `cr search "transformer"` returns results via FTS
- Config is fully loaded from `~/.ai-craftsman-kb/config/`

Dependencies:
```
click, httpx, pydantic, pyyaml, readability-lxml, lxml, html2text, openai, tiktoken
```

---

### Phase 2: All Pro Sources (Week 2)

**Goal**: Every pro-tier source is pulling, filtering, and storing content.

Tasks:
1. Substack ingestor — RSS feed parsing, full content extraction
2. RSS ingestor — generic RSS/Atom, follow links for full content if needed
3. YouTube ingestor — channel uploads via Data API, transcript pulling
4. Reddit ingestor — subreddit hot/top posts with top comments
5. ArXiv ingestor — keyword search, abstract + optional PDF extraction
6. DEV.to ingestor — tag/keyword API search
7. Adhoc URL ingestor — auto-detect type (YouTube URL → transcript, article URL → readability)
8. Incremental ingestion — `last_fetched_at` tracking, only fetch new content

Exit criteria:
- `cr ingest pro` pulls all 6+ source types
- `cr ingest-url "https://youtube.com/..."` ingests a single URL
- YouTube videos have full transcripts
- Re-running only fetches new content

New dependencies:
```
feedparser, youtube-transcript-api, google-api-python-client, praw, arxiv, pymupdf
```

---

### Phase 3: Semantic Search + Entities (Week 3)

**Goal**: Embed all content, extract entities, enable hybrid search and entity browsing.

Tasks:
1. Embedding pipeline — OpenAI and local (sentence-transformers) backends
2. Chunking — long documents split with overlap, short docs embedded whole
3. Qdrant local setup — collection creation, upsert, similarity search
4. Hybrid search — FTS + vector search merged via Reciprocal Rank Fusion
5. Entity extraction pipeline — LLM extracts people, companies, tech, events from each document
6. Entity deduplication — normalize names, merge variants
7. Entity search — FTS on entity names, co-occurrence queries
8. Auto-embed and extract on ingest — pipeline hooks into ingestor flow
9. CLI enhancements:
   - `cr search "GRPO" --source youtube --since 30d`
   - `cr entities --type person --top 20`
   - `cr search --entity "Karpathy"`

Exit criteria:
- All documents have embeddings
- Entities extracted for all documents
- `cr search "topic"` returns semantically relevant results
- `cr entities` shows extracted people, technologies, etc.

New dependencies:
```
qdrant-client, sentence-transformers (optional)
```

---

### Phase 4: Radar Engine (Week 4)

**Goal**: On-demand topic search across external sources as a callable capability.

Tasks:
1. Radar engine orchestrator — async fan-out to all radar sources
2. YouTube radar — API search + transcript pull for results
3. Reddit radar — global search across all subreddits
4. HN radar — Algolia search
5. ArXiv radar — keyword search in abstracts
6. DEV.to radar — tag/keyword API search
7. Result deduplication — by URL across sources
8. Auto-embed radar results — same pipeline as pro content
9. Promote/archive/delete operations on radar results
10. CLI: `cr radar "GRPO reinforcement learning" --sources youtube,reddit`

Exit criteria:
- `cr radar "topic"` searches 5+ sources concurrently
- YouTube results include transcripts
- Radar results are embedded and searchable alongside pro content
- Results can be promoted to pro tier

---

### Phase 5: Dashboard + MCP Server (Week 5-6)

**Goal**: Web dashboard with source management, search, stats, and MCP server for agent access.

Tasks:
1. **FastAPI backend** — REST API wrapping all CLI functionality
   - CRUD endpoints for sources (reads/writes sources.yaml)
   - Search, radar, briefing endpoints
   - Stats, entity browsing, document management endpoints
   - Adhoc URL ingestion endpoint
   - Source discovery endpoint
2. **MCP server** — Python MCP SDK, exposes all tools listed above
   - Runs on same process as FastAPI (or as separate `cr mcp` command)
3. **React dashboard** — Vite + Tailwind + shadcn/ui
   - Overview page with stats cards, source health, recent documents
   - Source editor — add, edit, remove, enable/disable sources visually
   - Search page — hybrid search with source/date filters, radar tab, adhoc URL tab
   - Entity explorer — browse entities, see co-occurrences, view related documents
   - Document manager — browse, tag, favorite, archive, delete, bulk actions
   - Briefing builder — topic input, LLM selection, generated output with export
   - Settings page — view/edit settings.yaml and filters.yaml
4. **Source discovery UI** — show suggested sources with accept/dismiss
5. **Document cleanup UI** — select and soft-delete documents, remove sources and their documents
6. **One-command startup**: `cr server` starts backend + serves dashboard

New dependencies:
```
# Backend
fastapi, uvicorn, mcp[server]

# Dashboard
react, vite, tailwindcss, @shadcn/ui, lucide-react, recharts (for stats charts)
```

Exit criteria:
- `cr server` opens a functional dashboard at localhost:3000
- All CRUD operations work through the dashboard
- MCP server responds to tool calls from Claude or other agents
- Sources can be added/removed from the dashboard and are reflected in YAML
- Adhoc URLs can be ingested from the dashboard

---

### Phase 6: Briefing + Discovery + Polish (Week 7)

**Goal**: Content briefing generator, source discovery, and daily-driver quality.

Tasks:
1. Briefing generator — full pipeline (ingest → radar → search → LLM synthesis)
2. Briefing prompt engineering — customizable templates in config/prompts/
3. Source discovery engine:
   - Extract outbound links from ingested articles
   - Identify frequently-cited authors, channels, publications
   - LLM-powered suggestions ("based on your reading, you'd like...")
   - Surface in dashboard with one-click add
4. Rich CLI output — progress bars, formatted tables, colored output via `rich`
5. Export functionality — search results → markdown, briefings → markdown/JSON
6. Error resilience — retry logic, graceful source failures, health monitoring
7. `cr doctor` — check API keys, source connectivity, disk usage, stale data
8. Database maintenance — `cr cleanup`, re-embed commands

New dependencies:
```
anthropic, rich
```

Exit criteria:
- `cr briefing "topic"` produces structured content briefs
- Dashboard shows source suggestions you can accept with one click
- System is resilient to individual source failures
- You actually enjoy using it daily

---

## Full Dependency List

### Python Backend — Core
```
click              # CLI framework
httpx              # async HTTP client
pydantic           # data validation
pyyaml             # config files
fastapi            # REST API
uvicorn            # ASGI server
mcp                # MCP server SDK
aiosqlite          # async SQLite
qdrant-client      # local vector store
openai             # embeddings + LLM
anthropic          # briefing LLM
tiktoken           # token counting
rich               # terminal formatting
readability-lxml   # article extraction
lxml               # HTML parsing
html2text          # HTML → plain text
feedparser         # RSS/Atom
```

### Python Backend — Source-specific
```
youtube-transcript-api
google-api-python-client
praw
arxiv
pymupdf
```

### Python Backend — Optional
```
sentence-transformers    # local embeddings
```

### Dashboard (TypeScript)
```
react, react-dom
vite
tailwindcss
@shadcn/ui components
lucide-react
recharts
```

---

## API Keys Needed

| Service | Cost | Purpose |
|---------|------|---------|
| OpenAI | ~$0.50/mo | Embeddings |
| OpenRouter | ~$0.50/mo | Filtering + entity extraction (Llama 3.1 8B) |
| Anthropic | ~$1/mo | Briefing generation (Claude Sonnet) |
| YouTube Data API v3 | Free | Video/channel metadata |
| Reddit API | Free | OAuth app for subreddit access |

**Total: ~$2/month.** Everything else is free.

---

## Timeline Summary

| Phase | Week | What you get |
|-------|------|-------------|
| 1. Foundation | 1 | CLI + SQLite + HN + LLM filtering + config system |
| 2. All Sources | 2 | Full pro ingestion across 7 source types + adhoc URL |
| 3. Search + Entities | 3 | Hybrid semantic search + entity extraction — the "magic" |
| 4. Radar | 4 | On-demand topic search as a capability |
| 5. Dashboard + MCP | 5-6 | Web UI + agent-accessible API — the "product" moment |
| 6. Briefing + Polish | 7 | Content generation pipeline + source discovery |

Each phase is independently useful. Phase 3 is when it becomes powerful. Phase 5 is when it becomes a product you and your agents can use together.

---

## Storage Budget (Revisited)

For your peace of mind, here's the worst-case 1-year projection:

```
SQLite (documents + entities + FTS):    ~800 MB
Qdrant vectors:                         ~600 MB
Logs:                                    ~50 MB
Config:                                   ~1 MB
────────────────────────────────────────────────
Total:                                  ~1.5 GB (1 year, heavy use)
```

Your MacBook likely has 256GB-1TB. This is <1% of storage. Not a concern.
