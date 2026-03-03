# Task 02: Config System (YAML Loader + Pydantic)

## Wave
Wave 2 (parallel with tasks: 03, 04, 05)
Domain: backend

## Objective
Implement the YAML-based configuration system with Pydantic validation and environment variable interpolation, producing a unified `AppConfig` object consumed throughout the app.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/config/loader.py` — YAML loading, `${ENV_VAR}` interpolation, merging
- `backend/ai_craftsman_kb/config/models.py` — All Pydantic config models
- `backend/ai_craftsman_kb/config/__init__.py` — Exports `load_config()`, `AppConfig`
- `config/sources.yaml` — Default source subscriptions
- `config/settings.yaml` — Default settings (LLM routing, embedding, server)
- `config/filters.yaml` — Per-source filter rules
- `config/prompts/briefing.md` — Briefing prompt template (placeholder)
- `config/prompts/entity_extraction.md` — Entity extraction prompt (placeholder)
- `config/prompts/content_filter.md` — Content filter prompt (placeholder)
- `config/prompts/source_discovery.md` — Source discovery prompt (placeholder)
- `backend/tests/test_config.py` — Unit tests

### Key interfaces / implementation details:

**`config/sources.yaml`** (exact format from plan.md):
```yaml
substack:
  - slug: karpathy
    name: "Andrej Karpathy"
  - slug: simonwillison
    name: "Simon Willison"

youtube_channels:
  - handle: "@AndrejKarpathy"
    name: "Andrej Karpathy"
  - handle: "@YannicKilcher"
    name: "Yannic Kilcher"

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
  - url: https://openai.com/blog/rss.xml
    name: "OpenAI Blog"

hackernews:
  mode: top
  limit: 30

arxiv:
  queries:
    - "cat:cs.CL AND abs:large language model"
    - "cat:cs.AI AND abs:reinforcement learning"
  max_results: 20

devto:
  tags:
    - ai
    - machinelearning
  limit: 20
```

**`config/settings.yaml`** (exact format from plan.md):
```yaml
data_dir: ~/.ai-craftsman-kb/data

embedding:
  provider: openai
  model: text-embedding-3-small
  chunk_size: 2000
  chunk_overlap: 200

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

youtube:
  api_key: ${YOUTUBE_API_KEY}
  transcript_langs: ["en"]

reddit:
  client_id: ${REDDIT_CLIENT_ID}
  client_secret: ${REDDIT_CLIENT_SECRET}

server:
  backend_port: 8000
  dashboard_port: 3000

search:
  default_limit: 20
  hybrid_weight_semantic: 0.6
  hybrid_weight_keyword: 0.4
```

**`config/filters.yaml`** (exact format from plan.md):
```yaml
hackernews:
  enabled: true
  strategy: llm
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
  keywords_exclude:
    - crypto
    - NFT
    - blockchain

reddit:
  enabled: true
  strategy: hybrid
  min_upvotes: 10
  keywords_exclude:
    - meme

substack:
  enabled: false

youtube:
  enabled: false

arxiv:
  enabled: true
  strategy: keyword
  keywords_include:
    - transformer
    - language model
    - reinforcement learning

rss:
  enabled: false

devto:
  enabled: true
  strategy: keyword
  min_reactions: 5
```

**Pydantic models** (`config/models.py`):
```python
# Sources config
class SubstackSource(BaseModel):
    slug: str
    name: str

class YoutubeChannelSource(BaseModel):
    handle: str   # e.g. "@AndrejKarpathy"
    name: str

class SubredditSource(BaseModel):
    name: str
    sort: Literal['hot', 'new', 'top', 'rising'] = 'hot'
    limit: int = 25

class RSSSource(BaseModel):
    url: str
    name: str

class HackerNewsConfig(BaseModel):
    mode: Literal['top', 'new', 'best'] = 'top'
    limit: int = 30

class ArxivConfig(BaseModel):
    queries: list[str]
    max_results: int = 20

class DevtoConfig(BaseModel):
    tags: list[str] = []
    limit: int = 20

class SourcesConfig(BaseModel):
    substack: list[SubstackSource] = []
    youtube_channels: list[YoutubeChannelSource] = []
    subreddits: list[SubredditSource] = []
    rss: list[RSSSource] = []
    hackernews: HackerNewsConfig | None = None
    arxiv: ArxivConfig | None = None
    devto: DevtoConfig | None = None

# Settings config
class EmbeddingConfig(BaseModel):
    provider: str = 'openai'
    model: str = 'text-embedding-3-small'
    chunk_size: int = 2000
    chunk_overlap: int = 200

class LLMTaskConfig(BaseModel):
    provider: str
    model: str

class LLMRoutingConfig(BaseModel):
    filtering: LLMTaskConfig
    entity_extraction: LLMTaskConfig
    briefing: LLMTaskConfig
    source_discovery: LLMTaskConfig

class ProviderConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None

class YoutubeAPIConfig(BaseModel):
    api_key: str | None = None
    transcript_langs: list[str] = ['en']

class RedditAPIConfig(BaseModel):
    client_id: str | None = None
    client_secret: str | None = None

class ServerConfig(BaseModel):
    backend_port: int = 8000
    dashboard_port: int = 3000

class SearchConfig(BaseModel):
    default_limit: int = 20
    hybrid_weight_semantic: float = 0.6
    hybrid_weight_keyword: float = 0.4

class SettingsConfig(BaseModel):
    data_dir: str = '~/.ai-craftsman-kb/data'
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMRoutingConfig
    providers: dict[str, ProviderConfig] = {}
    youtube: YoutubeAPIConfig = YoutubeAPIConfig()
    reddit: RedditAPIConfig = RedditAPIConfig()
    server: ServerConfig = ServerConfig()
    search: SearchConfig = SearchConfig()

# Filters config
class SourceFilterConfig(BaseModel):
    enabled: bool = True
    strategy: Literal['llm', 'hybrid', 'keyword'] = 'keyword'
    min_score: int | None = None       # for LLM strategy (1-10)
    min_upvotes: int | None = None     # for Reddit
    min_reactions: int | None = None   # for DEV.to
    keywords_include: list[str] = []
    keywords_exclude: list[str] = []
    llm_prompt: str | None = None

class FiltersConfig(BaseModel):
    hackernews: SourceFilterConfig = SourceFilterConfig()
    substack: SourceFilterConfig = SourceFilterConfig(enabled=False)
    youtube: SourceFilterConfig = SourceFilterConfig(enabled=False)
    reddit: SourceFilterConfig = SourceFilterConfig()
    arxiv: SourceFilterConfig = SourceFilterConfig()
    rss: SourceFilterConfig = SourceFilterConfig(enabled=False)
    devto: SourceFilterConfig = SourceFilterConfig()

class AppConfig(BaseModel):
    sources: SourcesConfig
    settings: SettingsConfig
    filters: FiltersConfig
```

**Loader** (`config/loader.py`):
```python
def load_config(
    config_dir: Path | None = None,
) -> AppConfig:
    """Load and merge sources.yaml + settings.yaml + filters.yaml.
    Config lookup order: argument → ~/.ai-craftsman-kb/ → bundled config/"""

def _interpolate_env_vars(data: Any) -> Any:
    """Recursively replace ${VAR_NAME} strings with os.environ values."""

def get_provider_api_key(config: AppConfig, provider: str) -> str | None:
    """Resolve API key for a given provider name."""
```

## Dependencies
- Depends on: task_01 (project structure + pyproject.toml exist)
- Packages needed: `pyyaml`, `pydantic` (already in pyproject.toml)

## Acceptance Criteria
- [ ] `load_config()` successfully loads and validates all three YAML files into `AppConfig`
- [ ] `${ENV_VAR}` syntax interpolated from environment at load time; missing vars log a warning but don't crash
- [ ] Config lookup order respected: CLI path → `~/.ai-craftsman-kb/` → bundled `config/`
- [ ] Pydantic validation errors include the field path and a human-readable message
- [ ] All four prompt files created under `config/prompts/` with placeholder content
- [ ] Unit tests cover: valid config, missing optional fields, env var interpolation, invalid field type

## Notes
- Use `yaml.safe_load()` only — never `yaml.load()`
- `AppConfig` loaded once at startup; pass it via dependency injection, not a global
- The `providers` dict key matches the `provider` string in `LLMTaskConfig` (e.g. `'openai'`, `'openrouter'`)
- `data_dir` should be `Path`-expanded with `~` at load time
