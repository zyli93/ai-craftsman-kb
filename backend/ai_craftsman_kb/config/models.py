"""Pydantic configuration models for AI Craftsman KB.

All configuration is structured as immutable Pydantic models validated at
load time. AppConfig is the top-level container consumed throughout the app.
"""

from typing import Literal

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sources config
# ---------------------------------------------------------------------------


class SubstackSource(BaseModel):
    """A single Substack newsletter to ingest."""

    slug: str
    name: str


class YoutubeChannelSource(BaseModel):
    """A YouTube channel to ingest transcripts from."""

    handle: str  # e.g. "@AndrejKarpathy"
    name: str


class SubredditSource(BaseModel):
    """A subreddit to monitor for content."""

    name: str
    sort: Literal["hot", "new", "top", "rising"] = "hot"
    limit: int = 25


class RSSSource(BaseModel):
    """An RSS/Atom feed to ingest."""

    url: str
    name: str


class HackerNewsConfig(BaseModel):
    """HackerNews ingestion settings."""

    mode: Literal["top", "new", "best"] = "top"
    limit: int = 30


class ArxivConfig(BaseModel):
    """ArXiv ingestion settings."""

    queries: list[str]
    max_results: int = 20


class DevtoConfig(BaseModel):
    """DEV.to ingestion settings."""

    tags: list[str] = []
    limit: int = 20


class SourcesConfig(BaseModel):
    """Top-level sources configuration — which feeds/channels/subreddits to ingest."""

    substack: list[SubstackSource] = []
    youtube_channels: list[YoutubeChannelSource] = []
    subreddits: list[SubredditSource] = []
    rss: list[RSSSource] = []
    hackernews: HackerNewsConfig | None = None
    arxiv: ArxivConfig | None = None
    devto: DevtoConfig | None = None


# ---------------------------------------------------------------------------
# Settings config
# ---------------------------------------------------------------------------


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    chunk_size: int = 2000
    chunk_overlap: int = 200


class LLMTaskConfig(BaseModel):
    """LLM provider + model for a specific task."""

    provider: str
    model: str


class LLMRoutingConfig(BaseModel):
    """Routes different LLM tasks to specific providers and models."""

    filtering: LLMTaskConfig
    entity_extraction: LLMTaskConfig
    briefing: LLMTaskConfig
    source_discovery: LLMTaskConfig


class ProviderConfig(BaseModel):
    """Credentials and base URL for a single LLM/API provider."""

    api_key: str | None = None
    base_url: str | None = None


class YoutubeAPIConfig(BaseModel):
    """YouTube Data API credentials and preferences."""

    api_key: str | None = None
    transcript_langs: list[str] = ["en"]


class RedditAPIConfig(BaseModel):
    """Reddit OAuth credentials."""

    client_id: str | None = None
    client_secret: str | None = None


class ServerConfig(BaseModel):
    """Port configuration for backend and dashboard servers."""

    backend_port: int = 8000
    dashboard_port: int = 3000


class SearchConfig(BaseModel):
    """Hybrid search weighting and default result limits."""

    default_limit: int = 20
    hybrid_weight_semantic: float = 0.6
    hybrid_weight_keyword: float = 0.4


class SettingsConfig(BaseModel):
    """Top-level application settings — LLM routing, embedding, server, credentials."""

    data_dir: str = "~/.ai-craftsman-kb/data"
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMRoutingConfig
    providers: dict[str, ProviderConfig] = {}
    youtube: YoutubeAPIConfig = YoutubeAPIConfig()
    reddit: RedditAPIConfig = RedditAPIConfig()
    server: ServerConfig = ServerConfig()
    search: SearchConfig = SearchConfig()


# ---------------------------------------------------------------------------
# Filters config
# ---------------------------------------------------------------------------


class SourceFilterConfig(BaseModel):
    """Per-source filter rules controlling which content is retained after ingestion."""

    enabled: bool = True
    strategy: Literal["llm", "hybrid", "keyword"] = "keyword"
    min_score: int | None = None  # for LLM strategy (1-10)
    min_upvotes: int | None = None  # for Reddit
    min_reactions: int | None = None  # for DEV.to
    keywords_include: list[str] = []
    keywords_exclude: list[str] = []
    llm_prompt: str | None = None


class FiltersConfig(BaseModel):
    """Content filter configuration for all supported sources."""

    hackernews: SourceFilterConfig = SourceFilterConfig()
    substack: SourceFilterConfig = SourceFilterConfig(enabled=False)
    youtube: SourceFilterConfig = SourceFilterConfig(enabled=False)
    reddit: SourceFilterConfig = SourceFilterConfig()
    arxiv: SourceFilterConfig = SourceFilterConfig()
    rss: SourceFilterConfig = SourceFilterConfig(enabled=False)
    devto: SourceFilterConfig = SourceFilterConfig()


# ---------------------------------------------------------------------------
# Top-level aggregate
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Root configuration object for AI Craftsman KB.

    Loaded once at startup and passed via dependency injection throughout
    the application. Never mutated after construction.
    """

    sources: SourcesConfig
    settings: SettingsConfig
    filters: FiltersConfig
