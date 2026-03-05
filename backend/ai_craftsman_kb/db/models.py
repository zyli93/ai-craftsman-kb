"""Pydantic row models for SQLite tables.

Each model maps directly to a database table and is used for
type-safe data serialization and deserialization throughout the app.
"""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utcnow_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class DocumentRow(BaseModel):
    """Maps to the documents table.

    Represents a single piece of content fetched from any source.
    JSON fields (metadata, user_tags) are stored as TEXT in SQLite and
    serialized/deserialized automatically in the query helpers.
    """

    id: str
    source_id: str | None = None
    origin: Literal["pro", "radar", "adhoc"]
    source_type: str
    url: str
    title: str | None = None
    author: str | None = None
    published_at: str | None = None  # ISO 8601
    fetched_at: str = Field(default_factory=utcnow_iso)
    content_type: str | None = None  # 'article' | 'video' | 'paper' | 'post'
    raw_content: str | None = None  # full extracted text
    word_count: int | None = None
    metadata: dict = Field(default_factory=dict)
    is_embedded: bool = False
    is_entities_extracted: bool = False
    is_keywords_extracted: bool = False
    filter_score: float | None = None
    filter_passed: bool | None = None
    is_favorited: bool = False
    is_archived: bool = False
    user_tags: list[str] = Field(default_factory=list)
    user_notes: str | None = None
    promoted_at: str | None = None
    deleted_at: str | None = None


class SourceRow(BaseModel):
    """Maps to the sources table.

    Represents a configured source (e.g., a Substack newsletter, a subreddit).
    JSON field (config) stores source-specific configuration snapshots.
    """

    id: str
    source_type: str  # 'substack','youtube','reddit','rss','hn','arxiv','devto'
    identifier: str  # slug, handle, subreddit name, feed URL, etc.
    display_name: str | None = None
    tier: str = "pro"  # 'pro' only for now
    enabled: bool = True
    last_fetched_at: str | None = None
    fetch_error: str | None = None
    config: dict = Field(default_factory=dict)  # source-specific config snapshot
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)


class EntityRow(BaseModel):
    """Maps to the entities table.

    Represents a named entity (person, company, technology, etc.) extracted
    from documents. Entities are deduplicated by (normalized_name, entity_type).
    """

    id: str
    name: str
    entity_type: str  # 'person'|'company'|'technology'|'event'|'book'|'paper'|'product'
    normalized_name: str  # lowercase, deduped version
    description: str | None = None
    first_seen_at: str | None = None
    mention_count: int = 1
    metadata: dict = Field(default_factory=dict)


class DocumentKeywordRow(BaseModel):
    """Maps to the document_keywords table.

    Represents a keyword extracted from a document. Keywords are normalized
    (lowercase, stripped) and deduplicated per document via the UNIQUE constraint
    on (document_id, keyword).
    """

    document_id: str
    keyword: str


class DiscoveredSourceRow(BaseModel):
    """Maps to the discovered_sources table.

    Represents a potential source discovered from document content,
    awaiting user review before being added as a tracked source.
    """

    id: str
    source_type: str
    identifier: str
    display_name: str | None = None
    discovered_from_document_id: str | None = None
    discovery_method: str | None = None  # 'outbound_link'|'citation'|'mention'|'llm_suggestion'
    confidence: float | None = None
    status: str = "suggested"  # 'suggested'|'added'|'dismissed'
    created_at: str = Field(default_factory=utcnow_iso)


class BriefingRow(BaseModel):
    """Maps to the briefings table.

    Represents a generated briefing (digest) of selected documents.
    source_document_ids is stored as a JSON array in SQLite.
    """

    id: str
    title: str
    query: str | None = None
    content: str  # markdown
    source_document_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow_iso)
    format: str = "markdown"
