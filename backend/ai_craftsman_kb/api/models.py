"""Shared Pydantic response models for the AI Craftsman KB API.

All response models use the ``Out`` suffix and are designed for JSON
serialisation. They are intentionally separate from the DB row models
(``DocumentRow``, ``SourceRow``, etc.) to allow the API to evolve
independently of the database schema.
"""
from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Document models
# ---------------------------------------------------------------------------


class DocumentOut(BaseModel):
    """API response model for a single document.

    Attributes:
        id: UUID of the document.
        title: Document title, or None if not available.
        url: Canonical URL.
        source_type: Source platform (e.g. 'hn', 'substack').
        origin: Ingestion origin ('pro', 'radar', 'adhoc').
        author: Author name, or None.
        published_at: ISO 8601 publication timestamp, or None.
        fetched_at: ISO 8601 timestamp when the document was ingested.
        word_count: Word count of the raw content, or None.
        is_embedded: Whether vectors have been stored in Qdrant.
        is_favorited: Whether the user has favorited this document.
        is_archived: Whether the document is archived.
        user_tags: List of user-applied tag strings.
        excerpt: First 300 characters of raw_content, or None.
    """

    id: str
    title: str | None
    url: str
    source_type: str
    origin: str
    author: str | None
    published_at: str | None
    fetched_at: str
    word_count: int | None
    is_embedded: bool
    is_favorited: bool
    is_archived: bool
    user_tags: list[str]
    excerpt: str | None  # first 300 chars of raw_content


class SearchResultOut(BaseModel):
    """API response model for a single search result.

    Attributes:
        document: The matching document.
        score: Combined RRF score (hybrid) or modality-specific score.
        mode_used: Which search mode was applied ('hybrid', 'semantic', 'keyword').
    """

    document: DocumentOut
    score: float
    mode_used: str  # 'hybrid' | 'semantic' | 'keyword'


# ---------------------------------------------------------------------------
# System models
# ---------------------------------------------------------------------------


class SystemStats(BaseModel):
    """API response model for aggregate system statistics.

    Attributes:
        total_documents: Total non-deleted documents in the DB.
        embedded_documents: Documents with is_embedded=True.
        total_entities: Total entities in the DB.
        active_sources: Number of enabled sources.
        total_briefings: Total briefings generated.
        vector_count: Total vectors stored in Qdrant.
        db_size_bytes: SQLite DB file size in bytes.
    """

    total_documents: int
    embedded_documents: int
    total_entities: int
    active_sources: int
    total_briefings: int
    vector_count: int
    db_size_bytes: int


class HealthCheckResult(BaseModel):
    """Result of a single health check.

    Attributes:
        status: Outcome of the check — 'ok', 'warn', or 'error'.
        message: Human-readable summary, including a fix hint on failure.
    """

    status: str  # 'ok' | 'warn' | 'error'
    message: str


class HealthOut(BaseModel):
    """API response model for the health check endpoint.

    Attributes:
        status: 'ok' if all critical components are healthy, else 'degraded'.
        db: Whether the SQLite database is reachable.
        qdrant: Whether the Qdrant vector store is reachable.
        checks: Extended diagnostics map (only present when ?full=true).
            Keys are check names (e.g. 'openai_key', 'hn_connectivity').
            Values are HealthCheckResult instances with status + message.
    """

    status: str
    db: bool
    qdrant: bool
    checks: dict[str, HealthCheckResult] | None = None


# ---------------------------------------------------------------------------
# Ingest models
# ---------------------------------------------------------------------------


class IngestReportOut(BaseModel):
    """API response model for an ingest operation.

    Attributes:
        source_type: The source type that was ingested.
        fetched: Total documents fetched from the source.
        stored: Documents stored to DB (new, non-duplicate).
        embedded: Documents embedded into Qdrant.
        errors: List of error message strings.
    """

    source_type: str
    fetched: int
    stored: int
    embedded: int
    errors: list[str]


# ---------------------------------------------------------------------------
# Source models
# ---------------------------------------------------------------------------


class SourceOut(BaseModel):
    """API response model for a source row.

    Attributes:
        id: UUID of the source.
        source_type: Source platform identifier (e.g. 'hn', 'substack').
        identifier: Source-specific identifier (slug, handle, URL, etc.).
        display_name: Human-readable name, or None.
        enabled: Whether the source is active for pro ingestion.
        last_fetched_at: ISO 8601 timestamp of last successful fetch, or None.
        fetch_error: Error message from last failed fetch, or None.
        created_at: ISO 8601 creation timestamp.
    """

    id: str
    source_type: str
    identifier: str
    display_name: str | None
    enabled: bool
    last_fetched_at: str | None
    fetch_error: str | None
    created_at: str


# ---------------------------------------------------------------------------
# Entity models
# ---------------------------------------------------------------------------


class EntityOut(BaseModel):
    """API response model for a named entity.

    Attributes:
        id: UUID of the entity.
        name: Display name of the entity.
        entity_type: Category (person, company, technology, etc.).
        normalized_name: Lowercased deduplicated name.
        description: Optional description text.
        mention_count: Number of documents that mention this entity.
        first_seen_at: ISO 8601 timestamp, or None.
    """

    id: str
    name: str
    entity_type: str
    normalized_name: str
    description: str | None
    mention_count: int
    first_seen_at: str | None


class EntityWithDocsOut(BaseModel):
    """API response model for an entity with its associated documents.

    Attributes:
        entity: The entity details.
        documents: List of documents that mention this entity.
    """

    entity: EntityOut
    documents: list[DocumentOut]


# ---------------------------------------------------------------------------
# Radar models
# ---------------------------------------------------------------------------


class RadarReportOut(BaseModel):
    """API response model for a radar search operation.

    Attributes:
        query: The search query that was executed.
        total_found: Total documents found across all sources.
        new_documents: New documents stored (not previously in DB).
        sources_searched: List of source_type strings that were searched.
        errors: Per-source error messages (keyed by source_type).
    """

    query: str
    total_found: int
    new_documents: int
    sources_searched: list[str]
    errors: dict[str, str]


# ---------------------------------------------------------------------------
# Briefing models
# ---------------------------------------------------------------------------


class BriefingOut(BaseModel):
    """API response model for a generated briefing.

    Attributes:
        id: UUID of the briefing.
        title: Briefing title.
        query: Search query used to generate the briefing, or None.
        content: Markdown-formatted briefing content.
        source_document_ids: UUIDs of documents included in the briefing.
        created_at: ISO 8601 creation timestamp.
        format: Content format (usually 'markdown').
    """

    id: str
    title: str
    query: str | None
    content: str
    source_document_ids: list[str]
    created_at: str
    format: str


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class IngestURLRequest(BaseModel):
    """Request body for POST /api/ingest/url.

    Attributes:
        url: The URL to ingest.
        tags: Optional list of user tags to apply to the document.
    """

    url: str
    tags: list[str] = []


class IngestProRequest(BaseModel):
    """Request body for POST /api/ingest/pro.

    Attributes:
        source: Optional source_type to restrict to a single source.
                When None, all enabled sources are ingested.
    """

    source: str | None = None


class CreateSourceRequest(BaseModel):
    """Request body for POST /api/sources.

    Attributes:
        source_type: Source platform identifier.
        identifier: Source-specific identifier (slug, handle, URL, etc.).
        display_name: Optional human-readable name.
    """

    source_type: str
    identifier: str
    display_name: str | None = None


class UpdateSourceRequest(BaseModel):
    """Request body for PUT /api/sources/{id}.

    Attributes:
        enabled: Whether to enable or disable the source.
        display_name: Optional updated display name.
    """

    enabled: bool | None = None
    display_name: str | None = None


class RadarSearchRequest(BaseModel):
    """Request body for POST /api/radar/search.

    Attributes:
        query: The topic search query.
        sources: Optional list of source_type strings to restrict the search.
        limit_per_source: Maximum results to request from each source.
    """

    query: str
    sources: list[str] | None = None
    limit_per_source: int = 10


class CreateBriefingRequest(BaseModel):
    """Request body for POST /api/briefings.

    Attributes:
        query: Search query to find relevant documents.
        limit: Maximum number of documents to include.
        run_radar: Whether to run a radar search for fresh content first.
        run_ingest: Whether to run pro ingest before generating the briefing.
    """

    query: str
    limit: int = 20
    run_radar: bool = False
    run_ingest: bool = False
