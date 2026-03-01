"""Abstract base ingestor and RawDocument model."""
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..config.models import AppConfig
    from ..db.models import DocumentRow


def _gen_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


class RawDocument(BaseModel):
    """Normalized output from any ingestor before DB storage.

    This is the common schema all source-specific ingestors return.
    Fields are mapped to DocumentRow when persisting to SQLite.
    """

    url: str
    title: str | None = None
    author: str | None = None
    raw_content: str | None = None  # full extracted text (may be None before extraction)
    content_type: str | None = None  # 'article' | 'video' | 'paper' | 'post'
    published_at: datetime | None = None
    source_type: str  # 'hn' | 'substack' | 'youtube' | 'reddit' | 'rss' | 'arxiv' | 'devto'
    origin: Literal["pro", "radar", "adhoc"] = "pro"
    word_count: int | None = None
    filter_score: float | None = None  # pre-filter score from ingestor (optional)
    metadata: dict = {}  # source-specific extras (HN points, upvotes, etc.)

    def to_document_row(self, source_id: str | None = None) -> "DocumentRow":
        """Convert to DB row model, generating a UUID.

        Args:
            source_id: Optional UUID of the source row in the sources table.

        Returns:
            A DocumentRow ready for persistence.
        """
        from ..db.models import DocumentRow

        return DocumentRow(
            id=_gen_uuid(),
            source_id=source_id,
            origin=self.origin,
            source_type=self.source_type,
            url=self.url,
            title=self.title,
            author=self.author,
            published_at=self.published_at.isoformat() if self.published_at else None,
            content_type=self.content_type,
            raw_content=self.raw_content,
            word_count=self.word_count,
            metadata=self.metadata,
            filter_score=self.filter_score,
        )


class BaseIngestor(ABC):
    """Abstract base for all source ingestors.

    Each source (HN, Substack, YouTube, Reddit, ArXiv, RSS, DEV.to) implements
    this interface. The caller is responsible for URL deduplication when
    persisting results to the database.
    """

    def __init__(self, config: "AppConfig") -> None:
        """Initialize the ingestor with application config.

        Args:
            config: The application configuration (see config.models.AppConfig).
        """
        self.config = config

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the source type identifier for this ingestor.

        Must match the source_type values used in the sources table.
        Examples: 'hn', 'substack', 'youtube', 'reddit', 'rss', 'arxiv', 'devto'.
        """
        ...

    @abstractmethod
    async def fetch_pro(self) -> list[RawDocument]:
        """Pull latest content from configured subscriptions.

        Called on a schedule for pro-tier ingestion.
        Returns deduplicated RawDocuments (URL uniqueness enforced by caller).
        """
        ...

    @abstractmethod
    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search this source for a query (Radar mode).

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            Up to `limit` results, sorted by relevance.
        """
        ...

    async def fetch_content(self, doc: RawDocument) -> RawDocument:
        """Fetch and populate raw_content for a doc that only has a URL.

        Default implementation uses ContentExtractor. Override for sources that
        return full content in their API (e.g. ArXiv abstracts, which already
        include the abstract text without needing to scrape the page).

        Args:
            doc: A RawDocument that may have raw_content=None.

        Returns:
            Updated RawDocument with raw_content, word_count, and title populated.
        """
        if doc.raw_content:
            return doc
        from ..processing.extractor import ContentExtractor

        async with ContentExtractor() as extractor:
            extracted = await extractor.fetch_and_extract(doc.url)
        return doc.model_copy(
            update={
                "raw_content": extracted.text,
                "word_count": extracted.word_count,
                "title": doc.title or extracted.title,
            }
        )
