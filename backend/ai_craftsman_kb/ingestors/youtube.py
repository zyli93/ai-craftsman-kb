"""YouTube ingestor using the YouTube Data API v3 and youtube-transcript-api."""
import asyncio
import logging
from datetime import datetime, timezone

import httpx

from ..config.models import AppConfig
from .base import BaseIngestor, RawDocument

logger = logging.getLogger(__name__)

BASE_URL = "https://www.googleapis.com/youtube/v3"


def _parse_iso_timestamp(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp string into a timezone-aware datetime.

    Args:
        ts: ISO-8601 string (e.g. '2025-01-15T10:00:00Z'), or None.

    Returns:
        A UTC-aware datetime, or None if ts is None or unparseable.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("Could not parse timestamp: %s", ts)
        return None


def _fetch_transcript_sync(video_id: str, langs: list[str]) -> str | None:
    """Synchronously fetch a YouTube transcript and join segments into plain text.

    This is a blocking function designed to be run in a thread executor because
    youtube-transcript-api is synchronous.

    Args:
        video_id: The YouTube video ID (e.g. 'dQw4w9WgXcQ').
        langs: Ordered list of preferred language codes (e.g. ['en']).

    Returns:
        The transcript as a single space-joined string, or None if unavailable.
    """
    try:
        from youtube_transcript_api import (  # type: ignore[import-untyped]
            NoTranscriptFound,
            TranscriptsDisabled,
            YouTubeTranscriptApi,
        )

        fetched = YouTubeTranscriptApi().fetch(video_id, languages=langs)
        return " ".join(snippet.text for snippet in fetched)
    except Exception as exc:
        # Covers NoTranscriptFound, TranscriptsDisabled, VideoUnavailable, and any
        # other error from the underlying HTTP request.
        logger.debug("Transcript unavailable for video %s: %s", video_id, exc)
        return None


class YouTubeIngestor(BaseIngestor):
    """Ingestor for YouTube channel videos using the YouTube Data API v3.

    Pro mode: fetches recent videos for all configured channels and pulls
    transcripts via youtube-transcript-api.

    Radar mode: searches YouTube by keyword and pulls transcripts for
    matching videos.

    Requires YOUTUBE_API_KEY in settings.youtube.api_key. Returns an empty
    list with a warning log if the key is missing, rather than crashing.

    Transcript fetching runs in a thread executor because youtube-transcript-api
    is synchronous. Transcripts unavailable due to language, privacy, or
    other constraints are logged at DEBUG level and produce raw_content=None.
    """

    source_type = "youtube"

    def __init__(self, config: AppConfig) -> None:
        """Initialize the ingestor with application config and httpx client.

        Args:
            config: The application configuration (see config.models.AppConfig).
        """
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=30.0,
            headers={"User-Agent": "ai-craftsman-kb/1.0"},
        )
        # In-memory cache: YouTube handle (e.g. "@AndrejKarpathy") → channel_id
        self._handle_cache: dict[str, str] = {}

    @property
    def _api_key(self) -> str | None:
        """Return the YouTube Data API key from settings, or None if absent."""
        return self.config.settings.youtube.api_key

    @property
    def _transcript_langs(self) -> list[str]:
        """Return the preferred transcript language codes from settings."""
        return self.config.settings.youtube.transcript_langs

    async def fetch_pro(self) -> list[RawDocument]:
        """Fetch recent videos for all configured YouTube channels.

        For each channel configured in config.sources.youtube_channels:
        1. Resolve the @handle to a channel_id (cached per session).
        2. Fetch recent videos via the YouTube search endpoint.
        3. Pull transcripts for each video.

        Returns an empty list if the API key is missing or if no channels
        are configured. Videos without available transcripts are still returned
        with raw_content=None.

        Returns:
            List of RawDocuments with source_type='youtube' and origin='pro'.
        """
        if not self._api_key:
            logger.warning("YouTube API key not configured — skipping YouTube pro fetch")
            return []

        channels = self.config.sources.youtube_channels
        if not channels:
            return []

        docs: list[RawDocument] = []
        for channel in channels:
            try:
                channel_id = await self._resolve_handle(channel.handle)
                if channel_id is None:
                    logger.warning(
                        "Could not resolve YouTube channel handle: %s", channel.handle
                    )
                    continue

                channel_docs = await self._fetch_channel_videos(
                    channel_id=channel_id,
                    channel_handle=channel.handle,
                )
                docs.extend(channel_docs)
            except Exception as exc:
                logger.error(
                    "Failed to fetch YouTube channel %s: %s", channel.handle, exc
                )

        return docs

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search YouTube for a keyword query and pull transcripts.

        Only videos with available transcripts are included in results.

        Args:
            query: The keyword or phrase to search for on YouTube.
            limit: Maximum number of results to return (capped at 50 per API).

        Returns:
            List of RawDocuments with origin='radar'. Returns [] if the API key
            is missing or on request failure.
        """
        if not self._api_key:
            logger.warning("YouTube API key not configured — skipping YouTube radar search")
            return []

        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": "relevance",
            "maxResults": min(limit, 50),
            "key": self._api_key,
        }

        try:
            resp = await self._client.get("/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            self._handle_quota_error(exc)
            logger.error("YouTube radar search failed for query '%s': %s", query, exc)
            return []
        except httpx.HTTPError as exc:
            logger.error("YouTube radar search failed for query '%s': %s", query, exc)
            return []

        docs: list[RawDocument] = []
        for item in data.get("items", [])[:limit]:
            doc = self._item_to_raw_doc(item, origin="radar")
            if doc is None:
                continue
            # Pull transcript; only include videos that have one
            transcript = await self._get_transcript(doc.metadata["video_id"])
            if transcript is not None:
                doc = doc.model_copy(
                    update={
                        "raw_content": transcript,
                        "word_count": len(transcript.split()),
                    }
                )
                docs.append(doc)

        return docs

    async def fetch_content(self, doc: RawDocument) -> RawDocument:
        """Override: use YouTube transcript instead of HTML extraction.

        If the document's metadata contains a video_id, fetches the transcript
        and populates raw_content + word_count. If the transcript is unavailable,
        returns the document unchanged (raw_content remains None).

        Args:
            doc: A RawDocument with metadata['video_id'] set.

        Returns:
            Updated RawDocument with raw_content and word_count if transcript
            is available, otherwise the original doc.
        """
        video_id = doc.metadata.get("video_id")
        if not video_id:
            return doc
        transcript = await self._get_transcript(video_id)
        if transcript:
            return doc.model_copy(
                update={
                    "raw_content": transcript,
                    "word_count": len(transcript.split()),
                }
            )
        return doc

    async def _get_transcript(self, video_id: str) -> str | None:
        """Fetch a YouTube transcript asynchronously using a thread executor.

        Runs the synchronous youtube-transcript-api call in the default thread
        pool executor so it does not block the event loop.

        Args:
            video_id: The YouTube video ID.

        Returns:
            The transcript as plain text, or None if unavailable.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _fetch_transcript_sync, video_id, self._transcript_langs
        )

    async def _resolve_handle(self, handle: str) -> str | None:
        """Resolve a YouTube channel @handle to a channel ID.

        Caches the mapping in _handle_cache so each handle is only resolved
        once per session (avoids burning API quota on repeated lookups).

        Args:
            handle: The channel handle including the '@' prefix (e.g. '@AndrejKarpathy').

        Returns:
            The channel ID string, or None if resolution fails.
        """
        if handle in self._handle_cache:
            return self._handle_cache[handle]

        params = {
            "part": "id",
            "forHandle": handle,
            "key": self._api_key,
        }
        try:
            resp = await self._client.get("/channels", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.error("Failed to resolve YouTube handle %s: %s", handle, exc)
            return None

        items = data.get("items", [])
        if not items:
            logger.warning("YouTube handle not found: %s", handle)
            return None

        channel_id: str = items[0]["id"]
        self._handle_cache[handle] = channel_id
        return channel_id

    async def _fetch_channel_videos(
        self,
        channel_id: str,
        channel_handle: str,
        max_results: int = 50,
    ) -> list[RawDocument]:
        """Fetch the most recent videos for a given channel_id.

        Args:
            channel_id: The YouTube channel ID.
            channel_handle: The @handle string (stored in metadata).
            max_results: Number of videos to fetch (max 50 per API call).

        Returns:
            List of RawDocuments with transcripts populated where available.
        """
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "order": "date",
            "type": "video",
            "maxResults": min(max_results, 50),
            "key": self._api_key,
        }

        try:
            resp = await self._client.get("/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            self._handle_quota_error(exc)
            logger.error(
                "Failed to fetch videos for channel %s (%s): %s",
                channel_handle,
                channel_id,
                exc,
            )
            return []
        except httpx.HTTPError as exc:
            logger.error(
                "Failed to fetch videos for channel %s (%s): %s",
                channel_handle,
                channel_id,
                exc,
            )
            return []

        docs: list[RawDocument] = []
        for item in data.get("items", []):
            doc = self._item_to_raw_doc(item, channel_handle=channel_handle, origin="pro")
            if doc is None:
                continue
            # Pull transcript; store None raw_content if unavailable (still store doc)
            transcript = await self._get_transcript(doc.metadata["video_id"])
            if transcript is not None:
                doc = doc.model_copy(
                    update={
                        "raw_content": transcript,
                        "word_count": len(transcript.split()),
                    }
                )
            docs.append(doc)

        return docs

    def _item_to_raw_doc(
        self,
        item: dict,
        channel_handle: str = "",
        origin: str = "pro",
    ) -> RawDocument | None:
        """Map a YouTube search API item to a RawDocument.

        Args:
            item: A single item dict from the YouTube search API response.
            channel_handle: The @handle of the channel, if known (may be empty
                            for radar results where we only have channelTitle).
            origin: The ingest origin ('pro' or 'radar').

        Returns:
            A RawDocument, or None if the item is missing a video ID.
        """
        # The search endpoint nests the video ID under id.videoId
        video_id: str | None = item.get("id", {}).get("videoId")
        if not video_id:
            return None

        snippet: dict = item.get("snippet", {})
        title: str | None = snippet.get("title")
        description: str = snippet.get("description", "")
        channel_title: str = snippet.get("channelTitle", "")
        channel_id: str = snippet.get("channelId", "")
        published_at = _parse_iso_timestamp(snippet.get("publishedAt"))

        return RawDocument(
            url=f"https://youtube.com/watch?v={video_id}",
            title=title,
            author=channel_title or channel_handle,
            raw_content=None,  # populated by _get_transcript
            content_type="video",
            published_at=published_at,
            source_type="youtube",
            origin=origin,  # type: ignore[arg-type]
            metadata={
                "video_id": video_id,
                "channel_handle": channel_handle,
                "channel_id": channel_id,
                "description": description,
            },
        )

    def _handle_quota_error(self, exc: httpx.HTTPStatusError) -> None:
        """Log a clear message when the YouTube API returns a quota exceeded error.

        Args:
            exc: The HTTP status error from httpx.
        """
        if exc.response.status_code == 403:
            try:
                body = exc.response.json()
                errors = body.get("error", {}).get("errors", [])
                for err in errors:
                    if err.get("reason") == "quotaExceeded":
                        logger.error(
                            "YouTube API quota exceeded. The 10,000 daily quota units "
                            "have been consumed. Results will be empty until quota resets."
                        )
                        return
            except Exception:
                pass

    async def __aenter__(self) -> "YouTubeIngestor":
        """Enter async context manager, returning self."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager, closing the httpx client."""
        await self._client.aclose()
