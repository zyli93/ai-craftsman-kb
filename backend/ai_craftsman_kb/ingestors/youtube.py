"""YouTube ingestor using the YouTube Data API v3 and yt-dlp for subtitles."""
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

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


def _strip_vtt_timestamps(vtt_text: str) -> str:
    """Strip VTT/SRT timestamps, cue headers, and style tags from subtitle text.

    Args:
        vtt_text: Raw VTT or SRT subtitle content.

    Returns:
        Plain text with timestamps and formatting removed, space-joined.
    """
    # Remove WEBVTT header, NOTE blocks, style blocks
    lines = vtt_text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        line = line.strip()
        # Skip header, timestamp lines, cue identifiers, and empty lines
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if "-->" in line:
            continue
        # Skip numeric cue identifiers (SRT format)
        if line.isdigit():
            continue
        # Remove HTML-style tags like <c>, </c>, <00:01:02.345>
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


def _extract_text_from_json3(json3_data: dict) -> str:
    """Extract plain text from YouTube's JSON3 subtitle format.

    Args:
        json3_data: Parsed JSON3 subtitle dict with 'events' key.

    Returns:
        Space-joined plain text from all subtitle segments.
    """
    segments: list[str] = []
    for event in json3_data.get("events", []):
        segs = event.get("segs", [])
        for seg in segs:
            text = seg.get("utf8", "").strip()
            if text and text != "\n":
                segments.append(text)
    return " ".join(segments)


def _fetch_transcript_sync(
    video_id: str, langs: list[str], cookies_file: str | None = None
) -> str | None:
    """Synchronously fetch a YouTube transcript via yt-dlp subtitle extraction.

    Uses yt-dlp's extract_info() to discover available subtitles, then fetches
    the subtitle URL using yt-dlp's own HTTP handler (which handles browser
    impersonation and avoids 429 rate limits). Prefers human-uploaded subtitles;
    falls back to auto-generated.

    This is a blocking function designed to be run in a thread executor.

    Args:
        video_id: The YouTube video ID (e.g. 'dQw4w9WgXcQ').
        langs: Ordered list of preferred language codes (e.g. ['en']).
        cookies_file: Path to a Netscape-format cookies.txt file, or None.

    Returns:
        The transcript as a single space-joined string, or None if unavailable.
    """
    import yt_dlp  # type: ignore[import-untyped]

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts: dict = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "logger": logger,
        "remote_components": ["ejs:github"],
    }

    if cookies_file:
        cookie_path = Path(cookies_file).expanduser()
        if cookie_path.exists():
            ydl_opts["cookiefile"] = str(cookie_path)
            logger.info("YouTube cookies loaded from %s", cookie_path)
        else:
            logger.warning("YouTube cookies file not found: %s", cookie_path)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return None

            # Priority: human-uploaded subtitles first, then auto-generated
            human_subs = info.get("subtitles") or {}
            auto_subs = info.get("automatic_captions") or {}

            for lang in langs:
                for sub_dict in (human_subs, auto_subs):
                    formats = sub_dict.get(lang, [])
                    if not formats:
                        continue

                    # Prefer json3 format, fall back to vtt, then anything
                    sub_entry = None
                    for preferred_ext in ("json3", "vtt"):
                        for fmt in formats:
                            if fmt.get("ext") == preferred_ext:
                                sub_entry = fmt
                                break
                        if sub_entry:
                            break
                    if not sub_entry:
                        sub_entry = formats[0]

                    sub_url = sub_entry.get("url")
                    if not sub_url:
                        continue

                    # Use yt-dlp's own HTTP handler (handles impersonation, avoids 429)
                    raw = ydl.urlopen(sub_url).read().decode("utf-8")

                    sub_ext = sub_entry.get("ext", "")
                    if sub_ext == "json3" or raw.strip().startswith("{"):
                        try:
                            json3 = json.loads(raw)
                            text = _extract_text_from_json3(json3)
                        except json.JSONDecodeError:
                            text = _strip_vtt_timestamps(raw)
                    else:
                        text = _strip_vtt_timestamps(raw)

                    if text.strip():
                        return text.strip()

            logger.debug("No subtitles found for video %s in langs %s", video_id, langs)
            return None

    except Exception as exc:
        logger.debug("Transcript unavailable for video %s: %s", video_id, exc)
        return None


class YouTubeIngestor(BaseIngestor):
    """Ingestor for YouTube channel videos using the YouTube Data API v3.

    Pro mode: fetches recent videos for all configured channels and pulls
    transcripts via yt-dlp.

    Radar mode: searches YouTube by keyword and pulls transcripts for
    matching videos.

    Requires YOUTUBE_API_KEY in settings.youtube.api_key. Returns an empty
    list with a warning log if the key is missing, rather than crashing.

    Transcript fetching runs in a thread executor because yt-dlp
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
        self._cookies_file = config.settings.youtube.cookies_file

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
        """Search YouTube for a keyword query and pull transcripts concurrently.

        Over-fetches (limit * 2) to account for videos without available
        transcripts. Transcripts are fetched concurrently via asyncio.gather.
        Videos without transcripts are still included using their description
        snippet as fallback content with metadata['has_transcript']=False.

        Args:
            query: The keyword or phrase to search for on YouTube.
            limit: Maximum number of results to return.

        Returns:
            List of RawDocuments with origin='radar', sorted by API relevance.
            Returns [] if the API key is missing. Returns partial results on
            quotaExceeded error (whatever was collected before the error).
        """
        if not self._api_key:
            logger.warning("YouTube API key not configured — skipping YouTube radar search")
            return []

        # Over-fetch to fill the limit even after filtering out no-transcript videos
        fetch_count = min(limit * 2, 50)
        search_results = await self._search_videos(query, fetch_count)
        if not search_results:
            return []

        # Fetch all transcripts concurrently — I/O bound so asyncio.gather shines here
        transcript_tasks = [
            self._get_transcript_safe(r["id"]["videoId"]) for r in search_results
        ]
        transcripts = await asyncio.gather(*transcript_tasks)

        docs: list[RawDocument] = []
        fallback_count = 0  # count of no-transcript videos included as fallback
        for result, transcript in zip(search_results, transcripts):
            doc = self._snippet_to_raw_doc(result, transcript)
            # Prefer videos with transcripts; only include no-transcript videos
            # as fallback to fill up to limit // 2 slots.
            if transcript is not None:
                docs.append(doc)
            elif fallback_count < limit // 2:
                docs.append(doc)
                fallback_count += 1
            if len(docs) >= limit:
                break

        return docs

    async def _search_videos(self, query: str, limit: int) -> list[dict]:
        """Call the YouTube Data API v3 search.list endpoint.

        Logs a warning on quota exceeded and returns an empty list rather than
        raising, so the caller receives partial results gracefully.

        Args:
            query: The search query string.
            limit: Maximum number of video results to request (capped at 50).

        Returns:
            List of raw API item dicts, or [] on error.
        """
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
            return data.get("items", [])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                self._handle_quota_error(exc)
            else:
                logger.error(
                    "YouTube radar search failed (HTTP %d) for query '%s': %s",
                    exc.response.status_code,
                    query,
                    exc,
                )
            return []
        except httpx.HTTPError as exc:
            logger.error("YouTube radar search failed for query '%s': %s", query, exc)
            return []

    async def _get_transcript_safe(self, video_id: str) -> str | None:
        """Fetch a YouTube transcript, returning None on any error.

        Wraps _get_transcript() to guarantee no exception propagation, making it
        safe to use inside asyncio.gather() without cancelling other transcript
        fetches if one fails.

        Args:
            video_id: The YouTube video ID.

        Returns:
            The transcript as plain text, or None if unavailable for any reason.
        """
        try:
            return await self._get_transcript(video_id)
        except Exception as exc:
            logger.debug("_get_transcript_safe: error for %s: %s", video_id, exc)
            return None

    def _snippet_to_raw_doc(
        self,
        search_result: dict,
        transcript: str | None,
    ) -> RawDocument:
        """Convert a YouTube search result and optional transcript to a RawDocument.

        When a transcript is available it becomes raw_content; otherwise the
        video's description snippet is used as a content stub. This ensures the
        document is still indexable and filterable even without a transcript.

        Args:
            search_result: A single item dict from the YouTube search API response.
            transcript: The transcript text, or None if unavailable.

        Returns:
            A RawDocument with origin='radar', content_type='video', and
            metadata fields: video_id, channel_id, channel_title, description,
            thumbnail_url, has_transcript.
        """
        video_id: str = search_result.get("id", {}).get("videoId", "")
        snippet: dict = search_result.get("snippet", {})
        title: str | None = snippet.get("title")
        description: str = snippet.get("description", "")
        channel_title: str = snippet.get("channelTitle", "")
        channel_id: str = snippet.get("channelId", "")
        published_at = _parse_iso_timestamp(snippet.get("publishedAt"))
        thumbnails: dict = snippet.get("thumbnails", {})
        # Prefer medium > default thumbnail URL
        thumbnail_url: str | None = (
            thumbnails.get("medium", {}).get("url")
            or thumbnails.get("default", {}).get("url")
        )

        has_transcript = transcript is not None
        raw_content = transcript if has_transcript else (description or None)
        word_count = len(raw_content.split()) if raw_content else None

        return RawDocument(
            url=f"https://youtube.com/watch?v={video_id}",
            title=title,
            author=channel_title,
            raw_content=raw_content,
            content_type="video",
            published_at=published_at,
            source_type="youtube",
            origin="radar",
            word_count=word_count,
            metadata={
                "video_id": video_id,
                "channel_id": channel_id,
                "channel_title": channel_title,
                "description": description,
                "thumbnail_url": thumbnail_url,
                "has_transcript": has_transcript,
            },
        )

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

        Runs the synchronous yt-dlp call in the default thread
        pool executor so it does not block the event loop.

        Args:
            video_id: The YouTube video ID.

        Returns:
            The transcript as plain text, or None if unavailable.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _fetch_transcript_sync, video_id, self._transcript_langs, self._cookies_file
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
