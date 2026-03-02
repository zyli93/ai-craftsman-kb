"""Reddit ingestor using the official Reddit OAuth2 API.

Pro mode: fetches hot/new/top posts from configured subreddits.
Radar mode: searches Reddit for a given query across all subreddits.

Authentication uses Reddit's script app OAuth2 flow (client credentials).
The token is cached and automatically refreshed when expired.

No PRAW dependency — all calls are direct httpx requests to the Reddit API.
"""
import asyncio
import base64
import logging
import time
from datetime import datetime, timezone

import httpx

from ..config.models import AppConfig
from .base import BaseIngestor, RawDocument

logger = logging.getLogger(__name__)

REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_BASE = "https://oauth.reddit.com"
USER_AGENT = "ai-craftsman-kb/1.0"

# Rate limit: 100 req/min per OAuth client → 0.6 s between requests
_RATE_LIMIT_DELAY = 0.6

# Minimum selftext length to consider a self-post worth storing
_MIN_SELFTEXT_LEN = 50


class RedditIngestor(BaseIngestor):
    """Ingestor for Reddit via the official OAuth2 REST API.

    Pro mode: for each subreddit configured in config.sources.subreddits,
    fetches posts using the specified sort (hot/new/top/rising) and limit.
    Self-posts use selftext as raw_content; link posts leave raw_content=None
    for the caller to populate via ContentExtractor.

    Radar mode: searches Reddit's /search endpoint with the given query and
    returns results sorted by relevance.

    Token is fetched once and cached; it is refreshed automatically before
    expiry.
    """

    source_type = "reddit"

    def __init__(self, config: AppConfig) -> None:
        """Initialize the Reddit ingestor.

        Args:
            config: Application configuration. Reddit credentials are read from
                    config.settings.reddit.client_id and .client_secret.
        """
        super().__init__(config)
        self._token: str | None = None
        # Monotonic timestamp at which the cached token expires
        self._token_expires_at: float = 0.0
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": USER_AGENT},
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_pro(self) -> list[RawDocument]:
        """Fetch posts from all configured subreddits.

        Iterates over config.sources.subreddits, authenticates (fetching a
        cached OAuth token), and requests each subreddit's feed. A 0.6 s
        delay is inserted between subreddit requests to respect Reddit's
        100 req/min rate limit.

        Self-posts with deleted or too-short selftext are skipped. Link posts
        have raw_content=None and must be fetched by the runner via
        fetch_content() / ContentExtractor.

        Returns:
            Combined list of RawDocuments across all configured subreddits.
            Returns [] if credentials are missing or subreddit list is empty.
        """
        if not self._has_credentials():
            logger.warning(
                "Reddit credentials not configured — skipping pro ingestion. "
                "Set settings.reddit.client_id and .client_secret in your config."
            )
            return []

        subreddits = self.config.sources.subreddits
        if not subreddits:
            return []

        docs: list[RawDocument] = []
        for i, subreddit_cfg in enumerate(subreddits):
            if i > 0:
                # Rate-limit delay between subreddit requests
                await asyncio.sleep(_RATE_LIMIT_DELAY)

            try:
                token = await self._authenticate()
                posts = await self._fetch_subreddit_posts(
                    token=token,
                    subreddit=subreddit_cfg.name,
                    sort=subreddit_cfg.sort,
                    limit=subreddit_cfg.limit,
                )
            except httpx.HTTPError as e:
                logger.error(
                    "Reddit pro fetch failed for r/%s: %s",
                    subreddit_cfg.name,
                    e,
                )
                continue

            min_upvotes = self.config.filters.reddit.min_upvotes or 0
            for post in posts:
                doc = self._post_to_raw_doc(post)
                if doc is None:
                    continue
                if doc.metadata.get("upvotes", 0) < min_upvotes:
                    continue
                docs.append(doc)

        return docs

    async def search_radar(self, query: str, limit: int = 20) -> list[RawDocument]:
        """Search Reddit for the given query (Radar mode).

        Uses Reddit's /search endpoint with sort=relevance and type=link.
        Searches across all subreddits (no subreddit restriction).

        Args:
            query: The search query string.
            limit: Maximum number of results (capped at 100 by Reddit).

        Returns:
            List of RawDocuments with origin='radar'. Returns [] if credentials
            are missing or the API call fails.
        """
        if not self._has_credentials():
            logger.warning(
                "Reddit credentials not configured — skipping radar search. "
                "Set settings.reddit.client_id and .client_secret in your config."
            )
            return []

        try:
            token = await self._authenticate()
        except httpx.HTTPError as e:
            logger.error("Reddit authentication failed during radar search: %s", e)
            return []

        params = {
            "q": query,
            "sort": "relevance",
            "limit": min(limit, 100),
            "type": "link",
        }
        try:
            resp = await self._client.get(
                f"{REDDIT_API_BASE}/search",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error("Reddit radar search failed for query '%s': %s", query, e)
            return []

        docs: list[RawDocument] = []
        children = data.get("data", {}).get("children", [])
        for child in children[:limit]:
            post = child.get("data", {})
            doc = self._post_to_raw_doc(post, origin="radar")
            if doc is not None:
                docs.append(doc)

        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _authenticate(self) -> str:
        """Obtain a valid OAuth2 bearer token for the Reddit API.

        Uses the client-credentials flow (script app). The token is cached
        in self._token and reused until within 60 seconds of expiry, at which
        point a new token is fetched.

        Returns:
            A valid bearer token string.

        Raises:
            httpx.HTTPError: If the token endpoint returns a non-2xx status or
                             is unreachable.
        """
        now = time.monotonic()
        # Refresh 60 s before actual expiry to avoid using a token that is
        # about to expire mid-request.
        if self._token is not None and now < self._token_expires_at - 60:
            return self._token

        client_id = self.config.settings.reddit.client_id or ""
        client_secret = self.config.settings.reddit.client_secret or ""
        credentials = f"{client_id}:{client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        resp = await self._client.post(
            REDDIT_TOKEN_URL,
            headers={
                "Authorization": f"Basic {encoded}",
                "User-Agent": USER_AGENT,
            },
            data={
                "grant_type": "client_credentials",
                "device_id": "DO_NOT_TRACK_THIS_DEVICE",
            },
        )
        resp.raise_for_status()
        payload = resp.json()

        self._token = payload["access_token"]
        expires_in = payload.get("expires_in", 3600)
        self._token_expires_at = now + expires_in

        return self._token

    async def _fetch_subreddit_posts(
        self,
        token: str,
        subreddit: str,
        sort: str,
        limit: int,
    ) -> list[dict]:
        """Fetch raw post dicts from a single subreddit.

        Args:
            token: A valid Reddit bearer token.
            subreddit: The subreddit name (without the r/ prefix).
            sort: Feed sort order — one of 'hot', 'new', 'top', 'rising'.
            limit: Maximum number of posts to fetch (capped at 100).

        Returns:
            List of raw post data dicts from the Reddit API response.

        Raises:
            httpx.HTTPError: On network or HTTP errors.
        """
        params: dict[str, str | int] = {"limit": min(limit, 100)}
        # 'top' sort requires a time filter; use 'day' to get daily top posts
        if sort == "top":
            params["t"] = "day"

        url = f"{REDDIT_API_BASE}/r/{subreddit}/{sort}"
        resp = await self._client.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        children = data.get("data", {}).get("children", [])
        return [child["data"] for child in children]

    def _post_to_raw_doc(
        self,
        post: dict,
        origin: str = "pro",
    ) -> RawDocument | None:
        """Map a Reddit API post dict to a RawDocument.

        For self-posts (is_self=True): raw_content is set from selftext (unless
        it is empty, '[deleted]', or shorter than _MIN_SELFTEXT_LEN characters,
        in which case None is returned to skip the post).

        For link posts (is_self=False): raw_content is None so the caller
        (IngestRunner) will invoke fetch_content() / ContentExtractor.

        The URL is always the canonical Reddit permalink:
        ``https://reddit.com{permalink}``

        The metadata dict contains:
        - subreddit: str — the subreddit name
        - post_id: str — the Reddit post ID (e.g. 'abc123')
        - upvotes: int — the post score/upvotes
        - comment_count: int — number of comments
        - is_self: bool — True for self/text posts, False for link posts
        - linked_url: str | None — the external URL for link posts, None for self-posts

        Args:
            post: A Reddit API post data dict (the 'data' key from a listing child).
            origin: Ingest origin label ('pro' or 'radar').

        Returns:
            A RawDocument, or None if the post should be skipped (e.g. deleted
            self-post with trivially short selftext).
        """
        post_id = post.get("id", "")
        permalink = post.get("permalink", "")
        # Canonical Reddit URL for the post discussion thread
        if permalink:
            url = f"https://reddit.com{permalink}"
        else:
            url = f"https://reddit.com/r/unknown/comments/{post_id}/"

        is_self: bool = bool(post.get("is_self", False))
        selftext: str = post.get("selftext", "") or ""

        if is_self:
            # Skip deleted or trivially short self-posts
            if selftext in ("[deleted]", "[removed]", "") or len(selftext) < _MIN_SELFTEXT_LEN:
                return None
            raw_content: str | None = selftext
            content_type = "post"
        else:
            raw_content = None
            content_type = "article"

        # Parse unix UTC timestamp from Reddit API (created_utc is a float)
        published_at: datetime | None = None
        created_utc = post.get("created_utc")
        if created_utc is not None:
            try:
                published_at = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
            except (ValueError, OSError):
                logger.debug("Could not parse created_utc timestamp: %s", created_utc)

        # linked_url is the external URL for link posts; None for self-posts
        linked_url: str | None = None
        if not is_self:
            linked_url = post.get("url") or None

        return RawDocument(
            url=url,
            title=post.get("title"),
            author=post.get("author"),
            raw_content=raw_content,
            content_type=content_type,
            published_at=published_at,
            source_type="reddit",
            origin=origin,
            metadata={
                "subreddit": post.get("subreddit", ""),
                "post_id": post_id,
                "upvotes": post.get("score", 0) or 0,
                "comment_count": post.get("num_comments", 0) or 0,
                "is_self": is_self,
                "linked_url": linked_url,
            },
        )

    def _has_credentials(self) -> bool:
        """Return True if Reddit OAuth credentials are present in config.

        Returns:
            True when both client_id and client_secret are non-empty strings.
        """
        reddit_cfg = self.config.settings.reddit
        return bool(reddit_cfg.client_id and reddit_cfg.client_secret)

    async def __aenter__(self) -> "RedditIngestor":
        """Enter async context manager, returning self."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager, closing the httpx client."""
        await self._client.aclose()
