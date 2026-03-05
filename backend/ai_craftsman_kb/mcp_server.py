"""MCP server for AI Craftsman KB.

Exposes all knowledge-base capabilities as MCP tools so AI agents (e.g. Claude
Desktop) can search, ingest, manage sources, and generate briefings directly.

The MCP server reuses the same service layer as the FastAPI server — the same
IngestRunner, HybridSearch, RadarEngine, and BriefingGenerator classes. Each
tool call opens a fresh DB connection to avoid cross-call state sharing.

Run via stdio transport (for Claude Desktop)::

    uv run python -m ai_craftsman_kb.cli mcp

Or directly::

    uv run python -c "from ai_craftsman_kb.mcp_server import run_mcp_server; \\
        from ai_craftsman_kb.config import load_config; \\
        run_mcp_server(load_config())"
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from .config.loader import load_config
from .config.models import AppConfig
from .db.sqlite import get_db, init_db
from .ingestors.runner import INGESTORS, IngestRunner
from .llm.router import LLMRouter
from .processing.embedder import Embedder
from .radar.engine import RadarEngine
from .search.hybrid import HybridSearch
from .search.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Module-level FastMCP instance — populated by create_mcp_server()
_mcp: FastMCP | None = None
_config: AppConfig | None = None
_db_path: Path | None = None
_vector_store: VectorStore | None = None
_embedder: Embedder | None = None
_llm_router: LLMRouter | None = None


def create_mcp_server(config: AppConfig) -> FastMCP:
    """Create and configure the MCP server with shared state.

    Initialises shared service objects (VectorStore, Embedder, LLMRouter) once
    and stores them as module-level singletons so each tool call can reuse them
    without incurring initialisation overhead. DB connections are opened per
    tool call via the async context manager to avoid cross-call state sharing.

    Args:
        config: Fully loaded AppConfig.

    Returns:
        A configured FastMCP instance with all tools registered.
    """
    global _mcp, _config, _db_path, _vector_store, _embedder, _llm_router

    _config = config
    _db_path = Path(config.settings.data_dir).expanduser().resolve() / "craftsman.db"
    _vector_store = VectorStore(config)
    _embedder = Embedder(config)
    _llm_router = LLMRouter(config)

    mcp = FastMCP("ai-craftsman-kb")
    _mcp = mcp

    # -----------------------------------------------------------------------
    # Register all tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    async def search(
        query: str,
        mode: str = "hybrid",
        sources: list[str] | None = None,
        since: str | None = None,
        limit: int = 20,
        entity_type: str | None = None,
    ) -> list[dict]:
        """Search indexed content across all sources.

        Performs hybrid, semantic, or keyword search over all ingested documents.
        Results are ranked by relevance and returned as a list of document summaries.

        Args:
            query: The search query string (required, non-empty).
            mode: Search mode — 'hybrid' (default, combines FTS5 + vector),
                  'semantic' (vector only), or 'keyword' (FTS5 only).
            sources: Optional list of source types to restrict results to,
                     e.g. ['hn', 'arxiv', 'substack'].
            since: Optional ISO 8601 date string (e.g. '2025-01-01') to filter
                   documents published on or after this date.
            limit: Maximum number of results to return (default 20).
            entity_type: Unused in search; reserved for future entity-filtered search.

        Returns:
            List of dicts, each with keys:
                id, title, url, source_type, author, published_at, excerpt, score.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")

        valid_modes = {"hybrid", "semantic", "keyword"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of: {sorted(valid_modes)}")

        await _ensure_db_initialised()

        searcher = HybridSearch(
            config=_config,
            vector_store=_vector_store,
            embedder=_embedder,
        )

        async with get_db(_db_path.parent) as conn:
            results = await searcher.search(
                conn,
                query=query,
                mode=mode,
                source_types=sources,
                since=since,
                limit=limit,
            )

        return [
            {
                "id": r.document_id,
                "title": r.title,
                "url": r.url,
                "source_type": r.source_type,
                "author": r.author,
                "published_at": r.published_at,
                "excerpt": r.excerpt,
                "score": r.score,
            }
            for r in results
        ]

    @mcp.tool()
    async def radar(
        query: str,
        sources: list[str] | None = None,
        since: str | None = None,
        max_results_per_source: int = 10,
    ) -> list[dict]:
        """Search the open web for a topic, ingest and index results.

        Fans out a search query to all radar-capable sources (HN, Reddit, ArXiv,
        YouTube, DEV.to) concurrently, deduplicates results, and stores new
        documents with origin='radar'. Useful for on-demand topic research that
        goes beyond the regularly ingested pro-tier content.

        Args:
            query: The topic or search query to look up across sources.
            sources: Optional list of source types to restrict the search to,
                     e.g. ['hn', 'arxiv']. All supported sources are searched
                     by default.
            since: Optional ISO 8601 date string — not currently applied to
                   radar results but reserved for future filtering.
            max_results_per_source: Maximum documents to fetch per source
                                    (default 10).

        Returns:
            List of newly found document dicts, each with keys:
                id, title, url, source_type, author, published_at, excerpt.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")

        await _ensure_db_initialised()

        ingestors = {st: cls(_config) for st, cls in INGESTORS.items()}
        engine = RadarEngine(config=_config, ingestors=ingestors)

        async with get_db(_db_path.parent) as conn:
            report = await engine.search(
                conn,
                query=query,
                sources=sources,
                limit_per_source=max_results_per_source,
            )

        logger.info(
            "Radar search for %r: %d found, %d new, errors=%s",
            query,
            report.total_found,
            report.new_documents,
            report.errors,
        )

        # Fetch the newly stored radar documents to return them
        async with get_db(_db_path.parent) as conn:
            async with conn.execute(
                """
                SELECT id, title, url, source_type, author, published_at, raw_content
                FROM documents
                WHERE origin = 'radar'
                  AND deleted_at IS NULL
                ORDER BY fetched_at DESC
                LIMIT ?
                """,
                (report.new_documents,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            {
                "id": row[0],
                "title": row[1],
                "url": row[2],
                "source_type": row[3],
                "author": row[4],
                "published_at": row[5],
                "excerpt": (row[6] or "")[:300] if row[6] else None,
            }
            for row in rows
        ]

    @mcp.tool()
    async def ingest(
        source_type: str | None = None,
    ) -> dict:
        """Pull latest content from pro-tier sources.

        Runs the full ingestion pipeline (fetch -> filter -> dedup -> store ->
        embed) for all configured pro-tier sources, or just one if source_type
        is specified. This is the equivalent of running 'cr ingest pro'.

        Args:
            source_type: Optional source type to restrict ingestion to, e.g.
                         'hn', 'substack', 'arxiv'. If None, all configured
                         sources are ingested.

        Returns:
            Dict with aggregated counts:
                fetched, stored, embedded, errors (list of error strings).

        Raises:
            ValueError: If source_type is specified but not a known source type.
        """
        if source_type is not None and source_type not in INGESTORS:
            raise ValueError(
                f"Unknown source_type: '{source_type}'. "
                f"Available: {sorted(INGESTORS.keys())}"
            )

        await _ensure_db_initialised()

        runner = IngestRunner(
            config=_config,
            llm_router=_llm_router,
            db_path=_db_path,
        )

        if source_type is not None:
            ingestor_cls = INGESTORS[source_type]
            ingestor = ingestor_cls(_config)
            reports = [await runner.run_source(ingestor)]
        else:
            reports, _skipped = await runner.run_all()

        total_fetched = sum(r.fetched for r in reports)
        total_stored = sum(r.stored for r in reports)
        total_embedded = sum(r.embedded for r in reports)
        all_errors: list[str] = []
        for r in reports:
            all_errors.extend(r.errors)

        return {
            "fetched": total_fetched,
            "stored": total_stored,
            "embedded": total_embedded,
            "errors": all_errors,
        }

    @mcp.tool()
    async def ingest_url(
        url: str,
        tags: list[str] | None = None,
    ) -> dict:
        """Ingest a single URL into the knowledge base.

        Automatically detects the URL type (YouTube video, ArXiv paper, web
        article, etc.) and extracts its content. The document is stored,
        embedded, and indexed for future search queries.

        Args:
            url: The full URL to ingest (must include http:// or https://).
            tags: Optional list of tag strings to apply to the document,
                  e.g. ['must-read', 'rag', 'research'].

        Returns:
            Dict with the ingested document metadata:
                id, title, url, source_type.

        Raises:
            ValueError: If the URL is empty or ingestion fails.
            RuntimeError: If the document could not be stored.
        """
        if not url.strip():
            raise ValueError("url must be non-empty")

        await _ensure_db_initialised()

        runner = IngestRunner(
            config=_config,
            llm_router=_llm_router,
            db_path=_db_path,
        )

        report = await runner.ingest_url(url, tags=tags)

        if report.errors:
            raise RuntimeError(f"Ingestion failed: {'; '.join(report.errors)}")

        # Fetch the stored document to return its metadata
        from .db.queries import get_document_by_url

        async with get_db(_db_path.parent) as conn:
            doc = await get_document_by_url(conn, url)

        if doc is None:
            raise RuntimeError("Document was not stored despite no errors")

        return {
            "id": doc.id,
            "title": doc.title,
            "url": doc.url,
            "source_type": doc.source_type,
        }

    @mcp.tool()
    async def briefing(
        topic: str,
        run_radar: bool = True,
        run_ingest: bool = True,
    ) -> dict:
        """Generate a content briefing on a topic.

        Optionally runs a radar search and/or a fresh pro ingest before
        generating, so the briefing reflects the most current content available.
        Uses the BriefingGenerator to search for relevant documents and
        synthesise a structured briefing via LLM.

        Args:
            topic: The topic or question to generate a briefing about.
            run_radar: If True (default), run a radar search on the topic first
                       to pull fresh content from the open web.
            run_ingest: If True (default), run a pro ingest before generating
                        to refresh subscribed sources. Can be slow (30-60s).

        Returns:
            Dict with briefing content:
                title, content (markdown), source_count, created_at.

        Raises:
            ValueError: If topic is empty.
            RuntimeError: If briefing generation fails.
        """
        if not topic.strip():
            raise ValueError("topic must be non-empty")

        await _ensure_db_initialised()

        # Optionally run radar search first
        if run_radar:
            try:
                ingestors = {st: cls(_config) for st, cls in INGESTORS.items()}
                engine = RadarEngine(config=_config, ingestors=ingestors)
                async with get_db(_db_path.parent) as conn:
                    await engine.search(conn, query=topic, limit_per_source=5)
            except Exception as e:
                logger.warning("Radar pre-search for briefing failed: %s", e)

        # Optionally run pro ingest
        if run_ingest:
            try:
                runner = IngestRunner(
                    config=_config,
                    llm_router=_llm_router,
                    db_path=_db_path,
                )
                await runner.run_all()  # return value unused for briefing pre-ingest
            except Exception as e:
                logger.warning("Pro ingest for briefing failed: %s", e)

        # Generate the briefing using BriefingGenerator
        from .briefing.generator import BriefingGenerator
        from .db.queries import insert_briefing

        hybrid_search = HybridSearch(
            config=_config,
            vector_store=_vector_store,
            embedder=_embedder,
        )
        ingestors = {st: cls(_config) for st, cls in INGESTORS.items()}
        radar_engine = RadarEngine(config=_config, ingestors=ingestors)
        ingest_runner = IngestRunner(
            config=_config,
            llm_router=_llm_router,
            db_path=_db_path,
        )

        generator = BriefingGenerator(
            config=_config,
            llm_router=_llm_router,
            hybrid_search=hybrid_search,
            radar_engine=radar_engine,
            ingest_runner=ingest_runner,
        )

        try:
            async with get_db(_db_path.parent) as conn:
                briefing_row = await generator.generate(
                    conn,
                    topic=topic,
                    run_radar=False,  # already handled above
                    run_ingest=False,  # already handled above
                    limit=20,
                )
        except Exception as e:
            logger.error("Briefing generation failed: %s", e)
            raise RuntimeError(f"Briefing generation failed: {e}") from e

        return {
            "title": briefing_row.title,
            "content": briefing_row.content,
            "source_count": len(briefing_row.source_document_ids),
            "created_at": briefing_row.created_at,
        }

    @mcp.tool()
    async def get_entities(
        query: str | None = None,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search and browse entities extracted from ingested documents.

        Entities are named concepts (people, companies, technologies, etc.)
        automatically extracted from document content. Use this to discover
        what subjects appear most frequently in your knowledge base.

        Args:
            query: Optional search string to filter entities by name. Uses
                   full-text search (FTS5) when provided.
            entity_type: Optional type filter. Must be one of:
                         'person', 'company', 'technology', 'event', 'book',
                         'paper', 'product'.
            limit: Maximum number of entities to return (default 20).

        Returns:
            List of entity dicts, each with keys:
                id, name, entity_type, mention_count.
        """
        valid_types = {
            None, "person", "company", "technology", "event",
            "book", "paper", "product",
        }
        if entity_type not in valid_types:
            raise ValueError(
                f"entity_type must be one of: {sorted(t for t in valid_types if t)}"
            )

        await _ensure_db_initialised()

        from .db.queries import search_entities_fts
        from .db.models import EntityRow
        from .db.queries import _row_to_dict

        async with get_db(_db_path.parent) as conn:
            if query:
                # FTS5 search
                entities = await search_entities_fts(conn, query, limit=limit + 20)
                if entity_type:
                    entities = [e for e in entities if e.entity_type == entity_type]
                entities = entities[:limit]
            else:
                # List all, ordered by mention_count
                conditions: list[str] = []
                params: list = []
                if entity_type:
                    conditions.append("entity_type = ?")
                    params.append(entity_type)
                where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
                params.extend([limit])
                async with conn.execute(
                    f"SELECT * FROM entities {where} ORDER BY mention_count DESC LIMIT ?",  # noqa: S608
                    params,
                ) as cursor:
                    rows = await cursor.fetchall()
                entities = [EntityRow(**_row_to_dict(row)) for row in rows]

        return [
            {
                "id": e.id,
                "name": e.name,
                "entity_type": e.entity_type,
                "mention_count": e.mention_count,
            }
            for e in entities
        ]

    @mcp.tool()
    async def get_stats() -> dict:
        """Get system statistics for the knowledge base.

        Returns aggregate counts for documents, entities, sources, embeddings,
        and vector store coverage. Useful for monitoring the health and
        completeness of the knowledge base.

        Returns:
            Dict with keys:
                total_documents: All ingested documents (pro + radar + adhoc).
                embedded_documents: Documents with vector embeddings in Qdrant.
                total_entities: All extracted named entities.
                active_sources: Configured pro-tier sources that are enabled.
                vector_count: Total vectors stored in Qdrant.
        """
        await _ensure_db_initialised()

        from .db.queries import get_stats as db_get_stats

        async with get_db(_db_path.parent) as conn:
            stats = await db_get_stats(conn)

            async with conn.execute(
                "SELECT COUNT(*) FROM sources WHERE enabled = TRUE"
            ) as cursor:
                row = await cursor.fetchone()
                active_sources = row[0] if row else 0

        vector_count = 0
        try:
            info = _vector_store.get_collection_info()
            vector_count = info.get("vectors_count", 0)
        except Exception as e:
            logger.debug("Could not get vector count: %s", e)

        return {
            "total_documents": stats.get("total_documents", 0),
            "embedded_documents": stats.get("embedded_documents", 0),
            "total_entities": stats.get("total_entities", 0),
            "active_sources": active_sources,
            "vector_count": vector_count,
        }

    @mcp.tool()
    async def manage_source(
        action: str,
        source_type: str,
        identifier: str,
        display_name: str | None = None,
    ) -> dict:
        """Add, remove, or modify a source subscription.

        Manages the pro-tier source list. Adding a source means it will be
        ingested on the next 'cr ingest pro' run. Disabling pauses ingestion
        without losing configuration.

        Args:
            action: What to do with the source. One of:
                    'add'    — create a new source subscription,
                    'remove' — permanently delete the source,
                    'enable' — re-enable a disabled source,
                    'disable' — pause ingestion for this source.
            source_type: The source type, e.g. 'hn', 'substack', 'youtube',
                         'reddit', 'arxiv', 'rss', 'devto'.
            identifier: Source-specific identifier (e.g. newsletter slug for
                        Substack, channel handle for YouTube, subreddit name
                        for Reddit).
            display_name: Optional human-readable label for the source.
                          Used in 'add' action; ignored for other actions.

        Returns:
            Dict with the updated source info:
                id, source_type, identifier, display_name, enabled.

        Raises:
            ValueError: If action is not one of the allowed values.
            RuntimeError: If the source is not found (for remove/enable/disable).
        """
        valid_actions = {"add", "remove", "enable", "disable"}
        if action not in valid_actions:
            raise ValueError(f"action must be one of: {sorted(valid_actions)}")

        await _ensure_db_initialised()

        from .db.models import SourceRow
        from .db.queries import upsert_source

        async with get_db(_db_path.parent) as conn:
            if action == "add":
                # Check for duplicates
                async with conn.execute(
                    "SELECT id FROM sources WHERE source_type = ? AND identifier = ?",
                    (source_type, identifier),
                ) as cursor:
                    existing = await cursor.fetchone()

                if existing is not None:
                    # Return existing source instead of creating a duplicate
                    async with conn.execute(
                        "SELECT * FROM sources WHERE source_type = ? AND identifier = ?",
                        (source_type, identifier),
                    ) as cursor:
                        row = await cursor.fetchone()

                    from .db.queries import _row_to_dict
                    source = SourceRow(**_row_to_dict(row))
                    return {
                        "id": source.id,
                        "source_type": source.source_type,
                        "identifier": source.identifier,
                        "display_name": source.display_name,
                        "enabled": source.enabled,
                    }

                source = SourceRow(
                    id=str(uuid.uuid4()),
                    source_type=source_type,
                    identifier=identifier,
                    display_name=display_name,
                    enabled=True,
                )
                await upsert_source(conn, source)
                logger.info(
                    "MCP: added source %s (%s/%s)",
                    source.id, source_type, identifier,
                )
                return {
                    "id": source.id,
                    "source_type": source.source_type,
                    "identifier": source.identifier,
                    "display_name": source.display_name,
                    "enabled": source.enabled,
                }

            # For remove/enable/disable, find the existing source
            async with conn.execute(
                "SELECT * FROM sources WHERE source_type = ? AND identifier = ?",
                (source_type, identifier),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                raise RuntimeError(
                    f"Source not found: source_type='{source_type}', "
                    f"identifier='{identifier}'"
                )

            from .db.queries import _row_to_dict
            source = SourceRow(**_row_to_dict(row))

            if action == "remove":
                await conn.execute(
                    "DELETE FROM sources WHERE id = ?", (source.id,)
                )
                await conn.commit()
                logger.info("MCP: removed source %s", source.id)
                return {
                    "id": source.id,
                    "source_type": source.source_type,
                    "identifier": source.identifier,
                    "display_name": source.display_name,
                    "enabled": False,
                }

            elif action == "enable":
                await conn.execute(
                    "UPDATE sources SET enabled = TRUE, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (source.id,),
                )
                await conn.commit()

            elif action == "disable":
                await conn.execute(
                    "UPDATE sources SET enabled = FALSE, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (source.id,),
                )
                await conn.commit()

            # Re-fetch the updated row
            async with conn.execute(
                "SELECT * FROM sources WHERE id = ?", (source.id,)
            ) as cursor:
                updated_row = await cursor.fetchone()

            updated = SourceRow(**_row_to_dict(updated_row))
            return {
                "id": updated.id,
                "source_type": updated.source_type,
                "identifier": updated.identifier,
                "display_name": updated.display_name,
                "enabled": updated.enabled,
            }

    @mcp.tool()
    async def tag_document(
        document_id: str,
        tags: list[str],
        action: str = "add",
    ) -> dict:
        """Tag or untag a document.

        Manages user-defined tags on documents. Tags are stored as a JSON
        array in the documents table and are returned alongside search results.

        Args:
            document_id: The UUID of the document to update.
            tags: List of tag strings to add, remove, or set. Tags are
                  case-sensitive and trimmed of whitespace.
            action: Tag operation to perform:
                    'add'  — append tags (duplicates are ignored),
                    'remove' — remove only the specified tags,
                    'set'  — replace all existing tags with the given list.

        Returns:
            Dict with the updated document metadata:
                id, title, url, source_type, user_tags.

        Raises:
            ValueError: If document_id or action is invalid.
            RuntimeError: If the document is not found.
        """
        valid_actions = {"add", "remove", "set"}
        if action not in valid_actions:
            raise ValueError(f"action must be one of: {sorted(valid_actions)}")

        if not document_id.strip():
            raise ValueError("document_id must be non-empty")

        # Normalise tags — strip whitespace, drop empty strings
        clean_tags = [t.strip() for t in tags if t.strip()]

        await _ensure_db_initialised()

        from .db.queries import get_document
        import json

        async with get_db(_db_path.parent) as conn:
            doc = await get_document(conn, document_id)
            if doc is None:
                raise RuntimeError(f"Document not found: '{document_id}'")

            existing_tags: list[str] = doc.user_tags or []

            if action == "add":
                new_tags = existing_tags + [t for t in clean_tags if t not in existing_tags]
            elif action == "remove":
                new_tags = [t for t in existing_tags if t not in clean_tags]
            else:  # set
                new_tags = clean_tags

            await conn.execute(
                "UPDATE documents SET user_tags = ? WHERE id = ?",  # noqa: S608
                (json.dumps(new_tags), document_id),
            )
            await conn.commit()

            # Re-fetch to get updated state
            updated = await get_document(conn, document_id)

        if updated is None:
            raise RuntimeError("Failed to retrieve updated document")

        return {
            "id": updated.id,
            "title": updated.title,
            "url": updated.url,
            "source_type": updated.source_type,
            "user_tags": updated.user_tags,
        }

    @mcp.tool()
    async def discover_sources(
        based_on: str = "recent",
        limit: int = 5,
    ) -> list[dict]:
        """Suggest new sources to follow based on existing knowledge base content.

        Analyses the content already in your knowledge base to identify sources
        that appear frequently or are referenced by the content you already read.
        Useful for expanding coverage of a topic area.

        Args:
            based_on: Basis for discovery:
                      'recent'   — analyse the most recently ingested documents,
                      'popular'  — analyse the most frequently referenced sources,
                      'entities' — look for sources related to top entities.
                      (All modes currently use the same discovery pipeline —
                      the parameter will drive more targeted heuristics in the future.)
            limit: Maximum number of source suggestions to return (default 5).

        Returns:
            List of suggestion dicts, each with keys:
                source_type, identifier, display_name, confidence, discovery_method.

        Raises:
            ValueError: If based_on is not a valid option.
        """
        valid_modes = {"recent", "popular", "entities"}
        if based_on not in valid_modes:
            raise ValueError(f"based_on must be one of: {sorted(valid_modes)}")

        await _ensure_db_initialised()

        from .db.queries import list_discovered_sources, list_documents
        from .processing.discoverer import SourceDiscoverer

        async with get_db(_db_path.parent) as conn:
            # Fetch recent documents to analyse for source discovery
            docs = await list_documents(conn, limit=50, include_archived=False)

            discoverer = SourceDiscoverer(config=_config, llm_router=_llm_router)

            # Run discovery — returns DiscoveredSourceRow list
            discovered = await discoverer.discover_from_documents(conn, docs)

            # Also query existing suggestions if discovery yields nothing new
            if not discovered:
                discovered_rows = await list_discovered_sources(conn, status="suggested")
                discovered = discovered_rows[:limit]

        return [
            {
                "source_type": d.source_type,
                "identifier": d.identifier,
                "display_name": d.display_name,
                "confidence": d.confidence,
                "discovery_method": d.discovery_method,
            }
            for d in discovered[:limit]
        ]

    return mcp


async def _ensure_db_initialised() -> None:
    """Ensure the SQLite database schema has been created.

    Idempotent — init_db() is safe to call multiple times. This is called
    at the start of every tool invocation to guarantee the schema exists,
    since the MCP server may start before the DB has been initialised.

    Raises:
        RuntimeError: If no config has been loaded (create_mcp_server not called).
    """
    if _db_path is None or _config is None:
        raise RuntimeError(
            "MCP server not configured. Call create_mcp_server(config) first."
        )
    await init_db(_db_path.parent)


def run_mcp_server(config: AppConfig) -> None:
    """Run the MCP server using stdio transport.

    Creates the MCP server, registers all tools, and starts listening on
    stdin/stdout. This is the standard transport for Claude Desktop integration.

    The server runs until the client disconnects or the process is killed.

    Args:
        config: Fully loaded AppConfig instance.
    """
    mcp = create_mcp_server(config)
    logger.info("Starting AI Craftsman KB MCP server (stdio transport)...")
    mcp.run(transport="stdio")
