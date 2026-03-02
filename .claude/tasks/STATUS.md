# Task Status Tracker

Last updated: [auto-updated by agents]

## Progress
- Total: 9/45 done
- Backend: 8/33 done (waves 1-3 merged)
- Frontend: 1/12 done (task_32 merged)

## Wave Execution Order

### Wave 1 — Scaffolding (sequential)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_01 | Project scaffolding + uv setup | backend | 1 | 🔀 merged | task/01-scaffolding | none |

### Wave 2 — Foundation (4 parallel)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_02 | Config system (YAML loader + Pydantic) | backend | 2 | 🔀 merged | task/02-config | 01 |
| task_03 | SQLite schema + migrations + queries | backend | 2 | 🔀 merged | task/03-database | 01 |
| task_04 | LLM provider abstraction | backend | 2 | 🔀 merged | task/04-llm-providers | 01 |
| task_05 | Base ingestor interface + content extractor | backend | 2 | 🔀 merged | task/05-base-ingestor | 01 |

### Wave 3 — First Ingestor + CLI (3 parallel)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_06 | HN ingestor (pro + radar) | backend | 3 | 🔀 merged | task/06-hn-ingestor | 02, 03, 05 |
| task_07 | Content filter (LLM + keyword) | backend | 3 | 🔀 merged | task/07-content-filter | 04, 05 |
| task_08 | CLI skeleton (Click) | backend | 3 | 🔀 merged | task/08-cli | 02, 03 |

### Wave 4 — Phase 1 Integration (sequential)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_09 | Phase 1 integration + tests | integration | 4 | 🔵 in-progress | task/09-phase1-integration | 06, 07, 08 |

### Wave 5 — All Ingestors (6 parallel)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_10 | Substack ingestor | backend | 5 | 🔲 todo | task/10-substack | 05 |
| task_11 | RSS ingestor | backend | 5 | 🔲 todo | task/11-rss | 05 |
| task_12 | YouTube ingestor + transcripts | backend | 5 | 🔲 todo | task/12-youtube | 05 |
| task_13 | Reddit ingestor | backend | 5 | 🔲 todo | task/13-reddit | 05 |
| task_14 | ArXiv ingestor | backend | 5 | 🔲 todo | task/14-arxiv | 05 |
| task_15 | DEV.to ingestor | backend | 5 | 🔲 todo | task/15-devto | 05 |

### Wave 6 — Adhoc + Integration (sequential)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_16 | Adhoc URL ingestor | backend | 6 | 🔲 todo | task/16-adhoc-url | 05, 12 |
| task_17 | Phase 2 integration + incremental fetch | integration | 6 | 🔲 todo | task/17-phase2-integration | 10-16 |

### Wave 7 — Search Foundation (3 parallel)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_18 | Embedding pipeline (OpenAI + local) | backend | 7 | 🔲 todo | task/18-embeddings | 04 |
| task_19 | Chunking system | backend | 7 | 🔲 todo | task/19-chunking | none |
| task_22 | Entity extraction pipeline | backend | 7 | 🔲 todo | task/22-entity-extraction | 04 |

### Wave 8 — Vector + Entity Store (2 parallel)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_20 | Qdrant local setup + vector store | backend | 8 | 🔲 todo | task/20-qdrant | 18 |
| task_23 | Entity dedup + FTS search | backend | 8 | 🔲 todo | task/23-entity-search | 03, 22 |

### Wave 9 — Search Assembly (sequential)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_21 | Hybrid search (FTS + vector + RRF) | backend | 9 | 🔲 todo | task/21-hybrid-search | 03, 20 |
| task_24 | Auto-embed + extract on ingest hook | backend | 9 | 🔲 todo | task/24-ingest-pipeline | 18, 19, 20, 22 |
| task_25 | Search CLI enhancements | backend | 9 | 🔲 todo | task/25-search-cli | 08, 21, 23 |

### Wave 10 — Radar Core (sequential)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_26 | Radar engine orchestrator (async fan-out) | backend | 10 | 🔲 todo | task/26-radar-engine | 05 |

### Wave 11 — Radar Sources (2 parallel)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_27 | YouTube radar search + transcript pull | backend | 11 | 🔲 todo | task/27-youtube-radar | 12, 26 |
| task_28 | Reddit + HN + ArXiv + DEV.to radar | backend | 11 | 🔲 todo | task/28-multi-radar | 13, 14, 15, 26 |

### Wave 12 — Radar CLI (sequential)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_29 | Radar CLI + promote/archive/delete | backend | 12 | 🔲 todo | task/29-radar-cli | 08, 26 |

### Wave 13 — API + Dashboard Scaffold (2 parallel)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_30 | FastAPI REST API layer | backend | 13 | 🔲 todo | task/30-rest-api | 03, 21, 26 |
| task_32 | Dashboard scaffolding (Vite + Tailwind + shadcn) | frontend | 13 | 🔀 merged | task/32-dashboard-scaffold | none |

### Wave 14 — All Dashboard Pages (7 parallel)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_33 | Overview page (stats, health, recent docs) | frontend | 14 | 🔲 todo | task/33-overview-page | 30, 32 |
| task_34 | Source editor page (CRUD + status) | frontend | 14 | 🔲 todo | task/34-source-editor | 30, 32 |
| task_35 | Search + Radar page | frontend | 14 | 🔲 todo | task/35-search-page | 30, 32 |
| task_36 | Entity explorer page | frontend | 14 | 🔲 todo | task/36-entity-page | 30, 32 |
| task_37 | Document manager page | frontend | 14 | 🔲 todo | task/37-document-page | 30, 32 |
| task_38 | Adhoc URL ingest + source discovery UI | frontend | 14 | 🔲 todo | task/38-adhoc-discovery | 30, 32 |
| task_39 | Briefing builder page | frontend | 14 | 🔲 todo | task/39-briefing-page | 30, 32 |

### Wave 15 — MCP + Server + Briefing (3 parallel)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_31 | MCP server (Python SDK) | backend | 15 | 🔲 todo | task/31-mcp-server | 30 |
| task_40 | `cr server` command (FastAPI + dashboard) | backend | 15 | 🔲 todo | task/40-server-cmd | 30, 32 |
| task_41 | Briefing generator engine | backend | 15 | 🔲 todo | task/41-briefing-engine | 21, 26 |

### Wave 16 — Polish (4 parallel)

| Task | Description | Domain | Wave | Status | Branch | Dependencies |
|------|-------------|--------|------|--------|--------|-------------|
| task_42 | Source discovery engine | backend | 16 | 🔲 todo | task/42-source-discovery | 03 |
| task_43 | Rich CLI output + progress bars | backend | 16 | 🔲 todo | task/43-rich-cli | 08 |
| task_44 | Export functionality | backend | 16 | 🔲 todo | task/44-export | 21 |
| task_45 | Error resilience + `cr doctor` | backend | 16 | 🔲 todo | task/45-resilience | all prior |

## Status Legend
- 🔲 todo — not started
- 🔵 in-progress — claimed by a runner, do not claim again
- ✅ done — completed, branch ready for merge
- 🔀 merged — merged into main
- ❌ blocked — failed or has unresolved issue
