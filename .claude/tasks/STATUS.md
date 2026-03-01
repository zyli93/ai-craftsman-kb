# Task Status Tracker

Last updated: [auto-updated by agents]

## Progress
- Total: 0/45 done
- Backend: 0/33 done
- Frontend: 0/12 done

## Wave Execution Order

### Wave 1 — Scaffolding (sequential)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_01 | Project scaffolding + uv setup | backend | 🔲 todo | task/01-scaffolding | none |

### Wave 2 — Foundation (4 parallel)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_02 | Config system (YAML loader + Pydantic) | backend | 🔲 todo | task/02-config | 01 |
| task_03 | SQLite schema + migrations + queries | backend | 🔲 todo | task/03-database | 01 |
| task_04 | LLM provider abstraction | backend | 🔲 todo | task/04-llm-providers | 01 |
| task_05 | Base ingestor interface + content extractor | backend | 🔲 todo | task/05-base-ingestor | 01 |

### Wave 3 — First Ingestor + CLI (3 parallel)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_06 | HN ingestor (pro + radar) | backend | 🔲 todo | task/06-hn-ingestor | 02, 03, 05 |
| task_07 | Content filter (LLM + keyword) | backend | 🔲 todo | task/07-content-filter | 04, 05 |
| task_08 | CLI skeleton (Click) | backend | 🔲 todo | task/08-cli | 02, 03 |

### Wave 4 — Phase 1 Integration (sequential)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_09 | Phase 1 integration + tests | integration | 🔲 todo | task/09-phase1-integration | 06, 07, 08 |

### Wave 5 — All Ingestors (6 parallel)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_10 | Substack ingestor | backend | 🔲 todo | task/10-substack | 05 |
| task_11 | RSS ingestor | backend | 🔲 todo | task/11-rss | 05 |
| task_12 | YouTube ingestor + transcripts | backend | 🔲 todo | task/12-youtube | 05 |
| task_13 | Reddit ingestor | backend | 🔲 todo | task/13-reddit | 05 |
| task_14 | ArXiv ingestor | backend | 🔲 todo | task/14-arxiv | 05 |
| task_15 | DEV.to ingestor | backend | 🔲 todo | task/15-devto | 05 |

### Wave 6 — Adhoc + Integration (sequential)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_16 | Adhoc URL ingestor | backend | 🔲 todo | task/16-adhoc-url | 05, 12 |
| task_17 | Phase 2 integration + incremental fetch | integration | 🔲 todo | task/17-phase2-integration | 10-16 |

### Wave 7 — Search Foundation (3 parallel)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_18 | Embedding pipeline (OpenAI + local) | backend | 🔲 todo | task/18-embeddings | 04 |
| task_19 | Chunking system | backend | 🔲 todo | task/19-chunking | none |
| task_22 | Entity extraction pipeline | backend | 🔲 todo | task/22-entity-extraction | 04 |

### Wave 8 — Vector + Entity Store (2 parallel)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_20 | Qdrant local setup + vector store | backend | 🔲 todo | task/20-qdrant | 18 |
| task_23 | Entity dedup + FTS search | backend | 🔲 todo | task/23-entity-search | 03, 22 |

### Wave 9 — Search Assembly (sequential)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_21 | Hybrid search (FTS + vector + RRF) | backend | 🔲 todo | task/21-hybrid-search | 03, 20 |
| task_24 | Auto-embed + extract on ingest hook | backend | 🔲 todo | task/24-ingest-pipeline | 18, 19, 20, 22 |
| task_25 | Search CLI enhancements | backend | 🔲 todo | task/25-search-cli | 08, 21, 23 |

### Wave 10 — Radar Core (sequential)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_26 | Radar engine orchestrator (async fan-out) | backend | 🔲 todo | task/26-radar-engine | 05 |

### Wave 11 — Radar Sources (2 parallel)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_27 | YouTube radar search + transcript pull | backend | 🔲 todo | task/27-youtube-radar | 12, 26 |
| task_28 | Reddit + HN + ArXiv + DEV.to radar | backend | 🔲 todo | task/28-multi-radar | 13, 14, 15, 26 |

### Wave 12 — Radar CLI (sequential)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_29 | Radar CLI + promote/archive/delete | backend | 🔲 todo | task/29-radar-cli | 08, 26 |

### Wave 13 — API + Dashboard Scaffold (2 parallel)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_30 | FastAPI REST API layer | backend | 🔲 todo | task/30-rest-api | 03, 21, 26 |
| task_32 | Dashboard scaffolding (Vite + Tailwind + shadcn) | frontend | 🔲 todo | task/32-dashboard-scaffold | none |

### Wave 14 — All Dashboard Pages (7 parallel)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_33 | Overview page (stats, health, recent docs) | frontend | 🔲 todo | task/33-overview-page | 30, 32 |
| task_34 | Source editor page (CRUD + YAML sync) | frontend | 🔲 todo | task/34-source-editor | 30, 32 |
| task_35 | Search + Radar page | frontend | 🔲 todo | task/35-search-page | 30, 32 |
| task_36 | Entity explorer page | frontend | 🔲 todo | task/36-entity-page | 30, 32 |
| task_37 | Document manager page | frontend | 🔲 todo | task/37-document-page | 30, 32 |
| task_38 | Adhoc URL ingest + source discovery UI | frontend | 🔲 todo | task/38-adhoc-discovery | 30, 32 |
| task_39 | Briefing builder page | frontend | 🔲 todo | task/39-briefing-page | 30, 32 |

### Wave 15 — MCP + Server + Briefing (3 parallel)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_31 | MCP server (Python SDK) | backend | 🔲 todo | task/31-mcp-server | 30 |
| task_40 | `cr server` command (FastAPI + dashboard) | backend | 🔲 todo | task/40-server-cmd | 30, 32 |
| task_41 | Briefing generator engine | backend | 🔲 todo | task/41-briefing-engine | 21, 26 |

### Wave 16 — Polish (4 parallel)

| Task | Description | Domain | Status | Branch | Dependencies |
|------|-------------|--------|--------|--------|-------------|
| task_42 | Source discovery engine | backend | 🔲 todo | task/42-source-discovery | 03 |
| task_43 | Rich CLI output + progress bars | backend | 🔲 todo | task/43-rich-cli | 08 |
| task_44 | Export functionality | backend | 🔲 todo | task/44-export | 21 |
| task_45 | Error resilience + `cr doctor` | backend | 🔲 todo | task/45-resilience | all prior |

## Status Legend
- 🔲 todo — not started
- 🔵 in-progress — claimed by a runner, do not claim again
- ✅ done — completed, branch ready for merge
- 🔀 merged — merged into main
- ❌ blocked — failed or has unresolved issue
