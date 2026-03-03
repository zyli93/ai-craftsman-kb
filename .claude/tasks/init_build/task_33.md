# Task 33: Overview Page (Stats, Health, Recent Docs)

## Wave
Wave 14 (parallel with tasks 34, 35, 36, 37, 38, 39; depends on tasks 30 and 32)
Domain: frontend

## Objective
Build the Overview dashboard page showing system stats, source health, and recent documents — the landing page users see when opening the dashboard.

## Scope

### Files to create/modify:
- `dashboard/src/pages/Overview.tsx` — Main page component
- `dashboard/src/components/StatsCards.tsx` — Stats cards grid
- `dashboard/src/components/SourceHealth.tsx` — Source health list
- `dashboard/src/hooks/useStats.ts` — Data fetching hook

### Key interfaces / implementation details:

**Wireframe** (from plan.md):
```
┌─────────────────────────────────────────────────────────┐
│  AI Craftsman KB Dashboard                    [Ingest Now] │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ 2,847        │ │ 14,203       │ │ 1,847        │    │
│  │ Documents    │ │ Entities     │ │ Embedded     │    │
│  │ ↑ 142 today  │ │ ↑ 89 today   │ │ 100%         │    │
│  └──────────────┘ └──────────────┘ └──────────────┘    │
│                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ 623 MB       │ │ 307 MB       │ │ 12           │    │
│  │ SQLite Size  │ │ Vector Store │ │ Active Srcs  │    │
│  └──────────────┘ └──────────────┘ └──────────────┘    │
│                                                          │
│  Source Health                                           │
│  ● Hacker News        Last pull: 2h ago    ✓ OK         │
│  ● Substack (3)       Last pull: 2h ago    ✓ OK         │
│  ● ArXiv              Last pull: 1d ago    ⚠ Stale      │
│  ● DEV.to             Last pull: 3d ago    ✗ Error      │
│                                                          │
│  Recent Documents                          [View All →]  │
│  "Scaling LLM Inference..."  │ Substack │ 2h ago         │
│  "New GRPO Paper Drops..."   │ ArXiv    │ 5h ago         │
└─────────────────────────────────────────────────────────┘
```

**API endpoints called**:
- `GET /api/stats` → `SystemStats`
- `GET /api/sources` → `Source[]` (for health status)
- `GET /api/documents?limit=5` → `Document[]` (recent docs)
- `POST /api/ingest/pro` → (triggered by "Ingest Now" button)

**Components**:

```typescript
// StatsCards.tsx
interface StatCardProps {
  label: string
  value: string | number
  subtext?: string
  icon?: React.ReactNode
}
function StatCard({ label, value, subtext, icon }: StatCardProps)
function StatsCards({ stats }: { stats: SystemStats })

// SourceHealth.tsx
interface SourceHealthProps {
  sources: Source[]
}
function SourceHealth({ sources }: SourceHealthProps)
// Status logic:
// - fetch_error != null → ✗ Error (red)
// - last_fetched_at > 48h ago → ⚠ Stale (yellow)
// - otherwise → ✓ OK (green)

// useStats.ts
function useStats() {
  return useQuery({ queryKey: ['stats'], queryFn: api.stats, refetchInterval: 30_000 })
}

// Overview.tsx
function Overview() {
  const { data: stats } = useStats()
  const { data: sources } = useQuery({ queryKey: ['sources'], queryFn: api.sources.list })
  const { data: recentDocs } = useQuery({
    queryKey: ['documents', 'recent'],
    queryFn: () => api.documents.list({ limit: 5 }),
  })
  const ingestMutation = useMutation({ mutationFn: () => api.ingest.pro() })
  ...
}
```

**shadcn/ui components** used: `Card`, `CardHeader`, `CardContent`, `Button`, `Badge`, `Separator`

**Auto-refresh**: `useStats` refetches every 30 seconds to keep counts current.

**"Ingest Now" button**: triggers `POST /api/ingest/pro`, shows loading state, toast on completion.

## Dependencies
- Depends on: task_30 (FastAPI `/api/stats`, `/api/sources`, `/api/documents`), task_32 (scaffold, api client, types)
- Packages needed: none new

## Acceptance Criteria
- [ ] Stats cards display all 6 metrics from `SystemStats`
- [ ] Source health shows ✓/⚠/✗ status for each configured source
- [ ] "Stale" threshold: last_fetched_at > 48 hours ago
- [ ] "Error" shown when `source.fetch_error` is not null
- [ ] Recent documents list shows last 5 documents with title + source_type + relative time
- [ ] "Ingest Now" button triggers `POST /api/ingest/pro` with loading spinner
- [ ] Stats auto-refresh every 30 seconds
- [ ] Responsive layout: cards wrap gracefully at narrow widths

## Notes
- Use `formatDistanceToNow` from `date-fns` (add to dependencies) for relative time display
- Source health status derived from `source.last_fetched_at` and `source.fetch_error` fields returned by `/api/sources`
- `db_size_bytes` from stats formatted as "623 MB" using a bytes formatter utility
- The "↑ 142 today" subtext requires a separate `since` parameter in the API — simplify: show `embedded_documents / total_documents` as percentage instead if "today" count isn't available
