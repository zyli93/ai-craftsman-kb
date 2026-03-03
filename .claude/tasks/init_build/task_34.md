# Task 34: Source Editor Page (CRUD + Status)

## Wave
Wave 14 (parallel with tasks 33, 35, 36, 37, 38, 39)
Domain: frontend

## Objective
Build the Source Editor page where users view, add, enable/disable, and delete configured sources, and see source health with last fetch status.

## Scope

### Files to create/modify:
- `dashboard/src/pages/Sources.tsx` — Main page component
- `dashboard/src/components/SourceEditor.tsx` — Add/edit source dialog
- `dashboard/src/components/DiscoveredSources.tsx` — Discovered sources panel

### Key interfaces / implementation details:

**Wireframe** (from plan.md):
```
┌─────────────────────────────────────────────────────────┐
│  Sources                        [+ Add Source] [Import] │
├─────────────────────────────────────────────────────────┤
│  Filter: [All Types ▾]  [Enabled Only ☑]                │
│                                                          │
│  Substack                                                │
│  ● Andrej Karpathy    │ karpathy    │ ✓ │ [Edit] [↓]   │
│  ● Simon Willison     │ simonw...   │ ✓ │ [Edit] [↓]   │
│                                                          │
│  YouTube Channels                                        │
│  ● Andrej Karpathy    │ @Andrej..   │ ✓ │ [Edit] [↓]   │
│                                                          │
│  Discovered Sources (5 suggestions)      [Review All →] │
│  💡 @laborjack (YouTube) - found in 3 articles          │
│     [Add to Pro] [Dismiss]                               │
└─────────────────────────────────────────────────────────┘
```

**API endpoints called**:
- `GET /api/sources` → `Source[]`
- `POST /api/sources` → `Source` (add new source)
- `PUT /api/sources/{id}` → `Source` (enable/disable, rename)
- `DELETE /api/sources/{id}` → (remove source)
- `POST /api/sources/{id}/ingest` → (trigger single source ingest)
- `GET /api/discover` → discovered sources

**Components**:

```typescript
// Sources.tsx
function Sources() {
  const { data: sources } = useQuery({ queryKey: ['sources'], queryFn: api.sources.list })
  const { data: discovered } = useQuery({ queryKey: ['discover'], queryFn: api.discover })
  // Group sources by source_type for section headers
  // e.g. { substack: [...], youtube: [...], reddit: [...] }
}

// SourceEditor.tsx — Dialog for adding a new source
interface AddSourceDialogProps {
  open: boolean
  onClose: () => void
}
// Form fields:
// - source_type: Select (hn | substack | youtube | reddit | rss | arxiv | devto)
// - identifier: Input (slug for substack, @handle for youtube, subreddit name, URL for rss)
// - display_name: Input (optional friendly name)

// DiscoveredSources.tsx — Panel showing suggested sources
function DiscoveredSources({ discoveries }: { discoveries: DiscoveredSource[] })
// Each suggestion: [Add to Pro] → POST /api/sources, [Dismiss] → PUT discovered_source status='dismissed'
```

**Source grouping**: Sources grouped by `source_type` with a section header. Within each group, sorted by `display_name`.

**Enable/disable toggle**: Inline toggle switch (shadcn/ui `Switch` or `Checkbox`) — calls `PUT /api/sources/{id}` with `{enabled: !current}`.

**Ingest now button**: `[↓]` icon button per source row → calls `POST /api/sources/{id}/ingest`, shows loading state, invalidates queries on completion.

**shadcn/ui components**: `Table`, `TableRow`, `TableCell`, `Dialog`, `DialogContent`, `Select`, `Input`, `Button`, `Badge`, `Switch`, `Card`

## Dependencies
- Depends on: task_30 (source CRUD endpoints), task_32 (scaffold + api client)
- Packages needed: none new

## Acceptance Criteria
- [ ] Sources listed grouped by type (Substack, YouTube Channels, Reddit, etc.)
- [ ] Each source shows: display_name, identifier, enabled status, last_fetched_at, fetch_error
- [ ] Enable/disable toggle calls `PUT /api/sources/{id}`
- [ ] "Add Source" dialog with source_type select + identifier input + display_name
- [ ] Per-source "Ingest Now" button triggers `POST /api/sources/{id}/ingest`
- [ ] Discovered sources panel shows suggestions with "Add to Pro" and "Dismiss" actions
- [ ] Source list refetches after any mutation
- [ ] Error state: show fetch_error in red badge if present

## Notes
- `[Import]` button (from wireframe) imports from sources.yaml — can be a read-only YAML display in a Dialog for now (full YAML sync is complex)
- Source identifier format varies by type: Substack=slug, YouTube=@handle, Reddit=subreddit name, RSS=URL, HN/ArXiv/DEV.to=no identifier (single source)
- Use `useMutation` + `queryClient.invalidateQueries(['sources'])` after each mutation
