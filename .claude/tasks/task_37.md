# Task 37: Document Manager Page

## Wave
Wave 14 (parallel with tasks 33, 34, 35, 36, 38, 39)
Domain: frontend

## Objective
Build the Document Manager page where users browse all indexed documents, filter by source/origin/tags, view document details, add user tags, archive, and delete documents.

## Scope

### Files to create/modify:
- `dashboard/src/pages/Documents.tsx` — Main document manager page

### Key interfaces / implementation details:

**Wireframe** (from plan.md):
```
┌─────────────────────────────────────────────────────────┐
│  Documents                                               │
├─────────────────────────────────────────────────────────┤
│  Filter: [All Sources ▾] [All Origins ▾] [Date Range]   │
│  Tags: [content-idea ×] [+]                              │
│  ☑ Show archived  ☐ Show deleted                        │
│                                                          │
│  [Select All] [Archive] [Delete] [Tag]  (bulk actions)  │
│                                                          │
│  ☐ 📄 "Scaling Laws for Neural..."                       │
│     ArXiv · Kaplan et al · 2024-12-01                   │
│     Tags: #scaling #research                            │
│     [Archive] [Delete] [View Full] [Open Source]        │
│                                                          │
│  ☐ 🎥 "I spent a mass on GRPO..."                        │
│     YouTube · @YannicKilcher · 2025-01-20               │
│     Origin: radar · [Promote to Pro] [Delete]           │
└─────────────────────────────────────────────────────────┘
```

**API endpoints called**:
- `GET /api/documents?origin=&source_type=&limit=&offset=&include_archived=` → `Document[]`
- `DELETE /api/documents/{id}` → soft delete
- `POST /api/radar/results/{id}/promote` → promote radar result to pro
- Bulk: archive/delete multiple documents (loop over selected IDs)

**Component structure**:
```typescript
function Documents() {
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [originFilter, setOriginFilter] = useState<string | null>(null)
  const [sourceFilter, setSourceFilter] = useState<string | null>(null)
  const [showArchived, setShowArchived] = useState(false)
  const [page, setPage] = useState(0)
  const PAGE_SIZE = 25

  const { data: docs } = useQuery({
    queryKey: ['documents', originFilter, sourceFilter, showArchived, page],
    queryFn: () => api.documents.list({
      origin: originFilter ?? undefined,
      source_type: sourceFilter ?? undefined,
      include_archived: showArchived,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    }),
  })

  // Bulk actions applied to all IDs in `selected` set
  const handleBulkDelete = async () => {
    await Promise.all([...selected].map(id => api.documents.delete(id)))
    queryClient.invalidateQueries(['documents'])
    setSelected(new Set())
  }
}
```

**Document row** (reuse `DocumentCard` from task_35 in compact variant):
- Checkbox for bulk selection
- Content type icon: 📄 article, 🎥 video, 📑 paper, 💬 post
- Title (linked to source URL)
- Source type + author + date
- Origin badge: `pro` (default, no badge), `radar` (yellow), `adhoc` (blue)
- User tags as inline badges
- Action buttons: Archive, Delete, View Full (modal), Open Source (link)
- For `origin='radar'`: show "Promote to Pro" button instead of Archive

**Pagination**: "Load more" button (not numbered pages) — increments offset.

**Tag editing**: Click `[+]` next to tags → inline input → `PUT /api/documents/{id}` with updated `user_tags`.

**shadcn/ui components**: `Checkbox`, `Button`, `Badge`, `Select`, `Popover`, `Input`, `Dialog`, `ScrollArea`

## Dependencies
- Depends on: task_30 (documents + radar promote endpoints), task_32 (scaffold, api client), task_35 (DocumentCard component)
- Packages needed: none new

## Acceptance Criteria
- [ ] Documents listed with pagination (load more)
- [ ] Filter by `origin` (pro/radar/adhoc) and `source_type`
- [ ] `show archived` toggle shows/hides archived documents
- [ ] Checkbox selection enables bulk Archive and Delete buttons
- [ ] Per-document Archive, soft-Delete, and "Open Source" actions work
- [ ] Radar documents show "Promote to Pro" button
- [ ] Tag add/remove updates `user_tags` via API
- [ ] Mutations invalidate document query to refresh list

## Notes
- Soft-delete sets `deleted_at` in DB — documents are hidden from all views by default (unless `show deleted` toggle is added later)
- "View Full" modal shows `raw_content` in a scrollable Dialog — useful for reading ingested articles
- The `origin` filter makes it easy to triage radar results: filter to `radar` → review → promote or delete
- Bulk delete should show a confirmation dialog before executing
