# Task 36: Entity Explorer Page

## Wave
Wave 14 (parallel with tasks 33, 34, 35, 37, 38, 39)
Domain: frontend

## Objective
Build the Entity Explorer page showing all extracted entities grouped by type, with a detail panel for each entity showing mention count, related entities, and linked documents.

## Scope

### Files to create/modify:
- `dashboard/src/pages/Entities.tsx` — Main entity explorer page

### Key interfaces / implementation details:

**Wireframe** (from plan.md):
```
┌─────────────────────────────────────────────────────────┐
│  Entities                                                │
├─────────────────────────────────────────────────────────┤
│  Filter: [All Types ▾]  Sort: [Most Mentioned ▾]        │
│  Search: [________________]                              │
│                                                          │
│  People (847)                                            │
│  Andrej Karpathy     │ 142 mentions │ [View Docs →]      │
│  Ilya Sutskever      │ 89 mentions  │ [View Docs →]      │
│                                                          │
│  Technologies (423)                                      │
│  GRPO                │ 34 mentions  │ [View Docs →]      │
│  FlashAttention      │ 21 mentions  │ [View Docs →]      │
│                                                          │
│  ── Entity Detail: "GRPO" ───────────────────────────── │
│  Type: Technology · 34 docs · First seen: 2025-01-15    │
│  Related: DeepSeek (15), PPO (12), RLHF (9)            │
│  📄 "DeepSeek's GRPO Paper..."  ArXiv  Jan 15           │
│  🎥 "GRPO Explained..."         YouTube Jan 18           │
└─────────────────────────────────────────────────────────┘
```

**API endpoints called**:
- `GET /api/entities?q=&entity_type=&limit=50` → `Entity[]`
- `GET /api/entities/{id}` → entity detail with co-occurring entities + documents

**Component structure**:
```typescript
// Entities.tsx
function Entities() {
  const [searchQuery, setSearchQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState<string | null>(null)
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null)  // entity id
  const [sortBy, setSortBy] = useState<'mention_count' | 'name'>('mention_count')

  const { data: entities } = useQuery({
    queryKey: ['entities', searchQuery, typeFilter],
    queryFn: () => api.entities.list({ q: searchQuery || undefined, entity_type: typeFilter || undefined, limit: 100 }),
  })

  const { data: entityDetail } = useQuery({
    queryKey: ['entity', selectedEntity],
    queryFn: () => api.entities.get(selectedEntity!),
    enabled: !!selectedEntity,
  })

  // Group by entity_type: { person: [...], company: [...], technology: [...], ... }
  const grouped = groupBy(entities ?? [], e => e.entity_type)
  // Entity types order: person, company, technology, product, paper, book, event
}
```

**Entity type icons** (Lucide):
- `person` → `User`
- `company` → `Building2`
- `technology` → `Cpu`
- `product` → `Package`
- `paper` → `FileText`
- `book` → `Book`
- `event` → `Calendar`

**Entity detail panel** (right panel or slide-over):
- Entity name + type badge + mention count
- First seen date
- Related entities (co-occurrences from `GET /api/entities/{id}`)
- Documents list (from `GET /api/entities/{id}`) — up to 10, sorted by `published_at DESC`
- Each document shown as compact row: icon + title (linked) + source_type + date

**Search behavior**: Debounced (300ms) — queries `GET /api/entities?q=` as user types.

**shadcn/ui components**: `Input`, `Select`, `Badge`, `Card`, `ScrollArea`, `Separator`, `Button`

## Dependencies
- Depends on: task_30 (entities endpoints), task_32 (scaffold + api client)
- Packages needed: none new (use `useMemo` for groupBy, no extra lib needed)

## Acceptance Criteria
- [ ] Entities grouped by type with section count headers ("People (847)")
- [ ] Search input filters entities via FTS (debounced 300ms)
- [ ] Type filter dropdown limits to one entity type
- [ ] Clicking an entity opens detail panel with co-occurrences + documents
- [ ] Entity detail shows mention count, first_seen_at, related entities
- [ ] Linked documents in detail panel open source URL in new tab
- [ ] Empty state when no entities match filter

## Notes
- No pagination needed for initial load — 100 entities is the limit; add "Load more" if needed later
- Co-occurrence data requires backend support from task_23 — the `GET /api/entities/{id}` response should include `co_occurring` list
- Entity type filter uses a `Select` from shadcn/ui with "All Types" as first option
- The detail panel can be a right-side Sheet (shadcn/ui `Sheet`) or an inline expanded section
