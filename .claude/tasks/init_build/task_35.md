# Task 35: Search + Radar Page

## Wave
Wave 14 (parallel with tasks 33, 34, 36, 37, 38, 39)
Domain: frontend

## Objective
Build the combined Search & Radar page with three tabs: Pro Search (hybrid/semantic/keyword search over indexed content), Radar Search (on-demand open-web search), and Adhoc URL ingest.

## Scope

### Files to create/modify:
- `dashboard/src/pages/Search.tsx` — Main page with tab navigation
- `dashboard/src/components/SearchBar.tsx` — Search input with mode selector
- `dashboard/src/components/DocumentCard.tsx` — Reusable document result card
- `dashboard/src/hooks/useSearch.ts` — Search query hook

### Key interfaces / implementation details:

**Wireframe** (from plan.md — Search & Radar page):
```
┌─────────────────────────────────────────────────────────┐
│  Search                                                  │
├─────────────────────────────────────────────────────────┤
│  🔍 [GRPO reinforcement learning LLM               ]    │
│  Mode: (•) Hybrid  ( ) Semantic  ( ) Keyword            │
│  Sources: [All ▾]  Since: [Any time ▾]  [Search]       │
│                                                          │
│  [Pro Results (23)] [Radar Search] [Adhoc URL]          │
│                                                          │
│  📄 "DeepSeek's GRPO: A New Approach..."                │
│     ArXiv · 3 days ago · Score: 0.94                    │
│     [★ Favorite] [🏷 Tag] [🔗 Open]                     │
└─────────────────────────────────────────────────────────┘
```

**API endpoints called**:
- `GET /api/search?q=&mode=&source_type=&since=&limit=` → `SearchResult[]`
- `POST /api/radar/search` → radar results
- `POST /api/ingest/url` → ingested document
- `POST /api/radar/results/{id}/promote` → promote a radar result

**Components**:

```typescript
// useSearch.ts
function useSearch(query: string, mode: string, filters: SearchFilters) {
  return useQuery({
    queryKey: ['search', query, mode, filters],
    queryFn: () => api.search({ q: query, mode, ...filters }),
    enabled: query.length > 0,
  })
}

// SearchBar.tsx
interface SearchBarProps {
  value: string
  onChange: (q: string) => void
  onSubmit: () => void
  mode: 'hybrid' | 'semantic' | 'keyword'
  onModeChange: (m: string) => void
}

// DocumentCard.tsx — reusable across Search + Document Manager pages
interface DocumentCardProps {
  document: Document
  score?: number
  showOriginBadge?: boolean
  onFavorite?: () => void
  onTag?: () => void
  onPromote?: () => void   // for radar results
}
function DocumentCard(props: DocumentCardProps)
// Shows: title (linked), source_type badge, author, relative date,
//        excerpt (first 200 chars), action buttons

// Search.tsx — Tabs component
function Search() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'hybrid'|'semantic'|'keyword'>('hybrid')
  const [sourceFilter, setSourceFilter] = useState<string | null>(null)
  const [sinceFilter, setSinceFilter] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'pro'|'radar'|'adhoc'>('pro')

  // Tab: Pro Results
  // Tab: Radar Search — POST /api/radar/search on submit
  // Tab: Adhoc URL — POST /api/ingest/url on submit
}
```

**Radar tab** behavior:
- Source checkboxes: HN, Reddit, ArXiv, DEV.to, YouTube (all checked by default)
- Clicking "Radar Search" → `POST /api/radar/search` with loading state per source
- Results shown as `DocumentCard` with `[Promote]` button
- `[Promote]` → `POST /api/radar/results/{id}/promote`

**Adhoc URL tab** behavior:
- Input URL → paste/type
- Auto-detect type from URL (show "Detected: YouTube Video")
- "Ingest & Index" → `POST /api/ingest/url`
- Show ingested document card on success

**shadcn/ui components**: `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent`, `Input`, `Button`, `RadioGroup`, `Select`, `Badge`, `Card`, `Checkbox`, `Skeleton` (loading state)

## Dependencies
- Depends on: task_30 (search + radar + ingest endpoints), task_32 (scaffold + api client)
- Packages needed: none new

## Acceptance Criteria
- [ ] `Pro Results` tab: search results display with score, source badge, excerpt, relative date
- [ ] Mode selector (Hybrid/Semantic/Keyword) changes `mode` param in API call
- [ ] Source filter and date filter applied to search
- [ ] `Radar Search` tab: source checkboxes + search → shows results with Promote button
- [ ] `Adhoc URL` tab: URL input → ingest → shows result DocumentCard
- [ ] `DocumentCard` reusable (imported by task_37 Document Manager)
- [ ] Loading skeleton shown during search
- [ ] Empty state message when no results

## Notes
- Search is triggered on form submit (Enter or button click), not on every keystroke
- `DocumentCard` is used in multiple pages — keep it generic with optional props
- Radar search results are stored in DB as `origin='radar'` — they appear in pro search results after `[Promote]`
- Use `Tabs` from shadcn/ui with `value` controlled by `activeTab` state
