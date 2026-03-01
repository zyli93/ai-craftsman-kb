# Task 38: Adhoc URL Ingest + Source Discovery UI

## Wave
Wave 14 (parallel with tasks 33, 34, 35, 36, 37, 39)
Domain: frontend

## Objective
Build the Adhoc & Discovery UI — a combined section (either its own page or a tab within Search) for manually ingesting single URLs and reviewing AI-suggested new sources to follow.

## Scope

### Files to create/modify:
- `dashboard/src/components/AdhocIngest.tsx` — URL input + ingest form
- `dashboard/src/components/DiscoveredSources.tsx` — Source discovery suggestion panel (also used in Sources page task_34)

### Key interfaces / implementation details:

**Adhoc Ingest component** (also embedded in Search tab from task_35):
```typescript
// AdhocIngest.tsx
interface AdhocIngestProps {
  onSuccess?: (doc: Document) => void
}

function AdhocIngest({ onSuccess }: AdhocIngestProps) {
  const [url, setUrl] = useState('')
  const [tags, setTags] = useState<string[]>([])
  const [detectedType, setDetectedType] = useState<string | null>(null)

  // URL type detection (client-side, no API call needed):
  const detectType = (url: string): string => {
    if (/youtube\.com|youtu\.be/.test(url)) return 'YouTube Video'
    if (/arxiv\.org/.test(url)) return 'ArXiv Paper'
    if (/substack\.com/.test(url)) return 'Substack Article'
    return 'Web Article'
  }

  const ingestMutation = useMutation({
    mutationFn: () => api.ingest.url(url, tags),
    onSuccess: (doc) => {
      onSuccess?.(doc)
      setUrl('')
      setTags([])
    },
  })

  return (
    // Layout:
    // [URL input field — full width]
    // Detected type: "YouTube Video" (auto-shown when URL typed)
    // [Tag input — add multiple tags]
    // [Ingest & Index button]
    // — Success: show DocumentCard of ingested doc
    // — Error: show error message
  )
}
```

**Discovered Sources panel**:
```typescript
// DiscoveredSources.tsx
interface DiscoveredSource {
  id: string
  source_type: string
  identifier: string
  display_name: string | null
  confidence: number
  discovery_method: string   // 'outbound_link' | 'citation' | 'mention' | 'llm_suggestion'
  status: 'suggested' | 'added' | 'dismissed'
}

function DiscoveredSources() {
  const { data: discoveries } = useQuery({
    queryKey: ['discover'],
    queryFn: api.discover,
  })

  const addMutation = useMutation({
    mutationFn: (d: DiscoveredSource) => api.sources.create({
      source_type: d.source_type,
      identifier: d.identifier,
      display_name: d.display_name,
    }),
    onSuccess: () => queryClient.invalidateQueries(['sources', 'discover']),
  })

  return (
    // Each discovery:
    // 💡 @laborjack (YouTube) — found in 3 articles [confidence: 85%]
    //    Discovery method: outbound_link
    //    [Add to Pro] [Dismiss]
  )
}
```

**Discovery method display**:
- `outbound_link` → "Found in outbound links"
- `citation` → "Cited in paper"
- `mention` → "Mentioned in articles"
- `llm_suggestion` → "AI suggestion"

**Confidence badge**: `> 0.8` → green, `0.5–0.8` → yellow, `< 0.5` → gray

**Integration**:
- `AdhocIngest` embedded in Search page (task_35) as the "Adhoc URL" tab content
- `DiscoveredSources` embedded in Sources page (task_34) as a panel below the sources table

**API endpoints called**:
- `POST /api/ingest/url` → `Document`
- `GET /api/discover` → `DiscoveredSource[]`
- `POST /api/sources` → add discovered source to pro tier
- `PUT /api/discover/{id}` → dismiss suggestion (if endpoint exists; otherwise just hide client-side)

**shadcn/ui components**: `Input`, `Button`, `Badge`, `Card`, `Separator`, `Progress`

## Dependencies
- Depends on: task_30 (ingest + discover endpoints), task_32 (scaffold + api client)
- Packages needed: none new

## Acceptance Criteria
- [ ] URL input detects type client-side (YouTube/ArXiv/Substack/Generic) and shows label
- [ ] Tag input allows multiple tags (comma or Enter separated)
- [ ] "Ingest & Index" calls `POST /api/ingest/url` and shows loading state
- [ ] Success shows ingested DocumentCard; error shows message
- [ ] `DiscoveredSources` lists all `status='suggested'` discoveries
- [ ] "Add to Pro" calls `POST /api/sources` and refreshes source list
- [ ] "Dismiss" hides the suggestion (client-side state or API call)
- [ ] Confidence shown as percentage badge

## Notes
- `DiscoveredSources` is a shared component used in both task_34 (Sources page) and optionally here
- If `GET /api/discover` is not yet implemented, show an empty state with "Run `cr ingest` to discover new sources"
- The discovery panel requires task_42 (source discovery engine) to have run at least once
