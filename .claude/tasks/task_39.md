# Task 39: Briefing Builder Page

## Wave
Wave 14 (parallel with tasks 33, 34, 35, 36, 37, 38)
Domain: frontend

## Objective
Build the Briefing Builder page where users generate AI briefings on a topic, view the generated markdown content, see source citations, and export the result.

## Scope

### Files to create/modify:
- `dashboard/src/pages/Briefing.tsx` — Briefing builder + history page

### Key interfaces / implementation details:

**Wireframe** (from plan.md):
```
┌─────────────────────────────────────────────────────────┐
│  Briefing Builder                                        │
├─────────────────────────────────────────────────────────┤
│  Topic: [LLM inference optimization                  ]   │
│  Options: ☑ Ingest fresh content   ☑ Run radar search   │
│  LLM: [Claude Sonnet ▾]                                 │
│  [Generate Briefing]                                     │
│                                                          │
│  ─── Generated Briefing ──────────────────────────────  │
│  Key Themes:                                             │
│  1. Speculative decoding is becoming mainstream...       │
│  2. KV cache optimization is the new bottleneck...      │
│                                                          │
│  Content Ideas:                                          │
│  1. "Why Your LLM Inference Stack is 10x Too Slow" →   │
│  2. "The Hidden Cost of Long Context Windows" →         │
│                                                          │
│  Sources Used: (18 documents)                            │
│  [📄 Source 1] [🎥 Source 2] ...                         │
│                                                          │
│  [📋 Copy as Markdown] [💾 Export] [🔄 Regenerate]       │
└─────────────────────────────────────────────────────────┘
```

**API endpoints called**:
- `POST /api/briefings` → `Briefing` (generate)
- `GET /api/briefings` → `Briefing[]` (history)
- `GET /api/briefings/{id}` → `Briefing` (view past briefing)
- `DELETE /api/briefings/{id}` → (delete)

**Component structure**:
```typescript
function Briefing() {
  const [topic, setTopic] = useState('')
  const [runRadar, setRunRadar] = useState(true)
  const [runIngest, setRunIngest] = useState(true)
  const [activeBriefing, setActiveBriefing] = useState<BriefingType | null>(null)

  const { data: history } = useQuery({
    queryKey: ['briefings'],
    queryFn: api.briefings.list,
  })

  const generateMutation = useMutation({
    mutationFn: () => api.briefings.create({
      query: topic,
      run_radar: runRadar,
      run_ingest: runIngest,
    }),
    onSuccess: (briefing) => {
      setActiveBriefing(briefing)
      queryClient.invalidateQueries(['briefings'])
    },
  })

  return (
    // Left: form + history list
    // Right: active briefing display
  )
}
```

**Briefing display**: Render `briefing.content` as Markdown. Use a lightweight markdown renderer (e.g. `react-markdown` — add to dependencies).

**Source citations**: `briefing.source_document_ids` is a JSON array of document IDs. For each ID, show a clickable badge that opens the document URL. Fetch documents in batch if needed (`GET /api/documents?ids=...` — or just store title/url in briefing content).

**Export**: "Copy as Markdown" → `navigator.clipboard.writeText(briefing.content)`. "Export" → `Blob` download as `.md` file.

**History list** (left panel):
- Previous briefings listed by title + date
- Click to load into display
- Delete button per item

**Loading state**: Briefing generation can take 10–30 seconds (LLM + ingest + radar). Show:
- Spinning indicator with status messages: "Ingesting fresh content...", "Running radar search...", "Generating briefing..."
- Use a simple `useInterval` that cycles through status messages while `generateMutation.isPending`

**shadcn/ui components**: `Textarea` (topic input), `Checkbox`, `Select`, `Button`, `Card`, `Badge`, `ScrollArea`, `Separator`, `Skeleton`

## Dependencies
- Depends on: task_30 (briefings endpoints), task_32 (scaffold + api client)
- Packages needed: `react-markdown` (add to dashboard/package.json)

## Acceptance Criteria
- [ ] Topic input + options form renders correctly
- [ ] "Generate Briefing" calls `POST /api/briefings` with loading state
- [ ] Generated briefing displayed as rendered Markdown
- [ ] Source document badges shown (at minimum: count of sources)
- [ ] "Copy as Markdown" copies `briefing.content` to clipboard
- [ ] "Export" downloads `briefing-{date}.md` file
- [ ] Briefing history listed; clicking loads previous briefing
- [ ] Delete button removes briefing from history
- [ ] Long generation time handled with visible loading progress

## Notes
- `react-markdown` renders the `briefing.content` markdown string — no HTML sanitization needed since content is LLM-generated for local use
- If `source_document_ids` is empty (LLM didn't cite sources), show "No sources cited"
- The "Regenerate" button reuses the same `topic` and options — calls `POST /api/briefings` again
- Two-column layout on wide screens: narrow left (form + history), wide right (briefing content)
