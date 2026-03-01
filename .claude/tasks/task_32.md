# Task 32: Dashboard Scaffolding (Vite + Tailwind + shadcn/ui)

## Wave
Wave 13 (parallel with task 30)
Domain: frontend

## Objective
Scaffold the React/TypeScript dashboard with Vite, Tailwind CSS v4, shadcn/ui, React Router, and a typed API client. All dashboard pages (tasks 33–39) build on this foundation.

## Scope

### Files to create/modify:
- `dashboard/package.json` — Dependencies + scripts
- `dashboard/vite.config.ts` — Vite + API proxy config
- `dashboard/tailwind.config.ts` — Tailwind config
- `dashboard/tsconfig.json`
- `dashboard/src/main.tsx` — React entry point
- `dashboard/src/App.tsx` — Router setup + layout shell
- `dashboard/src/api/client.ts` — Typed API client (all endpoints)
- `dashboard/src/api/types.ts` — TypeScript interfaces matching FastAPI response models
- `dashboard/src/components/layout/Sidebar.tsx` — Navigation sidebar
- `dashboard/src/components/layout/Layout.tsx` — Main layout wrapper
- `dashboard/src/pages/` — Empty placeholder files for each page

### Key interfaces / implementation details:

**`package.json`** dependencies:
```json
{
  "dependencies": {
    "react": "^18.3",
    "react-dom": "^18.3",
    "react-router-dom": "^6",
    "@tanstack/react-query": "^5",
    "lucide-react": "^0.400"
  },
  "devDependencies": {
    "vite": "^5",
    "@vitejs/plugin-react": "^4",
    "typescript": "^5",
    "tailwindcss": "^4",
    "@tailwindcss/vite": "^4",
    "shadcn": "latest"
  }
}
```

**shadcn/ui components** to initialize (run `npx shadcn init` then add):
```
button, card, input, select, badge, table, dialog, tabs,
textarea, checkbox, popover, dropdown-menu, toast, separator
```

**Vite config** — proxy API calls to FastAPI:
```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

**TypeScript API types** (`api/types.ts`) — matching FastAPI response models:
```typescript
export interface Document {
  id: string
  title: string | null
  url: string
  source_type: string
  origin: 'pro' | 'radar' | 'adhoc'
  author: string | null
  published_at: string | null
  fetched_at: string
  word_count: number | null
  is_embedded: boolean
  is_favorited: boolean
  is_archived: boolean
  user_tags: string[]
  excerpt: string | null
}

export interface SearchResult {
  document: Document
  score: number
  mode_used: string
}

export interface Entity {
  id: string
  name: string
  entity_type: string
  mention_count: number
  normalized_name: string
  first_seen_at: string | null
}

export interface SystemStats {
  total_documents: number
  embedded_documents: number
  total_entities: number
  active_sources: number
  total_briefings: number
  vector_count: number
  db_size_bytes: number
}

export interface Source {
  id: string
  source_type: string
  identifier: string
  display_name: string | null
  enabled: boolean
  last_fetched_at: string | null
  fetch_error: string | null
}

export interface Briefing {
  id: string
  title: string
  query: string | null
  content: string
  created_at: string
  format: string
}
```

**Typed API client** (`api/client.ts`):
```typescript
const BASE = '/api'

async function get<T>(path: string, params?: Record<string, string | number | boolean>): Promise<T>
async function post<T>(path: string, body?: unknown): Promise<T>
async function del<T>(path: string): Promise<T>
async function put<T>(path: string, body?: unknown): Promise<T>

export const api = {
  stats: () => get<SystemStats>('/api/stats'),
  health: () => get<{status: string}>('/api/health'),

  documents: {
    list: (params?: {...}) => get<Document[]>('/api/documents', params),
    get: (id: string) => get<Document>(`/api/documents/${id}`),
    delete: (id: string) => del(`/api/documents/${id}`),
  },

  search: (params: {q: string, mode?: string, source_type?: string, limit?: number}) =>
    get<SearchResult[]>('/api/search', params),

  ingest: {
    url: (url: string, tags?: string[]) => post<Document>('/api/ingest/url', {url, tags}),
    pro: (source?: string) => post('/api/ingest/pro', {source}),
  },

  sources: {
    list: () => get<Source[]>('/api/sources'),
    create: (data: {...}) => post<Source>('/api/sources', data),
    update: (id: string, data: {...}) => put<Source>(`/api/sources/${id}`, data),
    delete: (id: string) => del(`/api/sources/${id}`),
    ingest: (id: string) => post(`/api/sources/${id}/ingest`),
  },

  entities: {
    list: (params?: {q?: string, entity_type?: string, limit?: number}) =>
      get<Entity[]>('/api/entities', params),
    get: (id: string) => get<Entity & {documents: Document[]}>(`/api/entities/${id}`),
  },

  radar: {
    results: (params?: {status?: string}) => get<Document[]>('/api/radar/results', params),
    search: (body: {query: string, sources?: string[], limit_per_source?: number}) =>
      post('/api/radar/search', body),
    promote: (id: string) => post(`/api/radar/results/${id}/promote`),
    archive: (id: string) => post(`/api/radar/results/${id}/archive`),
  },

  briefings: {
    list: () => get<Briefing[]>('/api/briefings'),
    create: (body: {query: string, run_radar?: boolean}) => post<Briefing>('/api/briefings', body),
    get: (id: string) => get<Briefing>(`/api/briefings/${id}`),
    delete: (id: string) => del(`/api/briefings/${id}`),
  },

  discover: () => get('/api/discover'),
}
```

**React Router** (`App.tsx`):
```typescript
// Routes:
/           → Overview
/sources    → Source Editor
/search     → Search & Radar
/entities   → Entity Explorer
/documents  → Document Manager
/briefing   → Briefing Builder
```

**Sidebar navigation** links to all 6 routes with Lucide icons.

**TanStack Query** (`main.tsx`): Wrap app in `QueryClientProvider`.

## Dependencies
- Depends on: none (standalone scaffold)
- Packages needed: listed above in package.json

## Acceptance Criteria
- [ ] `pnpm install` succeeds with no peer dependency errors
- [ ] `pnpm dev` starts dev server at localhost:3000
- [ ] `pnpm build` produces `dist/` without TypeScript errors
- [ ] All 6 routes render without errors (placeholder pages are fine)
- [ ] API client typed — all endpoints return typed responses
- [ ] Sidebar navigation links work (no 404s in dev mode)
- [ ] `/api/*` proxied to `http://localhost:8000` in dev mode
- [ ] shadcn/ui `Button`, `Card`, `Input`, `Badge` importable

## Notes
- Use Tailwind CSS v4 (`@tailwindcss/vite` plugin) — NOT v3 PostCSS approach
- `shadcn init` generates `components/ui/` directory — commit these generated files
- TanStack Query handles caching and refetching; use `useQuery` and `useMutation` hooks in page components
- No `any` types in `api/types.ts` or `api/client.ts`
- The proxy in vite.config.ts means the dashboard makes all API calls to `/api/*` which get forwarded to FastAPI — no CORS issues in dev
