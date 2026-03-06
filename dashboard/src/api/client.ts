/**
 * Typed API client for communicating with the FastAPI backend.
 * All requests go to /api/* which is proxied to http://localhost:8000 in dev.
 */
import type {
  Document,
  SearchResult,
  Entity,
  EntityWithDocuments,
  SystemStats,
  Source,
  SourceCreate,
  SourceUpdate,
  Briefing,
  BriefingCreate,
  HealthStatus,
  IngestUrlRequest,
  IngestProRequest,
  RadarSearchRequest,
  DiscoveredSource,
  DiscoverListResponse,
  UsageSummary,
  UsageRecord,
} from './types'

async function get<T>(
  path: string,
  params?: Record<string, string | number | boolean>,
): Promise<T> {
  const url = new URL(path, window.location.origin)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`)
  return res.json() as Promise<T>
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`)
  return res.json() as Promise<T>
}

async function put<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`PUT ${path} failed: ${res.status}`)
  return res.json() as Promise<T>
}

async function patch<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`PATCH ${path} failed: ${res.status}`)
  return res.json() as Promise<T>
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(path, { method: 'DELETE' })
  if (!res.ok) throw new Error(`DELETE ${path} failed: ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  stats: () => get<SystemStats>('/api/stats'),
  health: () => get<HealthStatus>('/api/health'),

  documents: {
    list: (params?: {
      source_type?: string
      origin?: string
      is_archived?: boolean
      is_favorited?: boolean
      limit?: number
      offset?: number
    }) => get<Document[]>('/api/documents', params as Record<string, string | number | boolean>),
    get: (id: string) => get<Document>(`/api/documents/${id}`),
    update: (id: string, data: { user_tags?: string[]; is_archived?: boolean }) =>
      put<Document>(`/api/documents/${id}`, data),
    delete: (id: string) => del<void>(`/api/documents/${id}`),
  },

  search: (params: {
    q: string
    mode?: string
    source_type?: string
    since?: string
    limit?: number
  }) => get<SearchResult[]>('/api/search', params as Record<string, string | number | boolean>),

  ingest: {
    url: (req: IngestUrlRequest) => post<Document>('/api/ingest/url', req),
    pro: (req?: IngestProRequest) => post<void>('/api/ingest/pro', req),
  },

  sources: {
    list: () => get<Source[]>('/api/sources'),
    create: (data: SourceCreate) => post<Source>('/api/sources', data),
    update: (id: string, data: SourceUpdate) => put<Source>(`/api/sources/${id}`, data),
    delete: (id: string) => del<void>(`/api/sources/${id}`),
    ingest: (id: string) => post<void>(`/api/sources/${id}/ingest`),
  },

  entities: {
    list: (params?: { q?: string; entity_type?: string; limit?: number }) =>
      get<Entity[]>('/api/entities', params as Record<string, string | number | boolean>),
    get: (id: string) => get<EntityWithDocuments>(`/api/entities/${id}`),
  },

  radar: {
    results: (params?: { status?: string }) =>
      get<Document[]>('/api/radar/results', params as Record<string, string | number | boolean>),
    search: (body: RadarSearchRequest) => post<void>('/api/radar/search', body),
    promote: (id: string) => post<void>(`/api/radar/results/${id}/promote`),
    archive: (id: string) => post<void>(`/api/radar/results/${id}/archive`),
  },

  briefings: {
    list: () => get<Briefing[]>('/api/briefings'),
    create: (body: BriefingCreate) => post<Briefing>('/api/briefings', body),
    get: (id: string) => get<Briefing>(`/api/briefings/${id}`),
    delete: (id: string) => del<void>(`/api/briefings/${id}`),
  },

  usage: {
    summary: (params?: { since?: string }) =>
      get<UsageSummary>('/api/usage', params as Record<string, string | number | boolean>),
    recent: (params?: { limit?: number }) =>
      get<UsageRecord[]>('/api/usage/recent', params as Record<string, string | number | boolean>),
  },

  discover: {
    list: (params?: { status?: string }) =>
      get<DiscoverListResponse>('/api/discover', params as Record<string, string | number | boolean>),
    updateStatus: (id: string, status: string) =>
      patch<DiscoveredSource>(`/api/discover/${id}`, { status }),
  },
}
