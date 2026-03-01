/** TypeScript interfaces matching FastAPI response models */

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

export interface EntityWithDocuments extends Entity {
  documents: Document[]
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

export interface SourceCreate {
  source_type: string
  identifier: string
  display_name?: string
  enabled?: boolean
}

export interface SourceUpdate {
  display_name?: string
  enabled?: boolean
}

export interface Briefing {
  id: string
  title: string
  query: string | null
  content: string
  created_at: string
  format: string
}

export interface BriefingCreate {
  query: string
  run_radar?: boolean
}

export interface HealthStatus {
  status: string
}

export interface IngestUrlRequest {
  url: string
  tags?: string[]
}

export interface IngestProRequest {
  source?: string
}

export interface RadarSearchRequest {
  query: string
  sources?: string[]
  limit_per_source?: number
}
