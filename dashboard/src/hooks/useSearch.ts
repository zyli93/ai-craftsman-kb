/**
 * Custom hook for performing search queries against the backend.
 * Search is triggered only when query is non-empty and the hook is enabled.
 */
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { SearchResult } from '@/api/types'

export interface SearchFilters {
  /** Source type filter (e.g. 'hn', 'arxiv') or undefined for all */
  source_type?: string
  /** ISO date string to filter results since this date, or undefined */
  since?: string
  /** Maximum number of results to return */
  limit?: number
}

/**
 * Fetches search results from the Pro index with the given query, mode, and filters.
 * The query is only sent when query.length > 0.
 */
export function useSearch(
  query: string,
  mode: 'hybrid' | 'semantic' | 'keyword',
  filters: SearchFilters,
) {
  return useQuery<SearchResult[]>({
    queryKey: ['search', query, mode, filters],
    queryFn: () =>
      api.search({
        q: query,
        mode,
        source_type: filters.source_type,
        since: filters.since,
        limit: filters.limit ?? 20,
      }),
    enabled: query.length > 0,
  })
}
