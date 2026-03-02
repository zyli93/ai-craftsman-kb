/**
 * Custom hook for fetching system stats with auto-refresh.
 * Polls every 30 seconds to keep the Overview page up to date.
 */
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'

export function useStats() {
  return useQuery({
    queryKey: ['stats'],
    queryFn: api.stats,
    refetchInterval: 30_000,
  })
}
