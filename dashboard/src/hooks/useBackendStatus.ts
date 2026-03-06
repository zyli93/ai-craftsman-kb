import { useQuery } from '@tanstack/react-query'

export function useBackendStatus() {
  return useQuery({
    queryKey: ['backend-health'],
    queryFn: async () => {
      const res = await fetch('/api/health')
      if (!res.ok) throw new Error(`Backend returned ${res.status}`)
      return true
    },
    retry: false,
    refetchInterval: 10_000,
  })
}
