/**
 * DiscoveredSources component — panel showing automatically discovered source suggestions.
 * Users can add a discovery to Pro sources or dismiss it.
 */
import { useQueryClient, useMutation } from '@tanstack/react-query'
import { Lightbulb } from 'lucide-react'
import { api } from '@/api/client'
import type { DiscoveredSource } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

/** Maps source_type to a human-readable label */
const SOURCE_TYPE_LABELS: Record<string, string> = {
  hn: 'Hacker News',
  substack: 'Substack',
  youtube: 'YouTube',
  reddit: 'Reddit',
  rss: 'RSS',
  arxiv: 'ArXiv',
  devto: 'DEV.to',
}

interface DiscoveredSourcesProps {
  /** List of discovered source suggestions with status 'pending' */
  discoveries: DiscoveredSource[]
}

export function DiscoveredSources({ discoveries }: DiscoveredSourcesProps) {
  const queryClient = useQueryClient()

  /** Add discovered source to Pro sources via POST /api/sources */
  const addMutation = useMutation({
    mutationFn: (discovery: DiscoveredSource) =>
      api.sources.create({
        source_type: discovery.source_type,
        identifier: discovery.identifier,
        display_name: discovery.display_name ?? undefined,
        enabled: true,
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['sources'] })
      void queryClient.invalidateQueries({ queryKey: ['discover'] })
    },
  })

  /** Dismiss a discovered source via PUT /api/discover/{id} */
  const dismissMutation = useMutation({
    mutationFn: (id: string) => api.discover.dismiss(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['discover'] })
    },
  })

  const pendingDiscoveries = discoveries.filter((d) => d.status === 'pending')

  if (pendingDiscoveries.length === 0) {
    return null
  }

  return (
    <Card className="mt-6">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <Lightbulb className="h-4 w-4 text-yellow-500" />
            Discovered Sources
            <Badge variant="secondary" className="ml-1">
              {pendingDiscoveries.length} suggestions
            </Badge>
          </CardTitle>
          <Button variant="ghost" size="sm" className="text-xs text-muted-foreground">
            Review All
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {pendingDiscoveries.map((discovery) => (
          <div
            key={discovery.id}
            className="flex items-start justify-between gap-4 rounded-md border p-3"
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-medium text-sm truncate">
                  {discovery.display_name ?? discovery.identifier}
                </span>
                <Badge variant="outline" className="text-xs shrink-0">
                  {SOURCE_TYPE_LABELS[discovery.source_type] ?? discovery.source_type}
                </Badge>
                {discovery.mention_count > 0 && (
                  <span className="text-xs text-muted-foreground shrink-0">
                    found in {discovery.mention_count}{' '}
                    {discovery.mention_count === 1 ? 'article' : 'articles'}
                  </span>
                )}
              </div>
              {discovery.context && (
                <p className="text-xs text-muted-foreground mt-1 truncate">
                  {discovery.context}
                </p>
              )}
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Button
                size="sm"
                variant="outline"
                className="text-xs h-7"
                disabled={addMutation.isPending}
                onClick={() => addMutation.mutate(discovery)}
              >
                Add to Pro
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="text-xs h-7 text-muted-foreground"
                disabled={dismissMutation.isPending}
                onClick={() => dismissMutation.mutate(discovery.id)}
              >
                Dismiss
              </Button>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
