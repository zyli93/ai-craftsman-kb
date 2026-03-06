/**
 * DiscoveredSources component — panel showing automatically discovered source suggestions.
 * Users can add a discovery to Pro sources or dismiss it.
 * Shows confidence badges and discovery method labels.
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

/** Maps discovery_method to a human-readable description */
const DISCOVERY_METHOD_LABELS: Record<string, string> = {
  outbound_link: 'Found in outbound links',
  citation: 'Cited in paper',
  mention: 'Mentioned in articles',
  llm_suggestion: 'AI suggestion',
}

/**
 * Returns Tailwind color classes for a confidence badge.
 * >0.8 green, 0.5-0.8 yellow, <0.5 gray.
 */
function confidenceBadgeClass(confidence: number): string {
  if (confidence > 0.8) return 'bg-green-100 text-green-800 border-green-200'
  if (confidence >= 0.5) return 'bg-yellow-100 text-yellow-800 border-yellow-200'
  return 'bg-gray-100 text-gray-700 border-gray-200'
}

interface DiscoveredSourcesProps {
  /** List of discovered source suggestions */
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

  /** Dismiss a discovered source via PATCH /api/discover/{id} */
  const dismissMutation = useMutation({
    mutationFn: (id: string) => api.discover.updateStatus(id, 'dismissed'),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['discover'] })
    },
  })

  /** Filter to only show suggested items that haven't been acted on */
  const pendingDiscoveries = discoveries.filter(
    (d) => d.status === 'suggested',
  )

  if (discoveries.length === 0) {
    return (
      <Card className="mt-6">
        <CardContent className="py-8 text-center">
          <p className="text-sm text-muted-foreground">
            No source suggestions yet. Run{' '}
            <code className="text-xs font-mono bg-muted px-1 py-0.5 rounded">cr ingest</code>{' '}
            to discover new sources.
          </p>
        </CardContent>
      </Card>
    )
  }

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
              {pendingDiscoveries.length} suggestion{pendingDiscoveries.length !== 1 ? 's' : ''}
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

                {/* Confidence badge shown when confidence field is present */}
                {discovery.confidence != null && (
                  <Badge
                    variant="outline"
                    className={`text-xs shrink-0 ${confidenceBadgeClass(discovery.confidence)}`}
                  >
                    {Math.round(discovery.confidence * 100)}% confidence
                  </Badge>
                )}
              </div>

              {/* Discovery method label */}
              {discovery.discovery_method && (
                <p className="text-xs text-muted-foreground mt-1">
                  {DISCOVERY_METHOD_LABELS[discovery.discovery_method] ?? discovery.discovery_method}
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
