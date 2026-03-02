/**
 * Overview — main dashboard page.
 *
 * Shows:
 * - System stats (auto-refreshing every 30 s)
 * - Source health list
 * - Five most recently fetched documents
 * - "Ingest Now" button that triggers a Pro ingestion run
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { formatDistanceToNow } from 'date-fns'
import { ArrowRight, Loader2, RefreshCw } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '@/api/client'
import { useStats } from '@/hooks/useStats'
import { StatsCards } from '@/components/StatsCards'
import { SourceHealth } from '@/components/SourceHealth'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'

/** Simple inline toast notification — avoids pulling in a full toast library. */
function useToast() {
  const show = (message: string) => {
    // Use the native browser notification mechanism as a lightweight fallback.
    // A proper toast provider would be wired in a larger app.
    console.info('[toast]', message)
    // Dispatch a custom event so a future toast provider can pick it up.
    window.dispatchEvent(new CustomEvent('app:toast', { detail: { message } }))
  }
  return { show }
}

export function Overview() {
  const queryClient = useQueryClient()
  const toast = useToast()

  const { data: stats, isLoading: statsLoading } = useStats()
  const { data: sources, isLoading: sourcesLoading } = useQuery({
    queryKey: ['sources'],
    queryFn: api.sources.list,
  })
  const { data: recentDocs, isLoading: docsLoading } = useQuery({
    queryKey: ['documents', 'recent'],
    queryFn: () => api.documents.list({ limit: 5 }),
  })

  const ingestMutation = useMutation({
    mutationFn: () => api.ingest.pro(),
    onSuccess: () => {
      toast.show('Ingestion started successfully.')
      // Invalidate stats and docs so they refresh after the ingest completes.
      void queryClient.invalidateQueries({ queryKey: ['stats'] })
      void queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: () => {
      toast.show('Ingestion failed. Check the console for details.')
    },
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">Overview</h2>
        <Button
          onClick={() => ingestMutation.mutate()}
          disabled={ingestMutation.isPending}
        >
          {ingestMutation.isPending ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Ingesting…
            </>
          ) : (
            <>
              <RefreshCw className="mr-2 h-4 w-4" />
              Ingest Now
            </>
          )}
        </Button>
      </div>

      {/* Stats grid */}
      {statsLoading ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="h-28 rounded-xl border bg-card animate-pulse"
            />
          ))}
        </div>
      ) : stats ? (
        <StatsCards stats={stats} />
      ) : (
        <p className="text-sm text-muted-foreground">Stats unavailable.</p>
      )}

      {/* Source health */}
      {sourcesLoading ? (
        <div className="h-48 rounded-xl border bg-card animate-pulse" />
      ) : sources ? (
        <SourceHealth sources={sources} />
      ) : (
        <p className="text-sm text-muted-foreground">Sources unavailable.</p>
      )}

      {/* Recent documents */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle>Recent Documents</CardTitle>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/documents">
              View All <ArrowRight className="ml-1 h-4 w-4" />
            </Link>
          </Button>
        </CardHeader>
        <CardContent className="pt-2">
          {docsLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="h-8 rounded bg-muted animate-pulse" />
              ))}
            </div>
          ) : recentDocs && recentDocs.length > 0 ? (
            <div>
              {recentDocs.map((doc, idx) => (
                <div key={doc.id}>
                  <div className="flex items-center justify-between py-3 gap-4">
                    <p className="text-sm font-medium truncate flex-1">
                      {doc.title ?? doc.url}
                    </p>
                    <span className="text-xs text-muted-foreground shrink-0 capitalize">
                      {doc.source_type}
                    </span>
                    <span className="text-xs text-muted-foreground shrink-0">
                      {formatDistanceToNow(new Date(doc.fetched_at), {
                        addSuffix: true,
                      })}
                    </span>
                  </div>
                  {idx < recentDocs.length - 1 && <Separator />}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground py-4">
              No documents yet. Run an ingestion to get started.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
