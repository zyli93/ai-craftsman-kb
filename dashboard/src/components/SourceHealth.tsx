/**
 * SourceHealth — renders a list of sources with their health status.
 *
 * Status logic:
 * - fetch_error != null  → Error  (red)
 * - last_fetched_at > 48h ago (or null) → Stale (yellow)
 * - otherwise           → OK     (green)
 */
import { formatDistanceToNow } from 'date-fns'
import type { Source } from '@/api/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'

type HealthStatus = 'ok' | 'stale' | 'error'

/** Derive health status from a source record. */
function deriveStatus(source: Source): HealthStatus {
  if (source.fetch_error != null) return 'error'
  if (!source.last_fetched_at) return 'stale'

  const fetchedAt = new Date(source.last_fetched_at)
  const hoursSince = (Date.now() - fetchedAt.getTime()) / (1000 * 60 * 60)
  if (hoursSince > 48) return 'stale'

  return 'ok'
}

interface StatusBadgeProps {
  status: HealthStatus
}

/** Small coloured indicator dot and text label for health status. */
function StatusBadge({ status }: StatusBadgeProps) {
  const config: Record<HealthStatus, { dot: string; label: string; text: string }> = {
    ok: { dot: 'bg-green-500', label: 'OK', text: 'text-green-700' },
    stale: { dot: 'bg-yellow-400', label: 'Stale', text: 'text-yellow-700' },
    error: { dot: 'bg-red-500', label: 'Error', text: 'text-red-700' },
  }

  const { dot, label, text } = config[status]

  return (
    <span className={cn('flex items-center gap-1.5 text-xs font-medium', text)}>
      <span className={cn('inline-block h-2 w-2 rounded-full', dot)} />
      {status === 'ok' ? '✓' : status === 'stale' ? '⚠' : '✗'} {label}
    </span>
  )
}

interface SourceRowProps {
  source: Source
}

/** A single row showing source name, last fetch time, and health badge. */
function SourceRow({ source }: SourceRowProps) {
  const status = deriveStatus(source)
  const displayName =
    source.display_name ?? `${source.source_type} (${source.identifier})`

  const lastFetched = source.last_fetched_at
    ? formatDistanceToNow(new Date(source.last_fetched_at), { addSuffix: true })
    : 'Never'

  return (
    <div className="flex items-center justify-between py-3">
      <div className="flex items-center gap-2 min-w-0">
        <span
          className={cn(
            'inline-block h-2 w-2 shrink-0 rounded-full',
            status === 'ok'
              ? 'bg-green-500'
              : status === 'stale'
                ? 'bg-yellow-400'
                : 'bg-red-500',
          )}
        />
        <span className="text-sm font-medium truncate">{displayName}</span>
      </div>
      <div className="flex items-center gap-6 shrink-0">
        <span className="text-xs text-muted-foreground hidden sm:block">
          Last pull: {lastFetched}
        </span>
        <StatusBadge status={status} />
      </div>
    </div>
  )
}

interface SourceHealthProps {
  sources: Source[]
}

/**
 * Card listing all sources with health status indicators.
 * Sources are sorted: errors first, then stale, then ok.
 */
export function SourceHealth({ sources }: SourceHealthProps) {
  const statusOrder: Record<HealthStatus, number> = { error: 0, stale: 1, ok: 2 }
  const sorted = [...sources].sort(
    (a, b) => statusOrder[deriveStatus(a)] - statusOrder[deriveStatus(b)],
  )

  if (sources.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Source Health</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No sources configured.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Source Health</CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        {sorted.map((source, idx) => (
          <div key={source.id}>
            <SourceRow source={source} />
            {idx < sorted.length - 1 && <Separator />}
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
