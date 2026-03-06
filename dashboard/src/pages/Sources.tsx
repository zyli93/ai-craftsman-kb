/**
 * Sources page — main component for viewing, adding, enabling/disabling, and deleting sources.
 * Sources are grouped by source_type with section headers, sorted by display_name within each group.
 */
import { useState } from 'react'
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query'
import {
  Download,
  Pencil,
  Trash2,
  Plus,
  Upload,
  Circle,
  AlertCircle,
} from 'lucide-react'
import { api } from '@/api/client'
import type { Source } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { SourceEditor } from '@/components/SourceEditor'
import { DiscoveredSources } from '@/components/DiscoveredSources'

/** Human-readable section headers for each source type */
const SOURCE_TYPE_LABELS: Record<string, string> = {
  hn: 'Hacker News',
  substack: 'Substack',
  youtube: 'YouTube Channels',
  reddit: 'Reddit',
  rss: 'RSS Feeds',
  arxiv: 'ArXiv',
  devto: 'DEV.to',
}

/** All source types in the preferred display order */
const SOURCE_TYPE_ORDER = ['substack', 'youtube', 'reddit', 'rss', 'arxiv', 'hn', 'devto']

/**
 * Groups sources by source_type and sorts each group by display_name.
 * Returns entries in the preferred source type order.
 */
function groupSources(sources: Source[]): Array<[string, Source[]]> {
  const grouped: Record<string, Source[]> = {}

  for (const source of sources) {
    const type = source.source_type
    if (!grouped[type]) {
      grouped[type] = []
    }
    grouped[type].push(source)
  }

  // Sort each group by display_name (fallback to identifier)
  for (const type in grouped) {
    grouped[type].sort((a, b) => {
      const nameA = (a.display_name ?? a.identifier).toLowerCase()
      const nameB = (b.display_name ?? b.identifier).toLowerCase()
      return nameA.localeCompare(nameB)
    })
  }

  // Return in preferred order, then any unknown types alphabetically
  const ordered: Array<[string, Source[]]> = []
  for (const type of SOURCE_TYPE_ORDER) {
    if (grouped[type]) {
      ordered.push([type, grouped[type]])
    }
  }
  for (const type of Object.keys(grouped).sort()) {
    if (!SOURCE_TYPE_ORDER.includes(type)) {
      ordered.push([type, grouped[type]])
    }
  }

  return ordered
}

/** Formats a date string as a relative or absolute time label */
function formatLastFetched(dateStr: string | null): string {
  if (!dateStr) return 'Never'
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffHours = diffMs / (1000 * 60 * 60)
  if (diffHours < 1) return 'Just now'
  if (diffHours < 24) return `${Math.floor(diffHours)}h ago`
  const diffDays = Math.floor(diffHours / 24)
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return `${diffDays}d ago`
  return date.toLocaleDateString()
}

/**
 * Row component for a single source.
 * Shows: indicator dot, display_name, identifier, enabled switch, last_fetched_at, error badge, actions.
 */
function SourceRow({
  source,
  onEdit,
}: {
  source: Source
  onEdit: (source: Source) => void
}) {
  const queryClient = useQueryClient()

  const toggleMutation = useMutation({
    mutationFn: () =>
      api.sources.update(source.id, { enabled: !source.enabled }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['sources'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: () => api.sources.delete(source.id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['sources'] })
    },
  })

  const ingestMutation = useMutation({
    mutationFn: () => api.sources.ingest(source.id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['sources'] })
    },
  })

  const displayName = source.display_name ?? source.identifier

  return (
    <div className="flex items-center gap-3 px-4 py-2.5 hover:bg-muted/30 transition-colors rounded-md">
      {/* Status indicator dot */}
      <Circle
        className={`h-2 w-2 shrink-0 fill-current ${
          source.fetch_error
            ? 'text-destructive'
            : source.enabled
              ? 'text-green-500'
              : 'text-muted-foreground'
        }`}
      />

      {/* Display name */}
      <span className="flex-1 min-w-0 text-sm font-medium truncate" title={displayName}>
        {displayName}
      </span>

      {/* Identifier (truncated) */}
      <span
        className="w-32 shrink-0 text-xs text-muted-foreground truncate hidden sm:block"
        title={source.identifier}
      >
        {source.identifier || '—'}
      </span>

      {/* Last fetched time */}
      <span className="w-20 shrink-0 text-xs text-muted-foreground hidden md:block">
        {formatLastFetched(source.last_fetched_at)}
      </span>

      {/* Error badge */}
      {source.fetch_error && (
        <Badge variant="destructive" className="shrink-0 text-xs flex items-center gap-1">
          <AlertCircle className="h-3 w-3" />
          Error
        </Badge>
      )}

      {/* Enabled/disabled toggle */}
      <Switch
        checked={source.enabled}
        onCheckedChange={() => toggleMutation.mutate()}
        disabled={toggleMutation.isPending}
        aria-label={`${source.enabled ? 'Disable' : 'Enable'} ${displayName}`}
      />

      {/* Edit button */}
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={() => onEdit(source)}
        title="Edit source"
      >
        <Pencil className="h-3.5 w-3.5" />
      </Button>

      {/* Ingest now button */}
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={() => ingestMutation.mutate()}
        disabled={ingestMutation.isPending}
        title="Ingest now"
      >
        <Download
          className={`h-3.5 w-3.5 ${ingestMutation.isPending ? 'animate-bounce' : ''}`}
        />
      </Button>

      {/* Delete button */}
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7 text-muted-foreground hover:text-destructive"
        onClick={() => {
          if (window.confirm(`Delete source "${displayName}"?`)) {
            deleteMutation.mutate()
          }
        }}
        disabled={deleteMutation.isPending}
        title="Delete source"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}

/** Import dialog — shows read-only YAML representation of all sources */
function ImportDialog({
  open,
  onClose,
  sources,
}: {
  open: boolean
  onClose: () => void
  sources: Source[]
}) {
  const yaml = sources
    .map((s) => {
      const lines = [`  - source_type: ${s.source_type}`]
      if (s.identifier) lines.push(`    identifier: "${s.identifier}"`)
      if (s.display_name) lines.push(`    display_name: "${s.display_name}"`)
      lines.push(`    enabled: ${s.enabled}`)
      return lines.join('\n')
    })
    .join('\n')

  const yamlContent = `sources:\n${yaml || '  # No sources configured'}`

  return (
    <Dialog open={open} onOpenChange={(isOpen) => { if (!isOpen) onClose() }}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Sources YAML</DialogTitle>
          <DialogDescription>
            Read-only view of your configured sources in YAML format.
          </DialogDescription>
        </DialogHeader>
        <pre className="rounded-md bg-muted p-4 text-xs overflow-auto max-h-96 font-mono">
          {yamlContent}
        </pre>
      </DialogContent>
    </Dialog>
  )
}

/**
 * Main Sources page component.
 * Displays all configured sources grouped by type, with filtering, CRUD actions, and discovered sources panel.
 */
export function Sources() {
  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [editSource, setEditSource] = useState<Source | null>(null)
  const [importDialogOpen, setImportDialogOpen] = useState(false)
  const [filterType, setFilterType] = useState<string>('all')
  const [enabledOnly, setEnabledOnly] = useState(false)

  const { data: sources = [], isLoading: sourcesLoading, error: sourcesError } = useQuery({
    queryKey: ['sources'],
    queryFn: api.sources.list,
  })

  const { data: discoverResponse } = useQuery({
    queryKey: ['discover'],
    queryFn: () => api.discover.list({ status: 'suggested' }),
    retry: false,
  })
  const discovered = discoverResponse?.items ?? []

  // Collect all unique source types for the filter dropdown
  const sourceTypes = Array.from(new Set(sources.map((s) => s.source_type))).sort()

  // Apply filters
  const filteredSources = sources.filter((s) => {
    if (filterType !== 'all' && s.source_type !== filterType) return false
    if (enabledOnly && !s.enabled) return false
    return true
  })

  const groupedSources = groupSources(filteredSources)

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">Sources</h2>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setImportDialogOpen(true)}
            className="gap-2"
          >
            <Upload className="h-4 w-4" />
            Import
          </Button>
          <Button size="sm" onClick={() => setAddDialogOpen(true)} className="gap-2">
            <Plus className="h-4 w-4" />
            Add Source
          </Button>
        </div>
      </div>

      {/* Filters row */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Filter:</span>
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="h-8 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          >
            <option value="all">All Types</option>
            {sourceTypes.map((type) => (
              <option key={type} value={type}>
                {SOURCE_TYPE_LABELS[type] ?? type}
              </option>
            ))}
          </select>
        </div>
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <Switch
            checked={enabledOnly}
            onCheckedChange={setEnabledOnly}
            aria-label="Show enabled sources only"
          />
          <span>Enabled Only</span>
        </label>
      </div>

      {/* Loading state */}
      {sourcesLoading && (
        <div className="text-sm text-muted-foreground py-8 text-center">
          Loading sources...
        </div>
      )}

      {/* Error state */}
      {sourcesError && (
        <div className="text-sm text-destructive py-4">
          Failed to load sources. Please try again.
        </div>
      )}

      {/* Empty state */}
      {!sourcesLoading && !sourcesError && sources.length === 0 && (
        <div className="py-12 text-center">
          <p className="text-muted-foreground mb-4">No sources configured yet.</p>
          <Button onClick={() => setAddDialogOpen(true)} className="gap-2">
            <Plus className="h-4 w-4" />
            Add Your First Source
          </Button>
        </div>
      )}

      {/* Filtered empty state */}
      {!sourcesLoading && sources.length > 0 && filteredSources.length === 0 && (
        <div className="py-8 text-center text-sm text-muted-foreground">
          No sources match the current filters.
        </div>
      )}

      {/* Sources grouped by type */}
      {groupedSources.map(([type, typeSources]) => (
        <div key={type}>
          {/* Section header */}
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              {SOURCE_TYPE_LABELS[type] ?? type}
            </h3>
            <span className="text-xs text-muted-foreground">({typeSources.length})</span>
          </div>

          {/* Column headers */}
          <div className="flex items-center gap-3 px-4 py-1 text-xs text-muted-foreground">
            <span className="w-2 shrink-0" />
            <span className="flex-1">Name</span>
            <span className="w-32 shrink-0 hidden sm:block">Identifier</span>
            <span className="w-20 shrink-0 hidden md:block">Last Fetched</span>
            <span className="w-12 shrink-0" />
            {/* space for switch + buttons */}
            <span className="w-24 shrink-0" />
          </div>

          {/* Source rows */}
          <div className="rounded-lg border divide-y">
            {typeSources.map((source) => (
              <SourceRow
                key={source.id}
                source={source}
                onEdit={(s) => setEditSource(s)}
              />
            ))}
          </div>
        </div>
      ))}

      {/* Discovered sources panel */}
      {discovered.length > 0 && (
        <DiscoveredSources discoveries={discovered} />
      )}

      {/* Add source dialog */}
      <SourceEditor
        open={addDialogOpen}
        onClose={() => setAddDialogOpen(false)}
      />

      {/* Edit source dialog */}
      {editSource && (
        <SourceEditor
          open={editSource !== null}
          onClose={() => setEditSource(null)}
          source={editSource}
        />
      )}

      {/* Import YAML dialog */}
      <ImportDialog
        open={importDialogOpen}
        onClose={() => setImportDialogOpen(false)}
        sources={sources}
      />
    </div>
  )
}
