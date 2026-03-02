/**
 * Entity Explorer page — shows all extracted entities grouped by type.
 * Supports search (debounced 300 ms), type filter, sort, and a detail panel
 * for each entity showing mention count, related entities, and linked docs.
 */
import { useState, useMemo, useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  User,
  Building2,
  Cpu,
  Package,
  FileText,
  Book,
  Calendar,
  ExternalLink,
  X,
  Search,
  ChevronRight,
  Newspaper,
  Youtube,
} from 'lucide-react'
import { format } from 'date-fns'
import { api } from '@/api/client'
import type { Entity, EntityWithDocuments, CoOccurringEntity } from '@/api/types'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Canonical display order for entity types */
const ENTITY_TYPE_ORDER = [
  'person',
  'company',
  'technology',
  'product',
  'paper',
  'book',
  'event',
] as const

const TYPE_LABELS: Record<string, string> = {
  person: 'People',
  company: 'Companies',
  technology: 'Technologies',
  product: 'Products',
  paper: 'Papers',
  book: 'Books',
  event: 'Events',
}

const TYPE_ICONS: Record<string, React.ReactNode> = {
  person: <User className="h-4 w-4" />,
  company: <Building2 className="h-4 w-4" />,
  technology: <Cpu className="h-4 w-4" />,
  product: <Package className="h-4 w-4" />,
  paper: <FileText className="h-4 w-4" />,
  book: <Book className="h-4 w-4" />,
  event: <Calendar className="h-4 w-4" />,
}

const TYPE_BADGE_COLORS: Record<string, string> = {
  person: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  company: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
  technology: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  product: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
  paper: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
  book: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  event: 'bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200',
}

const SOURCE_ICONS: Record<string, React.ReactNode> = {
  youtube: <Youtube className="h-4 w-4 text-red-500" />,
  arxiv: <FileText className="h-4 w-4 text-blue-500" />,
  hn: <Newspaper className="h-4 w-4 text-orange-500" />,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Returns a Lucide icon for the given source type,
 * falling back to a generic document icon.
 */
function getSourceIcon(sourceType: string): React.ReactNode {
  return SOURCE_ICONS[sourceType] ?? <FileText className="h-4 w-4 text-muted-foreground" />
}

/**
 * Formats a date string as "MMM d, yyyy" or returns "Unknown" if null.
 */
function formatDate(dateStr: string | null): string {
  if (!dateStr) return 'Unknown'
  try {
    return format(new Date(dateStr), 'MMM d, yyyy')
  } catch {
    return dateStr
  }
}

/**
 * Groups an array of entities by their entity_type.
 * Returns a Map preserving insertion order.
 */
function groupByType(entities: Entity[]): Map<string, Entity[]> {
  const map = new Map<string, Entity[]>()
  for (const entity of entities) {
    const key = entity.entity_type
    if (!map.has(key)) map.set(key, [])
    map.get(key)!.push(entity)
  }
  return map
}

/**
 * Returns the plural display label for an entity type.
 * Capitalizes and appends "s" for unknown types as a fallback.
 */
function typeLabel(entityType: string): string {
  return (
    TYPE_LABELS[entityType] ??
    entityType.charAt(0).toUpperCase() + entityType.slice(1) + 's'
  )
}

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------

function EntitiesSkeleton() {
  return (
    <div className="space-y-6">
      {[1, 2].map((group) => (
        <div key={group} className="space-y-2">
          <Skeleton className="h-5 w-32" />
          {[1, 2, 3].map((row) => (
            <div key={row} className="flex items-center gap-4 py-2">
              <Skeleton className="h-4 w-40" />
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-6 w-24 rounded-full" />
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Detail panel skeleton
// ---------------------------------------------------------------------------

function DetailSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-6 w-48" />
      <Skeleton className="h-4 w-36" />
      <div className="space-y-2">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-4/5" />
        <Skeleton className="h-4 w-3/5" />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Entity detail panel
// ---------------------------------------------------------------------------

interface DetailPanelProps {
  entityId: string
  onClose: () => void
}

/**
 * Detail panel shown when an entity is selected.
 * Displays entity metadata, related entities, and linked docs.
 */
function EntityDetailPanel({ entityId, onClose }: DetailPanelProps) {
  const { data, isLoading } = useQuery<EntityWithDocuments>({
    queryKey: ['entity', entityId],
    queryFn: () => api.entities.get(entityId),
    enabled: !!entityId,
  })

  return (
    <Card className="sticky top-4">
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-3">
        <CardTitle className="text-base">
          {isLoading ? (
            <Skeleton className="h-5 w-32" />
          ) : (
            data?.name ?? 'Entity Detail'
          )}
        </CardTitle>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0 -mt-0.5 -mr-1"
          onClick={onClose}
          aria-label="Close detail panel"
        >
          <X className="h-4 w-4" />
        </Button>
      </CardHeader>

      <CardContent className="space-y-4">
        {isLoading ? (
          <DetailSkeleton />
        ) : data ? (
          <>
            {/* Metadata row */}
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span
                className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${
                  TYPE_BADGE_COLORS[data.entity_type] ?? 'bg-muted text-muted-foreground'
                }`}
              >
                {TYPE_ICONS[data.entity_type] ?? null}
                {data.entity_type.charAt(0).toUpperCase() + data.entity_type.slice(1)}
              </span>
              <span className="text-muted-foreground">
                {data.mention_count} mention{data.mention_count !== 1 ? 's' : ''}
              </span>
              {data.first_seen_at && (
                <span className="text-muted-foreground">
                  First seen {formatDate(data.first_seen_at)}
                </span>
              )}
            </div>

            <Separator />

            {/* Document count */}
            <p className="text-sm text-muted-foreground">
              {data.documents.length} linked document{data.documents.length !== 1 ? 's' : ''}
            </p>

            {/* Related entities (co-occurrences) */}
            {data.co_occurring && data.co_occurring.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm font-medium">Related entities</p>
                <div className="flex flex-wrap gap-1.5">
                  {data.co_occurring.slice(0, 8).map((rel: CoOccurringEntity) => (
                    <span
                      key={rel.entity_id}
                      className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs"
                    >
                      {rel.name}
                      <span className="text-muted-foreground">({rel.co_occurrence_count})</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Linked documents */}
            {data.documents.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm font-medium">Documents</p>
                <div className="space-y-1">
                  {data.documents.slice(0, 10).map((doc) => (
                    <a
                      key={doc.id}
                      href={doc.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-start gap-2 rounded-md p-1.5 text-sm hover:bg-muted/50 transition-colors group"
                    >
                      <span className="mt-0.5 shrink-0">
                        {getSourceIcon(doc.source_type)}
                      </span>
                      <span className="flex-1 min-w-0">
                        <span className="line-clamp-2 font-medium group-hover:underline">
                          {doc.title ?? doc.url}
                        </span>
                        <span className="flex items-center gap-1.5 text-xs text-muted-foreground mt-0.5">
                          <span className="capitalize">{doc.source_type}</span>
                          {doc.published_at && (
                            <>
                              <span>·</span>
                              <span>{formatDate(doc.published_at)}</span>
                            </>
                          )}
                        </span>
                      </span>
                      <ExternalLink className="h-3 w-3 mt-0.5 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                    </a>
                  ))}
                </div>
              </div>
            )}

            {data.documents.length === 0 && (
              <p className="text-sm text-muted-foreground">No linked documents found.</p>
            )}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Failed to load entity details.</p>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Entity row
// ---------------------------------------------------------------------------

interface EntityRowProps {
  entity: Entity
  isSelected: boolean
  onSelect: (id: string) => void
}

/** Single entity row within a type group section. */
function EntityRow({ entity, isSelected, onSelect }: EntityRowProps) {
  return (
    <div
      className={`flex items-center gap-3 rounded-lg px-3 py-2.5 cursor-pointer transition-colors hover:bg-muted/60 ${
        isSelected ? 'bg-muted ring-1 ring-border' : ''
      }`}
      onClick={() => onSelect(entity.id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onSelect(entity.id)
      }}
      aria-pressed={isSelected}
    >
      <span className="flex-1 min-w-0 text-sm font-medium truncate">{entity.name}</span>
      <span className="text-xs text-muted-foreground shrink-0">
        {entity.mention_count} mention{entity.mention_count !== 1 ? 's' : ''}
      </span>
      <Button
        variant="ghost"
        size="sm"
        className="h-7 shrink-0 text-xs gap-1"
        onClick={(e) => {
          e.stopPropagation()
          onSelect(entity.id)
        }}
        tabIndex={-1}
      >
        View Docs
        <ChevronRight className="h-3 w-3" />
      </Button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Type section
// ---------------------------------------------------------------------------

interface TypeSectionProps {
  entityType: string
  entities: Entity[]
  selectedEntityId: string | null
  onSelectEntity: (id: string) => void
}

/** Section grouping entities under a single entity type header. */
function TypeSection({ entityType, entities, selectedEntityId, onSelectEntity }: TypeSectionProps) {
  return (
    <div className="space-y-1">
      {/* Section header */}
      <div className="flex items-center gap-2 px-1 py-1">
        <span className="text-muted-foreground">{TYPE_ICONS[entityType] ?? null}</span>
        <h3 className="text-sm font-semibold">
          {typeLabel(entityType)}{' '}
          <span className="font-normal text-muted-foreground">({entities.length})</span>
        </h3>
      </div>

      {/* Entity rows */}
      {entities.map((entity) => (
        <EntityRow
          key={entity.id}
          entity={entity}
          isSelected={selectedEntityId === entity.id}
          onSelect={onSelectEntity}
        />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Debounce hook
// ---------------------------------------------------------------------------

/**
 * Returns a debounced version of value that only updates after delay ms.
 * Used to throttle search API calls as the user types.
 */
function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState<T>(value)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (timerRef.current !== null) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setDebounced(value), delay)
    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current)
    }
  }, [value, delay])

  return debounced
}

// ---------------------------------------------------------------------------
// Main EntitiesPage component
// ---------------------------------------------------------------------------

type SortMode = 'mention_count' | 'name'

/**
 * Entity Explorer page.
 *
 * Lists all extracted entities grouped by type. Supports debounced search
 * (300ms), type filter, sort by mention count or name, and an inline detail
 * panel showing co-occurrences and linked documents for a selected entity.
 */
export function EntitiesPage() {
  const [searchInput, setSearchInput] = useState('')
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [sortBy, setSortBy] = useState<SortMode>('mention_count')
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null)

  // Debounce the search input by 300 ms before sending to the API
  const debouncedSearch = useDebounced(searchInput, 300)

  const { data: entities, isLoading } = useQuery<Entity[]>({
    queryKey: ['entities', debouncedSearch, typeFilter === 'all' ? null : typeFilter],
    queryFn: () =>
      api.entities.list({
        q: debouncedSearch || undefined,
        entity_type: typeFilter !== 'all' ? typeFilter : undefined,
        limit: 100,
      }),
  })

  // Sort entities then group them by type, preserving canonical type order
  const groupedEntities = useMemo(() => {
    if (!entities) return new Map<string, Entity[]>()

    const sorted = [...entities].sort((a, b) => {
      if (sortBy === 'mention_count') return b.mention_count - a.mention_count
      return a.name.localeCompare(b.name)
    })

    const grouped = groupByType(sorted)

    // Re-order keys to match canonical type order; append any unknown types at end
    const ordered = new Map<string, Entity[]>()
    for (const type of ENTITY_TYPE_ORDER) {
      if (grouped.has(type)) ordered.set(type, grouped.get(type)!)
    }
    for (const [key, val] of grouped) {
      if (!ordered.has(key)) ordered.set(key, val)
    }

    return ordered
  }, [entities, sortBy])

  const totalCount = entities?.length ?? 0
  const hasResults = totalCount > 0

  function handleSelectEntity(id: string) {
    // Toggle: clicking the same entity again closes the detail panel
    setSelectedEntityId((prev) => (prev === id ? null : id))
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <h2 className="text-2xl font-bold tracking-tight">Entities</h2>

      {/* Filters row */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Search input */}
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search entities..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="pl-9"
          />
        </div>

        {/* Type filter */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground shrink-0">Type:</span>
          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger className="w-44">
              <SelectValue placeholder="All Types" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              {ENTITY_TYPE_ORDER.map((t) => (
                <SelectItem key={t} value={t}>
                  {typeLabel(t)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Sort selector */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground shrink-0">Sort:</span>
          <Select value={sortBy} onValueChange={(v) => setSortBy(v as SortMode)}>
            <SelectTrigger className="w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="mention_count">Most Mentioned</SelectItem>
              <SelectItem value="name">Alphabetical</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Result count badge */}
        {!isLoading && (
          <Badge variant="secondary" className="shrink-0">
            {totalCount} {totalCount === 1 ? 'entity' : 'entities'}
          </Badge>
        )}
      </div>

      {/* Main content - two-column layout when an entity is selected */}
      <div
        className={`grid gap-6 ${
          selectedEntityId ? 'lg:grid-cols-[1fr_380px]' : 'grid-cols-1'
        }`}
      >
        {/* Entity list column */}
        <div className="space-y-6">
          {isLoading ? (
            <EntitiesSkeleton />
          ) : hasResults ? (
            <>
              {Array.from(groupedEntities.entries()).map(([type, group]) => (
                <div key={type}>
                  <TypeSection
                    entityType={type}
                    entities={group}
                    selectedEntityId={selectedEntityId}
                    onSelectEntity={handleSelectEntity}
                  />
                  <Separator className="mt-4" />
                </div>
              ))}
            </>
          ) : (
            /* Empty state */
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <User className="h-12 w-12 text-muted-foreground mb-4 opacity-40" />
              <p className="text-muted-foreground font-medium">No entities found</p>
              {(debouncedSearch || typeFilter !== 'all') && (
                <p className="mt-1 text-sm text-muted-foreground">
                  Try adjusting your search or filter.
                </p>
              )}
              {!debouncedSearch && typeFilter === 'all' && (
                <p className="mt-1 text-sm text-muted-foreground">
                  Ingest and extract entities to see them here.
                </p>
              )}
            </div>
          )}
        </div>

        {/* Detail panel column - only visible when an entity is selected */}
        {selectedEntityId && (
          <EntityDetailPanel
            entityId={selectedEntityId}
            onClose={() => setSelectedEntityId(null)}
          />
        )}
      </div>
    </div>
  )
}
