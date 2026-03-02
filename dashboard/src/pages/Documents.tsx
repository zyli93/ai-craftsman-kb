/**
 * Document Manager page — browse, filter, tag, archive, and delete indexed documents.
 *
 * Features:
 *  - Filter by source type, origin, and archived status
 *  - Checkbox-based bulk selection with Archive / Delete / Tag actions
 *  - Per-row actions: Archive, Delete, View (open source), Promote to Pro (radar docs)
 *  - Inline tag editing via inline input
 *  - Offset-based "Load more" pagination
 *  - Bulk delete confirmation dialog
 */
import { useState, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  FileText,
  Youtube,
  BookOpen,
  Newspaper,
  Radio,
  ExternalLink,
  Archive,
  Trash2,
  ArrowUpCircle,
  Plus,
  ChevronDown,
  Loader2,
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { api } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import type { Document } from '@/api/types'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PAGE_SIZE = 25

const SOURCE_OPTIONS = [
  { label: 'All Sources', value: '' },
  { label: 'Hacker News', value: 'hn' },
  { label: 'Reddit', value: 'reddit' },
  { label: 'ArXiv', value: 'arxiv' },
  { label: 'DEV.to', value: 'devto' },
  { label: 'YouTube', value: 'youtube' },
  { label: 'Substack', value: 'substack' },
  { label: 'RSS', value: 'rss' },
]

const ORIGIN_OPTIONS = [
  { label: 'All Origins', value: '' },
  { label: 'Pro', value: 'pro' },
  { label: 'Radar', value: 'radar' },
  { label: 'Adhoc', value: 'adhoc' },
]

/** Maps source_type to Tailwind badge color classes */
const SOURCE_TYPE_COLORS: Record<string, string> = {
  hn: 'bg-orange-100 text-orange-800 border-orange-200',
  hackernews: 'bg-orange-100 text-orange-800 border-orange-200',
  substack: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  youtube: 'bg-red-100 text-red-800 border-red-200',
  reddit: 'bg-rose-100 text-rose-800 border-rose-200',
  arxiv: 'bg-blue-100 text-blue-800 border-blue-200',
  rss: 'bg-green-100 text-green-800 border-green-200',
  devto: 'bg-violet-100 text-violet-800 border-violet-200',
  adhoc: 'bg-gray-100 text-gray-800 border-gray-200',
}

const SOURCE_TYPE_LABELS: Record<string, string> = {
  hn: 'HN',
  hackernews: 'HN',
  substack: 'Substack',
  youtube: 'YouTube',
  reddit: 'Reddit',
  arxiv: 'ArXiv',
  rss: 'RSS',
  devto: 'DEV.to',
  adhoc: 'Adhoc',
}

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

/**
 * Returns an icon component for the given source_type.
 * Falls back to a generic FileText icon.
 */
function SourceIcon({ sourceType }: { sourceType: string }) {
  const cls = 'h-4 w-4 shrink-0 text-muted-foreground'
  switch (sourceType) {
    case 'youtube':
      return <Youtube className={cls} />
    case 'arxiv':
      return <BookOpen className={cls} />
    case 'rss':
    case 'substack':
      return <Newspaper className={cls} />
    case 'radar':
      return <Radio className={cls} />
    default:
      return <FileText className={cls} />
  }
}

/**
 * Formats a date string as a relative time (e.g. "3 days ago").
 * Returns an empty string for null or unparseable dates.
 */
function relativeDate(dateStr: string | null): string {
  if (!dateStr) return ''
  try {
    return formatDistanceToNow(new Date(dateStr), { addSuffix: true })
  } catch {
    return ''
  }
}

/**
 * Returns a Badge for the document's origin.
 * Pro documents get no badge; radar gets yellow; adhoc gets blue.
 */
function OriginBadge({ origin }: { origin: Document['origin'] }) {
  if (origin === 'pro') return null
  if (origin === 'radar') {
    return (
      <Badge className="text-xs bg-yellow-100 text-yellow-800 border-yellow-200 border">
        radar
      </Badge>
    )
  }
  return (
    <Badge className="text-xs bg-blue-100 text-blue-800 border-blue-200 border">
      adhoc
    </Badge>
  )
}

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------

function DocumentRowSkeleton() {
  return (
    <div className="flex items-start gap-3 px-4 py-3 border-b last:border-b-0">
      <Skeleton className="h-4 w-4 mt-0.5 shrink-0" />
      <Skeleton className="h-4 w-4 mt-0.5 shrink-0" />
      <div className="flex-1 space-y-1.5">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-3 w-1/3" />
        <Skeleton className="h-3 w-1/4" />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline tag editor sub-component
// ---------------------------------------------------------------------------

interface InlineTagEditorProps {
  /** Current tags on the document */
  tags: string[]
  /** Called with the updated tag list when the user submits */
  onSave: (tags: string[]) => void
  /** Whether a save mutation is currently in flight */
  isSaving: boolean
}

/**
 * Renders existing user tags and an inline input to add a new one.
 * Pressing Enter or clicking the plus icon commits the new tag.
 */
function InlineTagEditor({ tags, onSave, isSaving }: InlineTagEditorProps) {
  const [input, setInput] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  function handleAdd() {
    const trimmed = input.trim()
    if (!trimmed || tags.includes(trimmed)) {
      setInput('')
      return
    }
    onSave([...tags, trimmed])
    setInput('')
  }

  function handleRemove(tag: string) {
    onSave(tags.filter((t) => t !== tag))
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAdd()
    }
    if (e.key === 'Escape') {
      setInput('')
      inputRef.current?.blur()
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-1">
      {tags.map((tag) => (
        <Badge
          key={tag}
          variant="secondary"
          className="text-xs gap-1 cursor-pointer hover:bg-destructive/10"
          onClick={() => handleRemove(tag)}
          title="Click to remove tag"
        >
          #{tag}
          <span className="ml-0.5 opacity-60">x</span>
        </Badge>
      ))}
      <div className="flex items-center gap-1">
        <Input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="add tag"
          className="h-6 w-20 text-xs px-1.5 py-0"
          disabled={isSaving}
        />
        <Button
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0"
          onClick={handleAdd}
          disabled={!input.trim() || isSaving}
          aria-label="Add tag"
        >
          {isSaving ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Plus className="h-3 w-3" />
          )}
        </Button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Document row sub-component
// ---------------------------------------------------------------------------

interface DocumentRowProps {
  doc: Document
  isSelected: boolean
  onSelect: (id: string, checked: boolean) => void
  onArchive: (id: string) => void
  onDelete: (id: string) => void
  onPromote: (id: string) => void
  onTagSave: (id: string, tags: string[]) => void
  isMutating: boolean
}

/**
 * Single document row with checkbox, icon, title, meta, tags, and action buttons.
 */
function DocumentRow({
  doc,
  isSelected,
  onSelect,
  onArchive,
  onDelete,
  onPromote,
  onTagSave,
  isMutating,
}: DocumentRowProps) {
  const sourceLabel = SOURCE_TYPE_LABELS[doc.source_type] ?? doc.source_type
  const sourceColor =
    SOURCE_TYPE_COLORS[doc.source_type] ?? 'bg-gray-100 text-gray-800 border-gray-200'
  const date = relativeDate(doc.published_at ?? doc.fetched_at)

  return (
    <div
      className={`flex items-start gap-3 px-4 py-3 border-b last:border-b-0 hover:bg-muted/30 transition-colors ${
        doc.is_archived ? 'opacity-60' : ''
      }`}
    >
      {/* Selection checkbox */}
      <Checkbox
        checked={isSelected}
        onCheckedChange={(checked) => onSelect(doc.id, checked === true)}
        className="mt-0.5 shrink-0"
        aria-label={`Select ${doc.title ?? doc.url}`}
      />

      {/* Content type icon */}
      <span className="mt-0.5 shrink-0">
        <SourceIcon sourceType={doc.source_type} />
      </span>

      {/* Main content */}
      <div className="min-w-0 flex-1 space-y-1">
        {/* Title + origin badge */}
        <div className="flex flex-wrap items-center gap-2">
          <a
            href={doc.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium leading-tight hover:underline text-foreground line-clamp-1"
          >
            {doc.title ?? doc.url}
          </a>
          <OriginBadge origin={doc.origin} />
          {doc.is_archived && (
            <Badge variant="outline" className="text-xs text-muted-foreground">
              archived
            </Badge>
          )}
        </div>

        {/* Meta: source badge, author, date */}
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <Badge variant="outline" className={`text-xs ${sourceColor}`}>
            {sourceLabel}
          </Badge>
          {doc.author && <span>{doc.author}</span>}
          {date && (
            <>
              <span aria-hidden="true">·</span>
              <span>{date}</span>
            </>
          )}
        </div>

        {/* User tags (inline editable) */}
        <InlineTagEditor
          tags={doc.user_tags}
          onSave={(tags) => onTagSave(doc.id, tags)}
          isSaving={isMutating}
        />
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-1 shrink-0 flex-wrap justify-end">
        {doc.origin === 'radar' ? (
          /* Radar-only: Promote to Pro instead of Archive */
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1 text-xs text-green-700 hover:text-green-800"
            onClick={() => onPromote(doc.id)}
            disabled={isMutating}
            title="Promote to Pro index"
          >
            <ArrowUpCircle className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Promote</span>
          </Button>
        ) : (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={() => onArchive(doc.id)}
            disabled={isMutating}
            title={doc.is_archived ? 'Unarchive' : 'Archive'}
          >
            <Archive className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">
              {doc.is_archived ? 'Unarchive' : 'Archive'}
            </span>
          </Button>
        )}

        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1 text-xs"
          asChild
          title="Open source"
        >
          <a href={doc.url} target="_blank" rel="noopener noreferrer">
            <ExternalLink className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Open</span>
          </a>
        </Button>

        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1 text-xs text-destructive hover:text-destructive"
          onClick={() => onDelete(doc.id)}
          disabled={isMutating}
          title="Delete document"
        >
          <Trash2 className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Delete</span>
        </Button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Bulk delete confirmation dialog
// ---------------------------------------------------------------------------

interface BulkDeleteDialogProps {
  count: number
  open: boolean
  onConfirm: () => void
  onCancel: () => void
  isDeleting: boolean
}

function BulkDeleteDialog({
  count,
  open,
  onConfirm,
  onCancel,
  isDeleting,
}: BulkDeleteDialogProps) {
  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onCancel() }}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete {count} document{count !== 1 ? 's' : ''}?</DialogTitle>
          <DialogDescription>
            This will permanently delete {count} selected document
            {count !== 1 ? 's' : ''} from your knowledge base. This action
            cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isDeleting}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onConfirm}
            disabled={isDeleting}
          >
            {isDeleting ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : null}
            Delete {count} document{count !== 1 ? 's' : ''}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// Main Documents page
// ---------------------------------------------------------------------------

/**
 * Document Manager page — allows users to browse all indexed documents,
 * filter by source / origin / archived status, add tags, archive, delete,
 * and promote radar documents to the Pro index.
 */
export function DocumentsPage() {
  const queryClient = useQueryClient()

  // Filter state
  const [sourceFilter, setSourceFilter] = useState<string>('')
  const [originFilter, setOriginFilter] = useState<string>('')
  const [showArchived, setShowArchived] = useState(false)

  // Pagination
  const [page, setPage] = useState(0)
  /** All documents loaded so far — accumulated across pages */
  const [allDocs, setAllDocs] = useState<Document[]>([])

  // Selection
  const [selected, setSelected] = useState<Set<string>>(new Set())

  // Bulk delete dialog
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)

  // ---------------------------------------------------------------------------
  // Query key — resets page + allDocs when filters change
  // ---------------------------------------------------------------------------

  const queryKey = ['documents', sourceFilter, originFilter, showArchived, page]

  const { data: pageDocs, isFetching, isError } = useQuery({
    queryKey,
    queryFn: async () => {
      const params: Parameters<typeof api.documents.list>[0] = {
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }
      if (sourceFilter) params.source_type = sourceFilter
      if (originFilter) params.origin = originFilter
      if (showArchived) params.is_archived = true
      return api.documents.list(params)
    },
    staleTime: 30_000,
  })

  // Merge newly fetched page into allDocs.
  // When page resets to 0 (filter change), clear previous accumulation.
  const prevFiltersRef = useRef({ sourceFilter, originFilter, showArchived })
  const filtersChanged =
    prevFiltersRef.current.sourceFilter !== sourceFilter ||
    prevFiltersRef.current.originFilter !== originFilter ||
    prevFiltersRef.current.showArchived !== showArchived

  if (filtersChanged) {
    prevFiltersRef.current = { sourceFilter, originFilter, showArchived }
    if (allDocs.length > 0) {
      setAllDocs([])
    }
    if (page !== 0) {
      setPage(0)
    }
    setSelected(new Set())
  } else if (pageDocs && !isFetching) {
    // Append new docs if not already present (by id)
    const existingIds = new Set(allDocs.map((d) => d.id))
    const newDocs = pageDocs.filter((d) => !existingIds.has(d.id))
    if (newDocs.length > 0) {
      setAllDocs((prev) => [...prev, ...newDocs])
    }
  }

  const hasMore = pageDocs !== undefined && pageDocs.length === PAGE_SIZE

  // ---------------------------------------------------------------------------
  // Mutations
  // ---------------------------------------------------------------------------

  function invalidate() {
    // Invalidate all document queries to force a fresh fetch
    void queryClient.invalidateQueries({ queryKey: ['documents'] })
    setAllDocs([])
    setPage(0)
  }

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.documents.delete(id),
    onSuccess: () => invalidate(),
  })

  const archiveMutation = useMutation({
    mutationFn: ({ id, is_archived }: { id: string; is_archived: boolean }) =>
      api.documents.update(id, { is_archived }),
    onSuccess: () => invalidate(),
  })

  const tagMutation = useMutation({
    mutationFn: ({ id, user_tags }: { id: string; user_tags: string[] }) =>
      api.documents.update(id, { user_tags }),
    onSuccess: () => invalidate(),
  })

  const promoteMutation = useMutation({
    mutationFn: (id: string) => api.radar.promote(id),
    onSuccess: () => invalidate(),
  })

  /** Whether any mutation is currently in-flight */
  const anyMutating =
    deleteMutation.isPending ||
    archiveMutation.isPending ||
    tagMutation.isPending ||
    promoteMutation.isPending

  // Bulk delete state
  const [isBulkDeleting, setIsBulkDeleting] = useState(false)

  // ---------------------------------------------------------------------------
  // Selection helpers
  // ---------------------------------------------------------------------------

  function handleSelectOne(id: string, checked: boolean) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (checked) next.add(id)
      else next.delete(id)
      return next
    })
  }

  function handleSelectAll(checked: boolean) {
    if (checked) {
      setSelected(new Set(allDocs.map((d) => d.id)))
    } else {
      setSelected(new Set())
    }
  }

  const allSelected = allDocs.length > 0 && selected.size === allDocs.length
  const someSelected = selected.size > 0 && selected.size < allDocs.length

  // ---------------------------------------------------------------------------
  // Bulk actions
  // ---------------------------------------------------------------------------

  function handleBulkArchive() {
    const ids = Array.from(selected)
    ids.forEach((id) => {
      const doc = allDocs.find((d) => d.id === id)
      if (doc) {
        archiveMutation.mutate({ id, is_archived: !doc.is_archived })
      }
    })
    setSelected(new Set())
  }

  async function handleBulkDeleteConfirm() {
    setIsBulkDeleting(true)
    const ids = Array.from(selected)
    try {
      await Promise.all(ids.map((id) => api.documents.delete(id)))
    } finally {
      setIsBulkDeleting(false)
      setShowDeleteDialog(false)
      setSelected(new Set())
      invalidate()
    }
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const displayDocs = allDocs.length > 0 ? allDocs : pageDocs ?? []

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold tracking-tight">Documents</h2>

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-3">
        <Select
          value={sourceFilter}
          onValueChange={(v) => {
            setSourceFilter(v)
          }}
        >
          <SelectTrigger className="w-44">
            <SelectValue placeholder="All Sources" />
          </SelectTrigger>
          <SelectContent>
            {SOURCE_OPTIONS.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select
          value={originFilter}
          onValueChange={(v) => {
            setOriginFilter(v)
          }}
        >
          <SelectTrigger className="w-44">
            <SelectValue placeholder="All Origins" />
          </SelectTrigger>
          <SelectContent>
            {ORIGIN_OPTIONS.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Show archived toggle */}
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <Checkbox
            checked={showArchived}
            onCheckedChange={(v) => {
              setShowArchived(v === true)
            }}
          />
          Show archived
        </label>
      </div>

      {/* Bulk action bar — visible when at least one document is selected */}
      {selected.size > 0 && (
        <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-4 py-2">
          <span className="text-sm font-medium">
            {selected.size} selected
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={handleBulkArchive}
            disabled={anyMutating}
            className="gap-1"
          >
            <Archive className="h-3.5 w-3.5" />
            Archive
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowDeleteDialog(true)}
            disabled={anyMutating}
            className="gap-1 text-destructive border-destructive hover:bg-destructive/10"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Delete
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelected(new Set())}
            className="ml-auto"
          >
            Clear selection
          </Button>
        </div>
      )}

      {/* Document list */}
      <div className="rounded-md border">
        {/* Header row with Select All */}
        <div className="flex items-center gap-3 border-b px-4 py-2 bg-muted/30">
          <Checkbox
            checked={allSelected || (someSelected ? 'indeterminate' : false)}
            onCheckedChange={(checked) => handleSelectAll(checked === true)}
            aria-label="Select all documents"
          />
          <span className="text-xs text-muted-foreground font-medium">
            {allDocs.length > 0
              ? `${allDocs.length} document${allDocs.length !== 1 ? 's' : ''} loaded`
              : 'No documents'}
          </span>
        </div>

        {/* Loading state — first load */}
        {isFetching && allDocs.length === 0 && (
          <div>
            {[1, 2, 3, 4, 5].map((i) => (
              <DocumentRowSkeleton key={i} />
            ))}
          </div>
        )}

        {/* Error state */}
        {isError && (
          <div className="px-4 py-12 text-center">
            <p className="text-sm text-destructive">
              Failed to load documents. Please try again.
            </p>
          </div>
        )}

        {/* Empty state */}
        {!isFetching && !isError && displayDocs.length === 0 && (
          <div className="px-4 py-16 text-center">
            <p className="text-muted-foreground">No documents found.</p>
            {(sourceFilter || originFilter || showArchived) && (
              <p className="mt-1 text-sm text-muted-foreground">
                Try adjusting your filters.
              </p>
            )}
          </div>
        )}

        {/* Document rows */}
        {displayDocs.map((doc) => (
          <DocumentRow
            key={doc.id}
            doc={doc}
            isSelected={selected.has(doc.id)}
            onSelect={handleSelectOne}
            onArchive={(id) => {
              const d = allDocs.find((x) => x.id === id)
              archiveMutation.mutate({ id, is_archived: !(d?.is_archived ?? false) })
            }}
            onDelete={(id) => deleteMutation.mutate(id)}
            onPromote={(id) => promoteMutation.mutate(id)}
            onTagSave={(id, tags) => tagMutation.mutate({ id, user_tags: tags })}
            isMutating={anyMutating}
          />
        ))}

        {/* Load more */}
        {hasMore && (
          <div className="flex justify-center border-t px-4 py-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => p + 1)}
              disabled={isFetching}
              className="gap-1"
            >
              {isFetching ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <ChevronDown className="h-3.5 w-3.5" />
              )}
              Load more
            </Button>
          </div>
        )}
      </div>

      {/* Bulk delete confirmation dialog */}
      <BulkDeleteDialog
        count={selected.size}
        open={showDeleteDialog}
        onConfirm={() => void handleBulkDeleteConfirm()}
        onCancel={() => setShowDeleteDialog(false)}
        isDeleting={isBulkDeleting}
      />
    </div>
  )
}
