/**
 * Briefing Builder page — generate AI briefings on a topic, view rendered
 * markdown content, see source citations, and export the result.
 *
 * Layout: two-column — narrow left panel (form + history list), wide right
 * panel (rendered briefing content + actions).
 */
import { useState, useEffect, useRef } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import {
  Loader2,
  Trash2,
  Copy,
  Download,
  RefreshCw,
  FileText,
  Clock,
} from 'lucide-react'
import { api } from '@/api/client'
import type { Briefing } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Status messages cycled during briefing generation. */
const LOADING_MESSAGES = [
  'Searching knowledge base...',
  'Ingesting fresh content...',
  'Running radar search...',
  'Synthesising key themes...',
  'Generating briefing...',
  'Formatting sources...',
]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Formats an ISO timestamp to a human-readable short date.
 */
function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })
  } catch {
    return iso
  }
}

/**
 * Downloads a string as a Markdown file.
 */
function downloadMarkdown(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/markdown; charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

/**
 * Sanitises a briefing title into a filesystem-safe filename stem.
 */
function toFilename(title: string): string {
  return (
    title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 60) || 'briefing'
  )
}

// ---------------------------------------------------------------------------
// Loading indicator
// ---------------------------------------------------------------------------

/**
 * Animated loading indicator that cycles through status messages.
 * Intended for the long (10-30s) briefing generation operation.
 */
function GeneratingIndicator() {
  const [messageIndex, setMessageIndex] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % LOADING_MESSAGES.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex flex-col items-center justify-center gap-4 py-20">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
      <p className="text-sm text-muted-foreground transition-all duration-300">
        {LOADING_MESSAGES[messageIndex]}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// History item
// ---------------------------------------------------------------------------

interface HistoryItemProps {
  briefing: Briefing
  isActive: boolean
  onSelect: () => void
  onDelete: () => void
  isDeleting: boolean
}

/** Single row in the briefing history list. */
function HistoryItem({
  briefing,
  isActive,
  onSelect,
  onDelete,
  isDeleting,
}: HistoryItemProps) {
  return (
    <div
      className={cn(
        'group flex items-start gap-2 rounded-lg border p-3 cursor-pointer transition-colors',
        isActive
          ? 'border-primary bg-primary/5'
          : 'hover:bg-muted/50 hover:border-muted-foreground/30',
      )}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onSelect()
      }}
    >
      <FileText className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />

      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium leading-tight">
          {briefing.title}
        </p>
        <div className="mt-1 flex items-center gap-1.5 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" />
          <span>{formatDate(briefing.created_at)}</span>
          {briefing.source_document_ids.length > 0 && (
            <Badge variant="secondary" className="text-xs px-1 py-0">
              {briefing.source_document_ids.length} src
            </Badge>
          )}
        </div>
      </div>

      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive"
        onClick={(e) => {
          e.stopPropagation()
          onDelete()
        }}
        disabled={isDeleting}
        aria-label="Delete briefing"
      >
        {isDeleting ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : (
          <Trash2 className="h-3 w-3" />
        )}
      </Button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Right panel — briefing display
// ---------------------------------------------------------------------------

interface BriefingPanelProps {
  briefing: Briefing
  onRegenerate: () => void
  isRegenerating: boolean
}

/** Displays a rendered briefing with source citations and action buttons. */
function BriefingPanel({ briefing, onRegenerate, isRegenerating }: BriefingPanelProps) {
  const [copySuccess, setCopySuccess] = useState(false)

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(briefing.content)
      setCopySuccess(true)
      setTimeout(() => setCopySuccess(false), 2000)
    } catch {
      // Clipboard write failed — silently ignore
    }
  }

  function handleExport() {
    downloadMarkdown(`${toFilename(briefing.title)}.md`, briefing.content)
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 pb-4">
        <div className="min-w-0">
          <h3 className="text-lg font-semibold leading-tight">{briefing.title}</h3>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Generated {formatDate(briefing.created_at)}
            {briefing.query && (
              <> &middot; Query: <span className="italic">{briefing.query}</span></>
            )}
          </p>
        </div>

        {/* Source count badge */}
        <div className="shrink-0">
          {briefing.source_document_ids.length > 0 ? (
            <Badge variant="secondary">
              {briefing.source_document_ids.length} source
              {briefing.source_document_ids.length !== 1 ? 's' : ''}
            </Badge>
          ) : (
            <Badge variant="outline" className="text-muted-foreground">
              No sources cited
            </Badge>
          )}
        </div>
      </div>

      <Separator className="mb-4" />

      {/* Rendered markdown — scrollable */}
      <div className="flex-1 overflow-y-auto rounded-md border bg-muted/20 p-4">
        <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:font-semibold prose-headings:tracking-tight prose-p:leading-relaxed prose-li:leading-relaxed">
          <ReactMarkdown>{briefing.content}</ReactMarkdown>
        </div>
      </div>

      <Separator className="my-4" />

      {/* Action buttons */}
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={handleCopy}
          className={cn(copySuccess && 'text-green-700 border-green-400')}
        >
          <Copy className="mr-1.5 h-4 w-4" />
          {copySuccess ? 'Copied!' : 'Copy as Markdown'}
        </Button>

        <Button variant="outline" size="sm" onClick={handleExport}>
          <Download className="mr-1.5 h-4 w-4" />
          Export .md
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={onRegenerate}
          disabled={isRegenerating}
        >
          {isRegenerating ? (
            <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="mr-1.5 h-4 w-4" />
          )}
          Regenerate
        </Button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main BriefingPage component
// ---------------------------------------------------------------------------

/**
 * Briefing Builder page.
 *
 * Left panel: topic input form + briefing history list.
 * Right panel: rendered markdown content for the active briefing.
 */
export function BriefingPage() {
  const queryClient = useQueryClient()

  // Form state
  const [topic, setTopic] = useState('')
  const [runRadar, setRunRadar] = useState(true)
  const [runIngest, setRunIngest] = useState(false)

  // Active briefing (either just generated or selected from history)
  const [activeBriefing, setActiveBriefing] = useState<Briefing | null>(null)

  // Track which briefing is being deleted
  const [deletingId, setDeletingId] = useState<string | null>(null)

  // Ref to the topic textarea for focus management
  const topicRef = useRef<HTMLTextAreaElement>(null)

  // Fetch briefing history
  const { data: history, isLoading: historyLoading } = useQuery({
    queryKey: ['briefings'],
    queryFn: () => api.briefings.list(),
  })

  // Generate briefing mutation
  const generateMutation = useMutation({
    mutationFn: () =>
      api.briefings.create({
        query: topic.trim(),
        run_radar: runRadar,
        run_ingest: runIngest,
      }),
    onSuccess: (briefing) => {
      setActiveBriefing(briefing)
      void queryClient.invalidateQueries({ queryKey: ['briefings'] })
    },
  })

  // Delete briefing mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.briefings.delete(id),
    onSuccess: (_data, deletedId) => {
      // Clear active briefing if it was the one deleted
      if (activeBriefing?.id === deletedId) {
        setActiveBriefing(null)
      }
      setDeletingId(null)
      void queryClient.invalidateQueries({ queryKey: ['briefings'] })
    },
    onError: () => {
      setDeletingId(null)
    },
  })

  function handleGenerate() {
    if (!topic.trim() || generateMutation.isPending) return
    generateMutation.mutate()
  }

  function handleDelete(id: string) {
    setDeletingId(id)
    deleteMutation.mutate(id)
  }

  function handleRegenerate() {
    if (!activeBriefing?.query || generateMutation.isPending) return
    setTopic(activeBriefing.query)
    generateMutation.mutate()
  }

  // Refocus topic input after generation completes
  useEffect(() => {
    if (generateMutation.isSuccess) {
      topicRef.current?.focus()
    }
  }, [generateMutation.isSuccess])

  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col gap-4">
      <h2 className="text-2xl font-bold tracking-tight">Briefing Builder</h2>

      <div className="flex min-h-0 flex-1 gap-6">
        {/* ---------------------------------------------------------------- */}
        {/* Left panel — form + history                                       */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex w-72 shrink-0 flex-col gap-4">
          {/* Form card */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                New Briefing
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Topic input */}
              <div className="space-y-1.5">
                <label
                  htmlFor="briefing-topic"
                  className="text-sm font-medium"
                >
                  Topic
                </label>
                <Textarea
                  id="briefing-topic"
                  ref={topicRef}
                  placeholder="e.g. LLM inference optimization"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                      handleGenerate()
                    }
                  }}
                  className="min-h-[80px] resize-none"
                  disabled={generateMutation.isPending}
                />
              </div>

              {/* Options */}
              <div className="space-y-2">
                <p className="text-sm font-medium">Options</p>

                <label className="flex cursor-pointer items-center gap-2 text-sm">
                  <Checkbox
                    checked={runIngest}
                    onCheckedChange={(checked) =>
                      setRunIngest(checked === true)
                    }
                    disabled={generateMutation.isPending}
                  />
                  Ingest fresh content
                </label>

                <label className="flex cursor-pointer items-center gap-2 text-sm">
                  <Checkbox
                    checked={runRadar}
                    onCheckedChange={(checked) =>
                      setRunRadar(checked === true)
                    }
                    disabled={generateMutation.isPending}
                  />
                  Run radar search
                </label>
              </div>

              {/* Generate button */}
              <Button
                className="w-full"
                onClick={handleGenerate}
                disabled={!topic.trim() || generateMutation.isPending}
              >
                {generateMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : null}
                {generateMutation.isPending ? 'Generating...' : 'Generate Briefing'}
              </Button>

              {/* Keyboard tip */}
              <p className="text-xs text-muted-foreground">
                Tip: press Cmd+Enter to generate
              </p>

              {/* Error state */}
              {generateMutation.isError && (
                <p className="text-xs text-destructive">
                  Generation failed. Please try again.
                </p>
              )}
            </CardContent>
          </Card>

          {/* History list */}
          <Card className="flex min-h-0 flex-1 flex-col">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                History
                {history && history.length > 0 && (
                  <Badge variant="secondary" className="text-xs">
                    {history.length}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="min-h-0 flex-1 overflow-y-auto p-3 pt-0">
              {historyLoading && (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-14 w-full rounded-lg" />
                  ))}
                </div>
              )}

              {!historyLoading && (!history || history.length === 0) && (
                <p className="py-4 text-center text-xs text-muted-foreground">
                  No briefings yet. Generate your first one above.
                </p>
              )}

              {history && history.length > 0 && (
                <div className="space-y-2">
                  {history.map((b) => (
                    <HistoryItem
                      key={b.id}
                      briefing={b}
                      isActive={activeBriefing?.id === b.id}
                      onSelect={() => setActiveBriefing(b)}
                      onDelete={() => handleDelete(b.id)}
                      isDeleting={deletingId === b.id}
                    />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* Right panel — briefing content                                    */}
        {/* ---------------------------------------------------------------- */}
        <div className="min-w-0 flex-1">
          <Card className="flex h-full flex-col">
            <CardContent className="flex min-h-0 flex-1 flex-col p-6">
              {/* Generating state */}
              {generateMutation.isPending && <GeneratingIndicator />}

              {/* Active briefing display */}
              {!generateMutation.isPending && activeBriefing && (
                <BriefingPanel
                  briefing={activeBriefing}
                  onRegenerate={handleRegenerate}
                  isRegenerating={generateMutation.isPending}
                />
              )}

              {/* Initial empty state */}
              {!generateMutation.isPending && !activeBriefing && (
                <div className="flex flex-1 flex-col items-center justify-center gap-4 text-center">
                  <FileText className="h-12 w-12 text-muted-foreground/30" />
                  <div>
                    <p className="font-medium text-muted-foreground">
                      No briefing selected
                    </p>
                    <p className="mt-1 text-sm text-muted-foreground/70">
                      Enter a topic and click Generate Briefing, or select a
                      previous briefing from the history list.
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
