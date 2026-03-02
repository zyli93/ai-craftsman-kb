/**
 * AdhocIngest component — URL input form for manually ingesting a single URL
 * into the knowledge base with optional tags.
 *
 * Features:
 *  - URL input with client-side type detection (YouTube, ArXiv, Substack, etc.)
 *  - Tag input supporting comma-separated or Enter-separated entry
 *  - Shows ingested DocumentCard on success
 *  - Shows inline error message on failure
 */
import { useState, type KeyboardEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Loader2, Link as LinkIcon, X } from 'lucide-react'
import { api } from '@/api/client'
import type { Document } from '@/api/types'
import { DocumentCard } from '@/components/DocumentCard'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'

export interface AdhocIngestProps {
  /** Called with the ingested document on successful ingest */
  onSuccess?: (doc: Document) => void
}

/**
 * Detects the document type from a URL string using client-side regex matching.
 * Returns a human-readable label for display.
 */
function detectType(url: string): string {
  if (!url) return ''
  if (/youtube\.com|youtu\.be/.test(url)) return 'YouTube Video'
  if (/arxiv\.org/.test(url)) return 'ArXiv Paper'
  if (/substack\.com/.test(url)) return 'Substack Article'
  if (/reddit\.com/.test(url)) return 'Reddit Post'
  if (/news\.ycombinator\.com/.test(url)) return 'HN Thread'
  if (/dev\.to/.test(url)) return 'DEV.to Article'
  if (/\.pdf($|\?)/.test(url)) return 'PDF Document'
  return 'Web Article'
}

/**
 * Parses a tag string, splitting on commas and trimming whitespace.
 * Filters out empty strings.
 */
function parseTags(raw: string): string[] {
  return raw
    .split(',')
    .map((t) => t.trim())
    .filter((t) => t.length > 0)
}

/**
 * AdhocIngest — standalone form component for ingesting a single URL.
 * Can be embedded in any page that needs on-demand URL ingestion.
 */
export function AdhocIngest({ onSuccess }: AdhocIngestProps) {
  const [url, setUrl] = useState('')
  const [tagInput, setTagInput] = useState('')
  const [tags, setTags] = useState<string[]>([])
  const [ingestedDoc, setIngestedDoc] = useState<Document | null>(null)

  const detectedType = url ? detectType(url) : null

  const ingestMutation = useMutation({
    mutationFn: () => api.ingest.url({ url: url.trim(), tags: tags.length > 0 ? tags : undefined }),
    onSuccess: (doc) => {
      setIngestedDoc(doc)
      setUrl('')
      setTags([])
      setTagInput('')
      onSuccess?.(doc)
    },
  })

  /** Commits any pending text in the tag input to the tags list */
  function commitTagInput() {
    const newTags = parseTags(tagInput)
    if (newTags.length > 0) {
      setTags((prev) => {
        const combined = [...prev, ...newTags]
        // Deduplicate
        return Array.from(new Set(combined))
      })
      setTagInput('')
    }
  }

  /** Handles keydown events on the tag input field */
  function handleTagKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault()
      commitTagInput()
    } else if (e.key === 'Backspace' && tagInput === '' && tags.length > 0) {
      // Remove the last tag if backspace is pressed with an empty input
      setTags((prev) => prev.slice(0, -1))
    }
  }

  /** Removes a tag by index */
  function removeTag(index: number) {
    setTags((prev) => prev.filter((_, i) => i !== index))
  }

  function handleIngest() {
    // Commit any pending tag text before ingesting
    commitTagInput()
    if (!url.trim()) return
    setIngestedDoc(null)
    ingestMutation.mutate()
  }

  function handleUrlKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') handleIngest()
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Paste a URL to ingest and index it immediately into your knowledge base.
      </p>

      {/* URL input row */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <LinkIcon className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="url"
            placeholder="https://..."
            value={url}
            onChange={(e) => {
              setUrl(e.target.value)
              // Reset success state when URL changes
              if (ingestedDoc) setIngestedDoc(null)
            }}
            onKeyDown={handleUrlKeyDown}
            className="pl-9"
            aria-label="URL to ingest"
          />
        </div>
        <Button
          onClick={handleIngest}
          disabled={!url.trim() || ingestMutation.isPending}
        >
          {ingestMutation.isPending && (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          )}
          Ingest &amp; Index
        </Button>
      </div>

      {/* Auto-detected URL type hint */}
      {detectedType && (
        <p className="text-xs text-muted-foreground">
          Detected type:{' '}
          <span className="font-medium text-foreground">{detectedType}</span>
        </p>
      )}

      {/* Tag input */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground">
          Tags{' '}
          <span className="font-normal">(optional — press Enter or comma to add)</span>
        </label>
        <div className="flex flex-wrap items-center gap-1.5 rounded-md border border-input bg-background px-3 py-1.5 min-h-[36px] focus-within:ring-1 focus-within:ring-ring">
          {/* Existing tags as removable badges */}
          {tags.map((tag, idx) => (
            <Badge
              key={idx}
              variant="secondary"
              className="flex items-center gap-1 text-xs pr-1"
            >
              {tag}
              <button
                type="button"
                onClick={() => removeTag(idx)}
                className="rounded-full hover:bg-muted-foreground/20 p-0.5"
                aria-label={`Remove tag "${tag}"`}
              >
                <X className="h-2.5 w-2.5" />
              </button>
            </Badge>
          ))}

          {/* Tag text input */}
          <input
            type="text"
            value={tagInput}
            onChange={(e) => {
              const val = e.target.value
              // Auto-commit on trailing comma
              if (val.endsWith(',')) {
                const withoutComma = val.slice(0, -1).trim()
                if (withoutComma) {
                  setTags((prev) => Array.from(new Set([...prev, withoutComma])))
                  setTagInput('')
                } else {
                  setTagInput('')
                }
              } else {
                setTagInput(val)
              }
            }}
            onKeyDown={handleTagKeyDown}
            onBlur={commitTagInput}
            placeholder={tags.length === 0 ? 'e.g. ai, research, paper' : ''}
            className="flex-1 min-w-24 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            aria-label="Add tags"
          />
        </div>
      </div>

      {/* Error state */}
      {ingestMutation.isError && (
        <p className="text-sm text-destructive">
          Failed to ingest URL. Please check the URL and try again.
        </p>
      )}

      {/* Success state — show the ingested document */}
      {ingestedDoc && (
        <div className="space-y-2">
          <p className="text-sm font-medium text-green-700 dark:text-green-500">
            Document ingested successfully.
          </p>
          <DocumentCard document={ingestedDoc} showOriginBadge />
        </div>
      )}
    </div>
  )
}
