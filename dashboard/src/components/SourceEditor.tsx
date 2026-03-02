/**
 * SourceEditor component — dialog for adding or editing a source.
 * Handles form state for source_type, identifier, and display_name fields.
 */
import { useState } from 'react'
import { useQueryClient, useMutation } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { Source, SourceCreate } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

/** All supported source types with human-readable labels */
const SOURCE_TYPES = [
  { value: 'hn', label: 'Hacker News' },
  { value: 'substack', label: 'Substack' },
  { value: 'youtube', label: 'YouTube' },
  { value: 'reddit', label: 'Reddit' },
  { value: 'rss', label: 'RSS Feed' },
  { value: 'arxiv', label: 'ArXiv' },
  { value: 'devto', label: 'DEV.to' },
] as const

/** Placeholder/help text for the identifier field, keyed by source type */
const IDENTIFIER_PLACEHOLDERS: Record<string, string> = {
  hn: 'No identifier needed (global source)',
  substack: 'e.g. simonwillison (slug from URL)',
  youtube: 'e.g. @andrejkarpathy',
  reddit: 'e.g. machinelearning (subreddit name)',
  rss: 'e.g. https://example.com/feed.xml',
  arxiv: 'No identifier needed (global source)',
  devto: 'No identifier needed (global source)',
}

/** Source types that do not require an identifier */
const NO_IDENTIFIER_TYPES = new Set(['hn', 'arxiv', 'devto'])

interface SourceEditorProps {
  /** Whether the dialog is open */
  open: boolean
  /** Called when the dialog should close */
  onClose: () => void
  /** When provided, dialog is in edit mode for this source */
  source?: Source
}

/**
 * Dialog for adding a new source or editing an existing one.
 * On save, calls POST /api/sources (create) or PUT /api/sources/{id} (update).
 */
export function SourceEditor({ open, onClose, source }: SourceEditorProps) {
  const queryClient = useQueryClient()
  const isEditMode = source !== undefined

  const [sourceType, setSourceType] = useState<string>(source?.source_type ?? 'substack')
  const [identifier, setIdentifier] = useState<string>(source?.identifier ?? '')
  const [displayName, setDisplayName] = useState<string>(source?.display_name ?? '')
  const [error, setError] = useState<string | null>(null)

  const createMutation = useMutation({
    mutationFn: (data: SourceCreate) => api.sources.create(data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['sources'] })
      handleClose()
    },
    onError: (err: Error) => {
      setError(err.message)
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: { display_name?: string } }) =>
      api.sources.update(id, data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['sources'] })
      handleClose()
    },
    onError: (err: Error) => {
      setError(err.message)
    },
  })

  const isPending = createMutation.isPending || updateMutation.isPending

  function handleClose() {
    setError(null)
    if (!isEditMode) {
      setSourceType('substack')
      setIdentifier('')
      setDisplayName('')
    }
    onClose()
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)

    const needsIdentifier = !NO_IDENTIFIER_TYPES.has(sourceType)
    if (needsIdentifier && !identifier.trim()) {
      setError('Identifier is required for this source type.')
      return
    }

    if (isEditMode && source) {
      updateMutation.mutate({
        id: source.id,
        data: { display_name: displayName.trim() || undefined },
      })
    } else {
      createMutation.mutate({
        source_type: sourceType,
        identifier: identifier.trim(),
        display_name: displayName.trim() || undefined,
        enabled: true,
      })
    }
  }

  const noIdentifierNeeded = NO_IDENTIFIER_TYPES.has(sourceType)

  return (
    <Dialog open={open} onOpenChange={(isOpen) => { if (!isOpen) handleClose() }}>
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle>{isEditMode ? 'Edit Source' : 'Add Source'}</DialogTitle>
          <DialogDescription>
            {isEditMode
              ? 'Update the display name for this source.'
              : 'Configure a new content source to ingest.'}
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4 py-2">
          {/* Source type selector — disabled in edit mode */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium leading-none" htmlFor="source-type">
              Source Type
            </label>
            <Select
              value={sourceType}
              onValueChange={(val) => {
                setSourceType(val)
                setIdentifier('')
              }}
              disabled={isEditMode}
            >
              <SelectTrigger id="source-type">
                <SelectValue placeholder="Select a source type" />
              </SelectTrigger>
              <SelectContent>
                {SOURCE_TYPES.map(({ value, label }) => (
                  <SelectItem key={value} value={value}>
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Identifier field — hidden for global sources when adding */}
          {!isEditMode && (
            <div className="space-y-1.5">
              <label className="text-sm font-medium leading-none" htmlFor="identifier">
                Identifier
              </label>
              <Input
                id="identifier"
                value={identifier}
                onChange={(e) => setIdentifier(e.target.value)}
                placeholder={IDENTIFIER_PLACEHOLDERS[sourceType] ?? 'Enter identifier'}
                disabled={noIdentifierNeeded}
                className={noIdentifierNeeded ? 'opacity-50' : ''}
              />
              {noIdentifierNeeded && (
                <p className="text-xs text-muted-foreground">
                  This source type does not require an identifier.
                </p>
              )}
            </div>
          )}

          {/* Display name — optional for both add and edit */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium leading-none" htmlFor="display-name">
              Display Name
              <span className="ml-1 text-xs font-normal text-muted-foreground">(optional)</span>
            </label>
            <Input
              id="display-name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Friendly name for this source"
            />
          </div>

          {/* Error message */}
          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}

          <DialogFooter>
            <Button type="button" variant="outline" onClick={handleClose} disabled={isPending}>
              Cancel
            </Button>
            <Button type="submit" disabled={isPending}>
              {isPending ? 'Saving...' : isEditMode ? 'Save Changes' : 'Add Source'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
