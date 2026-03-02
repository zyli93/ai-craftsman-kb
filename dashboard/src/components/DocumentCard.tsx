/**
 * DocumentCard component — reusable card for displaying a Document.
 * Used across Search, Radar, and Document Manager pages.
 * Shows title (linked), source_type badge, author, relative date,
 * excerpt (first 200 chars), and optional action buttons.
 */
import { formatDistanceToNow } from 'date-fns'
import { Star, Tag, ExternalLink, ArrowUpCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import type { Document } from '@/api/types'

/** Maps source_type identifiers to display labels */
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

/** Maps source_type identifiers to Tailwind badge color classes */
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

export interface DocumentCardProps {
  /** The document to display */
  document: Document
  /** Relevance score (0-1) shown when coming from search results */
  score?: number
  /** When true, shows an origin badge (pro / radar / adhoc) */
  showOriginBadge?: boolean
  /** Called when the user clicks the Favorite button */
  onFavorite?: () => void
  /** Called when the user clicks the Tag button */
  onTag?: () => void
  /** Called when the user clicks the Promote button (Radar tab only) */
  onPromote?: () => void
}

/**
 * Truncates a string to at most `maxLength` characters, appending ellipsis.
 */
function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength).trimEnd() + '...'
}

/**
 * Formats a date string into a relative time string (e.g. "3 days ago").
 * Falls back to an empty string if the date is null or unparseable.
 */
function formatRelativeDate(dateStr: string | null): string {
  if (!dateStr) return ''
  try {
    return formatDistanceToNow(new Date(dateStr), { addSuffix: true })
  } catch {
    return ''
  }
}

/**
 * Reusable card component for displaying a single document result.
 * Accepts optional callbacks for favorite, tag, and promote actions.
 */
export function DocumentCard({
  document: doc,
  score,
  showOriginBadge = false,
  onFavorite,
  onTag,
  onPromote,
}: DocumentCardProps) {
  const sourceLabel = SOURCE_TYPE_LABELS[doc.source_type] ?? doc.source_type
  const sourceColor =
    SOURCE_TYPE_COLORS[doc.source_type] ?? 'bg-gray-100 text-gray-800 border-gray-200'
  const relativeDate = formatRelativeDate(doc.published_at ?? doc.fetched_at)
  const excerpt = doc.excerpt ? truncate(doc.excerpt, 200) : null

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="pt-4 pb-4">
        <div className="flex items-start justify-between gap-3">
          {/* Main content */}
          <div className="min-w-0 flex-1 space-y-1.5">
            {/* Title + badges row */}
            <div className="flex flex-wrap items-center gap-2">
              <a
                href={doc.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-semibold leading-tight hover:underline text-foreground line-clamp-2"
              >
                {doc.title ?? doc.url}
              </a>
            </div>

            {/* Meta row: source badge, author, date, score */}
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <Badge
                variant="outline"
                className={`text-xs ${sourceColor}`}
              >
                {sourceLabel}
              </Badge>

              {showOriginBadge && (
                <Badge variant="secondary" className="text-xs">
                  {doc.origin}
                </Badge>
              )}

              {doc.author && <span>{doc.author}</span>}

              {relativeDate && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>{relativeDate}</span>
                </>
              )}

              {score !== undefined && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>Score: {score.toFixed(2)}</span>
                </>
              )}
            </div>

            {/* Excerpt */}
            {excerpt && (
              <p className="text-xs text-muted-foreground leading-relaxed">
                {excerpt}
              </p>
            )}

            {/* User tags */}
            {doc.user_tags.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {doc.user_tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-xs">
                    {tag}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Action buttons */}
        {(onFavorite !== undefined || onTag !== undefined || onPromote !== undefined) && (
          <div className="mt-3 flex items-center gap-2">
            {onFavorite !== undefined && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onFavorite}
                className="h-7 gap-1 text-xs"
                aria-label={doc.is_favorited ? 'Unfavorite' : 'Favorite'}
              >
                <Star
                  className={`h-3.5 w-3.5 ${doc.is_favorited ? 'fill-yellow-400 text-yellow-400' : ''}`}
                />
                Favorite
              </Button>
            )}

            {onTag !== undefined && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onTag}
                className="h-7 gap-1 text-xs"
                aria-label="Tag document"
              >
                <Tag className="h-3.5 w-3.5" />
                Tag
              </Button>
            )}

            <Button
              variant="ghost"
              size="sm"
              asChild
              className="h-7 gap-1 text-xs"
            >
              <a href={doc.url} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-3.5 w-3.5" />
                Open
              </a>
            </Button>

            {onPromote !== undefined && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onPromote}
                className="h-7 gap-1 text-xs text-green-700 hover:text-green-800"
                aria-label="Promote to Pro"
              >
                <ArrowUpCircle className="h-3.5 w-3.5" />
                Promote
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
