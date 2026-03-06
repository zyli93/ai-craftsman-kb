/**
 * Search & Radar page — combined interface with three tabs:
 *  1. Pro Results   — searches the embedded Pro index
 *  2. Radar Search  — fan-out search across live sources (HN, Reddit, ArXiv, DEV.to, YouTube)
 *  3. Adhoc URL     — ingest a single URL on demand
 */
import { useState, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Loader2 } from 'lucide-react'
import { api } from '@/api/client'
import { useSearch } from '@/hooks/useSearch'
import { SearchBar, type SearchMode } from '@/components/SearchBar'
import { DocumentCard } from '@/components/DocumentCard'
import { AdhocIngest } from '@/components/AdhocIngest'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { Document } from '@/api/types'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ActiveTab = 'pro' | 'radar' | 'adhoc'

const ALL_SOURCES = '__all__'
const ALL_SINCE = '__all__'

const SOURCE_OPTIONS = [
  { label: 'All Sources', value: ALL_SOURCES },
  { label: 'Hacker News', value: 'hn' },
  { label: 'Reddit', value: 'reddit' },
  { label: 'ArXiv', value: 'arxiv' },
  { label: 'DEV.to', value: 'devto' },
  { label: 'YouTube', value: 'youtube' },
  { label: 'Substack', value: 'substack' },
  { label: 'RSS', value: 'rss' },
]

const SINCE_OPTIONS = [
  { label: 'Any time', value: ALL_SINCE },
  { label: 'Past week', value: '7d' },
  { label: 'Past month', value: '30d' },
  { label: 'Past 3 months', value: '90d' },
]

const RADAR_SOURCES = [
  { label: 'HN', value: 'hn' },
  { label: 'Reddit', value: 'reddit' },
  { label: 'ArXiv', value: 'arxiv' },
  { label: 'DEV.to', value: 'devto' },
  { label: 'YouTube', value: 'youtube' },
]

// ---------------------------------------------------------------------------
// Loading skeleton for search results
// ---------------------------------------------------------------------------

function ResultsSkeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="rounded-xl border p-4 space-y-2">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-3 w-1/3" />
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-4/5" />
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Radar tab
// ---------------------------------------------------------------------------

function RadarTab() {
  const [radarQuery, setRadarQuery] = useState('')
  const [selectedSources, setSelectedSources] = useState<string[]>(
    RADAR_SOURCES.map((s) => s.value),
  )
  const [radarResults, setRadarResults] = useState<Document[]>([])

  const radarMutation = useMutation({
    mutationFn: () =>
      api.radar.search({
        query: radarQuery,
        sources: selectedSources,
        limit_per_source: 10,
      }),
    onSuccess: () => {
      // Reload radar results from the API after the search completes
      api.radar.results({ status: 'pending' }).then(setRadarResults)
    },
  })

  const promoteMutation = useMutation({
    mutationFn: (id: string) => api.radar.promote(id),
    onSuccess: () => {
      // Remove promoted document from the list
      setRadarResults((prev) =>
        prev.filter((doc) => promoteMutation.variables !== doc.id),
      )
    },
  })

  function toggleSource(value: string) {
    setSelectedSources((prev) =>
      prev.includes(value)
        ? prev.filter((s) => s !== value)
        : [...prev, value],
    )
  }

  function handleSearch() {
    if (!radarQuery.trim() || selectedSources.length === 0) return
    setRadarResults([])
    radarMutation.mutate()
  }

  return (
    <div className="space-y-4 pt-2">
      <p className="text-sm text-muted-foreground">
        Search live across external sources and optionally promote results into
        your Pro index.
      </p>

      {/* Radar query input */}
      <div className="flex gap-2">
        <Input
          type="text"
          placeholder="Enter a topic to search live sources..."
          value={radarQuery}
          onChange={(e) => setRadarQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSearch()
          }}
          className="flex-1"
        />
        <Button
          onClick={handleSearch}
          disabled={
            !radarQuery.trim() ||
            selectedSources.length === 0 ||
            radarMutation.isPending
          }
        >
          {radarMutation.isPending ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : null}
          Radar Search
        </Button>
      </div>

      {/* Source checkboxes */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span className="text-sm text-muted-foreground">Sources:</span>
        {RADAR_SOURCES.map((src) => (
          <label
            key={src.value}
            className="flex cursor-pointer items-center gap-1.5 text-sm"
          >
            <Checkbox
              checked={selectedSources.includes(src.value)}
              onCheckedChange={() => toggleSource(src.value)}
            />
            {src.label}
          </label>
        ))}
      </div>

      {/* Loading state */}
      {radarMutation.isPending && (
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">
            Searching across sources...
          </p>
          <ResultsSkeleton />
        </div>
      )}

      {/* Error state */}
      {radarMutation.isError && (
        <p className="text-sm text-destructive">
          Radar search failed. Please try again.
        </p>
      )}

      {/* Results */}
      {radarResults.length > 0 && (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            {radarResults.length} result{radarResults.length !== 1 ? 's' : ''}{' '}
            found
          </p>
          {radarResults.map((doc) => (
            <DocumentCard
              key={doc.id}
              document={doc}
              showOriginBadge
              onPromote={() => promoteMutation.mutate(doc.id)}
            />
          ))}
        </div>
      )}

      {/* Empty state */}
      {radarMutation.isSuccess && radarResults.length === 0 && (
        <div className="py-12 text-center">
          <p className="text-muted-foreground">
            No radar results found for this query.
          </p>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main SearchPage component
// ---------------------------------------------------------------------------

/**
 * Search & Radar page with three tabs:
 *  - Pro Results: searches the embedded vector index
 *  - Radar Search: live fan-out search across external sources
 *  - Adhoc URL: ingest a single URL on demand
 */
export function SearchPage() {
  // Shared search state (Pro tab)
  const [inputValue, setInputValue] = useState('')
  const [submittedQuery, setSubmittedQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('hybrid')
  const [sourceFilter, setSourceFilter] = useState<string>(ALL_SOURCES)
  const [sinceFilter, setSinceFilter] = useState<string>(ALL_SINCE)
  const [activeTab, setActiveTab] = useState<ActiveTab>('pro')

  // Convert relative "7d"/"30d"/"90d" into ISO date strings for the backend
  const sinceDate = (() => {
    if (sinceFilter === ALL_SINCE) return undefined
    const match = sinceFilter.match(/^(\d+)d$/)
    if (!match) return undefined
    const d = new Date()
    d.setDate(d.getDate() - Number(match[1]))
    return d.toISOString().split('T')[0]
  })()

  const { data: results, isFetching, isError } = useSearch(
    submittedQuery,
    mode,
    {
      source_type: sourceFilter !== ALL_SOURCES ? sourceFilter : undefined,
      since: sinceDate,
    },
  )

  const handleSubmit = useCallback(() => {
    setSubmittedQuery(inputValue.trim())
  }, [inputValue])

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold tracking-tight">Search &amp; Radar</h2>

      {/* Search bar — shared across Pro and filters */}
      <SearchBar
        value={inputValue}
        onChange={setInputValue}
        onSubmit={handleSubmit}
        mode={mode}
        onModeChange={setMode}
      />

      {/* Filters row — only relevant on Pro tab */}
      {activeTab === 'pro' && (
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm text-muted-foreground">Sources:</span>
          <Select value={sourceFilter} onValueChange={setSourceFilter}>
            <SelectTrigger className="w-40">
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

          <span className="text-sm text-muted-foreground">Since:</span>
          <Select value={sinceFilter} onValueChange={setSinceFilter}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Any time" />
            </SelectTrigger>
            <SelectContent>
              {SINCE_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Tabs */}
      <Tabs
        value={activeTab}
        onValueChange={(v) => setActiveTab(v as ActiveTab)}
      >
        <TabsList>
          <TabsTrigger value="pro">
            Pro Results
            {results && results.length > 0 && (
              <Badge variant="secondary" className="ml-1.5 text-xs">
                {results.length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="radar">Radar Search</TabsTrigger>
          <TabsTrigger value="adhoc">Adhoc URL</TabsTrigger>
        </TabsList>

        {/* Pro Results tab */}
        <TabsContent value="pro" className="mt-4">
          {/* Loading skeleton */}
          {isFetching && <ResultsSkeleton />}

          {/* Error state */}
          {isError && !isFetching && (
            <p className="text-sm text-destructive">
              Search failed. Please check your query and try again.
            </p>
          )}

          {/* Results list */}
          {!isFetching && results && results.length > 0 && (
            <div className="space-y-3">
              {results.map(({ document: doc, score }) => (
                <DocumentCard
                  key={doc.id}
                  document={doc}
                  score={score}
                  showOriginBadge
                />
              ))}
            </div>
          )}

          {/* Empty state — after a query was submitted */}
          {!isFetching && !isError && submittedQuery && results?.length === 0 && (
            <div className="py-16 text-center">
              <p className="text-muted-foreground">
                No results found for "{submittedQuery}".
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                Try a different query or switch to Radar Search to search live
                sources.
              </p>
            </div>
          )}

          {/* Idle state — no query submitted yet */}
          {!submittedQuery && !isFetching && (
            <div className="py-16 text-center">
              <p className="text-muted-foreground">
                Enter a query above and press Search to get started.
              </p>
            </div>
          )}
        </TabsContent>

        {/* Radar Search tab */}
        <TabsContent value="radar" className="mt-4">
          <RadarTab />
        </TabsContent>

        {/* Adhoc URL tab — uses the AdhocIngest standalone component */}
        <TabsContent value="adhoc" className="mt-4">
          <AdhocIngest />
        </TabsContent>
      </Tabs>
    </div>
  )
}
