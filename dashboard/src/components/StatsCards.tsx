/**
 * StatsCards — displays a grid of key system metrics from SystemStats.
 * Each StatCard shows a label, a primary value, and optional subtext.
 */
import type { ReactNode } from 'react'
import type { SystemStats } from '@/api/types'
import { Card, CardContent } from '@/components/ui/card'
import { Database, Hash, Cpu, HardDrive, Layers, Radio } from 'lucide-react'

interface StatCardProps {
  label: string
  value: string | number
  subtext?: string
  icon?: ReactNode
}

/** A single metric card with icon, value, label, and optional subtext. */
function StatCard({ label, value, subtext, icon }: StatCardProps) {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-muted-foreground">{label}</p>
            <p className="text-2xl font-bold tracking-tight">{value}</p>
            {subtext && (
              <p className="text-xs text-muted-foreground">{subtext}</p>
            )}
          </div>
          {icon && (
            <div className="text-muted-foreground opacity-60">{icon}</div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

/** Formats a byte count into a human-readable string (MB or GB). */
function formatBytes(bytes: number): string {
  if (bytes >= 1_073_741_824) {
    return `${(bytes / 1_073_741_824).toFixed(1)} GB`
  }
  return `${(bytes / 1_048_576).toFixed(0)} MB`
}

interface StatsCardsProps {
  stats: SystemStats
}

/**
 * Grid of 6 stat cards derived from the SystemStats API response.
 * Displays documents, entities, embedded count, DB size, vector store size,
 * and active sources count.
 */
export function StatsCards({ stats }: StatsCardsProps) {
  const embeddedPct =
    stats.total_documents > 0
      ? Math.round((stats.embedded_documents / stats.total_documents) * 100)
      : 0

  const cards: StatCardProps[] = [
    {
      label: 'Documents',
      value: stats.total_documents.toLocaleString(),
      subtext: `${stats.embedded_documents.toLocaleString()} embedded`,
      icon: <Database className="h-5 w-5" />,
    },
    {
      label: 'Entities',
      value: stats.total_entities.toLocaleString(),
      icon: <Hash className="h-5 w-5" />,
    },
    {
      label: 'Embedded',
      value: stats.embedded_documents.toLocaleString(),
      subtext: `${embeddedPct}% of total`,
      icon: <Cpu className="h-5 w-5" />,
    },
    {
      label: 'SQLite Size',
      value: formatBytes(stats.db_size_bytes),
      icon: <HardDrive className="h-5 w-5" />,
    },
    {
      label: 'Vector Store',
      value: stats.vector_count.toLocaleString(),
      subtext: 'vectors indexed',
      icon: <Layers className="h-5 w-5" />,
    },
    {
      label: 'Active Sources',
      value: stats.active_sources,
      icon: <Radio className="h-5 w-5" />,
    },
  ]

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
      {cards.map((card) => (
        <StatCard key={card.label} {...card} />
      ))}
    </div>
  )
}
