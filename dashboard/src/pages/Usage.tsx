/**
 * Usage — LLM token usage tracking page.
 *
 * Shows:
 * - Summary cards: total input tokens, total output tokens, total requests (last 24h)
 * - Table: per-model rows with provider, model, task, input/output tokens, request count
 * - Auto-refresh every 30 seconds
 */
import { useQuery } from '@tanstack/react-query'
import { Activity } from 'lucide-react'
import { api } from '@/api/client'
import type { UsageSummary } from '@/api/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

/** Format large numbers with locale-aware separators. */
function fmtNum(n: number): string {
  return n.toLocaleString()
}

export function Usage() {
  const { data, isLoading, isError } = useQuery<UsageSummary>({
    queryKey: ['usage', 'summary'],
    queryFn: () => api.usage.summary(),
    refetchInterval: 30_000,
  })

  // Compute aggregate totals across all models.
  const totalInput = data?.summary.reduce((sum, r) => sum + r.total_input_tokens, 0) ?? 0
  const totalOutput = data?.summary.reduce((sum, r) => sum + r.total_output_tokens, 0) ?? 0
  const totalRequests = data?.summary.reduce((sum, r) => sum + r.request_count, 0) ?? 0

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Activity className="h-6 w-6 text-muted-foreground" />
        <h2 className="text-2xl font-bold tracking-tight">LLM Usage</h2>
      </div>

      {/* Summary cards */}
      {isLoading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="h-28 rounded-xl border bg-card animate-pulse"
            />
          ))}
        </div>
      ) : isError ? (
        <p className="text-sm text-muted-foreground">
          Failed to load usage data. Is the backend running?
        </p>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Input Tokens (24h)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{fmtNum(totalInput)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Output Tokens (24h)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{fmtNum(totalOutput)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Requests (24h)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{fmtNum(totalRequests)}</p>
              </CardContent>
            </Card>
          </div>

          {/* Per-model breakdown table */}
          <Card>
            <CardHeader>
              <CardTitle>Per-Model Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              {data && data.summary.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left text-muted-foreground">
                        <th className="pb-2 pr-4 font-medium">Provider</th>
                        <th className="pb-2 pr-4 font-medium">Model</th>
                        <th className="pb-2 pr-4 font-medium">Task</th>
                        <th className="pb-2 pr-4 font-medium text-right">Input Tokens</th>
                        <th className="pb-2 pr-4 font-medium text-right">Output Tokens</th>
                        <th className="pb-2 font-medium text-right">Requests</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.summary.map((row, idx) => (
                        <tr key={idx} className="border-b last:border-0">
                          <td className="py-2 pr-4 capitalize">{row.provider}</td>
                          <td className="py-2 pr-4 font-mono text-xs">{row.model}</td>
                          <td className="py-2 pr-4 capitalize">{row.task}</td>
                          <td className="py-2 pr-4 text-right tabular-nums">
                            {fmtNum(row.total_input_tokens)}
                          </td>
                          <td className="py-2 pr-4 text-right tabular-nums">
                            {fmtNum(row.total_output_tokens)}
                          </td>
                          <td className="py-2 text-right tabular-nums">
                            {fmtNum(row.request_count)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground py-4">
                  No usage data recorded in the last 24 hours.
                </p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
