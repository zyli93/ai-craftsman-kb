import { AlertTriangle, Terminal } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function BackendOffline() {
  return (
    <div className="flex items-center justify-center h-full">
      <Card className="max-w-lg w-full">
        <CardHeader className="space-y-1">
          <div className="flex items-center gap-2 text-amber-500">
            <AlertTriangle className="h-5 w-5" />
            <CardTitle className="text-lg">Backend Not Running</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            The dashboard can't reach the API server. Start the backend to see your data.
          </p>
          <div className="rounded-md bg-muted p-4 font-mono text-sm space-y-2">
            <div className="flex items-center gap-2 text-muted-foreground mb-2">
              <Terminal className="h-4 w-4" />
              <span>Run in your terminal:</span>
            </div>
            <pre className="text-foreground">
              uv run python -m ai_craftsman_kb.cli serve
            </pre>
          </div>
          <p className="text-xs text-muted-foreground">
            The dashboard will reconnect automatically once the backend is available.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
