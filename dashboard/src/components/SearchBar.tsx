/**
 * SearchBar component with a text input and search mode selector.
 * Supports hybrid, semantic, and keyword search modes.
 * Search is triggered on form submit (Enter key or button click), not on each keystroke.
 */
import { Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

export type SearchMode = 'hybrid' | 'semantic' | 'keyword'

export interface SearchBarProps {
  /** Current search query value */
  value: string
  /** Called when the input changes */
  onChange: (q: string) => void
  /** Called when the user submits the search form */
  onSubmit: () => void
  /** Currently selected search mode */
  mode: SearchMode
  /** Called when the user changes the search mode */
  onModeChange: (m: SearchMode) => void
}

const MODES: { label: string; value: SearchMode }[] = [
  { label: 'Hybrid', value: 'hybrid' },
  { label: 'Semantic', value: 'semantic' },
  { label: 'Keyword', value: 'keyword' },
]

/**
 * Search input with mode radio buttons.
 * Submits on Enter keypress or clicking the Search button.
 */
export function SearchBar({
  value,
  onChange,
  onSubmit,
  mode,
  onModeChange,
}: SearchBarProps) {
  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') {
      onSubmit()
    }
  }

  return (
    <div className="space-y-3">
      {/* Search input row */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search your knowledge base..."
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            className="pl-9"
          />
        </div>
        <Button onClick={onSubmit} disabled={value.trim().length === 0}>
          Search
        </Button>
      </div>

      {/* Mode selector row */}
      <div className="flex items-center gap-4">
        <span className="text-sm text-muted-foreground">Mode:</span>
        <div className="flex items-center gap-4">
          {MODES.map((m) => (
            <label
              key={m.value}
              className="flex cursor-pointer items-center gap-1.5 text-sm"
            >
              <input
                type="radio"
                name="search-mode"
                value={m.value}
                checked={mode === m.value}
                onChange={() => onModeChange(m.value)}
                className="accent-primary"
              />
              {m.label}
            </label>
          ))}
        </div>
      </div>
    </div>
  )
}
