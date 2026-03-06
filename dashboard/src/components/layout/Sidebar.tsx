import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Database,
  Search,
  Network,
  FileText,
  BookOpen,
  Activity,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/sources', label: 'Sources', icon: Database },
  { to: '/search', label: 'Search & Radar', icon: Search },
  { to: '/entities', label: 'Entities', icon: Network },
  { to: '/documents', label: 'Documents', icon: FileText },
  { to: '/briefing', label: 'Briefing', icon: BookOpen },
  { to: '/usage', label: 'LLM Usage', icon: Activity },
]

export function Sidebar() {
  return (
    <aside className="w-56 shrink-0 border-r bg-muted/40 flex flex-col h-full">
      <div className="px-4 py-5 border-b">
        <h1 className="text-sm font-semibold tracking-tight">AI Craftsman KB</h1>
      </div>
      <nav className="flex-1 px-2 py-4 space-y-1">
        {navItems.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
