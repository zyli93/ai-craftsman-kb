import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { Layout } from '@/components/layout/Layout'

const PageLoader = (
  <div className="flex items-center justify-center py-24">
    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
  </div>
)

const OverviewPage = lazy(() => import('@/pages/OverviewPage').then(m => ({ default: m.OverviewPage })))
const SourcesPage = lazy(() => import('@/pages/SourcesPage').then(m => ({ default: m.SourcesPage })))
const SearchPage = lazy(() => import('@/pages/SearchPage').then(m => ({ default: m.SearchPage })))
const EntitiesPage = lazy(() => import('@/pages/EntitiesPage').then(m => ({ default: m.EntitiesPage })))
const DocumentsPage = lazy(() => import('@/pages/DocumentsPage').then(m => ({ default: m.DocumentsPage })))
const BriefingPage = lazy(() => import('@/pages/BriefingPage').then(m => ({ default: m.BriefingPage })))
const UsagePage = lazy(() => import('@/pages/UsagePage').then(m => ({ default: m.UsagePage })))

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Suspense fallback={PageLoader}><OverviewPage /></Suspense>} />
          <Route path="sources" element={<Suspense fallback={PageLoader}><SourcesPage /></Suspense>} />
          <Route path="search" element={<Suspense fallback={PageLoader}><SearchPage /></Suspense>} />
          <Route path="entities" element={<Suspense fallback={PageLoader}><EntitiesPage /></Suspense>} />
          <Route path="documents" element={<Suspense fallback={PageLoader}><DocumentsPage /></Suspense>} />
          <Route path="briefing" element={<Suspense fallback={PageLoader}><BriefingPage /></Suspense>} />
          <Route path="usage" element={<Suspense fallback={PageLoader}><UsagePage /></Suspense>} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
