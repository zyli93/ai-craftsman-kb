import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/layout/Layout'

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
          <Route index element={<Suspense fallback={null}><OverviewPage /></Suspense>} />
          <Route path="sources" element={<Suspense fallback={null}><SourcesPage /></Suspense>} />
          <Route path="search" element={<Suspense fallback={null}><SearchPage /></Suspense>} />
          <Route path="entities" element={<Suspense fallback={null}><EntitiesPage /></Suspense>} />
          <Route path="documents" element={<Suspense fallback={null}><DocumentsPage /></Suspense>} />
          <Route path="briefing" element={<Suspense fallback={null}><BriefingPage /></Suspense>} />
          <Route path="usage" element={<Suspense fallback={null}><UsagePage /></Suspense>} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
