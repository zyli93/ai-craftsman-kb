import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/layout/Layout'
import { OverviewPage } from '@/pages/OverviewPage'
import { SourcesPage } from '@/pages/SourcesPage'
import { SearchPage } from '@/pages/SearchPage'
import { EntitiesPage } from '@/pages/EntitiesPage'
import { DocumentsPage } from '@/pages/DocumentsPage'
import { BriefingPage } from '@/pages/BriefingPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="sources" element={<SourcesPage />} />
          <Route path="search" element={<SearchPage />} />
          <Route path="entities" element={<EntitiesPage />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="briefing" element={<BriefingPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
