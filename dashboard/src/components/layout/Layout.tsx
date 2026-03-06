import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { useBackendStatus } from '@/hooks/useBackendStatus'
import { BackendOffline } from '@/components/BackendOffline'

export function Layout() {
  const { isError } = useBackendStatus()

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-auto p-6">
        {isError ? <BackendOffline /> : <Outlet />}
      </main>
    </div>
  )
}
