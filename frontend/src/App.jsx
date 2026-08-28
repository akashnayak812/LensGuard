import { Routes, Route, NavLink } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { Eye, History, Upload, Grid3X3, Sun, Moon, Zap } from 'lucide-react'
import clsx from 'clsx'

import UploadPage from './pages/UploadPage'
import ResultPage from './pages/ResultPage'
import HistoryPage from './pages/HistoryPage'
import BatchPage from './pages/BatchPage'

export default function App() {
  const [dark, setDark] = useState(true)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  const navItems = [
    { to: '/',        icon: Upload,    label: 'Analyze'  },
    { to: '/batch',   icon: Grid3X3,   label: 'Batch'    },
    { to: '/history', icon: History,   label: 'History'  },
  ]

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-surface-950 text-slate-900 dark:text-slate-100 transition-colors duration-200">
      {/* ── Top nav ─────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 border-b border-slate-200 dark:border-white/5 bg-white/80 dark:bg-surface-950/80 backdrop-blur-xl transition-colors">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            {/* Logo */}
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600">
                <Eye className="h-4.5 w-4.5 text-white" size={18} />
              </div>
              <span className="text-lg font-bold tracking-tight">
                Lens<span className="text-brand-400">Guard</span>
              </span>
            </div>

            {/* Nav links */}
            <nav className="flex items-center gap-1" aria-label="Main navigation">
              {navItems.map(({ to, icon: Icon, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === '/'}
                  className={({ isActive }) =>
                    clsx(
                      'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-all',
                      isActive
                        ? 'bg-brand-600/20 text-brand-400'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                    )
                  }
                >
                  <Icon size={15} />
                  <span className="hidden sm:inline">{label}</span>
                </NavLink>
              ))}
            </nav>

            {/* Theme toggle */}
            <button
              onClick={() => setDark(d => !d)}
              aria-label="Toggle dark mode"
              className="btn-ghost !px-2 !py-2"
            >
              {dark ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>
      </header>

      {/* ── Pages ───────────────────────────────────────────────────────── */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Routes>
          <Route path="/"              element={<UploadPage />} />
          <Route path="/result/:id"    element={<ResultPage />} />
          <Route path="/batch"         element={<BatchPage />} />
          <Route path="/history"       element={<HistoryPage />} />
        </Routes>
      </main>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <footer className="border-t border-white/5 py-6 text-center text-xs text-slate-600">
        <span className="flex items-center justify-center gap-1">
          <Zap size={11} className="text-brand-500" />
          LensGuard · All inference runs locally · No external APIs
        </span>
      </footer>
    </div>
  )
}
