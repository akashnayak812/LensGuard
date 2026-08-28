import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { History, SlidersHorizontal, ArrowUpDown, RefreshCw, AlertCircle } from 'lucide-react'
import { getAnalyses } from '../utils/api'
import { formatDate, labelBgColor, gaugeColor } from '../utils/helpers'
import { HistorySkeleton } from '../components/Skeletons'

const LABELS = ['', 'ACCEPTABLE', 'DEGRADED', 'DEFECTIVE']
const PAGE_SIZE = 20

export default function HistoryPage() {
  const navigate = useNavigate()
  const [state, setState] = useState('loading') // loading | done | error
  const [data, setData] = useState({ items: [], total: 0 })
  const [error, setError] = useState('')
  const [label, setLabel] = useState('')
  const [sortBy, setSortBy] = useState('created_at')
  const [order, setOrder] = useState('desc')
  const [page, setPage] = useState(0)

  const fetchData = useCallback(() => {
    setState('loading')
    getAnalyses({ limit: PAGE_SIZE, offset: page * PAGE_SIZE, label: label || undefined, sort_by: sortBy, order })
      .then(d => { setData(d); setState('done') })
      .catch(e => { setError(e.message); setState('error') })
  }, [label, sortBy, order, page])

  useEffect(() => { fetchData() }, [fetchData])

  const toggleSort = (field) => {
    if (sortBy === field) setOrder(o => o === 'asc' ? 'desc' : 'asc')
    else { setSortBy(field); setOrder('desc') }
    setPage(0)
  }

  const totalPages = Math.ceil(data.total / PAGE_SIZE)

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
            <History size={22} className="text-brand-400" />
            History
          </h1>
          <p className="text-slate-500 text-sm mt-0.5">
            {data.total} total analyses
          </p>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2 flex-wrap">
          <SlidersHorizontal size={14} className="text-slate-500" />
          {LABELS.map(l => (
            <button
              key={l || 'all'}
              onClick={() => { setLabel(l); setPage(0) }}
              className={`rounded-full px-3 py-1 text-xs font-medium border transition-all ${
                label === l
                  ? l ? labelBgColor(l) : 'bg-brand-600/20 text-brand-400 border-brand-500/30'
                  : 'border-white/10 text-slate-500 hover:text-slate-300 hover:border-white/20'
              }`}
              aria-pressed={label === l}
            >
              {l || 'All'}
            </button>
          ))}

          <button
            onClick={fetchData}
            className="btn-ghost !py-1 !px-2"
            aria-label="Refresh history"
          >
            <RefreshCw size={13} className={state === 'loading' ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Error */}
      {state === 'error' && (
        <div className="glass-card p-4 flex items-center gap-3 border-danger-500/20">
          <AlertCircle size={16} className="text-danger-400" />
          <p className="text-sm text-slate-400">{error}</p>
        </div>
      )}

      {/* Loading */}
      {state === 'loading' && <HistorySkeleton />}

      {/* Table */}
      {state === 'done' && data.items.length === 0 && (
        <div className="glass-card p-10 text-center text-slate-500 text-sm">
          No analyses yet. Upload an image to get started.
        </div>
      )}

      {state === 'done' && data.items.length > 0 && (
        <div className="glass-card overflow-hidden">
          {/* Sort header */}
          <div className="grid grid-cols-[1fr_auto_auto_auto] gap-4 px-4 py-2 border-b border-white/5 text-xs text-slate-500 font-medium">
            <span>Image</span>
            <button
              onClick={() => toggleSort('quality_score')}
              className="flex items-center gap-1 hover:text-slate-300 transition-colors"
              aria-label="Sort by score"
            >
              Score <ArrowUpDown size={10} />
            </button>
            <span>Label</span>
            <button
              onClick={() => toggleSort('created_at')}
              className="flex items-center gap-1 hover:text-slate-300 transition-colors"
              aria-label="Sort by date"
            >
              Date <ArrowUpDown size={10} />
            </button>
          </div>

          {/* Rows */}
          <div className="divide-y divide-white/3">
            {data.items.map((item) => (
              <button
                key={item.id}
                onClick={() => navigate(`/result/${item.id}`, { state: { result: item } })}
                className="w-full grid grid-cols-[1fr_auto_auto_auto] gap-4 px-4 py-3 items-center text-left hover:bg-white/2 transition-colors focus-visible:outline-none focus-visible:ring-inset focus-visible:ring-2 focus-visible:ring-brand-500"
                aria-label={`View analysis for ${item.filename}, score ${item.quality_score}, ${item.quality_label}`}
              >
                {/* Thumbnail + name */}
                <div className="flex items-center gap-3 min-w-0">
                  <div className="shrink-0 h-10 w-10 rounded-lg overflow-hidden bg-surface-900">
                    {item.thumbnail_url ? (
                      <img
                        src={`/api/v1/analyses/${item.id}/thumbnail`}
                        alt={`Thumbnail of ${item.filename}`}
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <div className="h-full w-full flex items-center justify-center text-slate-600 text-xs">?</div>
                    )}
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm text-slate-300 font-medium truncate">{item.filename}</p>
                    <p className="text-xs text-slate-600">{item.issues?.length || 0} issues</p>
                  </div>
                </div>

                {/* Score */}
                <span
                  className="text-sm font-bold tabular-nums"
                  style={{ color: gaugeColor(item.quality_score) }}
                >
                  {item.quality_score}
                </span>

                {/* Label */}
                <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium border ${labelBgColor(item.quality_label)}`}>
                  {item.quality_label}
                </span>

                {/* Date */}
                <span className="text-xs text-slate-600 hidden sm:block whitespace-nowrap">
                  {item.created_at ? formatDate(item.created_at) : '—'}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-6">
          <button
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
            className="btn-ghost !py-1.5 disabled:opacity-40"
          >
            Previous
          </button>
          <span className="text-xs text-slate-500">
            Page {page + 1} of {totalPages}
          </span>
          <button
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="btn-ghost !py-1.5 disabled:opacity-40"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
