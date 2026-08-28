import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, Grid3X3, Loader2, ImagePlus, X } from 'lucide-react'
import { analyzeBatch } from '../utils/api'
import { labelBgColor, gaugeColor } from '../utils/helpers'
import { BatchSkeleton } from '../components/Skeletons'

export default function BatchPage() {
  const navigate = useNavigate()
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [state, setState] = useState('idle') // idle | loading | done | error
  const [results, setResults] = useState([])
  const [failed, setFailed] = useState([])
  const [error, setError] = useState('')

  const addFiles = (newFiles) => {
    const valid = Array.from(newFiles).filter(f => f.type.startsWith('image/')).slice(0, 20 - files.length)
    const newPreviews = valid.map(f => ({ url: URL.createObjectURL(f), name: f.name }))
    setFiles(prev => [...prev, ...valid])
    setPreviews(prev => [...prev, ...newPreviews])
  }

  const removeFile = (idx) => {
    URL.revokeObjectURL(previews[idx].url)
    setFiles(prev => prev.filter((_, i) => i !== idx))
    setPreviews(prev => prev.filter((_, i) => i !== idx))
  }

  const runBatch = async () => {
    if (!files.length) return
    setState('loading')
    setError('')
    try {
      const data = await analyzeBatch(files)
      setResults(data.results || [])
      setFailed(data.failed || [])
      setState('done')
    } catch (err) {
      setError(err.message)
      setState('error')
    }
  }

  return (
    <div className="animate-fade-in">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
          <Grid3X3 size={22} className="text-brand-400" />
          Batch Analysis
        </h1>
        <p className="text-slate-500 text-sm mt-1">Upload up to 20 images and analyse them all at once.</p>
      </div>

      {state !== 'done' && (
        <div className="glass-card p-6 mb-6">
          {/* Drop zone */}
          <label
            htmlFor="batch-file-input"
            className="flex items-center justify-center gap-3 rounded-xl border-2 border-dashed border-white/10 p-6 cursor-pointer hover:border-brand-500/40 hover:bg-brand-600/5 transition-all mb-4"
            aria-label="Batch upload zone. Click or drag and drop multiple images."
          >
            <input
              id="batch-file-input"
              type="file"
              multiple
              accept="image/*"
              className="sr-only"
              onChange={e => addFiles(e.target.files)}
            />
            <ImagePlus size={20} className="text-brand-400" />
            <span className="text-sm text-slate-400">
              Click to add images <span className="text-slate-600">({files.length}/20 selected)</span>
            </span>
          </label>

          {/* Preview grid */}
          {previews.length > 0 && (
            <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-2 mb-4">
              {previews.map((p, i) => (
                <div key={i} className="relative group">
                  <img
                    src={p.url}
                    alt={`Preview of ${p.name}`}
                    className="h-20 w-full object-cover rounded-lg border border-white/5"
                  />
                  <button
                    onClick={() => removeFile(i)}
                    className="absolute top-1 right-1 h-5 w-5 rounded-full bg-surface-900/90 flex items-center justify-center text-slate-400 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"
                    aria-label={`Remove ${p.name}`}
                  >
                    <X size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={runBatch}
            disabled={!files.length || state === 'loading'}
            className="btn-primary w-full justify-center"
          >
            {state === 'loading' ? (
              <><Loader2 size={16} className="animate-spin" /> Analysing {files.length} images…</>
            ) : (
              <>Analyse {files.length || ''} Images</>
            )}
          </button>
        </div>
      )}

      {/* Skeleton */}
      {state === 'loading' && <BatchSkeleton />}

      {/* Error */}
      {state === 'error' && (
        <div className="glass-card p-4 border-danger-500/20 bg-danger-500/5 flex items-center gap-3">
          <AlertCircle size={18} className="text-danger-400" />
          <p className="text-sm text-slate-300">{error}</p>
        </div>
      )}

      {/* Results grid */}
      {state === 'done' && results.length > 0 && (
        <div>
          <p className="text-sm text-slate-500 mb-4">
            {results.length} analysed · {failed.length} failed
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
            {results.map((r) => (
              <button
                key={r.id}
                onClick={() => navigate(`/result/${r.id}`, { state: { result: r } })}
                className="glass-card p-3 text-left hover:border-white/15 hover:scale-[1.02] transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-400 rounded-2xl"
                aria-label={`View result for ${r.filename}, score ${r.quality_score}`}
              >
                <div className="relative mb-2">
                  {r.thumbnail_url ? (
                    <img
                      src={`/api/v1/analyses/${r.id}/thumbnail`}
                      alt={`Thumbnail of ${r.filename}`}
                      className="h-32 w-full rounded-xl object-cover bg-surface-900"
                    />
                  ) : (
                    <div className="h-32 w-full rounded-xl bg-surface-900 flex items-center justify-center text-slate-600 text-xs">
                      No preview
                    </div>
                  )}
                  {/* Score badge overlay */}
                  <div
                    className="absolute top-2 right-2 rounded-full px-2 py-0.5 text-xs font-bold"
                    style={{ background: `${gaugeColor(r.quality_score)}25`, color: gaugeColor(r.quality_score), border: `1px solid ${gaugeColor(r.quality_score)}40` }}
                  >
                    {r.quality_score}
                  </div>
                </div>
                <p className="text-xs font-medium text-slate-300 truncate">{r.filename}</p>
                <span className={`mt-1 inline-block rounded-full px-2 py-0.5 text-xs border ${labelBgColor(r.quality_label)}`}>
                  {r.quality_label}
                </span>
              </button>
            ))}
          </div>

          {/* Failed items */}
          {failed.length > 0 && (
            <div className="mt-4 space-y-2">
              <p className="text-xs text-slate-500 font-medium">Failed ({failed.length})</p>
              {failed.map((f, i) => (
                <div key={i} className="glass-card px-4 py-2 flex items-center gap-2 border-danger-500/20">
                  <AlertCircle size={14} className="text-danger-400" />
                  <span className="text-xs text-slate-400">{f.filename}</span>
                  <span className="text-xs text-slate-600 ml-auto">{f.error?.message || 'Failed'}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
