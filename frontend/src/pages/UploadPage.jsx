import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, RefreshCw, Sparkles } from 'lucide-react'
import DropZone from '../components/DropZone'
import { ResultSkeleton } from '../components/Skeletons'
import { analyzeImage } from '../utils/api'

export default function UploadPage() {
  const navigate = useNavigate()
  const [state, setState] = useState('idle') // idle | uploading | analyzing | error
  const [errorMsg, setErrorMsg] = useState('')
  const [progress, setProgress] = useState(0)

  const handleUpload = async (file) => {
    setState('uploading')
    setProgress(0)
    setErrorMsg('')

    // Simulate upload progress (we use fetch, so no real XHR progress)
    const tick = setInterval(() => {
      setProgress((p) => Math.min(p + 15, 85))
    }, 120)

    try {
      setState('analyzing')
      const result = await analyzeImage(file)
      clearInterval(tick)
      setProgress(100)
      navigate(`/result/${result.id}`, { state: { result } })
    } catch (err) {
      clearInterval(tick)
      setState('error')
      setErrorMsg(err.message || 'Analysis failed. Please try again.')
    }
  }

  const isLoading = state === 'uploading' || state === 'analyzing'

  return (
    <div className="animate-fade-in max-w-2xl mx-auto">
      {/* Hero header */}
      <div className="mb-8 text-center">
        <div className="inline-flex items-center gap-1.5 mb-3 rounded-full border border-brand-500/30 bg-brand-600/10 px-3 py-1 text-xs text-brand-400 font-medium">
          <Sparkles size={12} />
          Local AI · No external APIs
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-100 sm:text-4xl">
          Image Quality Detection
        </h1>
        <p className="mt-2 text-slate-400 text-sm sm:text-base max-w-md mx-auto">
          Upload any image to get an instant AI-powered quality score, defect breakdown, and Grad-CAM heatmap.
        </p>
      </div>

      {/* Upload card */}
      <div className="glass-card p-6 mb-4">
        <DropZone onUpload={handleUpload} isLoading={isLoading} />

        {/* Progress bar */}
        {isLoading && (
          <div className="mt-4 space-y-1">
            <div className="flex justify-between text-xs text-slate-500">
              <span>{state === 'uploading' ? 'Uploading…' : 'Running CV/ML pipeline…'}</span>
              <span>{progress}%</span>
            </div>
            <div className="h-1 rounded-full bg-white/5 overflow-hidden">
              <div
                className="h-full bg-brand-500 rounded-full transition-all duration-200 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Skeleton while analysing */}
      {state === 'analyzing' && (
        <div className="mt-6">
          <p className="text-xs text-slate-500 text-center mb-4 animate-pulse">
            Running classical feature extraction + MobileNetV2 inference…
          </p>
          <ResultSkeleton />
        </div>
      )}

      {/* Error state */}
      {state === 'error' && (
        <div className="glass-card p-4 border-danger-500/20 bg-danger-500/5 flex items-start gap-3">
          <AlertCircle size={18} className="text-danger-400 mt-0.5 shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-medium text-danger-300">Analysis failed</p>
            <p className="text-xs text-slate-500 mt-0.5">{errorMsg}</p>
          </div>
          <button
            onClick={() => setState('idle')}
            className="btn-ghost !py-1 !px-2 text-xs"
            aria-label="Try again"
          >
            <RefreshCw size={13} />
            Retry
          </button>
        </div>
      )}

      {/* Feature tiles */}
      {state === 'idle' && (
        <div className="mt-8 grid grid-cols-2 sm:grid-cols-3 gap-3">
          {[
            { icon: '🎯', title: 'Quality Score', desc: '0–100 score with ACCEPTABLE / DEGRADED / DEFECTIVE label' },
            { icon: '🔬', title: 'Defect Analysis', desc: 'Blur, noise, exposure, corruption, and visual defects' },
            { icon: '🧠', title: 'Grad-CAM Heatmap', desc: 'See exactly which regions triggered the detection' },
            { icon: '📊', title: 'Classical CV', desc: 'Laplacian variance, wavelet noise, DCT blocking' },
            { icon: '📈', title: 'History', desc: 'All analyses persisted and searchable' },
            { icon: '⚡', title: 'Batch Mode', desc: 'Upload up to 20 images at once' },
          ].map((f) => (
            <div key={f.title} className="glass-card p-4 hover:border-white/10 transition-all">
              <div className="text-2xl mb-2">{f.icon}</div>
              <p className="text-xs font-semibold text-slate-300 mb-0.5">{f.title}</p>
              <p className="text-xs text-slate-500 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
