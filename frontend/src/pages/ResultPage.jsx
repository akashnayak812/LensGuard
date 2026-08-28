import { useEffect, useState } from 'react'
import { useParams, useLocation, Link } from 'react-router-dom'
import {
  ArrowLeft, ChevronDown, ChevronUp, Eye, EyeOff, RefreshCw, AlertCircle
} from 'lucide-react'
import ScoreGauge from '../components/ScoreGauge'
import IssueCard from '../components/IssueCard'
import { ResultSkeleton } from '../components/Skeletons'
import { getAnalysis, heatmapUrl, thumbnailUrl } from '../utils/api'
import { formatDate, formatFloat, labelBgColor } from '../utils/helpers'

export default function ResultPage() {
  const { id } = useParams()
  const location = useLocation()
  const [result, setResult] = useState(location.state?.result || null)
  const [loading, setLoading] = useState(!result)
  const [error, setError] = useState(null)
  const [showStats, setShowStats] = useState(false)
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [heatmapLoaded, setHeatmapLoaded] = useState(false)

  useEffect(() => {
    if (!result) {
      getAnalysis(id)
        .then(setResult)
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false))
    }
  }, [id])

  if (loading) return (
    <div className="max-w-2xl mx-auto animate-fade-in">
      <ResultSkeleton />
    </div>
  )

  if (error) return (
    <div className="max-w-2xl mx-auto">
      <div className="glass-card p-6 flex flex-col items-center gap-3 text-center">
        <AlertCircle size={32} className="text-danger-400" />
        <p className="text-sm text-slate-300">{error}</p>
        <Link to="/" className="btn-primary text-sm">
          <ArrowLeft size={15} /> Upload New Image
        </Link>
      </div>
    </div>
  )

  if (!result) return null

  const {
    quality_score, quality_label, issues = [],
    image_stats = {}, model_version, filename,
    created_at, heatmap_url
  } = result

  const statRows = [
    ['Laplacian Variance', formatFloat(image_stats.laplacian_variance, 1), 'Blur proxy — higher = sharper'],
    ['Mean Luminance',     formatFloat(image_stats.mean_luminance, 1),     'Average pixel brightness (0–255)'],
    ['Noise Estimate',     formatFloat(image_stats.noise_estimate, 2),     'Wavelet-based noise σ'],
    ['Contrast RMS',       formatFloat(image_stats.contrast_rms, 4),       'RMS contrast (0–1)'],
    ['Block Artifact',     formatFloat(image_stats.block_artifact_score, 3),'JPEG blocking ratio (>1 = artifacts)'],
    ['Near Black %',       `${formatFloat((image_stats.pct_near_black || 0) * 100, 1)}%`, 'Underexposure signal'],
    ['Near White %',       `${formatFloat((image_stats.pct_near_white || 0) * 100, 1)}%`, 'Overexposure signal'],
    ['Mean Saturation',    formatFloat(image_stats.mean_saturation, 3),    'HSV saturation mean (0–1)'],
  ]

  return (
    <div className="max-w-2xl mx-auto space-y-5 animate-slide-up">
      {/* Back link */}
      <Link to="/" className="btn-ghost -ml-2 text-sm w-fit">
        <ArrowLeft size={15} /> New Analysis
      </Link>

      {/* Header */}
      <div className="glass-card p-5">
        <div className="flex flex-col sm:flex-row gap-6 items-center sm:items-start">
          {/* Gauge */}
          <ScoreGauge score={quality_score} label={quality_label} size={160} />

          {/* Meta */}
          <div className="flex-1 text-center sm:text-left">
            <h1 className="text-lg font-bold text-slate-100 mb-0.5 truncate" title={filename}>
              {filename}
            </h1>
            <p className="text-xs text-slate-500 mb-3">
              Analysed {created_at ? formatDate(created_at) : '—'} · Model {model_version}
            </p>

            <span className={`inline-block rounded-full px-3 py-1 text-xs font-semibold border ${labelBgColor(quality_label)}`}>
              {quality_label}
            </span>

            <div className="mt-4 text-xs text-slate-500">
              {issues.length === 0
                ? '✓ No quality issues detected.'
                : `${issues.length} issue${issues.length > 1 ? 's' : ''} detected`
              }
            </div>
          </div>
        </div>

        {/* Image + heatmap toggle */}
        {result.thumbnail_url && (
          <div className="mt-5 border-t border-white/5 pt-5">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400">Visual Inspection</span>
              {heatmap_url ? (
                <div className="inline-flex rounded-lg p-0.5 bg-slate-200/80 dark:bg-white/5 border border-slate-300 dark:border-white/10 text-xs">
                  <button
                    type="button"
                    onClick={() => setShowHeatmap(false)}
                    className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                      !showHeatmap
                        ? 'bg-white dark:bg-surface-800 text-slate-900 dark:text-slate-100 shadow-sm'
                        : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-300'
                    }`}
                  >
                    Original
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowHeatmap(true)}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded-md font-medium transition-all ${
                      showHeatmap
                        ? 'bg-brand-600 text-white shadow-sm'
                        : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-300'
                    }`}
                  >
                    <Eye size={12} />
                    Grad-CAM
                  </button>
                </div>
              ) : null}
            </div>

            <div className="relative overflow-hidden rounded-xl bg-black/30">
              <img
                src={thumbnailUrl(result.id)}
                alt={`Uploaded image: ${filename}`}
                className="w-full max-h-64 object-contain mx-auto block"
                style={{ display: showHeatmap ? 'none' : 'block' }}
              />
              {showHeatmap && (
                <img
                  src={heatmapUrl(result.id)}
                  alt={`Grad-CAM heatmap highlighting regions that triggered the defect detection for ${filename}`}
                  className="w-full max-h-64 object-contain mx-auto block"
                  onLoad={() => setHeatmapLoaded(true)}
                />
              )}
              {showHeatmap && !heatmapLoaded && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <RefreshCw size={18} className="text-slate-400 animate-spin" />
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Issues */}
      {issues.length > 0 && (
        <section aria-label="Detected issues">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Detected Issues
          </h2>
          <div className="space-y-3">
            {issues.map((issue, i) => (
              <IssueCard key={`${issue.type}-${i}`} issue={issue} />
            ))}
          </div>
        </section>
      )}

      {issues.length === 0 && (
        <div className="glass-card p-5 flex items-center gap-3 text-success-400">
          <div className="text-2xl">✓</div>
          <div>
            <p className="text-sm font-semibold">Image looks great!</p>
            <p className="text-xs text-slate-500 mt-0.5">No quality issues were detected.</p>
          </div>
        </div>
      )}

      {/* Raw stats panel (expandable) */}
      <div className="glass-card overflow-hidden">
        <button
          onClick={() => setShowStats(s => !s)}
          className="w-full flex items-center justify-between p-4 text-left hover:bg-white/2 transition-colors"
          aria-expanded={showStats}
          aria-controls="raw-stats-panel"
        >
          <span className="text-sm font-medium text-slate-400">Raw Feature Values</span>
          {showStats ? <ChevronUp size={16} className="text-slate-500" /> : <ChevronDown size={16} className="text-slate-500" />}
        </button>

        {showStats && (
          <div id="raw-stats-panel" className="border-t border-white/5">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-surface-900/50">
                  <th className="text-left px-4 py-2 text-slate-500 font-medium">Feature</th>
                  <th className="text-right px-4 py-2 text-slate-500 font-medium">Value</th>
                  <th className="text-left px-4 py-2 text-slate-500 font-medium hidden sm:table-cell">Description</th>
                </tr>
              </thead>
              <tbody>
                {statRows.map(([name, value, desc]) => (
                  <tr key={name} className="border-t border-white/3 hover:bg-white/2">
                    <td className="px-4 py-2 font-mono text-slate-400">{name}</td>
                    <td className="px-4 py-2 text-right font-mono text-brand-300">{value}</td>
                    <td className="px-4 py-2 text-slate-600 hidden sm:table-cell">{desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
