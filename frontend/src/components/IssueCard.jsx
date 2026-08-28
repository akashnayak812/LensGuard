import { AlertTriangle, Droplets, Eye, Sun, Sunset, Volume2, Zap } from 'lucide-react'
import clsx from 'clsx'
import { severityClass } from '../utils/helpers'

const ISSUE_ICONS = {
  blur:         Eye,
  noise:        Volume2,
  underexposed: Sun,
  overexposed:  Sunset,
  corruption:   Zap,
  defect:       AlertTriangle,
  default:      Droplets,
}

const ISSUE_DESCRIPTIONS = {
  blur:         'Image sharpness is reduced — edges and fine details are soft.',
  noise:        'Random luminance or colour variation detected across the image.',
  underexposed: 'Image is too dark; shadow regions are clipping to black.',
  overexposed:  'Image is too bright; highlight regions are clipping to white.',
  corruption:   'JPEG block artifacts or data corruption detected.',
  defect:       'Visible physical defects (scratches, blobs, occlusions) detected.',
}

/**
 * Card displaying a single detected issue with severity badge and confidence bar.
 */
export default function IssueCard({ issue }) {
  const { type, severity, confidence } = issue
  const Icon = ISSUE_ICONS[type] || ISSUE_ICONS.default
  const description = ISSUE_DESCRIPTIONS[type] || 'Quality issue detected.'
  const pct = Math.round(confidence * 100)

  return (
    <div className="glass-card p-4 flex gap-4 items-start animate-fade-in hover:border-white/10 transition-all">
      {/* Icon */}
      <div
        className={clsx(
          'flex h-10 w-10 shrink-0 items-center justify-center rounded-xl',
          severity === 'low'    && 'bg-success-500/15 text-success-400',
          severity === 'medium' && 'bg-warning-500/15 text-warning-400',
          severity === 'high'   && 'bg-danger-500/15  text-danger-400',
        )}
      >
        <Icon size={18} />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <span className="text-sm font-semibold capitalize text-slate-200">
            {type.replace('_', ' ')}
          </span>
          <span
            className={clsx(
              'rounded-full px-2 py-0.5 text-xs font-medium capitalize border',
              severityClass(severity)
            )}
          >
            {severity}
          </span>
        </div>

        <p className="text-xs text-slate-500 mb-2 leading-relaxed">{description}</p>

        {/* Confidence bar */}
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
            <div
              className={clsx(
                'h-full rounded-full transition-all duration-700 ease-out',
                severity === 'low'    && 'bg-success-500',
                severity === 'medium' && 'bg-warning-500',
                severity === 'high'   && 'bg-danger-500',
              )}
              style={{ width: `${pct}%` }}
              role="progressbar"
              aria-valuenow={pct}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`Confidence ${pct}%`}
            />
          </div>
          <span className="text-xs font-mono text-slate-500 w-9 text-right">
            {pct}%
          </span>
        </div>
      </div>
    </div>
  )
}
