/** Determine label from score */
export function scoreToLabel(score) {
  if (score >= 75) return 'ACCEPTABLE'
  if (score >= 45) return 'DEGRADED'
  return 'DEFECTIVE'
}

/** Tailwind colour class by label */
export function labelColor(label) {
  if (label === 'ACCEPTABLE') return 'text-success-400'
  if (label === 'DEGRADED')   return 'text-warning-400'
  return 'text-danger-400'
}

export function labelBgColor(label) {
  if (label === 'ACCEPTABLE') return 'bg-success-500/15 border-success-500/30 text-success-400'
  if (label === 'DEGRADED')   return 'bg-warning-500/15 border-warning-500/30 text-warning-400'
  return 'bg-danger-500/15 border-danger-500/30 text-danger-400'
}

export function gaugeColor(score) {
  if (score >= 75) return '#22c55e'   // green
  if (score >= 45) return '#f59e0b'   // amber
  return '#ef4444'                     // red
}

export function severityClass(sev) {
  if (sev === 'low')    return 'badge-low'
  if (sev === 'medium') return 'badge-medium'
  return 'badge-high'
}

export function formatDate(iso) {
  return new Intl.DateTimeFormat('en-GB', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(iso))
}

export function formatFloat(n, decimals = 1) {
  return typeof n === 'number' ? n.toFixed(decimals) : '—'
}
