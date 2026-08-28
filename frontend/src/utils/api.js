/** API client — all calls go through here */

const BASE = import.meta.env.VITE_API_BASE_URL || ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  const contentType = res.headers.get('content-type') || ''
  if (!res.ok) {
    let errMsg = `HTTP ${res.status}`
    if (contentType.includes('application/json')) {
      const body = await res.json()
      errMsg = body?.error?.message || errMsg
    }
    throw new Error(errMsg)
  }
  if (contentType.includes('application/json')) {
    return res.json()
  }
  return res
}

export async function analyzeImage(file) {
  const form = new FormData()
  form.append('file', file)
  return request('/api/v1/analyze', { method: 'POST', body: form })
}

export async function analyzeBatch(files) {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  return request('/api/v1/analyze/batch', { method: 'POST', body: form })
}

export async function getAnalyses({ limit = 20, offset = 0, label, sort_by, order } = {}) {
  const params = new URLSearchParams({ limit, offset })
  if (label) params.set('label', label)
  if (sort_by) params.set('sort_by', sort_by)
  if (order) params.set('order', order)
  return request(`/api/v1/analyses?${params}`)
}

export async function getAnalysis(id) {
  return request(`/api/v1/analyses/${id}`)
}

export function heatmapUrl(id) {
  return `${BASE}/api/v1/analyses/${id}/heatmap`
}

export function thumbnailUrl(id) {
  return `${BASE}/api/v1/analyses/${id}/thumbnail`
}
