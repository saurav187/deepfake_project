import type { AnalysisResult } from './types'

export function downloadReport(result: AnalysisResult) {
  const blob = new Blob([JSON.stringify(result, null, 2) + '\n'], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'deepfake-audio-check.json'
  document.body.append(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
