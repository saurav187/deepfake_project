import type { Job, ModelResponse } from './types'

async function request(path: string, options?: RequestInit) {
  const response = await fetch('/api' + path, options)
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.error?.message ?? 'Something went wrong. Try again.')
  }
  return response
}

export async function getModel(signal: AbortSignal): Promise<ModelResponse> {
  return (await request('/model', { signal })).json()
}

export async function submitFile(file: File): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('file', file)
  return (await request('/analyze', { method: 'POST', body: form })).json()
}

export async function getJob(id: string, signal: AbortSignal): Promise<Job> {
  return (await request('/jobs/' + id, { signal })).json()
}

export async function cancelJob(id: string) {
  await request('/jobs/' + id, { method: 'DELETE' })
}
