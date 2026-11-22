export type Verdict = 'likely_real' | 'uncertain' | 'likely_fake' | 'insufficient_audio'

export interface ModelInfo {
  name: string
  version: string
  experimental: boolean
  description: string
}

export interface Segment {
  start_s: number
  end_s: number
  fake_probability: number
}

export interface AnalysisResult {
  verdict: Verdict
  fake_probability: number | null
  segments: Segment[]
  duration_s: number
  speech_s: number
  warnings: string[]
  model: ModelInfo
}

export interface Limits {
  max_upload_mb: number
  max_duration_s: number
  accepted_extensions: string[]
}

export interface ModelResponse {
  model: ModelInfo
  limits: Limits
}

export interface Job {
  job_id: string
  status: 'queued' | 'processing' | 'done' | 'failed' | 'cancelled'
  progress: number | null
  result: AnalysisResult | null
  error: {code: string; message: string} | null
  created_at: string
  expires_at: string
}
