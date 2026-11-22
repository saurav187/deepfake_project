import { useEffect, useRef, useState } from 'react'
import { AnalysisView } from './AnalysisView'
import { cancelJob, getJob, getModel, submitFile } from './api'
import { DropZone } from './components/DropZone'
import { ProgressView } from './components/ProgressView'
import { privacyNote } from './copy'
import type { AnalysisResult, ModelResponse } from './types'

type View =
  | { stage: 'landing'; error?: string; cancelled?: boolean }
  | { stage: 'uploading' | 'analyzing'; file: File }
  | { stage: 'done'; file: File; result: AnalysisResult }

interface Run {
  controller: AbortController
  id?: string
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Something went wrong. Try again.'
}

export default function App() {
  const [model, setModel] = useState<ModelResponse>()
  const [modelError, setModelError] = useState<string>()
  const [attempt, setAttempt] = useState(0)
  const [view, setView] = useState<View>({ stage: 'landing' })
  const active = useRef<Run>()
  const title = useRef<HTMLHeadingElement>(null)
  const focusLanding = useRef(false)

  useEffect(() => {
    const controller = new AbortController()
    getModel(controller.signal).then((response) => {
      if (!controller.signal.aborted) setModel(response)
    }).catch((error) => {
      if (!controller.signal.aborted) setModelError(errorMessage(error))
    })
    return () => controller.abort()
  }, [attempt])

  useEffect(() => {
    if (view.stage === 'landing' && focusLanding.current) {
      title.current?.focus()
      focusLanding.current = false
    }
  }, [view.stage])

  async function analyze(file: File) {
    if (active.current) return
    const run: Run = { controller: new AbortController() }
    active.current = run
    setView({ stage: 'uploading', file })
    const signal = run.controller.signal
    try {
      const { job_id } = await submitFile(file)
      run.id = job_id
      if (signal.aborted) {
        await cancelJob(job_id)
        return
      }
      setView({ stage: 'analyzing', file })
      while (!signal.aborted) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000))
        if (signal.aborted) break
        const job = await getJob(job_id, signal)
        if (signal.aborted) break
        if (job.status === 'done' && job.result) {
          setView({ stage: 'done', file, result: job.result })
          break
        }
        if (job.status === 'cancelled') {
          focusLanding.current = true
          setView({ stage: 'landing', cancelled: true })
          break
        }
        if (job.status === 'failed' || job.status === 'done') {
          throw new Error(job.error?.message ?? 'Analysis failed. Try again.')
        }
      }
    } catch (error) {
      if (!signal.aborted) {
        focusLanding.current = true
        setView({ stage: 'landing', error: errorMessage(error) })
      }
    } finally {
      if (active.current === run) active.current = undefined
    }
  }

  async function cancel() {
    const run = active.current
    if (!run) return
    active.current = undefined
    run.controller.abort()
    focusLanding.current = true
    setView({ stage: 'landing', cancelled: true })
    if (!run.id) return
    try {
      await cancelJob(run.id)
    } catch {
      setView((current) => current.stage === 'landing' && current.cancelled
        ? { stage: 'landing', error: 'Could not cancel the analysis. Try again.' }
        : current)
    }
  }

  function reset() {
    focusLanding.current = true
    setView({ stage: 'landing' })
  }

  return (
    <main className="shell">
      <header className="site-header">
        <div className="brand-mark" aria-hidden="true"><i /><i /><i /><i /><i /></div>
        <h1 ref={title} tabIndex={-1}>Deepfake Audio Check</h1>
      </header>
      {view.stage === 'landing' && (
        <div className="landing stack">
          <p className="intro">Check an audio or video recording for signs of AI-generated speech.</p>
          {model ? (
            <DropZone limits={model.limits} serverError={view.error} onFile={analyze} />
          ) : modelError ? (
            <div className="panel">
              <p role="alert">{modelError}</p>
              <button type="button" onClick={() => {
                setModelError(undefined)
                setAttempt(attempt + 1)
              }}>Try again</button>
            </div>
          ) : <p role="status">Loading analysis settings…</p>}
          {view.cancelled && <p role="status">Analysis cancelled.</p>}
          <p className="privacy">{privacyNote}</p>
        </div>
      )}
      {(view.stage === 'uploading' || view.stage === 'analyzing') && (
        <ProgressView filename={view.file.name} stage={view.stage} onCancel={cancel} />
      )}
      {view.stage === 'done' && <AnalysisView file={view.file} result={view.result} onReset={reset} />}
    </main>
  )
}
