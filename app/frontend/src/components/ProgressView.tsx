import { useEffect, useRef } from 'react'

export function ProgressView({ filename, stage, onCancel }: {
  filename: string
  stage: 'uploading' | 'analyzing'
  onCancel: () => void
}) {
  const cancel = useRef<HTMLButtonElement>(null)
  useEffect(() => cancel.current?.focus(), [])

  return (
    <section className="panel progress-panel" aria-labelledby="progress-title">
      <div className="progress-meter" aria-hidden="true"><span /></div>
      <div role="status">
        <h2 id="progress-title">{stage === 'uploading' ? 'Uploading…' : 'Analyzing…'}</h2>
        <p className="filename">{filename}</p>
      </div>
      <button ref={cancel} type="button" onClick={onCancel}>Cancel</button>
    </section>
  )
}
