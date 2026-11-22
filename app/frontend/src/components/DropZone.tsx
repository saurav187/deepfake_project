import { useRef, useState } from 'react'
import { unreadable } from '../copy'
import type { Limits } from '../types'

interface Props {
  limits: Limits
  serverError?: string
  onFile: (file: File) => void
}

export function DropZone({ limits, serverError, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string>()
  const [showServerError, setShowServerError] = useState(true)

  function select(file?: File) {
    if (!file) return
    setShowServerError(false)
    if (file.size > limits.max_upload_mb * 1_000_000) {
      setError('Choose a file that is ' + limits.max_upload_mb + ' MB or smaller.')
      return
    }
    const extension = file.name.match(/\.[^.]+$/)?.[0].toLowerCase()
    if (!extension || !limits.accepted_extensions.includes(extension)) {
      setError(unreadable)
      return
    }
    setError(undefined)
    onFile(file)
  }

  const message = error ?? (showServerError ? serverError : undefined)
  return (
    <div>
      <div
        className={'drop-zone' + (dragging ? ' is-dragging' : '')}
        onDragEnter={(event) => {
          if (!event.dataTransfer.types.includes('Files')) return
          event.preventDefault()
          dragDepth.current += 1
          setDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1)
          if (!dragDepth.current) setDragging(false)
        }}
        onDrop={(event) => {
          event.preventDefault()
          dragDepth.current = 0
          setDragging(false)
          select(event.dataTransfer.files[0])
        }}
      >
        <span className="upload-mark" aria-hidden="true">↑</span>
        <h2>Drop an audio or video file here</h2>
        <input
          ref={input}
          type="file"
          aria-label="Audio or video file"
          accept={limits.accepted_extensions.join(',')}
          hidden
          onChange={(event) => {
            select(event.currentTarget.files?.[0])
            event.currentTarget.value = ''
          }}
        />
        <button type="button" className="primary" onClick={() => input.current?.click()}
          aria-describedby="file-limits">
          Choose file
        </button>
        <p className="limits" id="file-limits">
          {limits.max_upload_mb} MB max · {Math.floor(limits.max_duration_s / 60)} min max<br />
          {limits.accepted_extensions.map((extension) => extension.slice(1).toUpperCase()).join(', ')}
        </p>
      </div>
      {message && <p className="inline-error" role="alert">{message}</p>}
    </div>
  )
}
