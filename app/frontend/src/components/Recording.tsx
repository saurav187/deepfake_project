import { useEffect, useRef, useState } from 'react'
import type { AnalysisResult } from '../types'
import { waveformPath, waveformSize } from '../waveform'
import { Timeline } from './Timeline'

export function Recording({ file, result }: { file: File; result: AnalysisResult }) {
  const audio = useRef<HTMLAudioElement>(null)
  const [url, setUrl] = useState<string>()
  const [path, setPath] = useState<string>()
  const [waveformFailed, setWaveformFailed] = useState(false)
  const [playbackError, setPlaybackError] = useState<string>()

  useEffect(() => {
    let ignore = false
    const localUrl = URL.createObjectURL(file)
    setUrl(localUrl)
    waveformPath(file).then((points) => {
      if (!ignore) setPath(points)
    }).catch(() => {
      if (!ignore) setWaveformFailed(true)
    })
    return () => {
      ignore = true
      URL.revokeObjectURL(localUrl)
    }
  }, [file])

  function play(seconds: number) {
    const player = audio.current
    if (!player || !url) return
    setPlaybackError(undefined)
    try {
      player.currentTime = seconds
      void player.play().catch(() => setPlaybackError('Playback is unavailable for this file in this browser.'))
    } catch {
      setPlaybackError('Playback is unavailable for this file in this browser.')
    }
  }

  return (
    <section className="panel recording" aria-labelledby="recording-title">
      <h2 id="recording-title">Recording timeline</h2>
      <div className="waveform">
        {path ? (
          <svg role="img" aria-label="Recording waveform"
            viewBox={'0 0 ' + waveformSize.width + ' ' + waveformSize.height} preserveAspectRatio="none">
            <path d={path} />
          </svg>
        ) : <p role="status">{waveformFailed ? 'Waveform unavailable for this file in this browser.' : 'Preparing waveform…'}</p>}
      </div>
      <Timeline segments={result.segments} duration={result.duration_s} onPlay={play} />
      <audio ref={audio} src={url} aria-label="Original recording" controls preload="metadata"
        onError={() => setPlaybackError('Playback is unavailable for this file in this browser.')} />
      {playbackError && <p className="inline-error" role="status">{playbackError}</p>}
    </section>
  )
}
