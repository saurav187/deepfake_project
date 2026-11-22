import { useRef, useState } from 'react'
import { formatTime, percent } from '../format'
import type { Segment } from '../types'

const bands = ['Low', 'Medium', 'High'] as const

function probabilityBand(probability: number) {
  if (probability < 0.33) return 'Low'
  if (probability < 0.67) return 'Medium'
  return 'High'
}

interface Props {
  segments: Segment[]
  duration: number
  onPlay: (seconds: number) => void
}

export function Timeline({ segments, duration, onPlay }: Props) {
  const [focused, setFocused] = useState(0)
  const [selected, setSelected] = useState<number>()
  const controls = useRef<(SVGGElement | null)[]>([])
  const width = 1000
  const endTime = Math.max(...segments.map((segment) => segment.end_s), duration, 0.001)

  function play(index: number) {
    setSelected(index)
    onPlay(segments[index].start_s)
  }

  return (
    <div className="timeline">
      {segments.length > 0 ? (
        <>
          <svg className="timeline-chart" viewBox={'0 0 ' + width + ' 32'}
            preserveAspectRatio="none" role="group" aria-label="Timeline segments">
            {segments.map((segment, index) => {
              const band = probabilityBand(segment.fake_probability)
              const start = formatTime(segment.start_s, true)
              const end = formatTime(segment.end_s, true)
              const size = (segment.end_s - segment.start_s) / endTime * width
              return (
                <g
                  key={index}
                  ref={(element) => { controls.current[index] = element }}
                  role="button"
                  tabIndex={focused === index ? 0 : -1}
                  aria-label={start + ' to ' + end + ', ' + band + ' fake probability, ' + percent(segment.fake_probability)}
                  aria-pressed={selected === index}
                  onFocus={() => setFocused(index)}
                  onClick={() => play(index)}
                  onKeyDown={(event) => {
                    let next = index
                    switch (event.key) {
                      case 'Enter':
                      case ' ':
                        event.preventDefault()
                        play(index)
                        return
                      case 'ArrowLeft':
                      case 'ArrowUp': next = Math.max(0, index - 1); break
                      case 'ArrowRight':
                      case 'ArrowDown': next = Math.min(segments.length - 1, index + 1); break
                      case 'Home': next = 0; break
                      case 'End': next = segments.length - 1; break
                      default: return
                    }
                    event.preventDefault()
                    controls.current[next]?.focus()
                  }}
                >
                  <title>{start}–{end} · {percent(segment.fake_probability)}</title>
                  <rect x={segment.start_s / endTime * width} y="2" width={size} height="28"
                    className={'segment band-' + band.toLowerCase()} />
                </g>
              )
            })}
          </svg>
          <div className="time-scale" aria-hidden="true">
            {[0, 0.25, 0.5, 0.75, 1].map((position) => <span key={position}>{formatTime(endTime * position, endTime < 10)}</span>)}
          </div>
          <ul className="legend" aria-label="Fake probability legend">
            {bands.map((band) => (
              <li key={band}><span className={'swatch band-' + band.toLowerCase()} />{band}</li>
            ))}
          </ul>
          <p className="timeline-hint">Select a segment to play from that point. Use the arrow keys to move between segments.</p>
        </>
      ) : <p className="timeline-hint">No segments were scored.</p>}
    </div>
  )
}
