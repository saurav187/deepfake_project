import { formatTime, percent } from '../format'
import type { Segment } from '../types'

export function SegmentTable({ segments }: { segments: Segment[] }) {
  if (!segments.length) return null

  return (
    <section className="panel segment-panel" aria-labelledby="segments-title">
      <h2 id="segments-title">Segment details</h2>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Scored segments">
        <table>
          <thead><tr><th scope="col">Start</th><th scope="col">End</th><th scope="col">Fake probability</th></tr></thead>
          <tbody>
            {segments.map((segment, index) => (
              <tr key={index}>
                <td>{formatTime(segment.start_s, true)}</td>
                <td>{formatTime(segment.end_s, true)}</td>
                <td>{percent(segment.fake_probability)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
