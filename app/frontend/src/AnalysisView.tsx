import { useEffect, useRef } from 'react'
import { Caveat } from './components/Caveat'
import { ExperimentalBanner, ModelBadge } from './components/ModelBadge'
import { Recording } from './components/Recording'
import { SegmentTable } from './components/SegmentTable'
import { VerdictCard } from './components/VerdictCard'
import { downloadReport } from './report'
import type { AnalysisResult } from './types'

export function AnalysisView({ result, file, onReset }: {
  result: AnalysisResult
  file: File
  onReset: () => void
}) {
  const heading = useRef<HTMLHeadingElement>(null)
  useEffect(() => heading.current?.focus(), [])

  return (
    <div className="stack">
      <div className="result-header">
        <h2 ref={heading} tabIndex={-1}>Analysis complete</h2>
        <div className="result-actions">
          <button type="button" onClick={() => downloadReport(result)}>Download report</button>
          <button type="button" onClick={onReset}>Check another file</button>
        </div>
      </div>
      {result.model.experimental && <ExperimentalBanner />}
      <VerdictCard result={result} />
      <Recording file={file} result={result} />
      <Caveat />
      <ModelBadge model={result.model} />
      <SegmentTable segments={result.segments} />
    </div>
  )
}
