import { verdictCopy } from '../copy'
import type { AnalysisResult } from '../types'

export function VerdictCard({ result }: { result: AnalysisResult }) {
  const copy = verdictCopy[result.verdict]
  return (
    <section className={'panel verdict ' + result.verdict} aria-labelledby="verdict-title">
      <div className="verdict-heading">
        <span className="verdict-icon" aria-hidden="true">{copy.icon}</span>
        <h2 id="verdict-title">{copy.title}</h2>
        <p className="probability">
          Fake probability
          <strong>{result.fake_probability === null ? 'Unavailable' : Math.round(result.fake_probability * 100) + '%'}</strong>
        </p>
      </div>
      <p>{copy.explanation}</p>
      {result.warnings.length > 0 && (
        <ul className="warnings">{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
      )}
    </section>
  )
}
