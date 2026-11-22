import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import type { AnalysisResult, Verdict } from '../types'
import { VerdictCard } from './VerdictCard'

afterEach(cleanup)

it.each<[Verdict, string, string]>([
  ['likely_fake', 'Likely AI-generated', 'Large parts of this recording show strong signs of synthetic speech.'],
  ['uncertain', "Can't tell", 'The signs are mixed. Treat this recording with caution and verify it another way.'],
  ['likely_real', 'No strong signs of AI', "We didn't find strong signs of synthetic speech. This doesn't prove the recording is genuine."],
  ['insufficient_audio', 'Not enough speech', "We couldn't find enough clear speech to analyze. Try a longer or clearer recording."],
])('renders the exact %s copy', (verdict, title, explanation) => {
  const result: AnalysisResult = {
    verdict, fake_probability: verdict === 'insufficient_audio' ? null : 0.87,
    duration_s: 3, speech_s: 3, warnings: [], segments: [],
    model: { name: 'mock', version: '1.0', description: '', experimental: true },
  }
  render(<VerdictCard result={result} />)
  expect(screen.getByRole('heading', { name: title })).toBeTruthy()
  expect(screen.getByText(explanation)).toBeTruthy()
  expect(screen.getByText(verdict === 'insufficient_audio' ? 'Unavailable' : '87%')).toBeTruthy()
})
