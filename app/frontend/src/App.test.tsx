import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('rejects a 60 MB file without making an upload request', async () => {
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
    model: { name: 'mock', version: '1.0', description: '', experimental: true },
    limits: { max_upload_mb: 50, max_duration_s: 600, accepted_extensions: ['.wav'] },
  })))
  vi.stubGlobal('fetch', fetchMock)
  render(<App />)
  const chooser = await screen.findByLabelText('Audio or video file', {}, { timeout: 3000 })
  const file = new File([new Uint8Array(60_000_000)], 'large.wav', { type: 'audio/wav' })
  fireEvent.change(chooser, { target: { files: [file] } })
  expect(screen.getByRole('alert').textContent).toBe('Choose a file that is 50 MB or smaller.')
  expect(fetchMock).toHaveBeenCalledTimes(1)
  expect(fetchMock.mock.calls[0][0]).toBe('/api/model')
})
