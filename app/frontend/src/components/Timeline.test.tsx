import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { Timeline } from './Timeline'

afterEach(cleanup)

it('maps probabilities to three color bands with labeled, keyboard-operable segments', () => {
  const onPlay = vi.fn()
  render(<Timeline duration={3} onPlay={onPlay} segments={[
    { start_s: 0, end_s: 1, fake_probability: 0.329 },
    { start_s: 1, end_s: 2, fake_probability: 0.33 },
    { start_s: 2, end_s: 3, fake_probability: 0.67 },
  ]} />)
  const low = screen.getByRole('button', { name: '0:00.000 to 0:01.000, Low fake probability, 33%' })
  const medium = screen.getByRole('button', { name: '0:01.000 to 0:02.000, Medium fake probability, 33%' })
  const high = screen.getByRole('button', { name: '0:02.000 to 0:03.000, High fake probability, 67%' })
  expect(low.querySelector('.band-low')).toBeTruthy()
  expect(medium.querySelector('.band-medium')).toBeTruthy()
  expect(high.querySelector('.band-high')).toBeTruthy()

  low.focus()
  fireEvent.keyDown(low, { key: 'ArrowRight' })
  expect(document.activeElement).toBe(medium)
  fireEvent.keyDown(medium, { key: 'Enter' })
  expect(onPlay).toHaveBeenCalledWith(1)
  expect(medium.getAttribute('aria-pressed')).toBe('true')
  fireEvent.click(high)
  expect(onPlay).toHaveBeenLastCalledWith(2)
})
