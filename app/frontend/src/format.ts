export function formatTime(seconds: number, precise = false) {
  const scale = precise ? 1000 : 1
  const ticks = Math.round(seconds * scale)
  const mins = Math.floor(ticks / (60 * scale))
  const secs = (ticks % (60 * scale)) / scale
  return mins + ':' + secs.toFixed(precise ? 3 : 0).padStart(precise ? 6 : 2, '0')
}

export function percent(probability: number) {
  return Math.round(probability * 100) + '%'
}
