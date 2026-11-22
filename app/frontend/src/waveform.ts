const WIDTH = 480
const HEIGHT = 88

export const waveformSize = { width: WIDTH, height: HEIGHT }

export async function waveformPath(file: File): Promise<string> {
  const context = new OfflineAudioContext(1, 1, 8000)
  const buffer = await context.decodeAudioData(await file.arrayBuffer())
  const channels = Array.from({ length: Math.min(2, buffer.numberOfChannels) },
    (_, index) => buffer.getChannelData(index))
  const middle = HEIGHT / 2
  const step = buffer.length / WIDTH
  const bins = []

  for (let column = 0; column < WIDTH; column += 1) {
    const start = Math.floor(column * step)
    const end = Math.min(buffer.length, Math.max(start + 1, Math.floor((column + 1) * step)))
    const stride = Math.max(1, Math.floor((end - start) / 96))
    let peak = 0
    for (let sample = start; sample < end; sample += stride) {
      for (const channel of channels) peak = Math.max(peak, Math.abs(channel[sample]))
    }
    bins.push(peak)
  }

  const maxPeak = Math.max(0.03, ...bins)
  return bins.map((peak, x) => {
    const amplitude = Math.max(0.5, (peak / maxPeak) * (middle - 2))
    return 'M' + (x + 0.5) + ' ' + (middle - amplitude).toFixed(1) + 'v' + (amplitude * 2).toFixed(1)
  }).join('')
}
