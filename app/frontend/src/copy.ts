import type { Verdict } from './types'

export const verdictCopy: Record<Verdict, { title: string; explanation: string; icon: string }> = {
  likely_fake: {
    title: 'Likely AI-generated',
    explanation: 'Large parts of this recording show strong signs of synthetic speech.',
    icon: '!',
  },
  uncertain: {
    title: "Can't tell",
    explanation: 'The signs are mixed. Treat this recording with caution and verify it another way.',
    icon: '?',
  },
  likely_real: {
    title: 'No strong signs of AI',
    explanation: "We didn't find strong signs of synthetic speech. This doesn't prove the recording is genuine.",
    icon: '✓',
  },
  insufficient_audio: {
    title: 'Not enough speech',
    explanation: "We couldn't find enough clear speech to analyze. Try a longer or clearer recording.",
    icon: '–',
  },
}

export const privacyNote = "Your file is analyzed on our server and deleted immediately afterwards. We don't keep or share your audio."
export const caveat = "This is an automated estimate, not proof. Detectors can miss new voice-cloning tools and can be wrong on heavily compressed, noisy or edited audio. Don't make important decisions based on this result alone — verify through another channel, such as calling the person back on a known number."
export const unreadable = "We couldn't read this file as audio"
