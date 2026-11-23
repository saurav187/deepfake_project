# Deepfake Audio Check: Project Report

*An app that tells you whether a voice recording is real or made by AI.*

---

## 1. The problem

AI can now copy anyone's voice from a few seconds of audio. Tools like voice cloning and text-to-speech (TTS) make fake voice notes, fake phone calls and fake speeches that sound real. People use them for scams ("Mom, I'm in trouble, send money"), fake news and fraud.

Most people cannot hear the difference. **This project builds a tool that can.**

## 2. What the app does

1. You open the website and upload a recording: a voice note, an MP3, or even a video file.
2. The app listens to it and gives one of four answers:
   - 🟢 **Likely real**
   - 🔴 **Likely fake**
   - 🟡 **Uncertain**
   - ⚪ **Not enough audio** (too short, or no clear speech)
3. It shows a **fake probability** (0–100%) and a **timeline** coloured section by section, so you can see which parts sound suspicious.

Uploaded audio is analysed in memory and **never saved**.

## 3. How it works (simple version)

```
Upload → Clean up the audio → Cut into 4-second pieces → AI "listener" → Judge → Verdict
```

1. **Clean up:** convert any format to the same standard (16,000 samples per second, mono). Remove silence at the start and end. Make the loudness even.
2. **Cut into pieces:** split the speech into 4-second windows that cover the whole recording.
3. **AI listener (XLS-R):** a large speech model from Meta, trained on 436,000 hours of speech in 128 languages. It has learned how human speech works. We use it to turn each piece into a detailed "fingerprint".
4. **Judge (our trained part):** a small network we trained to spot the tiny traces that AI voice generators leave in these fingerprints. Three judges are trained separately and they vote. This is called an ensemble, and it makes the answer more stable.
5. **Verdict:** the scores of all pieces are averaged and turned into a probability. That probability is compared with two limits chosen in advance, which give the four answers above.

**Why this design?** Our first model learned from pictures of sound (spectrograms) from scratch. It scored well in the lab but failed on real-world audio. Starting from a model that already understands speech generalises much better.

## 4. Training data

| Dataset | What it is | Used for |
|---|---|---|
| **ASVspoof 5** (2024) | The newest international anti-spoofing challenge data: real speakers vs. many AI voice systems | Training |
| **IndieFake** | Indian speakers, real vs. fake | Training |
| **In-the-Wild** | 31,712 real and fake clips of celebrities and politicians collected from the internet | **Final test only** |
| **VA set** | 44 clips, real and fake | **Final test only** |

### How we kept the results honest

- **Different people in training and testing.** No speaker appears in both, so the model can't just memorise voices.
- **Sealed test sets.** In-the-Wild and VA were never used for training or tuning. They were run exactly once, at the end, and every run is logged.
- **Shortcut checks.** We found the model could "cheat": in one dataset, real recordings had silence at the end and fakes didn't, so the model was judging silence instead of voice. We removed that cue and added an automatic check that fails training if such a shortcut comes back. We do the same for file formats (WAV vs FLAC).
- **Real-world conditions.** During training, audio is randomly compressed (MP3, AAC, Opus, phone-quality), with noise and volume changes added, so the model works on WhatsApp-style audio, not just studio recordings.

## 5. Results

Tested on **In-the-Wild**: 31,712 internet clips the model had never seen.

| | Old model | **New model** |
|---|---:|---:|
| AUC (1.0 = perfect, 0.5 = guessing) | 0.839 | **0.957** |
| Error rate (EER, lower is better) | 23.4% | **11.3%** |

**The errors were cut by half.**

**Out of 100 uploaded files:**
- about **81 get the correct answer**
- about 9 get a wrong answer
- about 10 are "not enough audio" (clips shorter than about 1.5 seconds)

**Length matters a lot:**

| Speech in the clip | Fakes caught | Real voices wrongly flagged |
|---|---:|---:|
| Under 4 seconds | 60% | 5% |
| **4–16 seconds** | **98%** | 8% |
| Over 16 seconds | 99% | 4% |

👉 **Upload at least 5–10 seconds of speech for a reliable answer.**

On the small **VA set** the new model caught **19 of 22 fakes** with 1 false alarm. The old model caught only 6 of 22.

## 6. Live demo plan (about 5 minutes)

1. **Open the app** at <http://localhost:8000>. It runs fully offline on this laptop.
2. **Upload a real recording**, for example your own voice note of 10+ seconds. Expected: 🟢 Likely real.
3. **Upload a known fake** from the test sets (a clip the model catches). Expected: 🔴 Likely fake. Show the probability and the timeline.
4. **Upload a very short clip** (1–2 s). Expected: ⚪ Not enough audio. Explain that the app refuses to guess rather than give a wrong answer.
5. **Show the results table** (section 5) and explain the honest testing (section 4).
6. **Be upfront about the limits** (section 7). Examiners trust honest numbers more than perfect ones.

## 7. Honest limitations

- **Newest TTS tools are the weak spot.** Voices from very recent generators (2025–2026 systems) can fool the model, because it never saw them in training. Research papers report the same problem for other detectors: up to 30–55% error on the newest TTS. This is the main focus of the next version.
- **Short clips** (under 4 seconds) are much less reliable.
- **Mostly English training data.** Hindi and other Indian languages are not yet well tested.
- **Confident mistakes.** The "uncertain" zone is narrow, so a wrong answer usually looks confident.
- **It's an aid, not proof.** A detector's result should never be the only evidence that something is fake.

## 8. Next steps (version 2)

1. **Train on modern AI voices:** new public datasets (SpeechFake, MLAAD, IndicSynth for Hindi) that cover CosyVoice, GPT-SoVITS, Fish Speech, XTTS, OpenVoice and more.
2. **Our own test set:** real voices and fakes from tools people actually use today, in Hindi and English.
3. **Fine-tune the AI listener** itself instead of keeping it frozen.
4. **A wider "uncertain" zone,** so the app says "not sure" instead of being confidently wrong.
5. **A mobile version:** a smaller model (about 150 MB), or the phone sends audio to a server.

## 9. Technology used

| Part | Tools |
|---|---|
| AI model | PyTorch, Hugging Face Transformers, XLS-R 300M (Meta) |
| Audio processing | ffmpeg, soundfile, soxr |
| Backend | Python, FastAPI |
| Frontend | React, TypeScript, Vite |
| Training hardware | NVIDIA RTX 5060 (8 GB); the final 3 models trained in about 2.5 hours |
| Runs on | Any laptop; no GPU needed (a 28-second clip took about 3 seconds on a desktop CPU; laptops can be slower) |

## 10. Likely questions

**Q: How is this different from just listening carefully?**
AI voices are now good enough to fool people. The model detects statistical traces in the sound that humans can't hear.

**Q: Can it be fooled?**
Yes, especially by the newest generators it hasn't seen. That's why we test on data the model never saw, and why version 2 adds modern AI voices.

**Q: Why not 100% accuracy?**
No detector is perfect, and new generators keep appearing. We report the real error on unseen internet audio, not lab numbers. In the lab (same type of data as training) the model scores 99% AUC, but that number would be misleading.

**Q: Does it work on WhatsApp audio?**
Training included compressed and phone-quality audio, so it's designed for it. We are building a WhatsApp/Hindi test set to measure this properly.

**Q: Is my audio stored?**
No. It is analysed in memory and discarded.
