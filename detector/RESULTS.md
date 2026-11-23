# Detector v2: release v1 results

Release `detector/runs/release/v1` (release.json sha256 `bdb7748c…`), evaluated once per test set on 2026-11-23.
In-the-Wild and VA were sealed: they played no part in training, encoder choice, epoch selection, calibration or thresholds.

## Model

- Frozen `wav2vec2-xls-r-300m` encoder with 3 trained heads (seeds 42, 43, 44; best epochs 3, 2, 10). The recording score is the mean logit over heads and over every 4 s window.
- Training data: ASVspoof 5 train (both tars, 72,673 clips) and IndieFake train, with codec, noise, gain and edge-silence augmentation.
- Encoder choice: XLS-R had a mean dev EER of 0.051 against 0.081 for WavLM Base+ (one seed each, on half the training data), so it won outright.
- Calibration: Platt scaling, `p = sigmoid(0.688 * logit + 1.760)`, fitted on source-balanced dev audio (clean and MP3 32k).
- Thresholds, from a 5% false-alarm and 5% miss budget on dev: likely_real below 0.514, likely_fake at or above 0.562. At least 1 s of speech is required after trimming.

## Dev (selection and calibration data)

| Slice | n | AUC | EER |
|---|---:|---:|---:|
| Pooled, source-balanced | 8,800 | 0.990 | 5.4% |
| ASVspoof 5, clean / MP3 32k | 3,600 each | 1.000 / 1.000 | 0.3% / 0.8% |
| IndieFake, clean / MP3 32k | 800 each | 0.971 / 0.963 | 9.2% / 10.5% |

The IndieFake dev split has only 4 speakers, so its numbers are noisy.

**Edge reliance (dev, ASVspoof 5 recordings with 3 or more windows, n=167):** the shift of the first or last window from the interior windows separates real from fake with AUC 0.63 and 0.60. This is a mild residual cue, not a dominant one.

## Sealed test: In-the-Wild (31,712 recordings)

95% intervals come from 1,000 bootstrap resamples of whole speakers.

| | AUC | EER |
|---|---:|---:|
| Legacy audited baseline | 0.839 | 23.4% |
| **v1** | **0.957** [0.941, 0.972] | **11.3%** [8.9, 13.5] |

Outcomes at the frozen thresholds (all recordings in the denominator):

| | Likely fake | Uncertain | Likely real | Insufficient audio |
|---|---:|---:|---:|---:|
| Genuine (19,901) | 5.2% | 0.5% | 84.1% | 10.2% |
| Fake (11,811) | 75.4% | 0.9% | 14.6% | 9.1% |

By speech length after trimming:

| Speech | n | AUC | EER | Genuine flagged fake | Fakes flagged fake |
|---|---:|---:|---:|---:|---:|
| Under 4 s | 22,188 | 0.934 | 14.8% | 4.5% | 59.5% |
| 4–16 s | 9,111 | 0.992 | 4.4% | 7.5% | 98.3% |
| Over 16 s | 413 | 0.999 | 1.5% | 3.9% | 98.9% |

The 3,093 "insufficient audio" clips have a median raw duration of 1.2 s, so they are genuinely tiny clips, not a decoding failure.

## Sealed test: VA (44 recordings)

| | AUC | EER | Genuine flagged fake | Fakes flagged fake |
|---|---:|---:|---:|---:|
| Legacy (from its report) | — | — | 0/22 | 6/22 |
| **v1** | 0.963 [0.883, 1.000] | 6.8% | 1/22 | 19/22 (1 uncertain, 2 missed) |

Errors: `tts_1.m4a` and `tts_7.m4a` were called likely_real (p 0.33 and 0.35), `linus-to-margot.wav` came out uncertain (0.555), and `Speaker-16_bonafide17.wav` was a false alarm (0.92).

## App check

`DETECTOR_BACKEND=detector` loads v1 (`ssl-wav2vec2-xls-r-300m`, version `v1-bdb7748c`). Uploads through `/api/analyze` reproduce the sealed-test scores (for `Speaker-16_bonafide17.wav`, 0.920 in the app against 0.916 in the evaluation; the gap is from bf16 batching). The app correctly rejected a 115 MB file with `too_large`.

## What this means and what's next

- About 81 of every 100 In-the-Wild files get the correct verdict, about 9 are wrong and about 10 are too short to judge.
- Accuracy depends strongly on length. With 4 s or more of speech, the model catches 98% of fakes. Below 4 s it catches only about 60%, so the app should tell users to upload at least 5–10 s of speech.
- The uncertain band is very narrow (0.514–0.562), so errors show up as confident verdicts. Next, choose wider bands on a speaker-separate dev split, not on these test sets.
- Untested so far: Hindi speech, WhatsApp/phone audio, and the newest voice-cloning systems. A small private test set of the user's own recordings is the next evaluation.
- Pending ideas from the Codex v3 review: separate speaker splits for selection, calibration and thresholds; thresholds that depend on how much speech a clip has; and an Opus 16k stress channel.

Any change to the model or thresholds needs a new release, and these sealed results stay attached to v1.
