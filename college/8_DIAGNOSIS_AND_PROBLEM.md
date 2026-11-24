# Why v1 misses modern synthetic voices

## Finding from existing development trials

I recomputed recording-level statistics from the 7,984 rows of `detector/runs/compare/x1_vs_v1/dev_scores.csv`. Each row has paired clean, MP3 and 16→24→16 kHz views. These are seed-42 head logits. The shipped v1 combines three heads and has a separate calibration. Scores below refer to the single-head comparison.

| Development cohort | n | Clean median logit | MP3 32k median | Rate round trip median |
|---|---:|---:|---:|---:|
| ASVspoof 5 genuine | 400 | −11.05 | −9.62 | −10.76 |
| ASVspoof 5 fake | 3,200 | 11.31 | 10.91 | 11.44 |
| IndieFake genuine, stored 16 kHz | 379 | −2.31 | −1.98 | −2.24 |
| IndieFake fake, stored 16 kHz | 724 | 3.26 | 3.19 | 3.39 |
| IndieFake fake, stored 24 kHz | 594 | 9.30 | 8.94 | 9.41 |
| IndieFake genuine, stored 44.1 kHz | 1,221 | −6.77 | −6.17 | −6.80 |
| IndieFake fake, stored 44.1 kHz | 1,466 | 4.48 | 4.04 | 4.55 |

A known 24 kHz fake typically receives a strong fake score. There is no genuine 24 kHz IndieFake cell to identify an independent rate effect. Genuine scores also differ by more than four logit units between IndieFake rate groups. Speakers, capture history, content and bandwidth can all explain that difference. x1 improved clean IndieFake EER from 7.44% to 4.09% on the same rows, with small changes on ASVspoof 5 (0.86% to 0.69%). These use linear interpolation of the ROC crossing. The repository comparison uses the nearest operating point.

The owner previously observed two confident misses on Gemini and little change after transcoding. That observation motivates an experiment. Two examples cannot estimate a vendor miss rate. Taken together, the strongest explanation is inadequate coverage of synthesis family and recording style in the development distribution. Re-encoding alone has weak support as a remedy. The existing data cannot distinguish a representation that lacks the cue from a head that assigns it little weight. Paper 15 has a frozen-backbone counterexample and paper 13 motivates testing lower layers.

## Three-sentence problem statement

A Hindi or English voice note can contain human speech, synthesized speech or a cloned voice, and recording and forwarding change the acoustic evidence available to an audio-only detector. The project’s current English-trained detector separates its benchmark cohorts well yet has made confident errors on a modern commercial voice, while papers 3, 15 and 32 document substantial changes under domain or codec shift. This project asks whether matched bilingual training data and teacher-to-student distillation can deliver calibrated decisions on held-out voices and generators with a measured genuine false-alarm budget near 2% and a local model under 90 MB.

## Objectives and falsifiable questions

| Question | Controlled experiment on development data | Interpretation |
|---|---|---|
| Does clean read speech alone push v1 toward genuine? | Score fresh paired read vs conversational recordings by the same Hindi/English speakers and device. Compare within-speaker differences before/after Opus. Repeat with commercial speech reading the same texts. | A clean/phone score shift within both labels supports a style confound. The ASVspoof/IndieFake contrast above pools speaker and source. |
| Does generator coverage explain transfer? | Fit the same head on old data, then add matched new families with the same real pool and step budget. Reserve one whole synthesis lineage per language. Repeat vendor-disjoint and related-version trials. | A gain restricted to a trained service indicates family-specific learning. A held-family gain supports broader transfer. |
| Does frozen XLS-R retain a discriminative representation? | Compare each frozen layer, first 4/6 layer averages, small WavLM, and linear-frequency cepstra on identical speaker/voice/text partitions. Compare a linear probe with an MLP. | Success of a fresh probe where v1 misses implicates the old training/head. Failure of both probes motivates limited adaptation. It does not prove that the waveform has no evidence. |
| Can a small model keep that signal? | Compare exported int8 student and teacher on identical validation recordings, including short speech, actual forwarding and compression. | Report paired outcomes, false alarms, coverage, discrimination, model bytes and CPU time. |
| Does a probability travel across sources? | Fit file and window Platt curves on separate speakers, choose file thresholds on another group, assess calibration on untouched validation speakers. | Brier score, reliability and false-alarm/false-reassurance counts, including the abstention denominator. |

The intended scope is uploaded speech of roughly 3–30 seconds, including Indian English and Hindi. The timeline measures local evidence for synthesis. Known mixed or partial edits require segment annotations for calibrated localization. Watermarks, face analysis, speaker identification and live interception are outside the study.

Reproduce the table with `python -m pipeline_v3.diagnose --scores detector/runs/compare/x1_vs_v1/dev_scores.csv`. Use `pipeline_v3.v1_dev` for authorised fresh development trials and notebook 02 for the probe experiment. No new provider audio or model fit contributed to the findings above. [Historical release](../detector/RESULTS.md) · [Project review](../docs/PIPELINE_REVIEW.md)

### Paired check on the shipped three-head v1

I also ran the actual release on 32 existing development recordings with 5–9 s of usable speech, then on the same recordings passed through libopus at 16 kb/s. Each corpus contributed eight real and eight fake recordings. The sampler cycled over available speaker groups with seed 421. All 64 inputs were scorable.

| Corpus | Speakers in this panel | Median fake probability for genuine, original → Opus | Median for fake, original → Opus | AUC, original → Opus |
|---|---:|---:|---:|---:|
| ASVspoof 5 dev | 16 | 0.011% → 0.029% | 99.997% → 99.997% | 1.000 → 1.000 |
| IndieFake dev | 9 | 4.35% → 10.92% | 99.79% → 99.44% | 1.000 → 0.984 |

The same transform raises the typical genuine score much more in the IndieFake sample. Even after recoding, typical familiar fakes remain at the opposite end of the scale from the owner's previously reported modern-TTS misses. This is a paired codec control on a small development panel. It cannot isolate spontaneous delivery, identify a held-out commercial generator, or reproduce WhatsApp's full capture chain. The nine IndieFake speaker IDs belong to dev in both audit split plans. The older v1 results describe four speakers in their narrower reported dev cohort.

Reproduction from the repository root: `DEVICE=cpu HF_HUB_OFFLINE=1 venv/bin/python -m pipeline_v3.v1_channels --repo . --per-class 8`.
