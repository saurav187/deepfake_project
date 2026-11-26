# V2-90 protocol — ≥90% per family at ≤15% genuine false alarms

Written 26 Nov 2026, ~19:30 IST, by `v2-overnight-lead`, before any V2-90 data is scored.
It supersedes protocol 27 (70% target) for the owner's new target. Owner decisions (chat, 26 Nov):

- No paid synthesis.
- Non-commercial research licences are acceptable for this college project (MLAAD CC BY-NC,
  AntiDeepfake CC BY-NC-SA, GLOBE, LibriTTS-R, VCTK).
- Target 90% if achievable; ~150 CU available; A100 allowed.
- Shortcut-audit handling and familiarity rules are delegated to the lead. The decisions are
  recorded in section 5.

## 1. Claim

On held-out English clean recordings, one frozen rule detects at least 90% of each of MLAAD
`Chatterbox`, `ElevenLabs-v3`, `FishTTS` and `Gemini-3.1-Flash-TTS` **separately**, while
flagging at most 15% of genuine recordings.

- **Label.** "Familiar family; unseen text, unseen parent recording, and (ElevenLabs/Gemini)
  unseen stock voice."
- **Secondary results under the identical cutoff.** Simulated call channels:
  - G.711 µ-law 8 kHz;
  - AMR-NB;
  - Opus 12 kbps;
  - GSM.

  Real calls are **not** measured by this protocol.

## 2. Splits (fixed by deterministic rules before any model sees the data)

**Fakes (MLAAD, revision `30c3dec7`).**

- **Train.** Rows whose `original_file` is `en_US/...` from all downloaded folders. For
  ElevenLabs-* and Gemini, train uses stock voices in half A only.
  - The half is `sha256(voice) % 2 == 0`. Voice is `reference_speaker`, or the voice name
    from the file name when that is empty.
  - Half-B rows are dropped from training.
- **Test.** Rows whose `original_file` is `en_UK/...` in the four target folders. For
  ElevenLabs-v3 and Gemini, only voice half B counts.
  - `en_UK` texts and parents never occur in training: all en_UK rows of *every* folder are
    excluded from training.
  - Chatterbox and FishTTS each have one voice. Their test is unseen text/parent but a
    familiar voice, and is labelled as such.
- **Calibration fakes.** None. The cutoff is set on genuine audio only.

**Genuine.**

| Split | Sources (speaker-disjoint) |
|---|---|
| Train | M-AILABS en_US readers (Drive); LibriTTS-R train-clean-100; VCTK speakers with `sha256(spk)%10 < 6`; GLOBE v2 train shards |
| Calibration | LibriTTS-R dev-clean + dev-other; VCTK `%10 in {6,7}`; LibriSpeech-dev 841 (same readers as LibriTTS dev) |
| Test | LibriTTS-R test-clean + test-other; VCTK `%10 in {8,9}`; GLOBE v2 test |

Per corpus, training genuine is capped at 1,500 clips per corpus split and 40 clips per speaker.
Test and calibration use at most 10 clips per speaker, so that FA is not dominated by a few
speakers.

**One unit per recording per condition.** Channel siblings share the recording's split.

## 3. Model and selection (development only)

- **Common preprocessing for every clip.**
  - Decode, 16 kHz mono, trim edge silence, RMS-normalise.
  - **4 s** crop: centred for test and calibration, random for training. Clips shorter than
    4 s are excluded and counted.
- **Augmentation (both labels, same distribution).**
  - Random low-pass at 4–8 kHz (p 0.3).
  - Codec pool (p 0.4): µ-law, A-law, GSM, AMR-NB, Opus 8–24 kbps, MP3.
  - Gain ±6 dB.
  - Light noise (p 0.2).
  - **AMR-NB is held out of training** to measure an unseen codec.
- **Candidates.**
  - **C1.** Frozen XLS-R-300m (the local copy used by release v1) multi-layer mean/std
    features with a logistic head.
  - **C2.** XLS-R-300m fine-tuned (top layers + attentive-stat head), ≤3 epochs, with
    checkpoints.
  - **C3** (optional, only if time allows). AntiDeepfake checkpoint fine-tuned the same way.
- **Selection.**
  - Selection uses a *development* split carved from **train**: 10% of en_US parents and
    VCTK/LibriTTS speakers.
  - The criterion is the minimum over the four target families of development detection at
    the development genuine 15% cutoff.
  - The primary is frozen before calibration.

## 4. Cutoff and evaluation

- **Cutoff.** The lowest cutoff with ≤ ⌊0.15·n⌋ calibration genuine flagged. The calibration
  pool is LibriTTS-R dev + VCTK cal + LibriSpeech-dev, clean channel. Ties are resolved by
  `≥`.
- **Freeze.** The primary model, its sha256, the cutoff and the test manifest sha256 go into
  `v2/v2-90/signoff.json` before any test score is computed.
- **Evaluation.**
  - Computed once.
  - Reports, per family and per genuine source and speaker: raw counts, Wilson 95% intervals
    and a speaker/voice cluster bootstrap.
  - Release v1 is scored on the same test rows, at its shipped cutoff (0.5623) and at the
    same calibration rule, as reference.
- **Verdict.**
  - **PASS.** Genuine FA ≤ 15% *and* every family ≥ 90%, at point estimates.
  - **"Not robust".** A PASS whose interval crosses a limit is reported that way.
  - **MISS.** Any cell fails. Each gap is given in recordings.

## 5. Lead decisions delegated by the owner

1. **Shortcut audit does not block.**
   - Before training, simple-feature probes are fitted on train and scored on development:
     duration before crop, speech ratio, RMS, spectral roll-off, upper-band energy, tilt and
     the native sample-rate flag.
   - Any probe above 0.80 AUC is a **red flag** printed next to the result. Between 0.60 and
     0.80 it is reported.
   - The common preprocessing and the label-symmetric low-pass/codec augmentation are the
     mitigation.
2. **Familiar family with unseen voice and text counts** toward the target and is labelled as
   such. Family-held-out generalisation is not claimed.
3. **Versions.** "FishTTS" and "Gemini TTS" mean exactly the MLAAD folders `FishTTS` and
   `Gemini-3.1-Flash-TTS` at the pinned revision. The sibling folders are training-only
   context: Fish-S2-Pro, Openaudio-S1-Mini, Chatterbox-Turbo, ElevenLabs-Turbo-v2.5 and
   ElevenLabs-v2-Multilingual.

## 6. Known limitations, recorded in advance

- All en_UK test texts come from one reader's books (Elizabeth Klett).
- Test genuine is from different corpora than the test fakes, so FA measures read speech from
  LibriTTS-R, VCTK and GLOBE.
- Chatterbox and FishTTS each have a single, familiar voice.
- LibriTTS-R is a restored, vocoder-processed corpus. Its "genuine" audio has passed through a
  neural vocoder, which may raise FA. This is reported per source.
- A Chatterbox watermark (Perth) may be a learnable cue. It is acceptable for detecting the
  family but is noted.
