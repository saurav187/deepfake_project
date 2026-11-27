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

## Amendment 1 (26 Nov, ~19:05 IST, before any V2-90 data was preprocessed or scored)

Colab's ffmpeg lacks an AMR-NB encoder. The held-out unseen call codec is therefore
**G.726 32 kbps (8 kHz)** instead of AMR-NB, both in training (excluded) and in evaluation.
Evaluation channels: clean, G.711 µ-law 8 kHz, G.726 (unseen), Opus 12 kbps, GSM.

## Amendment 2 (26 Nov, ~19:30 IST, before any V2-90 data was preprocessed or scored)

At the owner's request (diversity of recordings and speakers), two genuine sources and one fake
system are added.

- **AMI meetings** (`edinburghcstr/ami`): headset (IHM) and distant-mic (SDM) segments,
  spontaneous speech. Split by speaker: `sha256("ami/"+spk)%5 == 0` test, `== 1` calibration,
  else train. Both mics of one speaker share a split.
- **IndieFake** (local copy; capped subset packed by `pack_indiefake.py`, sha256 `256ee052…`):
  Indian-accented English.
  - 13 speakers are held out for test (genuine and fakes, 10 clips each).
  - Of the rest, speakers with `sha256%10 == 1` give calibration genuine (fakes dropped) and
    `== 0` give development; the remainder train.
  - IndieFake fakes are a descriptive cell, not a target.
- **MLAAD `RVC`** (voice conversion), en_US rows, training only.

**Primary genuine FA** is now pooled over *all* test genuine sources: LibriTTS-R, VCTK, GLOBE,
AMI and IndieFake. It is also reported per source, plus a "read speech only" pool (LibriTTS-R,
VCTK, GLOBE). The calibration pool adds the AMI and IndieFake calibration speakers.

## Amendment 3 (26 Nov, ~19:45 IST, before any V2-90 data was preprocessed or scored)

The owner authorised extra steps and compute to reach the target. Candidates, all trained on
the same train split with the same sampler and augmentation:

- **C2.** `facebook/wav2vec2-xls-r-300m`, full fine-tune (CNN frozen), batch 32.
- **C3.** `nii-yamagishilab/xls-r-1b-anti-deepfake`, an XLS-R-1B encoder post-trained for
  deepfake detection (CC BY-NC-SA). The bottom 24 of its 48 layers are frozen, batch 16. Only
  the encoder is used; a new head is trained.
- **C4.** The equal-weight mean of C2 and C3 z-scores, with z from development genuine
  mean/sd.

**Selection (`v2_90.py select`).**

- Score development (clean) with each option.
- Pick the maximum of the minimum-over-target-families detection at the development genuine 15%
  cutoff.
- Ties go to the higher pooled detection, then to fewer members.

The chosen option is frozen in `selection.json` before calibration. Calibration, sign-off and
evaluation use it unchanged. C1 (frozen-feature head) is dropped.

## Amendment 4 (26 Nov, ~20:35 IST, before any V2-90 data was preprocessed or scored)

The owner states this is the final model and asks for the widest coverage. Additions:

- **German ElevenLabs-v3** (MLAAD `fake/de`, all 1,000 clips). Training and development only;
  stock-voice half B excluded, so the English test voices stay unseen.
- **Hindi.** Reported as a **separate result**. It is not part of the primary English verdict.
  - Fakes: MLAAD `fake/hi`, 9 systems, 300 clips each: IndicF5, Indri-TTS, OmniVoice, Ringg
    Squirrel, Veena, VoxCPM2, Voxtral, Bark, XTTS-v2. Their M-AILABS text roles match English
    (en_US train/development, en_UK test).
  - Fakes: IndicSynth Hindi (XTTS-v2, VITS, FreeVC24 voice conversion; Drive copy). Target
    speakers are Kathbath speakers, so the two share a namespace.
  - Genuine: Kathbath Hindi (8 of 32 shards) and IndicVoices Hindi (5 shards; read, extempore,
    conversation).
  - Split by speaker: `sha256(spk)%10` in {0,1} test, 2 calibration (genuine), 3 development,
    else train.
  - Caps: 40 genuine clips per speaker in train, 10 in evaluation, 3,000 IndicSynth training
    fakes.
- **Cutoff.** Calibrated on **English** calibration genuine only, unchanged. Hindi genuine FA
  and Hindi fake detection are reported at that same cutoff (`genuine_hindi` cell). The primary
  `genuine` cell is English test genuine only (this replaces the all-sources pool of
  amendment 2 only by removing Hindi; AMI and IndieFake stay in it).
- **Known caveat.** An IndicSynth FreeVC fake converts a *source* speaker's utterance. The
  source speaker's genuine audio may fall in another split. This is content overlap, not
  identity overlap, and is noted.

## Amendment 6 (26 Nov, ~22:15 IST, before any V2-90 data was preprocessed or scored)

This amendment follows the Codex review of commit `26ff20f`. It aligns the protocol with the
code at commit `e9a0229` and later.

**Actual split rules.** These supersede the earlier text where they differ.

| Source | Role | Split unit and rule |
|---|---|---|
| MLAAD (all folders) | fakes | en_US originals: train, with dev when `sha256(original)%10==0`. en_UK originals: test, for the four targets and the Hindi folders only. Stock-voice families use voice half A for train/dev and half B for test. German EL-v3: train/dev only. One fake per (family, original) |
| LibriTTS (original) | genuine | train-clean-100 → train (dev by speaker hash); dev-clean/other → calibration; test-clean/other → test |
| VCTK | genuine | by speaker: `%10<6` train, 6–7 calibration, 8–9 test. One recording per (speaker, text_id) when the column exists |
| AMI (IHM + SDM) | genuine | by **connected component of meetings and speakers**, so all mic views of a meeting and every meeting a speaker attends stay in one split. `%5`: 0 test, 1 calibration, else train. One recording per (meeting, speaker, begin time), so IHM and SDM siblings are not double-counted |
| EdAcc | genuine | by **conversation** (`EDACC-Cxx`). `%5`: 0 test, 1 calibration, else train |
| M-AILABS en_US | genuine | train/dev only, by reader |
| IndieFake subset | both | 13 speakers test; the rest by speaker: `%10==0` dev, `==1` calibration genuine, else train |
| Kathbath / IndicVoices / IndicSynth | Hindi | by speaker, `%10`: {0,1} test, 2 calibration, 3 dev, else train. IndicSynth uses the Kathbath target speaker |

- **Removed.** LibriSpeech-dev 841 was never loaded. LibriTTS-R and GLOBE v2 are excluded
  (amendment 5).
- **VoxENES.** Excluded from the primary run (`--include-voxenes` is off). It has no source or
  target speaker lineage, so it cannot be checked against the reserved LibriTTS/LibriSpeech
  readers.

**Integrity checks (prep fails if any is non-zero):**

- duplicate recording ids;
- repeated originals within a family;
- any recording parent, split group, or genuine speaker present in more than one split;
- test fake parents in train/dev;
- test/calibration genuine speakers in train/dev;
- calibration/test genuine speaker overlap;
- stock voices of the test half seen in train/dev.

**Prepared data.**

- Prep is staged in a run directory.
- `PREPARED.json` is written only after every shard's row count, row→audio references and
  sha256 are verified. It is then swapped in atomically, and a previous set is moved aside.
- Train, select and finish re-verify it.
- Checkpoints carry the dataset fingerprint and the candidate configuration. Resume with
  anything different fails.

**Minimum coverage (preflight; training and selection refused below these).**

| Split | Fakes per target family (unique parents) | English genuine |
|---|---|---|
| train | 200 | 3,000 recordings / 100 speakers |
| dev | 25 | 200 / 20 |
| cal | — | 500 / 50 |
| test | 100 | 500 / 60 |

- A development metric with any target family below the dev minimum is *insufficient*. It can
  never be selected.
- A held-out test cell below its minimum makes the verdict **INSUFFICIENT**.

**Shortcut audit (advisory, unchanged policy).** Probes:

- original duration and native sample rate, recorded before cropping;
- RMS, roll-off, upper-band energy, tilt, speech ratio and ZCR on the treated crop;
- the joint probe.

Findings are reproduced in the final report.

**Unresolved identities, stated in advance.**

- MLAAD Chatterbox and FishTTS: one unnamed voice each.
- ElevenLabs/Gemini: stock voice names only.
- IndieFake: speaker labels from the publisher folders.
- VCTK and LibriTTS: speaker ids only. Overlap with release v1's ASVspoof 5 (MLS) speakers
  cannot be checked.

**Matched genuine/fake panel.** The en_UK test fakes' parents (M-AILABS en_UK, Elizabeth Klett)
are not on Drive, so no matched test panel exists. Fake detection is therefore measured on the
current familiar-family, unseen-text/voice panel. Genuine FA is measured on independent
corpora, and the claim is limited accordingly.

## Amendment 5 (26 Nov, ~21:00 IST, before any V2-90 data was preprocessed or scored)

Reliability review of the genuine sources (owner: "reliable and modern data; fix V1's mistakes").
Two sources are **not clean genuine** and are removed from every split:

- **LibriTTS-R.** Restored by Miipher, a neural vocoder-based restoration model.
- **GLOBE v2.** "Enhanced" and super-sampled to 44.1 kHz.

Neural processing leaves synthetic traces on "genuine" labels. That would teach the detector
that vocoder artifacts mean *real*. Replacements, unprocessed:

- **LibriTTS (original, `mythicinfinity/libritts`, CC BY 4.0).** Same train-clean-100 / dev /
  test roles as LibriTTS-R had.
- **EdAcc (`edinburghcstr/edacc`).** Edinburgh International Accents of English: conversational
  recordings across ~40 accents. Split by speaker: `sha256("edacc/"+spk)%5` = 0 test, 1
  calibration, else train.

The read-speech-only pool becomes LibriTTS + VCTK. VoxENES (Kaggle, CC BY 4.0) is added for
modern voice conversion (Seed-VC, OpenVoice v2, RVC v2), **training only**. Any fake whose
target or source speaker is a LibriTTS/LibriSpeech test or calibration speaker is excluded,
and so is all VoxENES bona fide audio whose speaker cannot be shown disjoint.
