# modern-tts-70-at-15 — release v1 × Teacher V1 score combination

Owner target: at least 70% detection on each of clean Chatterbox, FishTTS, ElevenLabs-v3 and
Gemini-3.1-Flash-TTS at one frozen cutoff, with at most 15% of genuine clips flagged.

**Status (26 Nov 2026, 02:05 IST): the target is not certified.** The frozen combination meets
70% on every family on the development panel. With a cutoff set off-panel it flags 15.4% of
genuine clips (107/697), against an allowance of 104/697. Independent evaluation is being run by
`independent-panel-1` (job 435e59bf) at the frozen cutoff. Its result is reported below when it
arrives, and is otherwise **unmeasured**.

## Where things are

- **Code.** Branch `worktree-modern-tts-70-at-15`:
  - `pipeline_v3/modern_tts_70_at_15.py`: overlap, crossfit and offpanel phases.
  - `pipeline_v3/modern_tts_70_at_15_libri.py`: release v1 scoring of libri-dev crops.
  - `pipeline_v3/modern_tts_70_at_15_freeze.py`: candidate freeze, then cutoff freeze.
  - Each protocol was committed before it was run (commits in git log).
- **Outputs.** `MyDrive/audio-v3/v2/modern-tts-70-at-15/`, with `SHA256SUMS` verified on Drive.
- **Baseline left intact.** `v2/day-target-1/` and `fallback-detector-v1/` were only read.
- **Not touched.** V2-L stays paused. No A100 or new VM was allocated.
  - v3-t4p was used for read-only file pulls and one coordinated upload.
  - release v1 scoring of the libri-dev crops ran on the local RTX 5060.

## Models and provenance

| Model | Identity | Training | Score used |
|---|---|---|---|
| release v1 (fallback) | `detector/runs/release/v1/release.json` sha256 `bdb7748c…`, members `e0ade0e6`, `97c6ce44`, `a2b2ed57`, wav2vec2-xls-r-300m | ASVspoof 5 train (bona fide + A01–A08) and IndieFake train. No MLAAD/M-AILABS or panel generators; exact-PCM overlap 0/1,398. Speaker overlap cannot be checked (ASVspoof 5 anonymised) | `fake_probability`, "paired" view (fixed 4 s crop + frozen `norm`), clean |
| Teacher V1 (exploratory) | teacher sha256 `8b4625fb…`, scores CSV `240f22bf…` | Exploratory, failed nuisance audit (owner exception `exploratory-failed-audit-v1`). Not releasable | Platt p on the same crop, clean |

Inputs that were read: `fallback-detector-v1/result.json` `446ec99a…` and `inputs.json` `ebec2ef2…`.

## Data roles

| Role | Data | Genuine | Fakes | Readers / speakers |
|---|---|---|---|---|
| Development (fit; repeatedly inspected) | 3-reader MLAAD panel, clean | 697 | Chatterbox 166, ElevenLabs-v3 181, FishTTS 173, Gemini 181 | 3 M-AILABS readers (elliot_miller 511 clips, judy_bieber 455, mary_ann 432) |
| Calibration (genuine only) | libri-dev panel, LibriSpeech dev-clean + dev-other | 841 | — | 73 speakers, none shared with MLAAD |
| Independent evaluation | `v2/independent-panel-1` (job 435e59bf): unused en_UK MLAAD fakes with voices disjoint from the cal/dev voices, plus fresh genuine | pending | pending | pending |

## 1. Score overlap (descriptive, in-sample on the dev panel)

Each model is at its own in-sample 15% cutoff, i.e. 104/697 genuine flagged. The release v1
numbers reproduce `fa15-descriptive.json`: 150/166, 173/173, 105/181, 99/181.

| Family | n | 70% needs | Release catches | Release gap | Teacher-only catches | Caught by both | Caught by neither |
|---|---|---|---|---|---|---|---|
| Chatterbox | 166 | 117 | 150 | 0 | 3 | 22 | 13 |
| ElevenLabs-v3 | 181 | 127 | 105 | **22** | **48** (judy 22, elliot 20, mary_ann 6) | 46 | 28 |
| FishTTS | 173 | 122 | 173 | 0 | 0 | 13 | 0 |
| Gemini | 181 | 127 | 99 | **28** | **58** (elliot 22, mary_ann 19, judy 17) | 59 | 24 |
| Genuine (false alarms) | 697 | — | 104 | — | **+87 additional** | 17 | — |

- **The two models are complementary.** Their ranks are uncorrelated or negatively correlated:
  Spearman −0.28 on ElevenLabs, −0.19 on Gemini, −0.06 on genuine.
- **The teacher's extra catches more than cover both gaps.** The cost is that its false alarms
  barely overlap the release's (17 shared). The OR rule at both 15% cutoffs gives 191/697
  (27.4%) false alarms, with detection of 153, 153, 173 and 157.
- **In-sample sweep, not a result.** An OR rule with each model at 7.9% false alarms gives
  103/697, with 145/166, 133/181, 173/173 and 133/181. This only shows the combination is worth
  testing.

## 2. Prespecified combination, reader-disjoint rotation (dev panel)

- **Rule.**
  - Each model's score is z = (logit p − μ) / σ, with μ and σ taken from the fit reader's genuine.
  - The combined score is w·z_release + (1−w)·z_teacher.
  - w comes from {0, .25, .5, .75, 1}, chosen on the fit reader to maximise worst-family
    detection at its own 15% cutoff.
  - The 15% cutoff is set on a second reader and evaluated on the third.
  - All 6 orderings were run. w = 0.5 was selected in all 6.

| Fit → cal → eval | Genuine FA | Chatterbox | ElevenLabs | Fish | Gemini | Pass |
|---|---|---|---|---|---|---|
| elliot → judy → mary_ann | 11/215 | 53/58 | 33/43 | 50/50 | 58/66 | yes |
| elliot → mary_ann → judy | 80/227 | 47/51 | 64/68 | 52/52 | 55/57 | no (FA) |
| judy → elliot → mary_ann | 0/215 | 43/58 | 22/43 | 49/50 | 45/66 | no (det.) |
| judy → mary_ann → elliot | 136/255 | 54/57 | 66/70 | 71/71 | 54/58 | no (FA) |
| mary_ann → elliot → judy | 6/227 | 36/51 | 47/68 | 51/52 | 38/57 | no (det.) |
| mary_ann → judy → elliot | 100/255 | 51/57 | 61/70 | 70/71 | 48/58 | no (FA) |

Pooled over the rotations, each clip is counted twice: 333/1,394 genuine false alarms (23.9%).
Detection is 284/332, 293/362, 343/346 and 298/362. Under the same protocol, release alone and
teacher alone each pass 1 of 6 and 0 of 6 rotations.

**Finding.** Detection is not the limiting factor. The genuine score level shifts by reader, so a
cutoff calibrated on one M-AILABS reader does not transfer to another: realised false alarms
range from 0% to 53%. Three readers cannot set a reliable 15% cutoff.

## 3. Frozen candidate, off-panel cutoff (development evaluation)

- **Normaliser.** Frozen in `frozen-candidate.json` (sha256 `796107b8…`), fitted on the 697
  dev-panel genuine clips:
  - release: μ −7.7008, σ 2.4700
  - teacher: μ −0.7112, σ 1.1244
- **Score.** w = 0.5. A clip is flagged when 0.5·z_release + 0.5·z_teacher ≥ cutoff.
- **Cutoff.** Frozen in `frozen-candidate-cutoff.json` (sha256 `8c237786…`) at **0.69257**. It is
  the lowest cutoff that flags at most 126 of the 841 libri-dev genuine clips (73 speakers). The
  libri-dev release scores are in `libri-release.json` (`056aac71…`).

The cutoff was applied unchanged to the MLAAD dev panel:

| Model (cutoff from libri-dev) | Genuine FA | Chatterbox | ElevenLabs-v3 | FishTTS | Gemini | Pooled fakes |
|---|---|---|---|---|---|---|
| **Combo z-mean** | **107/697 (15.4%, 95% CI 12.9–18.2)** | 144/166 (86.7%) | 141/181 (77.9%) | 171/173 (98.8%) | 150/181 (82.9%) | 606/701 (86.4%) |
| Release v1 alone (p ≥ 0.00921) | 90/697 (12.9%) | 147/166 | 98/181 (54.1%) | 173/173 | 90/181 (49.7%) | 508/701 |
| Teacher V1 alone (p ≥ 0.5619) | 141/697 (20.2%) | 39/166 | 107/181 | 17/173 | 123/181 | 286/701 |

- **False alarms by reader (combo).** elliot_miller 77/255 (30.2%), judy_bieber 26/227,
  mary_ann 4/215. The shift between readers seen in section 2 is still there.
- **Status.** This is a development measurement: the panel was inspected and its genuine set the
  normaliser. It is **not** a pass. It misses the 15% limit by 3 clips, and the confidence
  interval spans both sides of the limit.

## 4. Independent evaluation

The cutoff and candidate were frozen before any independent evaluation scoring. The full spec
was sent to `independent-panel-1`. It reports the combo and both single models at their frozen
libri-dev cutoffs.

_Result: pending (see the update log below)._

## 5. Modern-data training comparison

See the update log. It depends on the `modern-data-intake` 2 s vs 4 s SpeechFake eligibility
result from session 8b8b1ec6.

## Update log

- 01:37 IST: started. Score files pulled read-only from v3-t4p.
- 01:45: overlap measured. Combination protocol committed, then cross-fit run.
- 01:50–01:58: libri-dev crops extracted on v3-t4p (CPU, read-only); release v1 scored locally.
  Off-panel phase committed, then run. Candidate and cutoff frozen. Outputs published to Drive
  (coordinated with the v3-t4p operator).
