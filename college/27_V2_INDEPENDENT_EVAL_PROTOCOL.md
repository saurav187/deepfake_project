# V2 independent evaluation protocol — modern TTS 70% at ≤15% FA

Written 26 Nov 2026, 02:04 IST, by `v2-overnight-lead`, before any evaluation-split score
exists. Owner assignment: `college/26_V2_OVERNIGHT_ASSIGNMENT.md`. This document is committed
before scoring. A sign-off manifest (`protocol-signoff.json`) binds it to the frozen candidate
and the evaluation manifest by sha256. Scoring of the evaluation split may start only after that
manifest exists. Nothing in this document may change after evaluation scores are read. Any
deviation is reported as a deviation, never folded into the primary result.

## 1. Claim being tested

On independent English clean recordings, one fixed decision rule detects at least 70% of each
of Chatterbox, ElevenLabs-v3, FishTTS and Gemini-3.1-Flash-TTS **separately**, while flagging at
most 15% of genuine recordings. This is a teacher research milestone. It is not a product
verdict and makes no claim about probabilities, Hindi, voice conversion or real calls.

## 2. Frozen primary candidate

| Item | Value |
|---|---|
| Rule | `score = 0.5·z_release + 0.5·z_teacher`; flag if `score ≥ 0.6925721578327001` |
| z | `(logit(clip(p, 1e-6, 1−1e-6)) − μ) / σ`; release μ −7.700825013856512, σ 2.4700144421708905; teacher μ −0.7111816606022586, σ 1.1244474413254268 |
| p_release | release v1 `fake_probability` (`release.json` sha256 `bdb7748c…ab4fc`, members `e0ade0e6`, `97c6ce44`, `a2b2ed57`) on the fixed 4 s crop + frozen RMS norm (`clip_cache.treat`, fingerprint `fixed4s-rmsnorm-v1`). One crop per recording, no window averaging. |
| p_teacher | exploratory Teacher V1 (sha256 `8b4625fb…e711`) Platt probability on the same crop (`model_view` pair `phonemedf`, `mean_logit`) |
| Candidate file | `v2/modern-tts-70-at-15/frozen-candidate.json` sha256 `796107b8…9f68` |
| Cutoff file | `v2/modern-tts-70-at-15/frozen-candidate-cutoff.json` sha256 `8c237786…a22e` (code commit `8c29f22`) |
| Cutoff rule | Lowest cutoff with ≤ ⌊0.15·841⌋ = 126 of the 841 libri-dev genuine clips (73 LibriSpeech dev-clean/dev-other readers) flagged; ties resolved by `≥`, cutoff = nextafter(127th-highest score, +∞) |

The candidate was chosen on development data (weight 0.5 selected in 6/6 reader rotations). The
cutoff was set on genuine-only calibration data, disjoint by speaker from development and
evaluation. No eligible trained alternative exists before the test opens (see section 9), so
this frozen combination is the **only primary**.

**Known weakness, stated in advance.** The cutoff sits exactly at 126/841 (14.98%) on
calibration, with no margin. On a new genuine population the expected false-alarm rate is
about 15%, and its sampling spread is wide. The development panel at this cutoff gave 107/697
(15.4%). A miss on false alarms is therefore a likely outcome and will be reported as such.

## 3. References (same recordings, reported separately, never selected after the fact)

| Name | Rule | Role |
|---|---|---|
| R1 release v1, calibrated | `p_release ≥ 0.009209248773623443` (same libri-dev 126/841 rule) | Baseline at the independently calibrated cutoff |
| R2 release v1, shipped | `p_release ≥ 0.5623497986447662` (`release.json` `likely_fake_at_or_above`) | Baseline at the shipped cutoff, single 4 s crop view |
| T teacher V1 alone | `p_teacher ≥ 0.561926035165614` (same libri-dev rule) | Constituent |

The shipped release averages all 4 s windows of a recording. R2 here uses the single paired crop
and is labelled as such.

## 4. Data roles

| Role | Content | Status |
|---|---|---|
| Development (inspected repeatedly; never test) | 3-reader MLAAD en_US panel (697 genuine / 701 fakes, readers elliot_miller, judy_bieber, mary_ann), its texts and parents; Teacher V1 validate panel | Fit normaliser and weight |
| Calibration (genuine sets the cutoff) | libri-dev 841 clips, 73 readers. Teacher V1 aggregates were previously seen, so this is calibration material, not test | Done, frozen |
| Calibration fakes (descriptive only) | MLAAD en_US rows outside the development sample, texts and parents; ElevenLabs/Gemini stock-voice half A | Characterises detection; does not move the cutoff |
| **Evaluation — primary** | Fakes: MLAAD en_UK rows (Elizabeth Klett books; never in the stage-0 MLAAD diagnostic, which is en_US only); ElevenLabs/Gemini stock-voice half B. Genuine: RVCBench ground-truth recordings of the 21 LibriTTS **test-clean** speakers | Sealed until sign-off |
| Evaluation — secondary genuine | VCC 2018 genuine (DAPS speakers) | Reported separately; not part of the pass rule |
| Evaluation — proxy | OpenAudio S1-mini (RVCBench) on the test-clean speakers | Fish-family proxy, never counted as MLAAD FishTTS |

Split membership is fixed by `independent_panel.py build` (worktree-independent-panel, commit
`cc002f4` or later with identical manifests). The sign-off binds `manifest-evaluation.csv` by
sha256. Splits are by speaker/voice, text and recording parent, and are fixed before scoring.
Alternate channel encodings of a recording share its row and fold.

## 5. Unit of counting

- **One score per original recording per condition (channel).**
  - Genuine: one row per recording (`parent` unique).
  - Fakes: within a family, one row per MLAAD `original_file`. If a family has several fakes
    of one original, keep the row with the lowest sha256(id) and record the others as
    `duplicate_parent` exclusions.
- **Eligibility.** The fixed 4 s crop must succeed. Shorter recordings are excluded and counted
  per cell. Rows whose decoded PCM exactly matches a release v1 training or dev hash are
  excluded and counted. RVCBench rows are read from the cache, which holds no raw PCM, so their
  exact-overlap status is reported as `unchecked`.
- **Missing scores.** If either constituent score is missing for a row, the row is excluded from
  all rules, so every rule is compared on the identical row set.

## 6. Decision rule (primary, clean channel)

**PASS** when all five hold at the frozen cutoff, on the primary evaluation split, clean channel:

1. Genuine flagged ≤ 0.15 × n_genuine (LibriTTS test-clean genuine).
2. Each of the four MLAAD families has detected ≥ 0.70 × n_family.

Report the outcome as follows:

- **MISS.** Any condition fails. Name every failing cell and its gap in recordings, i.e.
  `ceil(0.70·n) − detected`, or `flagged − floor(0.15·n)`.
- **INSUFFICIENT INDEPENDENT EVIDENCE.**
  - Any family has fewer than 30 eligible independent fakes, or the genuine stratum has fewer
    than 10 speakers or fewer than 100 recordings. Then the point result is still reported.
  - A family entirely missing is reported as **untested**, with the reason.

Point estimates decide the outcome, as the owner defined the target. Uncertainty is reported
next to them, not used to rescue or fail a result:

- 95% Wilson interval per cell.
- 95% speaker-cluster percentile bootstrap (2,000 resamples, seed 20261126) for genuine FA,
  resampling LibriTTS speakers with their recordings.
- Per-family detection resampled by voice where voices are stated. Where voices are unstated
  (Chatterbox, FishTTS), the clustering is unknown and the Wilson interval is labelled
  "voice clustering unknown".

A **pass with a CI that crosses a limit** is reported as "pass at point estimate; not robust".

## 7. Secondary measurements (same cutoff, never used for the decision)

- Opus and simulated phone channels (`pipeline_v3.audio.channel(x, "opus"|"phone")` applied to
  the treated crop, as in teacher_eval), identical cutoff and normaliser.
- VCC 2018 genuine FA. Per-speaker genuine FA. Proxy-Fish detection. Calibration-fake detection
  (descriptive).
- References R1, R2 and T on every cell.

## 8. Known limitations recorded before the result

- **Source confound.** Evaluation genuine (LibriTTS, audiobook) and fakes (MLAAD, from
  M-AILABS en_UK texts) come from different corpora. The matched en_UK genuine parents are not
  on Drive. FA measures LibriTTS reader speech only.
- **Familiar voices.** ElevenLabs/Gemini stock voices in half B also occur in the development
  panel, which fixed the normaliser and weight. Voices are disjoint from calibration only.
  Chatterbox and FishTTS voices are unstated, so voice independence is unverifiable for them.
- **One genuine source parent.** All en_UK MLAAD parents are Elizabeth Klett books. Text and
  parent are disjoint from development and calibration, but the reader of the parent text is
  one person. This matters only for the prompt and content of the fakes.
- **Release v1 speaker overlap.** ASVspoof 5 (MLS-derived) speakers are anonymised. Overlap of
  LibriTTS test-clean speakers with release v1 training cannot be verified here. MLS is built to
  exclude LibriSpeech dev/test speakers; that is the publisher's construction, not a check
  made here.
- **Teacher V1 is exploratory.** It failed the nuisance audit under owner exception
  `exploratory-failed-audit-v1`, so the combination inherits a non-releasable constituent.
- **Small genuine population.** About 21 speakers, so FA is speaker-clustered and its interval is
  wide.

## 9. Training arm (assignment step 3) and the order of operations

The intake reports (2 s and 4 s eligibility) are not final at writing. At 4 s, RVCBench has:

- VCTK train candidates: 11–12 eligible pairs per generator.
- LibriTTS: evaluation/descriptive only.

That is far below any coverage gate. Unless the finished 2 s/4 s and SpeechFake reports show
eligible matched modern TTS coverage that passes the existing coverage and shortcut gates, no
adaptation runs and the frozen combination stays the only primary. If an adaptation does run,
it is development-only and is **not** evaluated on this split under this protocol. That would
need a new sign-off before scoring, and the primary here would not change.

## 10. Execution

1. `independent_panel.py build` produces manifests and `build.json` (owner: independent-panel).
2. `v2_overnight_eval.py signoff` writes `protocol-signoff.json` with a UTC/IST timestamp,
   sha256 of this file, the manifests, the candidate and cutoff files, the code commit and the
   rules. It copies that manifest to Drive `v2/overnight-v2/`.
3. `independent_panel.py score --splits evaluation` (owner: independent-panel, clean).
   `v2_overnight_eval.py channels` (owner: v2-overnight-lead, opus and phone) after step 2.
4. `v2_overnight_eval.py report`: one pass, writes `v2/overnight-v2/result.json` and
   `result.md`. It refuses to run if any bound hash differs from the sign-off.
