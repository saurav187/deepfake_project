# Internal project audit: deepfake audio detector state as of 26 Nov 2026, ~18:30 IST

Scope: read-only audit of local repo, worker branches (via `git show`), local result copies and job timelines. All times IST (timeline timestamps are UTC; +5:30 applied). "Measured" = number read from a result file or recomputed here; "claim" = a statement in a handoff/timeline not re-verified. Source links are local file paths. Abbreviations: REPO = `/home/saurav/deepfake_project`, WT = `REPO/.claude/worktrees/v2-overnight`, JOBS = `/home/saurav/.claude/jobs`.

New owner goal (from JOBS/8820e109/timeline.jsonl, 18:09 IST 26 Nov): >=90% detection separately on Chatterbox, ElevenLabs v3, FishTTS, Gemini TTS at <=15% genuine FA. The previous target, which is what all existing artifacts were built for, was 70% at <=15% FA ([26_V2_OVERNIGHT_ASSIGNMENT.md](file://WT/college/26_V2_OVERNIGHT_ASSIGNMENT.md)).

## Q1. What are the models (release v1, Teacher V1, day-target probes, V2-L/V2-A)?

### Takeaway
Release v1 is a frozen XLS-R-300M encoder with 3 small trained heads, trained on ASVspoof 5 + IndieFake, with none of the four target generators in training. Teacher V1 is XLS-R with its last 2 layers fine-tuned on PhonemeDF XTTS-v2 vs LibriSpeech. It runs under a failed-audit exception and cannot be released. Nothing has been trained on modern TTS. V2-L is paused at a cache checkpoint. V2-A is blocked on A08 metadata.

### Cited Findings
- **Release v1 identity:** `release.json` sha256 `bdb7748cd609bc7a30b86ab83dceb3c3b53c80d710826fa8c6d50d2a3ecab4fc` (recomputed here). It was created 2026-11-23 02:30 UTC. Encoder `wav2vec2-xls-r-300m`. Members: seeds 42/43/44, best epochs 3/2/10, checkpoints `e0ade0e6…`, `97c6ce44…`, `a2b2ed57…`. — [release.json](file://REPO/detector/runs/release/v1/release.json)
- **Release v1 architecture:** frozen SSL encoder (`requires_grad_(False)`), a learned softmax mix over all hidden layers, a projection, attentive-stats pooling, and an MLP head. — [detector/model/ssl.py](file://REPO/detector/model/ssl.py)
- **Release v1 training recipe:** argparse defaults are 12 epochs, 600 steps/epoch, batch 32, lr 1e-3 and patience 4. The epoch with the lowest mean per-source dev EER (clean + MP3 32k) is kept. — [detector/train/train.py](file://REPO/detector/train/train.py)
- **Release v1 data and augmentation:** ASVspoof 5 train (both tars, 72,673 clips) plus IndieFake train, with codec/noise/gain/edge-silence augmentation. The recording score is the mean logit over heads and every 4 s window. — [detector/RESULTS.md](file://REPO/detector/RESULTS.md)
- **Release v1 calibration and thresholds:** Platt a=0.688, b=1.760 on clean + mp3_32k dev. Thresholds: likely_real < 0.5145, likely_fake >= 0.5623, from a 5% FA / 5% miss budget. Min speech 1 s. — [release.json](file://REPO/detector/runs/release/v1/release.json)
- **Release v1 dev results:** pooled source-balanced AUC 0.990, EER 5.4% (n=8,800). ASVspoof5 clean EER 0.34%. IndieFake clean EER 9.25%. — [release.json](file://REPO/detector/runs/release/v1/release.json)
- **Release v1 provenance vs the target families:** no MLAAD/M-AILABS or panel generators in training. Exact-PCM overlap with the 1,398-recording dev panel is 0/1,398 (checked against 138,180 train/dev hashes). ASVspoof 5 speakers are anonymised, so speaker overlap is unverifiable. — [26_MODERN_TTS_70_AT_15.md on worktree-modern-tts-70-at-15](file://WT) (`git show worktree-modern-tts-70-at-15:college/26_MODERN_TTS_70_AT_15.md`); [13_HANDOFF on worktree-day-target-1](file://WT)
- **Teacher V1 architecture:** HF `AutoModel` (the XLS-R path `models/xlsr`). All of it is frozen except `encoder.layers[-2:]`. Linear head on [mean, std] of the last hidden state. AdamW lr 1e-5. The source/label-balanced sampler uses 6/14 clean plus opus/mp3/aac/phone/rate/playback/noise/rawboost augmentations. Training stops after 3 stale epochs. — [pipeline_v3/models.py](file://REPO/pipeline_v3/models.py), [pipeline_v3/train.py](file://REPO/pipeline_v3/train.py)
- **Teacher V1 training data:** PhonemeDF English, XTTS-v2 fakes vs genuine LibriSpeech parents with the frozen `norm` treatment. Chatterbox is held out for select. The XTTS-only match had 2,282 real/126 readers and 2,282 XTTS fakes (4 voices) in train. The lead also reports a 10,613-row training selection with no speaker overlap with libri-dev or the MLAAD panel readers (claim). — [13_HANDOFF (day-target-1 branch)](file://WT); [JOBS/8820e109/timeline.jsonl](file://JOBS/8820e109/timeline.jsonl)
- **Why Teacher V1 is "exploratory/failed audit"**
  - The formal nuisance audit of the exact `norm` bytes failed the 0.60 AUC gate: upper_band 0.817, rolloff 0.779, tilt 0.758, joint 0.758, speech_ratio 0.659.
  - PhonemeDF fakes are band-limited and darker than LibriSpeech. Upper_band >4 kHz is 0.0040–0.0041 in select fakes vs 0.0307 in genuine.
  - The owner approved an exploratory run under exception `exploratory-failed-audit-v1`. Student/export/evaluate/sealed steps refuse that run.
  - Source: [13_HANDOFF (day-target-1 branch)](file://WT)
- **Teacher V1 run:** pilot 20:04–20:59 IST 25 Nov. Teacher trained 20:59–21:51 IST on A100. Chatterbox select EER by epoch: 0.463, 0.479, 0.422, 0.288 (best, epoch 3), 0.343. Checkpoint `teacher.best.pt` sha256 `8b4625fb…`, 1.26 GB. Platt on 1,231 calibrate clips. Thresholds 0.197/0.907. — [13_HANDOFF (day-target-1 branch)](file://WT)
- **Day-target-1 probes** (frozen WavLM features, linear heads; `clean` and `clean_opus`)
  - They separated PhonemeDF select perfectly (AUC 1.000, EER 0.000, n=375).
  - On MLAAD they caught 4–23% per family at ~2% FA. They are effectively dead candidates.
  - Source: [13_HANDOFF (day-target-1 branch)](file://WT)
- **V2-L (LibriSeVoc-only, `v2/v2l-lsv-norm-1`), paused at 23:05 IST 25 Nov**
  - Cache state: 1,040 clips, checksum-verified; 40,285 remain (~173 min at 3.88 clips/s, Drive-ZIP-bound).
  - Lineage: 13,201 utterances, 221 readers (all also PhonemeDF readers).
  - Train-only shortcut check: WaveNet 0.655 and DiffWave 0.607 fail the 0.60 gate.
  - Resume command: `bash pipeline_v3/run_v2l.sh prepare`.
  - Source: [13_HANDOFF (day-target-1 branch)](file://WT)
- **V2-A:** blocked. A08 confirmation passed at 0.558, but ASVspoof 5 publishes no transcripts or MLS reader mapping, so 0 A08 rows are admissible. — [13_HANDOFF (day-target-1 branch)](file://WT); [22_V2A_CLAUDE_HANDOFF.md](file://WT/college/22_V2A_CLAUDE_HANDOFF.md)

### Inferences
- No existing model has seen Chatterbox, ElevenLabs v3, FishTTS or Gemini TTS audio in training. Every number below is zero-shot transfer.
- Release v1's architecture (frozen encoder, small head) is cheap to retrain or extend with new data. Its training code exists in `detector/`, but AGENTS.md forbids modifying `detector/`. Retraining would need a copy or an import from `app/`/`pipeline_v3/`.

### Gaps
- Release v1 training wall-clock time and GPU were not found in the files read.
- The exact Teacher V1 train-fold counts for the exploratory `norm` pack (vs the XTTS-only match table) were not re-verified.

## Q2. Results so far per family at ~15% FA on the 3-reader MLAAD dev panel

### Takeaway
The best measured operating point is the frozen release×teacher z-mean combo at the libri-dev-calibrated cutoff: 107/697 (15.4%) FA with 144/166, 141/181, 171/173 and 150/181 detection (86.7/77.9/98.8/82.9%). Only FishTTS exceeds 90%. My in-sample recomputation shows that reaching >=90% on every family on this panel costs 25.5% FA for the combo, 37.0% for release v1 full-recording, and 48.5% for release v1 single-crop. Reader-dependent genuine score shift is the dominant limiter.

### Cited Findings
- **Dev panel composition:** 697 genuine clips from 3 M-AILABS en_US readers (elliot_miller 255, judy_bieber 227, mary_ann 215). Fakes: Chatterbox 166, ElevenLabs-v3 181, FishTTS 173, Gemini-3.1-Flash-TTS 181, out of a 250-per-family sample before the >=4 s crop eligibility filter. The panel has been inspected repeatedly and is development only. — recount from [teacher-eval-clips.csv](file://JOBS/63d812f9/tmp/data/stage0-en-ecapa-xtts-norm-exploratory/runs/teacher-eval-clips.csv); [27 protocol](file://WT/college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md)
- **Release v1, single 4 s crop ("paired" view), shipped cutoff 0.5623, clean** (detected/total)

  | Genuine FA | Chatterbox | ElevenLabs-v3 | FishTTS | Gemini |
  |---|---|---|---|---|
  | 1/697 | 67/166 (40.4%) | 11/181 (6.1%) | 155/173 (89.6%) | 13/181 (7.2%) |

  Full-recording view: FA 0/697, Chatterbox 36.7–40.4%, ElevenLabs 4.4–9.4%, FishTTS 97.7–98.8%, Gemini 2.2–3.9%. Scoring took 57 min on T4, 00:30–01:27 IST 26 Nov. — [13_HANDOFF (day-target-1 branch)](file://WT); [fallback result.json](file://JOBS/63d812f9/tmp/data/v2/day-target-1/fallback-detector-v1/result.json)
- **Each model at its own in-sample 15% cutoff (post hoc, descriptive; 104/697 FA)**

  | Model | Chatterbox | ElevenLabs | FishTTS | Gemini |
  |---|---|---|---|---|
  | Release v1 paired | 150/166 | 105/181 | 173/173 | 99/181 |
  | Teacher V1 | 25/166 | 94/181 | 13/173 | 117/181 |
  | Probe clean_opus | 52/166 | 102/181 | 68/173 | 77/181 |

  — [fa15-descriptive.json](file://JOBS/63d812f9/tmp/data/v2/day-target-1/fa15-descriptive.json)
- **Frozen combo, libri-dev cutoff 0.69257, applied to the dev panel**

  | Model (cutoff) | Genuine FA | Chatterbox | ElevenLabs | FishTTS | Gemini |
  |---|---|---|---|---|---|
  | Combo z-mean | 107/697 (15.4%, CI 12.9–18.2) | 144/166 (86.7%) | 141/181 (77.9%) | 171/173 (98.8%) | 150/181 (82.9%) |
  | Release alone (p >= 0.00921) | 90/697 | 147/166 | 98/181 (54.1%) | 173/173 | 90/181 (49.7%) |
  | Teacher alone (p >= 0.5619) | 141/697 | 39/166 | 107/181 | 17/173 | 123/181 |

  I reproduced the combo FA and per-family counts exactly from local score files. — [26_MODERN_TTS_70_AT_15.md](file://WT) (`git show worktree-modern-tts-70-at-15:…`); recomputed with /tmp/claude-audit/calc.py from local [fallback scores](file://JOBS/63d812f9/tmp/data/v2/day-target-1/fallback-detector-v1/scores) and [teacher-eval-clips.csv](file://JOBS/63d812f9/tmp/data/stage0-en-ecapa-xtts-norm-exploratory/runs/teacher-eval-clips.csv)
- **Combo cutoff definition:** `0.5·z_release + 0.5·z_teacher` with z = (logit p − μ)/σ. Release μ −7.7008, σ 2.4700; teacher μ −0.7112, σ 1.1244, both fitted on the 697 dev genuine. The cutoff is the lowest score that flags <=126/841 libri-dev genuine (73 readers), so it has zero margin. — [27 protocol](file://WT/college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md)
- **New derived numbers (in-sample, dev panel, clean): FA needed for every family to reach >=X%**

  | Model | All families >=70% | All families >=90% |
  |---|---|---|
  | Release paired | 157/697 (22.5%) | 338/697 (48.5%) |
  | Release full-recording | 135/697 (19.4%) | 258/697 (37.0%) |
  | Combo | 59/697 (8.5%) | 178/697 (25.5%) |
  | Teacher | 573/697 (82.2%) | 652/697 (93.5%) |

  Release full-recording at in-sample 104/697: 161/166, 127/181, 173/173, 110/181. This is in-sample and optimistic. — recomputed with /tmp/claude-audit/calc.py from the same local files
- **Complementarity:** the release and teacher ranks are uncorrelated or negative. Spearman −0.28 on ElevenLabs, −0.19 on Gemini, −0.06 on genuine. The teacher alone catches 48 ElevenLabs and 58 Gemini clips that the release misses, but adds 87 false alarms. — [26_MODERN_TTS_70_AT_15.md](file://WT)
- **Reader shift**
  - In 6 reader-disjoint rotations (fit → cal → eval), w=0.5 was chosen 6/6, but only 1/6 rotations passed 70%@15%.
  - Realised FA ranged 0/215 to 136/255 (0–53%). The pooled rotation FA was 333/1,394 (23.9%).
  - At the frozen cutoff, combo FA by reader: elliot_miller 77/255 (30.2%), judy_bieber 26/227, mary_ann 4/215.
  - The combo genuine z-median by reader is +0.283 / −0.028 / −0.351 (recomputed).
  - Source: [26_MODERN_TTS_70_AT_15.md](file://WT); calc.py
- **Secondary channels at the frozen clean cutoff**

  | Channel | Combo FA | Chatterbox | ElevenLabs | FishTTS | Gemini |
  |---|---|---|---|---|---|
  | Opus | 193/697 (27.7%) | 149/166 | 152/181 | 170/173 | 150/181 |
  | Phone | 183/697 (26.3%) | 145/166 | 159/181 | 170/173 | 167/181 |

  — [26_MODERN_TTS_70_AT_15.md](file://WT)
- **Teacher V1 evaluation (own thresholds 0.197/0.907)**
  - Clean "not flagged" rate: ElevenLabs 84%, Gemini 81%, Fish 98%, MLAAD Chatterbox 99%.
  - Validate AUC: XTTS 0.88, Chatterbox 0.71.
  - Skipped <4 s: LibriSpeech dev 619, MLAAD 503.
  - Source: [13_HANDOFF (day-target-1 branch)](file://WT)
- **Independent evaluation:** never run. The protocol (commit `4dfda40`) and script (`c1fccda`, report step untested on mock) exist. No sign-off file and no evaluation score exist. All workers stopped at ~02:08–02:10 IST 26 Nov on usage limits. — [JOBS/8820e109/timeline.jsonl](file://JOBS/8820e109/timeline.jsonl) (messages 02:09 and 18:01 IST)

### Inferences
- The ElevenLabs and Gemini misses are the binding constraint for 90%. At its calibrated cutoff, release v1 alone catches ~50–54% of these two families. The combo catches 78–83%, and that is on the panel used to fit its normaliser.
- Even in-sample, no existing model or combination reaches 90% on all four families at 15% FA. The best is 25.5% FA. A 90% target therefore needs new training signal on these families, or very different genuine calibration, not threshold tuning.
- Chatterbox is a secondary gap for the combo (86.7%). FishTTS is already near-solved by release v1.

### Gaps
- No independent (non-dev-panel) number exists for any model.
- Per-voice breakdown of ElevenLabs/Gemini misses was not computed here.

## Q3. Data: MLAAD per family, M-AILABS genuine, other intake sources, and what was lost

### Takeaway
Locally (metadata) there are exactly 1,000 MLAAD fakes per target family:

| Family | en_US | en_UK |
|---|---|---|
| Chatterbox | 680 | 320 |
| ElevenLabs-v3 | 666 | 334 |
| FishTTS | 655 | 345 |
| Gemini | 666 | 334 |

All en_UK fakes derive from one reader (Elizabeth Klett). ElevenLabs/Gemini en_UK use the same 26/30 stock voices as en_US. Before cropping, the 250/family dev sample consumed part of en_US. Matched en_UK M-AILABS genuine parents are not on Drive. Modern-TTS training-eligible matched pairs from RVCBench are ~9–12 per generator. Part of the intake outputs existed only on the dead intake VM.

### Cited Findings
- **MLAAD metadata copies:** each of Chatterbox.csv, ElevenLabs-v3.csv, FishTTS.csv and Gemini-3.1-Flash-TTS.csv has 1,000 rows and 1,000 unique `original_file`.
- **Reader split, recounted here**

  | Family | mary_ann | elliot_miller | judy_bieber | elizabeth_klett (en_UK) |
  |---|---|---|---|---|
  | Chatterbox | 220 | 232 | 228 | 320 |
  | ElevenLabs-v3 | 180 | 262 | 224 | 334 |
  | FishTTS | 197 | 229 | 229 | 345 |
  | Gemini | 213 | 223 | 230 | 334 |

- **Durations:** median raw duration is 6.35 s (Chatterbox), 7.11 s (ElevenLabs), 7.66 s (Fish) and 8.68 s (Gemini). Clips under 5 s raw: 298, 306, 221 and 148 respectively. The 4 s crop is applied after silence trim, so eligibility is lower than these counts suggest.
- Source: [JOBS/435e59bf/tmp/meta/mlaad/*.csv](file://JOBS/435e59bf/tmp/meta/mlaad)
- **Voices:** ElevenLabs-v3 has 26 stock voices in en_US and the same 26 in en_UK (0 UK-only). Gemini has 30 and 30 (0 UK-only), e.g. Kore, Puck, Charon, Achernar. Chatterbox and FishTTS rows show `-` (voice unstated). — recount from the same CSVs; [27 protocol §8](file://WT/college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md)
- **Dev-panel use and pinned release:** dev used 250/family from en_US (all 701 dev fakes are in that sample). The stage-0 MLAAD diagnostic CSV has 680/666/655/666 en_US rows. MLAAD revision `30c3dec7…` sits under Drive `development/mlaad/fake/en`. — [independent_panel.py header (worktree-independent-panel)](file://WT); [mlaad-diagnostic.csv](file://JOBS/435e59bf/tmp/meta/mlaad-diagnostic.csv); [17_STAGE0_REPORT.md](file://WT/college/17_STAGE0_REPORT.md)
- **Planned independent-evaluation fakes:** 1,333 en_UK clips (sum 320+334+345+334). The planned genuine side is the RVCBench ground truth for the 21 LibriTTS test-clean speakers (327 usable pairs across models), with VCC2018 DAPS as secondary. Matched en_UK genuine parents are "not on Drive". — [27 protocol §4, §8](file://WT/college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md); [JOBS/8b8b1ec6 timeline 01:55 IST](file://JOBS/8b8b1ec6/timeline.jsonl)
- **M-AILABS genuine:** en_US.tgz is on Drive (`mailabs/en_US.tgz` in mlaad-diagnostic). The dev panel uses its 3 readers. — [mlaad-diagnostic.csv](file://JOBS/435e59bf/tmp/meta/mlaad-diagnostic.csv)
- **LibriSpeech:** dev-clean has 2,703 members/40 readers and dev-other 2,864/33. The libri-dev panel is 841 clips (480 dev-clean + 361 dev-other) from 73 readers. It is used as the calibration genuine. — [17_STAGE0_REPORT.md](file://WT/college/17_STAGE0_REPORT.md); recount of teacher-eval-clips.csv
- **LibriTTS train-clean-100:** only the transcripts were kept (33,236 files in `v2/libritts-text/`). — [13_HANDOFF (day-target-1)](file://WT)
- **RVCBench** (Nanboy/RVCBench-Cloned-Speech @ `365897b0…`, CC BY-NC 4.0, research-only)
  - 2,799 records fetched, 738 usable at 4 s.
  - Training-eligible VCTK pairs at 4 s:

    | Generator | Pairs | Speakers | Genuine recordings |
    |---|---|---|---|
    | CosyVoice2 | 12 | 11 | 12 |
    | F5-TTS | 11 | 9 | 11 |
    | OpenVoice V2 | 9 | 8 | 9 |
    | VibeVoice | 10 | 9 | 10 |

    Qwen3-TTS 0. Combined: 11 speakers, 13 genuine recordings.
  - LibriTTS: 82–93 pairs/model across 34–37 speakers, evaluation-only. Fish has no usable pairs on the reserved speakers. OpenAudio S1-mini on test-clean is a Fish proxy only.
  - Source: [JOBS/8b8b1ec6 timeline 00:47 and 01:55 IST](file://JOBS/8b8b1ec6/timeline.jsonl)
- **SpeechFake** (inclusionAI/SPEECHFAKE): a bounded range-read slice of 7,148 objects (GPT-SoVITS + VC_OpenVoice chosen). 93 of the first 3,000 were usable. Later batches were ~15% usable, mostly genuine. Rejects are mostly too short after trimming. — [JOBS/8b8b1ec6 timeline 00:39–01:08 IST](file://JOBS/8b8b1ec6/timeline.jsonl)
- **AMI:** 2 meetings (ES2004a 58, IS1009a 37), 95 segments with both headset and distant-mic crops usable. Genuine FA measurement only. — [JOBS/8b8b1ec6 timeline 02:00 IST](file://JOBS/8b8b1ec6/timeline.jsonl)
- **SCDF:** 10 speakers, diagnostic only. Genuine 162, XTTS-v2 186, OpenVoice v2 192, DDDM-VC 165, F5-TTS 160. Simple acoustic features separate real/fake at AUC 0.70, which fails the 0.60 gate. — [JOBS/8b8b1ec6 timeline 02:00 IST](file://JOBS/8b8b1ec6/timeline.jsonl)
- **VCC2018:** 20,871 converted clips from 12 DAPS speakers (8 source, 4 target), MD5-verified. Copied to Drive `development/vcc2018/`. — [13_HANDOFF (day-target-1)](file://WT)
- **Gated or blocked sources:** RTCFake and SpoofCeleb need owner HF access requests (no token configured). IndicTTS has no speaker/generator columns. — [JOBS/8b8b1ec6 timeline 00:39 IST](file://JOBS/8b8b1ec6/timeline.jsonl)
- **What is on Drive (claim from intake):** 59 files published at `development/modern-intake/v1/` (01:08 IST). These are shards and manifests 000000–000028: 5,799 records, i.e. all of RVCBench plus the first 3,000 SpeechFake objects. sha256 was verified at every hop. — [JOBS/8b8b1ec6 timeline 01:08 IST](file://JOBS/8b8b1ec6/timeline.jsonl)
- **Probably lost with the intake VM** (claim by the lead at 18:01 IST, not verified on Drive)
  - The second closed list (`closed-20261125T203019Z.sha256`, 60 files) was prepared at 02:00 IST. It is apparently not on Drive. It covered:
    - SpeechFake shards 000029–000049, AMI 000050–000051 and SCDF 000052–000055;
    - RVCBench `pairs.jsonl`/`lineage.json` (sha `81cc468c…`) and the SpeechFake and AMI lineage files.
  - Also lost: the 2 s/3 s/4 s trim report (`/content/mi/v1/measure/trim_report.json`) and the 5-fold acoustic check.
  - Both VMs (`v3-t4p`, `intake-cpu`) are gone.
  - The lead says `pairs.jsonl` can be regenerated from pinned metadata with committed `modern_intake.py` in ~15–30 min CPU.
  - Source: [JOBS/8820e109 timeline 18:01 IST](file://JOBS/8820e109/timeline.jsonl); [JOBS/8b8b1ec6 timeline 02:00 IST](file://JOBS/8b8b1ec6/timeline.jsonl)
- **Safe on Drive (claim):** the frozen candidate, the release v1 fallback scores and the published intake clip cache. — [JOBS/8820e109 timeline 18:01 IST](file://JOBS/8820e109/timeline.jsonl)

### Inferences
- **No training pool exists for the four target families.** The only labelled audio of Chatterbox/ElevenLabs/FishTTS/Gemini available is MLAAD: ~1,000 per family, 4 readers' texts (3 US + 1 UK), single-sentence clips.
- **Training on MLAAD needs a reservation.** It means "familiar-family training" under the owner rule. Reserve readers, texts and voices (e.g. train on en_US, test on en_UK), and label familiarity. ElevenLabs/Gemini voices cannot be held out between US and UK, because the voice sets are identical. A voice-disjoint split must be made within the 26/30 voices.
- **Genuine/fake corpus confound is a serious risk for any MLAAD training.** Fakes are MLAAD and genuine is M-AILABS (US only on Drive) or LibriSpeech, so a 0.60 audit could fail on channel or corpus cues, as PhonemeDF did.
- **RVCBench/SpeechFake/SCDF counts are too small or the wrong families** to move the four target families.

### Gaps
- The actual Drive contents were not inspected: whether `development/modern-intake/v1/` has exactly 59 files, and whether any of the second relay landed. These need a Drive listing.
- The count of 4 s-eligible MLAAD fakes per family in en_UK and in the unused en_US remainder was not computed. It needs the trimmed-duration step on audio. Raw durations suggest roughly 70–85% eligibility (inference only).
- Whether MLAAD on Drive is the full English release, including other generators (e.g. older MLAAD TTS families useful as extra training), was not checked.

## Q4. Training and scoring infrastructure, timings, CU, cache and audit gates, Gemini generation

### Takeaway
Measured timings on T4: scoring 1,398 recordings × 3 channels with release v1 took ~57 min, and fitting a frozen-feature probe took 29 min. On A100, a full Teacher V1 fine-tune took 52 min. Colab rates: A100 8.16 CU/h, T4 1.24 CU/h, CPU 0.08 CU/h. Last recorded balance was 148.83 CU at 01:28 IST 26 Nov. Gemini TTS generation code exists (`gemini-3.1-flash-tts-preview`) with a cost dry-run, but no evidence was found that any Gemini/Azure synthesis was executed.

### Cited Findings
- **A100 (`v3-a100`, A100-SXM4-40GB, 12 vCPU):** 8.16 CU/h. 1 h 23 min cost 10.88 CU (195.81 → 184.93). The crop cache of 10,433 crops took 12.8 min. — [13_HANDOFF (day-target-1)](file://WT)
- **Teacher V1 on A100:** pilot 55 min (20:04–20:59), teacher 52 min (20:59–21:51), evaluation (`teacher-eval.json`) 59 min (21:56–22:55). The account used ~31.5 CU from 18:47 to 23:06 IST 25 Nov and ~46 CU that day. — [13_HANDOFF (day-target-1)](file://WT)
- **T4 `v3-t4p`:** 1.24 CU/h. Day-target chain 56 min: fit 29 min (Drive-I/O-bound reading 5,000 WavLM `.npz`) and evaluate 27 min. The release v1 fallback took 57 min for 4,194 trials. Balance went 151.43 → 148.83 CU (2.60 CU) from 23:32 to 01:28 IST. — [13_HANDOFF (day-target-1)](file://WT)
- **Other throughput:** T4 2-vCPU scoring measured 2.7 clips/s and caching ~7 crops/s (1.49 CU over 75 min). Intake CPU ran at 0.08 CU/h, and the balance was 150.26 at ~00:47 IST. — [13_HANDOFF (day-target-1)](file://WT); [JOBS/8b8b1ec6 timeline](file://JOBS/8b8b1ec6/timeline.jsonl)
- **Shared cache `v2/clip-cache/fixed4s-rmsnorm-v1/`:**
  - USTAR shards + JSONL manifests (source, id, locator, preprocessing version, eligibility/reason, sha256, shard, offset).
  - Atomic publish, fingerprinted to the preprocessing code, one writer.
  - The preprocessing is `clip_cache.treat`: the central 64,000 samples (4 s @16 kHz) after a shared edge trim, then the frozen RMS norm `x·0.05/rms(x)`. Spans under 4 s are ineligible.
  - Source: [13_HANDOFF (day-target-1)](file://WT); [17_STAGE0_REPORT.md](file://WT/college/17_STAGE0_REPORT.md); [27 protocol](file://WT/college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md)
- **Audit and shortcut gate**
  - Probes are fitted on train and scored on held-out select.
  - Fields: duration, active-frame ratio, edge silence, RMS, upper-band energy, spectral tilt, 95% rolloff, codec/rate, source corpus, pair, and the joint combination.
  - Fail if AUC > 0.60 in either orientation, undefined, or missing a matched parent. A one-audit rule forbids retrying treatments on select.
  - Source: [17_STAGE0_REPORT.md](file://WT/college/17_STAGE0_REPORT.md); [13_HANDOFF (day-target-1)](file://WT)
- **Gemini/Azure generation**
  - `pipeline_v3/generate.py` supports engines azure, gemini, edge, xtts and chatterbox. Gemini uses `google-genai` with model `gemini-3.1-flash-tts-preview` (stage0_jobs.py, english.py).
  - The dry-run reports pending requests and estimated cost (`seconds*0.0005` USD at $20/M output tokens). The docs estimate 240 × 8 s ≈ $0.96 plus input.
  - Stage-0 plans list 1,104 Azure + 336 Gemini clips. "All synthesis remains an owner operation." "New paid synthesis … require an explicit spending decision."
  - Source: [pipeline_v3/generate.py](file://REPO/pipeline_v3/generate.py); [pipeline_v3/stage0_jobs.py](file://REPO/pipeline_v3/stage0_jobs.py); [17_STAGE0_REPORT.md](file://WT/college/17_STAGE0_REPORT.md); [26 assignment](file://WT/college/26_V2_OVERNIGHT_ASSIGNMENT.md)
- **Local GPU:** release v1 libri-dev scoring ran on the local RTX 5060. V1 weights were copied to `~/deepfake_models/v1-exploratory/`, outside the repo. — [26_MODERN_TTS_70_AT_15.md](file://WT); [13_HANDOFF (day-target-1)](file://WT)

### Inferences
- **Training time is not the bottleneck.** A release-v1-style head on a frozen XLS-R, or a Teacher-style last-2-layer fine-tune on a few thousand 4 s clips, fits in ~1 h on an A100 (~8–10 CU) or a few hours on a T4. Data preparation and Drive I/O (ZIP reads at ~1.8 s/clip/worker) dominated past runs. Copying archives to local disk first was suggested and is unmeasured.
- **Gemini audio** beyond MLAAD would need an owner spending decision. The cost is small (order of dollars per few hundred clips per the stored estimate). ElevenLabs v3 has no generation code path in the repo.

### Gaps
- No record of release v1's training duration.
- The current CU balance after 01:28 IST 26 Nov is unknown.
- No evidence was found of any executed Gemini/Azure generation run or of owner-held API keys.

## Q5. Owner rules and constraints a new plan must respect

### Takeaway
Train on familiar families only with reserved speakers, prompts and versions, labelled familiarity. Freeze candidate and cutoff on separate calibration data before an independent test. Use a one-writer, write-once sign-off. No audio, weights or archives in git. V2-L stays paused. Paid synthesis and new accelerator bookings need explicit owner approval. Keep `deepfake_model/` and `detector/` unmodified.

### Cited Findings
- **Training priorities:** "Give training-eligible matched modern TTS and genuine controls first claim on new training time. Reserve modern generator families, or when deliberately training on familiar families, reserve speakers, prompts and versions and label familiarity." — [23/24/25 owner update](file://WT/college/23_ONE_DAY_DETECTION_HANDOFF.md)
- **Calibration and freezing:** "Choose a deployment candidate's cutoff on separate calibration data and freeze it before testing on independent speakers and prompts." One fixed decision rule. Count each original recording once per condition. Opus and phone are secondary. — [26 assignment](file://WT/college/26_V2_OVERNIGHT_ASSIGNMENT.md)
- **Compute, spending and sessions:** "Keep V2-L paused and use current compute authorizations. New paid synthesis or additional accelerator bookings require an explicit spending decision." Keep release weights, private uploads and sealed evaluations safe. Local-only commits; the lead pushed at the owner's request at 17:54 IST. — [26 assignment](file://WT/college/26_V2_OVERNIGHT_ASSIGNMENT.md); [JOBS/8820e109 timeline](file://JOBS/8820e109/timeline.jsonl)
- **New owner message (18:09 IST):** "If needed we can use more CU and also download more data." This is a broad spending signal. Whether it covers paid TTS synthesis is not explicit. — [JOBS/8820e109 timeline](file://JOBS/8820e109/timeline.jsonl)
- **Repository rules:** app code under `app/`. Do not modify `deepfake_model/` or `detector/`. Use `venv/` (Python 3.14) and do not upgrade pinned packages. Never commit audio, `*.pth`, archives, `*/datasets/` or `detector/runs/`. Never log or persist uploaded audio. — [AGENTS.md](file://WT/AGENTS.md)
- **Sealed sets:** `app_testing_dataset/` (44 files) was fully unsealed by the owner on 25 Nov. A new sealed final test set is needed before V2's final comparison. In-the-Wild and VA were sealed for release v1. — [13_HANDOFF (day-target-1)](file://WT); [detector/RESULTS.md](file://REPO/detector/RESULTS.md)
- **Exception scope:** the exception `exploratory-failed-audit-v1` lets Teacher V1 exist, but student, export, evaluate and sealed steps refuse it. Any combination therefore inherits a non-releasable constituent. — [27 protocol §8](file://WT/college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md)
- **Protocol 27 rules:** pass/miss/insufficient (<30 fakes per family, or <10 genuine speakers / <100 recordings). Wilson and speaker-cluster bootstrap (2,000, seed 20261126). Nothing changes after evaluation scores are read. It was written for 70%. — [27 protocol](file://WT/college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md)

### Inferences
- A 90% plan changes the claim, so it needs a new protocol and sign-off. Protocol 27 explicitly says any adaptation "is not evaluated on this split under this protocol" without a new sign-off.
- If en_UK MLAAD is used as the independent test, MLAAD training must exclude the en_UK texts and parent books. For ElevenLabs/Gemini it must either exclude held-out voices or label "familiar voice".

### Gaps
- Whether the owner's "use more CU / download more data" message authorises paid Gemini/ElevenLabs synthesis is not stated.

## Q6. Reusable code paths: (a) train on new data with the shared cache, (b) score panels, (c) calibrate thresholds

### Takeaway
Reusable pieces exist for intake into the cache, teacher training, frozen-model panel scoring and calibration/report. They are spread across several branches (pipeline-v3-scratch/main checkout, worktree-modern-intake, worktree-independent-panel, worktree-modern-tts-70-at-15, worktree-v2-overnight). Nothing yet wires "MLAAD fakes + genuine → cache → train → calibrate" end to end.

### Cited Findings
- **(a) Training**
  - `pipeline_v3/train.py`: Teacher/Student training on a staged pack, with resume and checkpoint provenance binding.
  - `pipeline_v3/cli.py stage|pilot|teacher` and `run_exploratory.sh`: the catalog → pack → stage → pilot → teacher chain.
  - `pipeline_v3/day_target.py fit`: frozen-feature linear heads.
  - `v2l.py` / `run_v2l.sh` and `v2a.py`: experiments that read the shared cache and own their splits, audit and pack.
  - `pipeline_v3/clip_cache.py`: the cache writer/reader with fingerprint `fixed4s-rmsnorm-v1`.
  - `detector/train/train.py` (release recipe). Read/import only; do not modify.
  - Source: [pipeline_v3/](file://REPO/pipeline_v3); [13_HANDOFF (day-target-1)](file://WT)
- **(a) Intake into the cache, worktree-modern-intake** (commits dd5b64f…047364a)
  - `modern_intake.py`: RVCBench fetch, lineage, diagnostic, publish, and an M-AILABS name crosswalk.
  - `speechfake_slice.py` + `remote_zip.py`: range-reads from remote ZIPs.
  - `measure_trim.py`: 2/3/4 s eligibility.
  - Also `ami_sample.py` and `scdf_slice.py`.
  - Source: `git show worktree-modern-intake` log/tree
- **(b) Panel scoring**
  - `pipeline_v3/teacher_eval.py`: freeze (Platt + thresholds) and evaluate (validate × 8 simulated channels, libri-dev panel, MLAAD panel at 250/family).
  - `pipeline_v3/day_fallback.py` (worktree-day-target-1): release v1 scoring on identical trial IDs with paired (4 s crop) and full views and PCM-overlap checks.
  - `independent_panel.py build|score|report` (worktree-independent-panel, commits cc002f4, 065c95b): manifests from metadata only, frozen scoring, a dev-score reproduction check, and a per-original uniqueness check.
  - `pipeline_v3/score_files.py`: ad-hoc files.
  - Source: [pipeline_v3/](file://REPO/pipeline_v3); independent_panel.py header
- **(c) Calibration and decision**
  - `modern_tts_70_at_15_freeze.py` and `modern_tts_70_at_15.py` (worktree-modern-tts-70-at-15): z-normaliser, weight selection, reader rotation, and the libri-dev "<=floor(0.15n)" cutoff rule with nextafter tie handling.
  - `pipeline_v3/modern_tts_target.py`: FA cost of reaching X% per family.
  - `pipeline_v3/v2_overnight_eval.py signoff|channels|report` (WT, c1fccda): write-once sign-off with sha256 bindings. Report step untested.
  - `pipeline_v3/metrics.py`.
  - Source: [26_MODERN_TTS_70_AT_15.md](file://WT); [27 protocol §10](file://WT/college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md); [JOBS/8820e109 timeline](file://JOBS/8820e109/timeline.jsonl)
- **(d) Generation:** `pipeline_v3/generate.py --jobs <csv> --dry-run|--execute` and `stage0_jobs.py` to plan jobs. — [pipeline_v3/generate.py](file://REPO/pipeline_v3/generate.py)

### Inferences
- The fastest reusable route to a 90% attempt has three parts:
  - Write an MLAAD→cache intake modelled on `modern_intake.py`.
  - Train a release-style head (frozen XLS-R plus a small head) or a Teacher-style fine-tune on reserved-split MLAAD en_US fakes plus genuine (M-AILABS en_US readers and LibriSpeech), with the existing augmentation list.
  - Reuse `independent_panel.py` scoring and the libri-dev cutoff rule.
- The 0.60 audit gate is likely to fail on a MLAAD-vs-M-AILABS/LibriSpeech pairing unless genuine comes from the same M-AILABS parents. Plan an owner decision on gating before compute.

### Gaps
- Not verified whether `train.py` can consume cache shards directly or only staged packs. The V2-L/V2-A runners suggest a cache→pack path exists, but it was not read in detail.
- Test status of the worker-branch code beyond the stated counts (89 local tests at `3906c8b`; 53, 65 and 70 at earlier commits) was not re-run, by design (read-only).
