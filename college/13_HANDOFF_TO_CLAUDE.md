# Pipeline v3 handoff for Claude

<!-- OVERNIGHT-STAGE1:START -->
## Overnight Stage 1 run — 25 Nov 2026 (Claude, unattended; owner approved Option A)

*Live section, updated during the night. MEASURED = observed on the owner's Colab/Drive; PROJECTED = estimate.*

**State at last update (22:50 UTC / 04:20 IST):** Stage 1 chain running on the existing CPU-only `v3-data` session (2 vCPU, 12 GB RAM, 0.08 CU/h, Drive mounted at `/content/drive`). Inventory DONE; `wavlm-sv` present; **voice labelling RUNNING** (XTTSv2 ≥512/28,539 at 22:45 UTC). Codec probe v1 DONE; extended probe v2 (model-view treatments) running at nice 15.

**Code:** branch `pipeline-v3-scratch`; remote bundle = `bdfac3a` + Option A files (`audit.py`, `cli.py`, `stage0.py`, `codec_probe.py` from local commit `3cf1eca` and the uncommitted codec probe). Nothing pushed.

**What was done**
1. Refreshed `/content/v3-src` from `bdfac3a` (copy also at `MyDrive/audio-v3/v3-code.tar`); Colab already satisfies `requirements-colab.txt` (transformers 5.22.1, huggingface-hub 1.33.0, pyarrow 23.0.1).
2. Started ONE stage 1 chain (`/content/stage1_chain.sh`, PID-file guard `/content/start_stage1.sh`): inventory → `wavlm-sv` download if missing → `voice_labels --batch 8` on CPU. Log: `/content/stage1.log`, copied to `MyDrive/audio-v3/stage0/stage1.log`.
3. Option A implemented locally and committed (`3cf1eca`): policy `acoustic-gate-v2-provenance-warning`. Provenance fields (source, pair, archive, path) are reported with their AUC as `provenance_warning` and excluded from blocking and from the joint predictor; every decoded-signal control **and** stored codec/sample rate still block at 0.60 in both directions; matched-parent, reader, voice, text and generator checks unchanged; pilot/teacher/student, catalog and pack reject any other policy (old `source-strict-v1` audits cannot authorize training). 59 tests pass. Known Hindi result recorded: selection has **811 genuine Kathbath** and **1,596 IndicSynth** fakes; **source-only AUC 1.00** (now a warning, not a block).
4. Hindi codec probe (`pipeline_v3/codec_probe.py`): 150 matched pairs per fold (train/select), baseline vs AAC 64 kb/s vs Opus 16 kb/s VoIP, applied identically to both labels. Output: `MyDrive/audio-v3/stage0/codec_probe_hi.json`. *(results below when finished)*
5. **Hindi FreeVC limitation:** the held-family FreeVC24 selection panel has **210 fake clips from one source reader** (10 target IDs). Any Hindi unseen-family number from it describes one reader.

**How to inspect without starting a duplicate**
```bash
colab --auth=oauth2 ssh -s v3-data -- 'cat /content/stage1.log; cat /content/stage1.pid; ps -eo pid,etime,args | grep -E "[p]ipeline_v3"'
```
Restart only through the guard (no-op if running): `colab --auth=oauth2 ssh -s v3-data -- 'bash /content/start_stage1.sh'`. Voice labelling resumes from `MyDrive/audio-v3/stage0/sv.sqlite`.

**Measurements**
- Inventory (MEASURED): 21:57:59 → 22:11:02 UTC (13 min), `MyDrive/audio-v3/stage0/inventory.csv` = 119,734 lines.
- `wavlm-sv` (MEASURED): downloaded 22:11, 386 MB, `MyDrive/audio-v3/models/wavlm-sv/`.
- Voice labelling progress sample (MEASURED): started 22:11:09 UTC; references 10/10; **XTTSv2 1,030/28,539 committed to `stage0/sv.sqlite` at 23:14 UTC** (63 min incl. model load). Log milestones (IST): 512 by ~04:13, 768 by ~04:28, 1,024 by ~04:44. **Overall 980 clips/h**; ~1,000 clips/h while probe v2 shared the 2 vCPUs (it runs at nice 15, yet still contends), with bursts of up to ~5,000 clips/h earlier. Compute: 0.08 CU/h displayed; balance 198.38 CU at 23:14 UTC.
  - **Clean rate (MEASURED, no probe running): 2,058 → 2,570 clips between 00:11:25 and 00:30:45 UTC ≈ 1,600–1,700 clips/h** (one 256-clip commit every ~9 min).
  - **PROJECTED (clean rate): ~83,000 clips remain → ~50 h → does NOT finish before this session's ~24 h limit (~20:30 UTC 25 Nov / 02:00 IST 26 Nov).** Earlier contended projections, kept for the record:
  - PROJECTED finish for the remaining ~84,600 clips (XTTSv2 + ChatterboxTTS + MeloTTS): **~85 h at 1,000/h (would NOT finish in this session)** vs **~17 h at 5,000/h (≈ 16:00–17:00 IST)**. Re-measured after probe v2 ends (see below). Projected cost at 0.08 CU/h: < 2 CU for this session.
  - Colab session limit: `v3-data` was created ~20:30 UTC on 24 Nov; Colab typically recycles a VM after ~24 h, i.e. around **20:30 UTC / 02:00 IST on 26 Nov**. Progress is committed every 256 clips to Drive, so a recycled VM loses at most ~256 clips.
- Codec probe v1 (MEASURED; `MyDrive/audio-v3/stage0/codec_probe_hi.json`): 300 matched Hindi pairs (150 train / 150 select; 291 XTTS-v2 + 9 FreeVC24 fakes; distinct real readers 8 train / 4 select).
  - **Stored payloads verified:** all 300 genuine = **FLAC, 16 kHz, mono** (despite `.m4a` source names); all 300 fakes = **WAV PCM16, 24 kHz, mono, 384 kb/s**. No AAC/M4A payload was found in the sample, so there is no verified historical lossy-codec trace to "match".
  - Acoustic control AUCs (train-fit, select-scored; FAIL if outside 0.40–0.60):

    | Treatment (both labels) | codec | rate | duration | rms | edge | upper_band | speech_ratio | rolloff | tilt | joint | gate |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | baseline (stored) | 1.00 | 1.00 | 0.49 | 0.85 | 0.89 | 0.71 | 0.60 | 0.61 | 0.62 | 1.00 | FAIL |
    | AAC 64 kb/s | 0.50 | 0.50 | 0.50 | 0.84 | 0.87 | 0.71 | 0.61 | 0.61 | 0.62 | 0.94 | FAIL |
    | Opus 16 kb/s VoIP | 0.50 | 0.50 | 0.49 | 0.85 | 0.90 | 0.71 | 0.54 | 0.59 | 0.61 | 0.94 | FAIL |

  - Reading: shared re-encoding removes the container cue but **not** the dominant cues — level (rms), edge silence and upper-band energy. Note the Hindi rows are audited on stored bytes (only PhonemeDF goes through `fixed_crop`), whereas the model input trims and level-normalises; probe v2 measures crop / level-normalised / 7 kHz low-pass / Opus views to separate storage cues from what the model can see.
- Codec probe v2 (MEASURED; `MyDrive/audio-v3/stage0/codec_probe_hi_v2.json`, same 300 pairs; crop rows = 494 because 106 clips have < 4 s of active speech):

    | Treatment (both labels) | rms | edge | upper_band | speech_ratio | rolloff | tilt | joint | gate |
    |---|---|---|---|---|---|---|---|---|
    | crop (PhonemeDF fixed 4 s centre crop) | 0.79 | 0.52 | 0.51 | **0.73** | 0.55 | 0.55 | 0.86 | FAIL |
    | crop + unit level | 0.53 | 0.52 | 0.51 | **0.73** | 0.55 | 0.55 | 0.72 | FAIL |
    | crop + unit level + 7 kHz low-pass | 0.46 | 0.52 | 0.51 | **0.73** | 0.55 | 0.55 | 0.72 | FAIL |
    | crop + unit level + Opus 16 kb/s | 0.52 | 0.50 | 0.51 | **0.74** | 0.54 | 0.55 | 0.73 | FAIL |

  - Reading: the model-view preprocessing (trim + centre crop + level) removes the edge, bandwidth and level cues (the upper-band difference was the silent edges, not the speech spectrum). **One cue remains: `speech_ratio` ≈ 0.73** (share of active 10 ms frames in the 4 s window — pause/rhythm structure of Kathbath reading vs IndicSynth voices). Re-encoding and low-pass do not change it. Under the approved policy, **Hindi fails the acoustic gate** even in the best treatment; thresholds were NOT loosened and Hindi was NOT dropped.
  - Whether `speech_ratio` is a nuisance (corpus reading style) or a genuine synthesis cue (TTS pausing) is a research judgement for the owner. Options for the morning: (a) choose crops by matched speech activity (sample the 4 s window from the most-voiced region of BOTH labels, or match windows on speech_ratio) and re-measure; (b) add Hindi data whose real and fake share recording style (owner's Hindi recordings + Hindi Gemini/Microsoft fakes of the same texts); (c) keep the pilot English-only and report Hindi as blocked.

**Next steps / decisions for the owner (morning)**
1. **Voice labelling speed.** On this 2-vCPU CPU VM it runs at ~1,000 clips/h (MEASURED), so the ~85,600 fakes would take ~3.5 days — longer than the session lives. It resumes from `stage0/sv.sqlite` on Drive. Recommended (needs your OK + an interactive Drive mount, so not done tonight): finish on a GPU session. T4 ≈ 1.2 CU/h (third-party rate); expected well under an hour of GPU, i.e. ~1–2 CU. **The SQLite cache allows one writer: stop the CPU job first.**
   ```bash
   # 1) stop the CPU job cleanly (progress is committed every 256 clips)
   colab --auth=oauth2 ssh -s v3-data -- 'kill $(cat /content/stage1.pid); sleep 5; pkill -f "[p]ipeline_v3.voice_labels"; ps -eo pid,args | grep -c "[v]oice_labels"'
   # 2) GPU session (interactive Drive mount required)
   colab --auth=oauth2 new -s v3-gpu --gpu T4
   colab --auth=oauth2 drivemount -s v3-gpu
   git archive --format=tar --output=/tmp/v3-code.tar pipeline-v3-scratch pipeline_v3 colab college/DOWNLOAD_PLAN.sh
   colab --auth=oauth2 upload -s v3-gpu /tmp/v3-code.tar /content/v3-code.tar
   colab --auth=oauth2 ssh -s v3-gpu -- 'mkdir -p /content/v3-src && tar -xf /content/v3-code.tar -C /content/v3-src && cd /content/v3-src && nohup python3 -m pipeline_v3.voice_labels --data /content/drive/MyDrive/audio-v3 --batch 32 > /content/voice.log 2>&1 &'
   ```
   Then `match → audit → catalog → pack` (commands in `17_STAGE0_REPORT.md`, all with `--data /content/drive/MyDrive/audio-v3`), on the CPU or GPU session.
2. **Hindi gate** (see codec probe v2): Hindi fails on `speech_ratio` ≈ 0.73 even with crop + level normalisation. Decide between matched-activity cropping (code change + re-measure), new Hindi data with shared recording style, or an English-only pilot with Hindi reported as blocked.
3. Untracked owner files left untouched: `college/14_CODEX_ENGLISH_DATA_PROMPT.md`, `college/16_CODEX_STAGE0_PROMPT.md`.
<!-- OVERNIGHT-STAGE1:END -->


**24 November 2026 · Local branch `pipeline-v3-scratch` · Base `detector-v2` at `f1f7e59`**

## Stage 0 handoff · 25 November 2026

Use the [Stage 0 report, exact commands and resource estimates](17_STAGE0_REPORT.md) for the current `v3-data` session. The owner reports Drive is mounted. Refresh the code bundle from this branch first. The earlier LibriSpeech commercial recipe is now an optional Stage 3 add-on.

The complete local listing replay joined 28,539 genuine PhonemeDF recordings and transcripts to all 85,617 synthetic renditions. The 251 PhonemeDF readers share no IDs with the 73 reserved LibriSpeech dev readers. None of the 28,537 distinct normalized transcripts crosses reader folds. Accepted English counts await the resumable target-voice inference job and four-second crop check. Chatterbox is held out for transfer. Unknown voices are excluded; compact Melo built-in clusters form a single training cohort.

**Decision before GPU spending.** The unchanged Hindi selection gives a source-only AUC of 1.00 on 811 genuine and 1,596 fake selection recordings. This exceeds the requested gate of 0.60; bilingual training therefore remains blocked. The English preparation can proceed independently. Resolve the Hindi data composition or explicitly revise the research protocol before spending on a bilingual pilot. Preserve the one-source-reader denominator for the 210 Hindi FreeVC24 selection fakes. Stage 0 ran locally on metadata and test fixtures. All 53 backend and pipeline tests passed (five warnings), and the fetch pins matched files.tsv. There are no cloud voice labels, passed audio audit or trained v3 results from this step.

## English repair after the first Drive run

Use the [English repair report and commands](15_ENGLISH_DATA_FIX.md) before packing. Local replay matched all 16,201 saved metadata rows. Only three US readers have parents on Drive. All 1,333 UK references add one other reader. A family-blocked fallback retained 1,016 English fakes but still left selection and validation empty. Keep MLAAD as a diagnostic panel.

The new pilot uses OpenSLR dev-clean and dev-other (about 0.65 GB total), with a projected 768/192/192/144/144 English pairs in train/select/calibrate/threshold/validate. The plan requires 60 distinct readers, disjoint texts and stock voices, and 1,104 Azure plus 336 Gemini clips. The owner runs acquisition and generation from the updated setup notebook. The planner checks real transcript and member eligibility before making a jobs CSV. No repaired English clips have yet been generated or scored in this work. Pilot transfer holds out Gemini and FreeVC24. Hindi selection contains four real readers overall, but its 210 FreeVC24 fakes have only one source reader. The release recipe admits Gemini training voices and reports its familiar-family status.

The combined backend and pipeline suite passed **47 tests** with five warnings after this repair. Local checks covered the metadata replay, quota planning, fold isolation, a missing-reader failure, paired-parent metrics, and archive-to-catalog conversion. Real LibriSpeech eligibility, provider availability, channel controls and detection accuracy still need the owner-run stages.

### Earlier progress update · 25 November 2026

The owner asked Codex to re-check the prompt and continue. The prompt has no working-tree edits, and its requirements match this handoff. A local adapter improvement now recognizes both Hindi Parquet audio column names (`audio` and `audio_filepath`); regression coverage was added. The v3 test suite passes **14 tests** and Ruff passes for the changed files. PyArrow is not installed in the local venv, so actual Parquet reads remain to be verified in the authorized Colab data setup. The earlier 41-test combined-suite result has not been rerun after this addition. No downloads, synthesis, cloud sessions, training or sealed-set access occurred.

The owner wants a small Hindi/English audio deepfake detector that catches modern TTS and voice cloning on short phone recordings. The revised prompt adds Google Colab CLI 0.8.0 alongside notebooks. The immediate decision is whether a cheap frozen-feature pilot on matched data justifies spending A100 time. There is a runnable implementation and a local diagnosis. Trained v3 weights, modern-family recall and a release decision still require the owner's Colab run. [Owner's prompt](7_CODEX_FULL_RESEARCH_AND_PIPELINE_PROMPT.md) · [Runbook](../pipeline_v3/README.md)

## What the measurements establish

- Recomputed single-head comparisons on **7,984 development recordings** from ASVspoof 5 and IndieFake. The median clean fake logit for 594 familiar IndieFake fakes stored at 24 kHz was **9.30**, above the **3.26** for 16 kHz fakes. Recovering high-rate examples and masking rate in x1 improved clean IndieFake EER from **7.44% to 4.09%** on these rows. A 24 kHz input by itself is a poor explanation for the owner's earlier Gemini misses.
- Ran the **released three-head v1**, on CPU, on **32 development recordings** and their 32 matched 16 kb/s Opus views. In the IndieFake panel, median fake probability on genuine recordings rose from **4.35% to 10.92%** after Opus. The familiar fakes moved from **99.79% to 99.44%**. This was a small paired codec test. Clean vs conversational style and previously unseen synthesis remain unresolved.
- The leading hypothesis is a mismatch in generator families and recording styles. The owner's earlier Gemini observations are historical. Establishing the cause requires fresh matched Gemini and Microsoft development recordings and a comparison of frozen probes, the original head and a handcrafted baseline. All reproduction commands and denominators are in the [diagnosis](8_DIAGNOSIS_AND_PROBLEM.md).
- An **untrained int8 student fixture** occupied **151,661 bytes**. On a desktop Ryzen 5 7600X using two threads, the bytes-to-verdict path took **0.045 s warm median** for synthetic 30 s PCM. This measures architecture cost; trained accuracy, installed runtime footprint, laptop performance and a mobile port remain open. [Design and benchmark conditions](12_PIPELINE_V3_DESIGN.md)

## What is available

| Deliverable | Where to look |
|---|---|
| 54 unique paper entries, full-text findings and comparison table | [Analysis](9_PAPER_ANALYSIS.md) and [CSV](9_paper_analysis.csv). 40 and 40b are distinct. The CSV records selected results and flags unverified fields. |
| Research gaps and evidence | [Gap table](10_RESEARCH_GAPS.md). Paper 34's 24.32% belongs to a cited RawNet2 baseline; VeriVoice reports 99.66% on its own OpenVoice corpus. Paper 15's frozen XLS-R plus a large head achieved 7.10% external ITW EER. |
| Data selection and acquisition | [Dataset and consent plan](11_DATASETS_V3.md). Minimum **35.409 decimal GB**: M-AILABS US 8.017, four English MLAAD families 1.359, Kathbath Hindi 15.976, IndicSynth Hindi 7.122, IndicVoices Hindi 2.935. These are transfer bytes. Effective matched counts must be measured from the authorised files. |
| Training and serving | [Runbook](../pipeline_v3/README.md). Modules cover source adapters, matching, shard packing/staging, shortcut tests, synthesis planning, pilot probes, teacher, distillation, int8 export, calibration, validation and CPU benchmark. |
| Colab and CLI entry points | [CPU setup](../colab/01_setup_and_data.ipynb), [pilot](../colab/02_pilot.ipynb), [teacher](../colab/03_teacher.ipynb), [student/export](../colab/04_student_export.ipynb), [evaluation](../colab/05_evaluate.ipynb), and [shared runner](../colab/run_stage.py). |
| App integration | The opt-in backend `DETECTOR_BACKEND=v3` reads `V3_RELEASE`. It handles uploaded bytes in anonymous Linux memory. The existing `DETECTOR_BACKEND=detector` selects v1. The exporter needs only the student and release metadata at serving time. |

## What to do next

1. **Use a CPU session for acquisition and joins.** Follow the three opening steps in the runbook. Inspect the dry-run transfer budget and obtain the owner's approval before acquisition. After matching, count real/fake clips, distinct source and target speakers, sentence families and generators in each language and fold. Missing reference voice metadata or too few US audiobook readers can empty a fold. Resolve this with additional matched, consented speech.
2. **Run the pilot before the teacher.** Start with 1,000 clips per class. The current decision rule requires held-family EER ≤20%, an advantage of ≥3 percentage points over the cepstral baseline, an Opus penalty ≤5 points, Hindi and English EER each ≤25%, and scoring coverage ≥95%. A passing nuisance audit is also required. Check absolute family counts and speaker intervals before interpreting a pass. [Pilot implementation](../pipeline_v3/pilot.py)
3. **Use new development clips for diagnosis.** Score clean and phone-delivered speech, human and synthetic, on the same texts and matched channels. Evaluate v1 and the new probes on the same cases. Record provider, model revision, locale, voice and acquisition date. The existing private examples are reserved for final evaluation.
4. **If the pilot passes**, run partial XLS-R adaptation, distil, export, and evaluate the deployed int8 model. Record validation false accusations and false reassurances, abstention, teacher/student difference and target-laptop latency before freezing a release. A 2% false-alarm budget needs independent speakers and denominators.

## Verification and ownership

The combined backend and pipeline suite passed all **41 tests** (five warnings). The 12 v3 tests cover source matching, speaker/voice/text/parent leakage, held-family checks, shortcut controls, shard corruption/recovery, in-memory container decoding and the quantized PRD adapter. Ruff and Python compilation passed. Notebook cells were syntax-checked locally. External synthesis, authenticated dataset joins, cloud acceleration and an end-to-end trained export have not been exercised. PyTorch emits deprecation warnings for the dynamic quantization/TorchScript path, which passed the local CPU fixture.

Preserve pre-existing work in `.gitignore`, the x1 changes under `detector/`, `detector/datagen/`, `detector/eval/compare_dev.py`, `google_collab/` and the owner's prompt. The research implementation lives in `pipeline_v3/`; app integration lives in `app/`. Do not write to the frozen pipelines, push the branch, or commit audio, weights, datasets, keys or papers. The owner controls cloud spending, service credentials and voice-cloning consent. All sealed sets, including `app_testing_dataset/`, stay outside development.
