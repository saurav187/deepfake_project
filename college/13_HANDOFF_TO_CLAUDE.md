# Pipeline v3 handoff for Claude

## English XTTS-only catalog and pack review (25 Nov 2026)

Use the reviewed code on the VM before running the pilot. The old pilot replaced the language list with EER numbers after the first candidate and crashed on the second. Feature caches for the same index remain usable. The local suite passed 39 tests. The status of the live Drive audit was not observed during this review.

- Existing selection, eligible manifest, passing prep audit and catalog bytes remain usable when their hashes match. Confirm the digest of `$D/stage0/inventory.csv` equals the hash in `$D/stage0-ecapa/voice-inventory.sha256`. An old match checked the ECAPA inventory copy, even if the active inventory had changed. Investigate any mismatch before proceeding.
- Audit and catalog separately crop the original member. Catalog verifies the resulting hash. Pack verifies the file against that hash, staging checks each requested shard, and payload verifies the exact indexed range. The model decodes the stored four-second crop and applies the requested view and normalization to both labels. See `stage0.py:29`, `data.py:125`, `data.py:161`, `data.py:192`, `audio.py:133` and `pilot.py:47`.
- The parent intersection only removes rows. Viability and identity checks run after it and at pack time. Training admits only XTTS fakes in this experiment. The default recipe still requires XTTS and Melo. Evaluation requires Chatterbox in every fold. XTTS presence in evaluation is reported separately and is not a required cell. See `stage0.py:40`, `phonemedf.py:173`, `data.py:87`, `pilot.py:87` and `train.py:43`. Voice disjointness applies to inferred IDs.
- A rerun of matching that changes the manifest, or a later failed preparation audit, now invalidates the older training authorization. The stage command checks before copying shards. Teacher requires a passing pilot on the same index and audited languages.
- With one writer, interrupted catalog writes can be retried with the same command. Existing correct files are retained, but the source archives are scanned and crops recomputed. Pack streams one member at a time, holds row metadata in RAM and writes shards under a temporary directory. A hard kill before rename can leave unused temporary data. After rename, a retry compares the published index against the current audited catalog before restoring the audit binding. Stage verifies shard checksums as it copies. Source archive streaming and small Drive files dominate I/O. The matched IDs are 20 ASCII hex characters, so the USTAR member uses one 512-byte header, even at a new shard.
- Only the development archives and inventories enter this route. The MLAAD diagnostic is a separate metadata output. No sealed audio or Colab runtime was accessed in the review.

Set the paths and run only the interrupted or pending command, in order. Stop on a failed audit or pilot.

```bash
D=/content/drive/MyDrive/audio-v3
W="$D/stage0-en-ecapa-xtts"
L=/content/v3-local-en
python3 -m pipeline_v3.stage0 catalog --data "$D" --languages en --voices ecapa --xtts-only
python3 -m pipeline_v3.stage0 pack --data "$D" --languages en --voices ecapa --xtts-only
python3 -m pipeline_v3.cli stage --data "$W" --packed "$W/packed" --local "$L" --splits train select
python3 -m pipeline_v3.cli pilot --data "$W" --local "$L" --encoder "$D/models/xlsr" --small-encoder "$D/models/wavlm-base" --languages en
python3 -m pipeline_v3.cli teacher --data "$W" --local "$L" --encoder "$D/models/xlsr" --languages en
```

<!-- ENGLISH-PILOT-RUN:START -->
## English pilot run — 25 Nov 2026 (Claude, unattended; owner approvals 08:18, ~09:00, ~09:10 IST)

*Live section. MEASURED = observed on Colab/Drive. Start 08:18 IST (02:48 UTC).*

**State (04:20 UTC / 09:50 IST): STOPPED at English match. Failed gate: `stage0.viable()`, "Missing English generator cell in train". Train has XTTS-v2 but no MeloTTS.** Last completed stage: **ECAPA voice labels (full)**. Audit, catalog, pack, pilot and Teacher V1 were **not run**. No threshold was changed. Student deferred. **T4 `v3-gpu` released 04:20 UTC (09:50 IST)** after the ECAPA MeloTTS embeddings were cached (28,539, 70.7 clips/s). **Compute: balance 198.10 → 196.15 CU = 1.95 CU this run** (T4 ~1.5 h). `v3-data` (CPU, 0.08 CU/h) is still up.

**Why train has no MeloTTS (MEASURED):** the Melo built-in-voice cohort rule (`voice_labels.py`) only clusters clips whose max WavLM cosine to every VCTK reference is < 0.65, needs ≥ 50 such clips, then requires cluster cosine ≥ 0.88. Under WavLM-SV the Melo max-reference cosine has p1 0.682, p5 0.705, median 0.837, p95 0.95. Only **0.05% (≈15 clips) fall below 0.65**, so no clustering runs and **all 28,539 Melo labels are `unknown`**. This is the same WavLM score-scale problem as the 0.86 floor, now in the cohort constants (0.65 / 0.80 / 0.88). The owner approved an ECAPA rule for *VCTK target* acceptance only and said to keep WavLM's Melo results, so I did not invent an ECAPA cohort rule.

**English match under the frozen ECAPA rule (MEASURED, counts are [clips, readers, inferred target voices]):**

| Fold | genuine | XTTS-v2 | Chatterbox (held out) |
|---|---|---|---|
| train | 2,282 / 126 | 2,282 / 126 / 4 (p243, p258, p259, p273; p230 absent) | 0 (held out) |
| select | 1,338 / 44 | 657 / 44 / 2 (p228, p233) | 843 / 44 / 2 |
| calibrate | 615 / 33 | 333 / 33 / 1 (p236) | 333 / 33 / 1 |
| threshold | 359 / 19 | 196 / 19 / 1 (p244) | 203 / 19 / 1 |
| validate | 559 / 29 | 300 / 29 / 1 (p254) | 313 / 29 / 1 |

Every reader and voice minimum holds. The only failing cell is MeloTTS in train. ECAPA acceptance (whole corpus): XTTS-v2 15,924/28,539 (55.8%; strict 0.69 floor 30.2%); Chatterbox 23,329/28,539 (81.7%; strict 64.7%). Details: `MyDrive/audio-v3/stage0-ecapa/label-report-full.json`, `voices.csv`; failing match log `MyDrive/audio-v3/stage0-en-ecapa/match-FAILED.log`.

**Drive artifacts:** `MyDrive/audio-v3/stage0-ecapa/` (rule.json, refs-report.json, label-report-{sample,full}.json, descriptive-argmax.json, voices.csv, sv-ecapa.sqlite with all 4 groups complete); `stage0/` (WavLM sv.sqlite complete, voices.csv, hindi_density.json); `stage0-en-ecapa/match-FAILED.log`.

**Resume after the owner's choice:** a new T4 with Drive mounted (`colab new -s v3-gpu --gpu T4`, then `colab drivemount -s v3-gpu` and press Enter after granting). Upload the code (`git archive HEAD pipeline_v3`). Then on the VM, with the Drive tree directly (no symlinks needed):
```bash
cd /content/v3-src; D=/content/drive/MyDrive/audio-v3
python3 -m pipeline_v3.stage0 match   --data $D --languages en --voices ecapa
python3 -m pipeline_v3.stage0 audit   --data $D --languages en --voices ecapa
python3 -m pipeline_v3.stage0 catalog --data $D --languages en --voices ecapa
python3 -m pipeline_v3.stage0 pack    --data $D --languages en --voices ecapa
python3 -m pipeline_v3.cli stage --data $D/stage0-en-ecapa --packed $D/stage0-en-ecapa/packed --local /content/v3-local-en --splits train select
python3 -m pipeline_v3.cli pilot --data $D/stage0-en-ecapa --local /content/v3-local-en --encoder $D/models/xlsr --small-encoder $D/models/wavlm-base --languages en
python3 -m pipeline_v3.cli teacher --data $D/stage0-en-ecapa --local /content/v3-local-en --encoder $D/models/xlsr   # only runs if pilot go
```
(For option (a), first add and freeze the ECAPA Melo cohort rule, then rerun `voice_ecapa label`.)

**Owner decision needed (pick one, then resume):**
- (a) Approve an **ECAPA Melo cohort rule** defined like the target rule, from reference trials: a Melo clip joins the train-only cohort only if the frozen ECAPA rule assigns it **no** VCTK voice (both windows below floor 0.64 or margin < 0.08), with cluster constants re-derived on ECAPA scale. The ECAPA Melo embeddings are already cached, so this is about 5 min of CPU plus the chain (~30–60 min including the pilot).
- (b) Approve an XTTS-only train cell (changes the `viable()` generator requirement).
- (c) Keep as is: English pilot blocked.

**Errors fixed during the run (not gates):**
1. Twice, the WavLM writer died with `sqlite3.OperationalError: database is locked`. Causes: read-only status queries, then my own 5-min online backup holding a read lock past the 5 s busy timeout. Resumed from the last commit each time (no restart from zero). The concurrent backup was removed and runs once at the end.
2. `safe_path` rejected Drive symlinks for the base selection and `inventory.jsonl`. Replaced them with byte-checked local copies.

**Sessions:** `v3-gpu` (T4, created 02:50 UTC; displayed rate 1.32 CU/h including the CPU session). `v3-data` (CPU) is idle apart from checks. The CPU voice writer was **stopped** at a commit at 03:04:23 UTC (6,436 XTTSv2 + 10 references, `integrity_check` ok, sha256 `03c5c8d8…`). The checkpoint was copied byte-identically to the T4 (`/content/w/stage0/sv.sqlite`), with one writer only. Backups to `MyDrive/audio-v3/stage0/sv.sqlite` every 5 min. Overnight watchdog stopped.

**GPU local tree** `/content/w` on `v3-gpu`: PhonemeDF archives staged locally and md5-verified against Zenodo `files.tsv`. Other `development/*` entries and `models` are symlinks to Drive. `stage0/inventory.csv` is a byte copy (sha256 `97cf0396…`).

**WavLM voices (MEASURED): collapse.** GPU end-to-end ~33 clips/s. On 19,773 XTTS-v2 clips, 92.5% were unknown and accepted labels were almost all p236. Train had 6 clips from 5 readers. Cause: reference cosines of up to 0.959 between *different* VCTK references under WavLM-SV (same reference, two views 0.93–0.98), so a margin of ≥0.08 is unreachable. Kept for the record; MeloTTS built-in cohort labels still come from WavLM (`stage0/voices.csv`).

**ECAPA (`speechbrain/spkrec-ecapa-voxceleb` rev `0f99f2d0…`, `MyDrive/audio-v3/models/ecapa-voxceleb`)**, separate cache `stage0-ecapa/sv-ecapa.sqlite`:
- References: cross-reference cosine median 0.12, max 0.56 (p228–p244). Same reference, two views 0.66–0.80 (p230 0.35).
- Sample (1 in 16 by hash, 1,767 + 1,767), measured 11.2–11.7 clips/s end-to-end while sharing the T4 with WavLM. Under the original 0.86 floor: **0 accepted** (best-score median 0.65 / 0.71).
- **Frozen rule `ecapa-reference-floor-v1`** (`stage0-ecapa/rule.json`), set from known-identity reference trials *before* use on the corpus:
  - Method: three non-overlapping thirds of each speech-active reference. One third enrolls all 10 references; the other two are the two query windows. 3 rotations give 30 trials, each against the correct reference plus 9 wrong ones, and again with the correct reference removed.
  - Rule: both windows choose the same voice, margin ≥ 0.08, floor = **0.64** (smallest 0.01 step above the maximum wrong-reference window score, 0.631).
  - Results: genuine accepted **30/30**; correct-removed false accepts **0/30**. Hardest pair p228–p244 (0.598 overall; worst window 0.631).
  - Strict sensitivity floor 0.69 (report only, never used for tuning): genuine 28/30, 0 false accepts.
  - Labels are **INFERRED**. Caveat: ten single recordings, and same-recording matching is easier than matching synthesized speech.
- Early application (snapshot: 12,531 XTTS / 1,767 Chatterbox embedded): XTTS acceptance 55% (strict 30%); Chatterbox 81% (strict 64%). Every fold is filled under both floors (Chatterbox threshold fold has 5 readers at strict). Final counts follow from the full label step.

**Hindi speech-density experiment (MEASURED, `MyDrive/audio-v3/stage0/hindi_density.json`, `pipeline_v3/hindi_density.py`):**
- Caliper |Δ speech_ratio| ≤ δ within each matched pair, on the crop_norm view. δ = **0.08** was chosen on train only (train speech_ratio AUC 0.461), frozen, and applied once to select.
- Before: 354 train / 425 select pairs. speech_ratio 0.724, joint 0.724: FAIL.
- After: 200 train pairs (56%, 8 readers) / 170 select pairs (40%, **4 readers**, 155 XTTS + 15 FreeVC). All controls ≤ 0.60: speech_ratio 0.539, rolloff 0.579, tilt 0.569, edge 0.564, joint **0.597**.
- **Hindi stays PENDING/BLOCKED for this pilot.** Select has 4 readers (< 5), the joint AUC sits at the gate, and the caliper is not implemented in the pack pipeline's own audit.

**English inventory joins (MEASURED):** each system has 28,539 fakes, all with a real parent and a non-empty transcript. 251 PhonemeDF readers, 73 LibriSpeech dev readers, 0 overlap.

**New code (local commits on `pipeline-v3-scratch`, not pushed):**
- `ffb471d` English-only path (`--languages en`, products in `stage0-en*/`; the audit records `languages_audited`; pilot must match) + Hindi experiment
- `cfdd9cb` `--subset` deterministic reader round-robin quotas (unused so far)
- `0511b21`/`87de36c`/`3e152c9` ECAPA path, calibration, frozen-rule labelling (`stage0 --voices ecapa`, products in `stage0-en-ecapa/`)
- 35 tests pass.

**Inspect (no duplicates are started by these):**
```bash
colab --auth=oauth2 ssh -s v3-gpu -- 'tail -20 /content/en_chain.log; tail -2 /content/ecapa_full.log; tail -2 /content/voice.log'
```
<!-- ENGLISH-PILOT-RUN:END -->

<!-- OVERNIGHT-STAGE1:START -->
## Overnight Stage 1 run — 25 Nov 2026 (Claude, unattended; owner approved Option A)

*Live section, updated during the night. MEASURED = observed on the owner's Colab/Drive; PROJECTED = estimate.*

**State at last update (01:00 UTC / 06:30 IST, 25 Nov):** Stage 1 chain healthy on the CPU-only `v3-data` session. **Last completed stage: inventory** (+ `wavlm-sv` download). **Voice labelling RUNNING** — 3,083 XTTSv2 embeddings committed at 00:57 UTC; clean rate ~1,600–1,700 clips/h; **~50 h remain on CPU, so it cannot finish in this session** (recommend finishing on a GPU session; commands under "Next steps"). **match → audit → catalog → pack were NOT run** (they need the finished voice labels). Hindi codec probes v1–v3 done: **Hindi fails the acoustic gate on `speech_ratio` ≈ 0.72–0.73** in every treatment tested; English cannot be audited until voice labels exist. A local watchdog checks every 10 min and resumes the job once if it stops; nothing else is scheduled.

**Code:** branch `pipeline-v3-scratch`, local commits `3cf1eca` (Option A audit) → `0a746ad`/`57b8851`/`2545ee9` (codec probe) → handoff commits; nothing pushed. Remote `/content/v3-src` = bundle `bdfac3a` plus uploaded `audit.py`, `cli.py`, `stage0.py`, `codec_probe.py` from these commits (voice labelling runs the unchanged `bdfac3a` code). Tests: 59 pass locally.

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
  - Codec probe v3 (MEASURED; `stage0/codec_probe_hi_v3.json`): **matched-activity window** (both labels: the 4 s window with the most active frames, unit level) → speech_ratio **0.72**, joint 0.74, all other controls 0.48–0.56 → **FAIL**. Window placement is not the cause; the speech-density difference is intrinsic to the two corpora's audio. Option (a) as a simple cropping fix is therefore ruled out.
  - Whether `speech_ratio` is a nuisance (corpus reading style) or a genuine synthesis cue (TTS pausing) is a research judgement for the owner. Options for the morning: (a) ~~matched-activity crop~~ (tested in v3, does not help); (a2) explicitly balance the Hindi training/selection sample on speech_ratio (stratify pairs so both labels share its distribution) and re-measure — changes which clips are used, not the threshold; (b) add Hindi data whose real and fake share recording style (owner's Hindi recordings + Hindi Gemini/Microsoft fakes of the same texts); (c) keep the pilot English-only and report Hindi as blocked.

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
