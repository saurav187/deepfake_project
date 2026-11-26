# V2-90 handoff — 26/27 Nov 2026 (IST)

Owner target: **≥90% detection on each of Chatterbox, ElevenLabs-v3, FishTTS and Gemini-3.1-Flash-TTS,
≤15% genuine false alarms, one cutoff frozen before testing.** Protocol: `college/28_V2_90_PROTOCOL.md`
(amendments 1–6). Branch `worktree-v2-overnight`.

## Final Held-Out Verdict (Plain Language)

- **Primary clean English verdict: `MISS`** (formal protocol verdict) — missed **solely on the English genuine false-alarm budget**, not on fake detection:
  - **Target modern TTS detection (4/4 met ≥90%):** Chatterbox **232/232 (100.0%)**, ElevenLabs-v3 **118/120 (98.3%)**, FishTTS **283/283 (100.0%)**, Gemini-3.1-Flash-TTS **125/125 (100.0%)** — pooled **758/760 (99.74%)**.
  - **English genuine false alarms:** **158/825 (19.15%)** vs. ≤15.0% limit (`≤123/825`, gap **+35 recordings**). Over **70% of false alarms (112/158)** came from **LibriTTS audiobooks (`112/391 = 28.64%`)**, whereas `vctk` had `0/135 (0.00%)`, `ami` `1/24 (4.17%)`, `indiefake` `14/98 (14.29%)`, and `edacc` `31/177 (17.51%)`.
- **Simulated call codecs (`mulaw8k`, unseen `g726`, `opus12`, `gsm`, same frozen cutoff `0.8830`):** All four returned **`MISS`** on English genuine FA (`15.15%–25.82%`; `gsm` missed by only 2 clips at `125/825 = 15.15%`), while target fake detection stayed **98.3%–100.0%** across every codec — confirming detection does **not** collapse when 8 kHz phone downsampling strips frequencies above 4 kHz.
- **Hindi (secondary, same English-calibrated cutoff):** Genuine FA was **7.47% (`97/1,298`)** on clean (`4.16%–8.71%` across codecs), with **97.3%–100.0%** detection across all 5 MLAAD Hindi TTS systems (`161/162` clean) and **99.3% (`149/150`)** on `IndicSynth/freevc24`.
- **Release v1 comparison (exact same 3,263 test clips, clean):** V2-90 dramatically outperforms Release v1 on modern TTS. At its shipped cutoff (`0.56235`), Release v1 detected **36.2%** Chatterbox, **9.2%** ElevenLabs-v3, **88.3%** FishTTS, and **3.2%** Gemini (`349/760 = 45.9%` pooled) at `4.48%` English FA. Even when independently calibrated on the identical 634 English calibration clips (`cutoff = 0.02785`), Release v1 hit **20.36% (`168/825`) English FA** (worse than V2-90's `19.15%`) and **36.59% (`475/1,298`) Hindi FA** while detecting only **78.0%** Chatterbox, **49.2%** ElevenLabs-v3, **99.3%** FishTTS, and **26.4%** Gemini (`554/760 = 72.9%` pooled).
- **Post-hoc descriptive check (NOT the primary verdict):** Because target fake separation is strong (`pooled AUC = 0.9958`, `EER = 2.72%` on clean), a post-hoc clean cutoff set at **1.94% English FA (`16/825`)** still detects **95.92% (`729/760`)** pooled across the four targets (`Chatterbox 96.6%`, `ElevenLabs-v3 85.8%`, `FishTTS 99.7%`, `Gemini 96.0%`; `Hindi FA 0.62%`). At **4.97% English FA (`41/825`)**, every target exceeds 90% (`91.7%–100.0%`, pooled `98.42%`). Any recalibrated cutoff would require a fresh sealed test set for a formal pass.
- **Claim scope:** Familiar generator families on unseen `en_UK` texts/parents and (for ElevenLabs/Gemini) unseen stock voices; **not** unseen-family generalization and **not** real telephone calls.

---

## Run Identity, Provenance & Artifact Verification

| Item | Verified Value |
|---|---|
| Deployed code commit (`COMMIT`) | `a4a7068` (`pipeline_v3/v2_90.py` SHA-256 `ab92cc837af76cf2…`) |
| Protocol (`college/28_V2_90_PROTOCOL.md`) | SHA-256 `9400c251da06654f…` (amendments 1–6) |
| Prepared dataset fingerprint | `f8eb0fe0eaf42ebe25ce4128086d3619ba349e89434e86640f93d457a3fadcea` |
| Test binding SHA-256 (`test-0000.npy`) | `0ab21d6e0e0051ad…` (`test-0000.npy` SHA-256 `4641fc07ccc62078…`) |
| Primary selection (`selection.json`) | **C4 = `["c2", "c3"]`** equal-weight z-mean (SHA-256 `6168996193cd4fac…`) |
| Member `c2` (`facebook/wav2vec2-xls-r-300m`) | Best step 500/6,000, SHA-256 `3c180a32dab32dcd…` (`mu = -4.6351`, `sd = 1.9971`) |
| Member `c3` (`nii-yamagishilab/xls-r-1b-anti-deepfake`) | Rev `60c96543…` (806 tensors), best step 500/4,500, SHA-256 `3c688d8a3d474111…` (`mu = -3.9402`, `sd = 1.2250`) |
| Frozen cutoff (`cutoff.json`) | **`0.8829656848957964`** (SHA-256 `997d697226d1fdf8…`; `95/634 = 14.98%` cal EN genuine) |
| Sign-off (`signoff.json`) | Signed `2026-11-27 01:26:17 IST`, SHA-256 `c879b05ce98c0f1c…` |
| Evaluation (`result.json`, `eval-scores.jsonl`) | `result.json` SHA-256 `27d609616b5df1b6…`, `eval-scores.jsonl` `85991f2fcd199e2e…` (1.69 MB) |
| Release v1 baseline (`baseline.json`) | `release_sha256` `bdb7748cd609bc7a…`, `baseline.json` SHA-256 `0235faf43fc8ad82…` |
| Binding & Drive parity check | **`bindings_ok = True`** — `result.json` and `baseline.json` both match `signoff_sha256`; all 7 run outputs byte-identical between A100 local and Drive (`MyDrive/audio-v3/v2/v2-90/run/`) |

---

## Verified Dataset Coverage & Split Integrity

All split-integrity and leakage checks in `prep.json` returned **0** (`test_fake_parent_in_train=0`, `test_genuine_speaker_in_train=0`, `cal_genuine_speaker_in_train=0`, `cal_test_genuine_speaker_overlap=0`, `test_stock_voice_in_train=0`, `split_crossing_{parent,group,genuine_speaker}=0`). Preflight passed with zero failures.

| Split | Chatterbox | ElevenLabs-v3 | FishTTS | Gemini-3.1-Flash | English Genuine (rec / spk) | Hindi Genuine (rec / spk) |
|---|---:|---:|---:|---:|---:|---:|
| **Train** | 424 (13 v) | 247 (15 v) | 447 (13 v) | 276 (17 v) | 5,254 / 276 | 2,575 / 317 |
| **Dev** | 49 (12 v) | 26 (11 v) | 54 (13 v) | 31 (16 v) | 600 / 32 | 1,459 / 122 |
| **Cal** | — | — | — | — | 634 / 126 | 618 / 122 |
| **Test** | 232 (2 cl) | 120 (11 v) | 283 (2 cl) | 125 (13 v) | 825 / 150 | 1,298 / 267 |

*(Total held-out test set: **3,263** 4 s treated clips across 18 cells, including `IndieFake` 103, `IndicSynth/freevc24` 150, and 5 MLAAD Hindi TTS systems totaling 162 clips.)*

---

## Development Selection & Calibration (Strictly Separate from Test)

- **Training checkpoints (`dev` clean, 15% dev genuine cutoff):**
  - `c2` (L4, 6,000 steps, `23:53:25–01:16:28 IST`, 83 min): hit `key = 1.0` on dev at step 500 (`pooled = 0.9777`), rising to `0.9989` at steps 2500–5000 and `0.9978` at step 6000. Because `best.pt` updates strictly on `m["key"] > best["key"]`, `best.pt` remained at step 500.
  - `c3` (A100, 4,500 steps, `00:08:18–01:11:47 IST`, 63 min): hit `key = 1.0` at step 500 (`pooled = 0.9565`), peaking at `0.9844` at step 4000 (`0.9833` at step 4500); `best.pt` remained at step 500.
  - Smoke tags (`smoke3`, `smoke3fixed`) were excluded from selection.
- **Prespecified selection (`01:22:17–01:24:25 IST`):** All three options (`c2`, `c3`, `["c2", "c3"]`) achieved `key = 1.0` on dev (`Chatterbox 49/49`, `ElevenLabs-v3 26/26`, `FishTTS 54/54`, `Gemini 31/31`). The equal-weight z-mean ensemble **C4 (`["c2", "c3"]`)** won on the prespecified pooled-detection tie-break: **`0.9822`** vs. `c2` `0.9777` and `c3` `0.9565`.
- **Calibration (`01:24:29–01:25:49 IST`, clean English genuine only):** Cutoff **`0.8829656848957964`** flagged **95/634 (14.98%)** across 126 speakers: `libritts 78/354 (22.0%)`, `edacc 16/103 (15.5%)`, `ami 1/22 (4.5%)`, `vctk 0/142 (0.0%)`, `indiefake 0/13 (0.0%)`.

---

## Held-Out Test Results — Primary Clean & All Simulated Call Codecs

All five channels were scored once on the signed test set (`01:26:17–03:14:49 IST`) under the identical frozen cutoff **`0.8829656848957964`**.

### 1. Primary Clean Channel (`verdict: MISS`, gap: `+35` English genuine FA)

| Cell | Flagged / N | Rate | Wilson 95% CI | Clusters | Cluster Boot 95% CI |
|---|---:|---:|---|---:|---|
| **Chatterbox** *(target)* | 232 / 232 | **100.00%** | [98.37%, 100.0%] | 2 | [100.0%, 100.0%] |
| **ElevenLabs-v3** *(target)* | 118 / 120 | **98.33%** | [94.13%, 99.54%] | 11 | [95.59%, 100.0%] |
| **FishTTS** *(target)* | 283 / 283 | **100.00%** | [98.66%, 100.0%] | 2 | [100.0%, 100.0%] |
| **Gemini-3.1-Flash-TTS** *(target)* | 125 / 125 | **100.00%** | [97.02%, 100.0%] | 13 | [100.0%, 100.0%] |
| **English Genuine (pooled)** *(limit ≤15%)* | **158 / 825** | **19.15%** | [16.61%, 21.98%] | 150 | [14.14%, 24.43%] |
| └ `genuine/libritts` | 112 / 391 | 28.64% | [24.39%, 33.32%] | 71 | [19.60%, 38.21%] |
| └ `genuine/edacc` | 31 / 177 | 17.51% | [12.62%, 23.79%] | 36 | [12.25%, 23.16%] |
| └ `genuine/indiefake` | 14 / 98 | 14.29% | [8.70%, 22.56%] | 11 | [1.94%, 32.22%] |
| └ `genuine/ami` | 1 / 24 | 4.17% | [0.74%, 20.24%] | 12 | [0.00%, 15.00%] |
| └ `genuine/vctk` | 0 / 135 | 0.00% | [0.00%, 2.77%] | 20 | [0.00%, 0.00%] |
| `genuine_read_only` (LibriTTS + VCTK) | 112 / 526 | 21.29% | [18.01%, 24.99%] | 91 | [14.09%, 28.60%] |
| `IndieFake` (fakes, descriptive) | 95 / 103 | 92.23% | [85.42%, 96.01%] | 12 | [81.65%, 100.0%] |
| **Hindi Genuine** *(secondary)* | **97 / 1,298** | **7.47%** | [6.16%, 9.03%] | 267 | [5.09%, 10.17%] |
| └ `genuine/indicvoices` | 89 / 1,226 | 7.26% | [5.94%, 8.85%] | 257 | [4.89%, 9.93%] |
| └ `genuine/kathbath` | 8 / 72 | 11.11% | [5.74%, 20.42%] | 10 | [1.61%, 24.24%] |
| `IndicSynth/freevc24` *(Hindi VC)* | 149 / 150 | 99.33% | [96.32%, 99.88%] | 15 | [98.00%, 100.0%] |
| `hi__IndicF5` | 36 / 37 | 97.30% | [86.18%, 99.52%] | 36 | [91.67%, 100.0%] |
| `hi__OmniVoice` / `Veena` / `VoxCPM2` / `Voxtral` | 125 / 125 | 100.00% | [87.1%–90.4%, 100%] | 2–29 | [100.0%, 100.0%] |

*(Speaker concentration on clean English genuine: 92 of 150 speakers had 0% FA, while 26 speakers had ≥50% FA; 13 of the 15 worst speakers were LibriTTS readers.)*

### 2. All Five Channels Summary (Identical Frozen Cutoff `0.8830`)

| Channel | Verdict (FA gap) | Chatterbox | ElevenLabs-v3 | FishTTS | Gemini | EN Gen FA | HI Gen FA | HI MLAAD (5 sys) | FreeVC24 | IndieFake |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **clean** | MISS (+35) | 100.0% | 98.3% | 100.0% | 100.0% | **19.15%** | 7.47% | 99.4% | 99.3% | 92.2% |
| **mulaw8k** | MISS (+51) | 100.0% | 100.0% | 100.0% | 100.0% | **21.09%** | 7.55% | 100.0% | 99.3% | 92.2% |
| **g726** *(unseen)* | MISS (+80) | 100.0% | 100.0% | 100.0% | 100.0% | **24.61%** | 8.63% | 99.4% | 98.7% | 92.2% |
| **opus12** | MISS (+90) | 100.0% | 99.2% | 100.0% | 100.0% | **25.82%** | 8.71% | 99.4% | 100.0% | 91.3% |
| **gsm** | MISS (+2) | 98.7% | 98.3% | 99.7% | 100.0% | **15.15%** | 4.16% | 99.4% | 97.3% | 87.4% |

---

## Release v1 Baseline Comparison (Exact Same Test Clips, Clean)

Release v1 (`detector/runs/release/v1`, `release_sha256` `bdb7748c…`) scored on the identical 4 s treated crops (`03:14:49–03:17:06 IST`):

| Cell | V2-90 C4 (`cut 0.8830`) | Release v1 Shipped (`cut 0.56235`) | Release v1 Calibrated (`cut 0.02785`) |
|---|---:|---:|---:|
| **Chatterbox** | **232/232 (100.0%)** | 84/232 (36.2%) | 181/232 (78.0%) |
| **ElevenLabs-v3** | **118/120 (98.3%)** | 11/120 (9.2%) | 59/120 (49.2%) |
| **FishTTS** | **283/283 (100.0%)** | 250/283 (88.3%) | 281/283 (99.3%) |
| **Gemini-3.1-Flash-TTS** | **125/125 (100.0%)** | 4/125 (3.2%) | 33/125 (26.4%) |
| **Pooled 4 Targets** | **758/760 (99.7%)** | **349/760 (45.9%)** | **554/760 (72.9%)** |
| **English Genuine FA** | **158/825 (19.15%)** | **37/825 (4.48%)** | **168/825 (20.36%)** |
| └ `libritts` / `vctk` | 28.6% / 0.0% | 0.3% / 0.0% | 5.6% / 1.5% |
| └ `edacc` / `ami` / `indiefake` | 17.5% / 4.2% / 14.3% | 9.6% / 12.5% / 16.3% | 49.2% / 70.8% / 40.8% |
| **Hindi Genuine FA** | **97/1,298 (7.47%)** | **72/1,298 (5.55%)** | **475/1,298 (36.59%)** |
| `IndicSynth/freevc24` | **149/150 (99.3%)** | 106/150 (70.7%) | 136/150 (90.7%) |
| `IndieFake` (fakes) | 95/103 (92.2%) | 97/103 (94.2%) | 101/103 (98.1%) |
| Hindi MLAAD (5 sys) | **161/162 (99.4%)** | 74/162 (45.7%) | 134/162 (82.7%) |

---

## Post-Hoc Descriptive Operating-Point Analysis (Not Primary Verdict)

Computed directly from `eval-scores.jsonl` to understand score separation and lower false-alarm regimes (post-hoc descriptive only; cannot change the frozen `MISS` verdict):

| Channel | Pooled 4-Target AUC / EER | At ~2% EN FA (`16/825`): CB / EL / FT / GM (Pooled) | At ~5% EN FA (`41/825`): CB / EL / FT / GM (Pooled) |
|---|---|---|---|
| **clean** | **0.9958 / 2.72%** | 96.6% / 85.8% / 99.7% / 96.0% (**95.92%**, HI FA 0.62%) | 99.1% / 91.7% / 100% / 100% (**98.42%**, HI FA 2.16%) |
| **mulaw8k** | 0.9939 / 3.66% | 86.6% / 84.2% / 99.3% / 91.2% (**91.71%**, HI FA 0.54%) | 95.3% / 93.3% / 99.7% / 99.2% (**97.24%**, HI FA 1.46%) |
| **g726** | 0.9929 / 4.48% | 84.9% / 82.5% / 98.6% / 93.6% (**91.05%**, HI FA 0.54%) | 92.7% / 92.5% / 99.7% / 98.4% (**96.18%**, HI FA 1.69%) |
| **opus12** | 0.9930 / 3.97% | 87.1% / 81.7% / 98.9% / 89.6% (**91.05%**, HI FA 0.46%) | 97.0% / 89.2% / 99.7% / 97.6% (**96.84%**, HI FA 1.23%) |
| **gsm** | 0.9877 / 5.62% | 75.0% / 79.2% / 97.5% / 87.2% (**86.05%**, HI FA 0.15%) | 87.5% / 90.8% / 98.2% / 95.2% (**93.29%**, HI FA 0.85%) |

---

## Shortcut Audit Findings (Reported Prominently per Protocol §5)

Advisory train→dev simple-feature probes (`audit.json`, `22:54:22 IST`):
- **Red flags (`AUC > 0.80`):**
  - **Chatterbox:** `joint` (`0.8218`)
  - **ElevenLabs-v3:** `tilt` (`0.8215`), `zcr` (`0.8730`), `joint` (`0.8898`)
  - **FishTTS:** `joint` (`0.8126`)
  - **Gemini-3.1-Flash-TTS:** `tilt` (`0.8291`), `zcr` (`0.8178`), `joint` (`0.8823`)
  - **Pooled:** no red flags (`joint = 0.6378`, `native_sample_rate = 0.7313`).
- **Notable 0.60–0.80 cues:** `native_sample_rate` (`0.7086` across all four targets), `speech_ratio` (`0.6863–0.7608`), `upper_band` (`0.7770` ElevenLabs, `0.7921` Gemini). No target family exceeded `0.80` on `native_duration` or `native_sample_rate`.
- **Interpretation:** Acoustic/spectral shortcuts exist in the clean MLAAD vs. multi-corpus genuine comparison; however, detection held at **98.3%–100.0%** (and `AUC 0.9877–0.9939`) across 8 kHz phone codecs (`mulaw8k`, unseen `g726`, `gsm`) and 12 kbps Opus, indicating substantial reliance on cues surviving severe band-limiting and lossy compression.

---

## Limitations & Known Caveats

1. **Familiar-family scope:** All four target families were present in training (on disjoint `en_US` texts/parents and half-A voices); this run does **not** prove generalization to brand-new unseen TTS families.
2. **Unmatched test panel:** `en_UK` genuine parents (M-AILABS `en_UK`, Elizabeth Klett) were not on Drive; genuine FA is measured on independent corpora (LibriTTS, VCTK, AMI, EdAcc, IndieFake).
3. **LibriTTS false-alarm concentration:** Calibration (`22.0%` LibriTTS FA) shifted to `28.64%` on test LibriTTS (112/158 English FAs), causing the overall `19.15%` FA miss. Audiobook recording characteristics appear partially confounded with MLAAD audiobook-style fakes.
4. **Best-checkpoint selection quirk:** Both `c2` and `c3` hit `key = 1.0` on dev at step 500; strict `>` kept step-500 weights rather than later higher-pooled checkpoints (steps 2500–5000).
5. **Low cluster counts on Chatterbox & FishTTS (`cl = 2`):** Each has minimal voice variety in test; `ElevenLabs-v3` (`cl = 11`) and `Gemini` (`cl = 13`) test unseen stock voices.
6. **No real telephone calls & 4 s minimum:** Only simulated `ffmpeg` codecs were evaluated, and clips <4 s after trimming were excluded.
7. **Unresolved cross-corpus IDs:** LibriTTS/VCTK overlap with Release v1's ASVspoof 5 (MLS) speakers cannot be verified.

---

## Compute, Elapsed Time & Session Accounting

- **Prep (`v2-l4`):** `22:16:09–22:54:23 IST` (38 min).
- **C3 smoke (`smoke3fixed`, A100):** `23:02:05–23:09:50 IST` (8 min).
- **C2 training (`v2-l4`, 6,000 steps):** `23:53:25–01:16:28 IST` (83 min). `v2-l4` verified (SHA-256 `3c180a32…`) and **released at 01:26 IST**.
- **C3 training (`v2-a100`, 4,500 steps):** `00:08:18–01:11:47 IST` (63 min).
- **Finish chain (`v2-a100`):** `select` (`01:22:17`), `calibrate` (`01:24:29`), `signoff` (`01:25:51`), `evaluate` (`01:26:17–03:14:49 IST`, 108 min — bottlenecked by serial CPU `ffmpeg`), `baseline` (`03:14:49–03:17:06 IST`, 2.3 min).
- **Compute balance:** `129.26 CU` (post-outage recovery at ~00:03 IST) → `119.15 CU` (01:02 IST) → `113.89 CU` (after releasing L4 at 01:28 IST) → **`96.89 CU`** (03:36 IST, **~32.4 CU** used from recovery through full training, 5-channel evaluation, and Release v1 baseline).
- **Active sessions:** `v2-a100` is **idle (0% GPU / 0 MiB)** and awaiting owner confirmation to release (`8.16 CU/hr`).

## Recommended Next Actions

1. **Release `v2-a100` immediately** (all artifacts verified on Drive and locally).
2. **Fix calibration / LibriTTS domain gap before any V2 release:** Include matched genuine audiobook calibration/training balance or per-domain calibration checks, and consider selecting checkpoints with a combined `(key, pooled, EER)` tie-break rather than `key` alone (both `c2` and `c3` stopped updating `best.pt` at step 500).
3. **Parallelize `ffmpeg` in `v2_90.py` evaluation** (cache channel-processed waveforms once across ensemble members using multiprocessing) to cut 5-channel evaluation from ~108 min down to ~3 min.
4. **Collect a fresh sealed confirmation set** (including real phone/WhatsApp recordings and held-out unseen modern TTS families) before making any deployment or student-distillation promotion claim.
