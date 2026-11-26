# V2-90 handoff — 26/27 Nov 2026 (IST)

Owner target: **≥90% detection on each of Chatterbox, ElevenLabs-v3, FishTTS and Gemini-3.1-Flash-TTS,
≤15% genuine false alarms, one cutoff frozen before testing.** Protocol: `college/28_V2_90_PROTOCOL.md`
(amendments 1–6). Branch `worktree-v2-overnight` (local commits; pushed only when the owner asks).

## Status

_Updated as the run progresses; every number below is from real data unless marked._

| Item | State |
|---|---|
| Codex review of `26ff20f` | All 5 findings confirmed; fixed in `e9a0229` (pre-training) and `e538fd1` (evaluation) |
| Tests | 110 collected tests passed locally under Python 3.14 after the C3 loader fix |
| Local smoke (synthetic audio, RTX 5060) | prep → publish → preflight refuses tiny set → train interrupted at step 2 → resumed at 2 → resume with different LR refused |
| Prep on real data (L4) | restarted at 22:16 IST on `6c3c4c9`, published at 22:52, coverage passed and audit completed at 22:54 |
| GPU smoke on real data | corrected C3 run passed at 23:09:50 IST, resumed at step 20 and scored dev through step 60 |
| Full training | C2 started at 23:53 IST on L4. C3 started at 00:08 IST on the replacement A100 |
| Held-out test | sealed (no sign-off yet) |

At takeover, the L4 prep on `15d384a` raised `KeyError: 'ami-ihm/FEO070/34'` before publishing a prepared set. Commit `6c3c4c9` uses AMI audio IDs and guards against repeated IDs. It is deployed on both GPUs. The local regression in `4c92800` validates reused row IDs and shared microphone parents. The idle download CPU was released. The A100 smoke queue logs to `/content/codex-v290-smoke.log`.

### Prepared set and advisory audit

- Prepared fingerprint `f8eb0fe0eaf42ebe25ce4128086d3619ba349e89434e86640f93d457a3fadcea`. All split integrity checks returned zero. Preflight returned no failures.
- Train and development unique parents by target were Chatterbox 424/49, ElevenLabs v3 247/26, FishTTS 447/54, and Gemini 276/31. English genuine coverage was 5,254 recordings from 276 speakers in train, 600 from 32 in development, and 634 from 126 in calibration.
- Joint audit AUC flagged Chatterbox (0.8218), ElevenLabs v3 (0.8898), FishTTS (0.8126), and Gemini (0.8823). ElevenLabs v3 also flagged spectral tilt (0.8215) and zero-crossing rate (0.8730). Gemini flagged spectral tilt (0.8291) and zero-crossing rate (0.8178). Original duration and native sample rate had no target-family flags above 0.80.
- The first smoke launcher inherited a working directory replaced during redeployment and failed before training. The relaunched smoke job opened the current code directory after verification. It uses the same gate, prepared set and candidate configuration.

### C3 checkpoint recovery and GPU smoke

The original smoke run loaded the published C3 Fairseq checkpoint directly into Transformers and left encoder tensors uninitialized. Development produced a NaN cutoff. Commit `a4a7068` maps the pinned upstream weights to the Transformers encoder, loads them strictly, and rejects non-finite training losses and scores. The invalid `smoke3` checkpoint is retained for diagnosis and excluded from candidate selection.

The corrected `smoke3fixed` loaded all 806 encoder tensors, checkpointed at step 20, resumed at step 20, and completed development scoring at steps 40 and 60 without an OOM. Each check scored all four target families. The Drive log is `run/train-smoke3fixed.log`. The resulting development values are smoke evidence. No held-out audio was scored.

After the workstation outage, the L4 still had the verified prepared set and deployed `a4a7068`. Colab pruned the earlier A100 assignment. C2 started with the planned batch of 32 and a 6,000-step schedule. Compute balance at recovery was 129.26 CU.

Colab returned Service Unavailable for the initial A100 allocation attempts. A single retry loop secured the replacement A100-SXM4-40GB at 00:02:59 IST and exited. The owner mounted Drive. The replacement used the verified `a4a7068` bundle and matching prepared fingerprint. C3 started with batch 16 and a 4,500-step schedule, chosen before launch to allow time for evaluation after the outage. Its first 100 training steps took 36 s, excluding model loading and development scoring. C2 had a durable step-500 checkpoint and reached step 950 during the recovery.

## Review findings → fixes

1. **Incomplete dev/train sets.**
   - Fix: `dev_metric` is insufficient (key −1) unless all four families have ≥25 dev recordings
     and English genuine exists. `select` refuses insufficient options.
   - `preflight` enforces the minimums before training.
   - Test cells below minimum make the verdict INSUFFICIENT.
   - Tests: `test_dev_metric_one_family_is_insufficient`, `test_dev_metric_all_families_sufficient`.
2. **Prep and resume.**
   - Prep is staged in `stage-<UTC>`, verified (row counts, row→audio refs, sha256) and gets
     `PREPARED.json` last. It is swapped in atomically; an old set is moved aside.
   - Train, select and finish call `verify_prepared`.
   - Checkpoints carry `cfg` (dataset fingerprint, encoder, freeze, batch, LRs, steps). A
     mismatched resume fails. `best.json` binds `best.pt` sha to its cfg.
   - Tests: `test_prepared_manifest_detects_tampering`, `test_unpublished_prep_is_rejected`,
     `test_changed_shard_index_is_rejected`.
3. **Lineage and splits.**
   - AMI is split by meeting∪speaker connected component, with IHM/SDM siblings deduped.
   - EdAcc is split by conversation, and VCTK deduped by (speaker, text_id).
   - One MLAAD fake per (family, original).
   - Duplicate ids, repeated originals, and any parent, group or genuine speaker in more than
     one split make prep fail.
   - VoxENES is excluded (no speaker lineage).
   - Test: `test_duplicate_recording_and_split_crossing_rejected`.
4. **Evaluation binding.**
   - Sign-off hashes every test field incl. shard and idx, plus the bytes of each referenced
     shard, the prepared fingerprint, selection, member weights, cutoff, code and protocol.
   - `check_signoff` re-verifies before scoring.
   - Results and scores are written atomically; a completed evaluation is preserved on retry.
   - Tests: `test_signoff_rejects_changed_shard_bytes`,
     `test_signoff_binding_changes_with_shard_index`, `test_duplicate_test_recording_rejected`,
     `test_repeated_evaluation_is_preserved`.
5. **Audit and baseline.**
   - Native duration and sample rate are recorded before cropping and probed train→dev.
   - Audit red flags are copied into `result.json`.
   - `baseline` scores release v1 on the same test rows at the shipped cutoff (0.5623) and at a
     cutoff set by the same calibration rule, with FA by genuine source.
   - Matched genuine/fake test panel: **unavailable**. The en_UK parents (M-AILABS en_UK,
     Elizabeth Klett) are not on Drive, so the claim is limited to the familiar-family,
     unseen-text/voice panel.

## Still unverified on real data

- Eligible counts and coverage preflight (prep running).
- Shortcut audit results.
- GPU resume on real data.
- Release v1 baseline (needs sign-off first).
