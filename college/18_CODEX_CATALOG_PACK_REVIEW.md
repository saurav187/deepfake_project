# Codex prompt: review the catalog → pack → stage path for the English XTTS-only run

Run on a high-reasoning model from the repository root (`~/deepfake_project`, branch `pipeline-v3-scratch`). Paste everything inside the box.

````text
You are reviewing pipeline v3 (pipeline_v3/; docs college/8–17; handoff college/13_HANDOFF_TO_CLAUDE.md). This is a CODE REVIEW of the Stage 1 → Stage 2 hand-over: catalog → pack → stage → pilot loader. A live Colab run is about to execute this code, so the priority is to find defects that would silently corrupt or leak data, or crash after hours of I/O. Do not redesign the pipeline.

## The run being reviewed (owner-approved, 25 Nov 2026)
- Separately named experiment "English XTTS-only" (owner option B), commit ad13ba8 and later:
  python3 -m pipeline_v3.stage0 {match,audit,catalog,pack} --data $D --languages en --voices ecapa --xtts-only
  Work dir: $D/stage0-en-ecapa-xtts (D = /content/drive/MyDrive/audio-v3, a Google Drive FUSE mount).
- Train: PhonemeDF real + XTTS-v2 only (MeloTTS excluded). Chatterbox TTS is fully held out (select/calibrate/threshold/validate only). The default two-generator recipe (TRAIN_FAMILIES) must remain the default and unchanged.
- Voice labels: $D/stage0-ecapa/voices.csv from the frozen ECAPA rule ecapa-reference-floor-v1 (both windows agree, margin ≥ 0.08, floor 0.64). Voice-disjoint fake folds; reader-, text- and parent-disjoint real folds.
- Match passed. Measured counts (clips / readers / voices):
  train real 2,282/126; train XTTS 2,282/126/4
  select real 1,338/44; XTTS 657/44/2; Chatterbox 843/44/2
  calibrate 615/33; 333/33/1; 333/33/1 · threshold 359/19; 196/19/1; 203/19/1 · validate 559/29; 300/29/1; 313/29/1
- The audit (stage0.audit_selection → prep-audit.json, eligible.csv) runs on the exact training bytes: the PhonemeDF 4 s fixed crop (audio.fixed_crop) applied identically to real and fake. Gate: every acoustic/metadata probe AUC ≤ 0.60 on select.
- Next: catalog writes $D/development/stage0-en-ecapa-xtts/bytes/<id>.audio and catalog.csv; pack writes $D/stage0-en-ecapa-xtts/packed (tar shards + index.json + checksums.json) and audit.json; then on an A100 VM:
  python3 -m pipeline_v3.cli stage --data $W --packed $W/packed --local /content/v3-local-en --splits train select
  python3 -m pipeline_v3.cli pilot --data $W --local /content/v3-local-en --encoder $D/models/xlsr --small-encoder $D/models/wavlm-base --languages en
  and, only if the pilot passes, python3 -m pipeline_v3.cli teacher (same --data/--local/--encoder).

## Files to review (read fully)
pipeline_v3/stage0.py (viable, subset, audit_selection, catalog, load_labels, main incl. --xtts-only and the pack branch), pipeline_v3/data.py (FIELDS, digest, safe_path, check, pack, stage, entries, payload), pipeline_v3/audit.py (POLICY, panels, languages_audited), pipeline_v3/adapters.py (selected_members), pipeline_v3/audio.py (fixed_crop, prepare, decode), pipeline_v3/phonemedf.py (match, counts, write_csv), pipeline_v3/cli.py and pipeline_v3/pilot.py (how the audit/index binding and the languages check are enforced before training), and the tests under pipeline_v3/tests.

## Questions to answer, each with file:line evidence
1. Byte identity. Are the bytes the audit hashed exactly the bytes catalog writes, pack stores and stage/pilot feed to the model? Trace sha256 from transformed() → eligible.csv → catalog.csv → index.json → staged files → payload(). Is any decode/resample/crop applied a second time, or applied to one label only?
2. Binding. Can pack, stage, pilot or teacher run on a selection, eligible set, catalog or index other than the one the passing audit covered (stale files from an earlier run, the -en-ecapa work dir, a re-run of match after audit)? Check every digest comparison, the policy/languages_audited checks, and audit.json's index_sha256. Is the XTTS-only experiment kept from mixing with stage0-en-ecapa or stage0 artifacts (work-dir names, catalog_path = development/<work.name>-catalog.csv, bytes dir)?
3. Split integrity after the join. audit_selection drops rows whose parent lacks a real/fake partner. Can that drop break reader/text/parent/voice disjointness, empty a required cell, or leave Chatterbox in train? Is viable(kept, train_need) re-checked on what is actually packed? Does --xtts-only filtering in main() keep every genuine row whose parent still has an XTTS or Chatterbox fake, and drop only MeloTTS rows and orphaned genuines?
4. Held-out family. Prove no Chatterbox row can reach the train split or the train-time sampler; prove pilot.py's holdout set and the staged splits (train, select only) agree with this experiment.
5. Robustness on Drive FUSE. Atomicity and resume: catalog's tmp → replace per file, pack's TemporaryDirectory + rename, "Choose an unused pack destination" on rerun, partial shards, write_csv/audit.tmp replacement. What happens if the run is killed mid-catalog or mid-pack, and what is the correct resume command? Any O(n²) or memory issue for about 11,000 clips (all payloads in memory? index.json size?)? Is tarfile USTAR safe for these ids (name length ≤ 100)? Is the offset (shard.offset + 512) correct for every member and for the shard-rotation boundary?
6. Leakage and policy. Is there any read of sealed sets (In-the-Wild, VA, VoxENES, Deepfake-Eval, app_testing_dataset/, owner recordings)? Is any threshold, minimum or gate weakened by the --xtts-only change (ad13ba8), e.g. is TRAIN_FAMILIES still the default everywhere and the pilot gate unchanged?
7. Tests. Which of the above are untested? Write the minimal tests that would catch each real defect you find.

## Rules
- Review first. Report findings ranked by severity: blocker (corrupts or leaks data, or fails the live run), major, minor. Give a concrete failure scenario for each; no style nits.
- You may commit fixes with tests on pipeline-v3-scratch only if a finding is a blocker or major. Fixes must not change any threshold, sample minimum, voice rule, fold assignment, the 4 s crop or the audited bytes. For each fix, state whether the already-produced artifacts in stage0-en-ecapa-xtts (selection.csv, eligible.csv, prep-audit.json, catalog bytes) stay valid, or exactly which stage must be re-run.
- Don't start Colab sessions, download data, call paid APIs, or open archives. Never commit audio, data, listings, inventories, keys or papers. Never push. Keep deepfake_model/ and detector/ untouched.
- All tests must pass: venv/bin/python -m pytest -q pipeline_v3/tests

## Final reply (short)
- findings by severity, each with file:line, a failure scenario and the fix (or "not fixed: why")
- whether the live XTTS-only run can proceed through catalog → pack → stage → pilot → teacher as is
- any stage that must be re-run, with its exact command
- tests added and the pytest result
````
