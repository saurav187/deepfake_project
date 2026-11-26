# Codex prompt: final review of the exploratory English Teacher V1 pipeline before training

Run on a high-reasoning model from the repository root (`~/deepfake_project`, branch `pipeline-v3-scratch`). Paste everything inside the box.

````text
You are doing the FINAL pre-training review of pipeline v3's exploratory English Teacher V1 baseline. Read college/13_HANDOFF_TO_CLAUDE.md (top sections) first. Nothing has been run yet; a single paid GPU run will execute pipeline_v3/run_exploratory.sh once you sign off. Find defects that would waste the GPU run, silently invalidate results, or weaken a gate. Do not redesign.

## What the owner approved (25 Nov 2026, verbatim intent)
"A separately named exploratory English Teacher V1 baseline using the existing frozen, normalised PhonemeDF selection despite its documented failed nuisance audit. Preserve the failed report and the default gate. Carry the exception explicitly in the baseline's run record. Keep Chatterbox held out, retain the pilot gate, and train Teacher V1 if the pilot passes. Freeze the teacher and threshold before evaluating on non-sealed development data. Report false alarms on genuine speech and detection rates by generator and channel, with clip and speaker counts. Mark missing evaluations as unmeasured. Keep the sealed sets for the final comparison and defer the student."

## Facts (measured; see the handoff)
- Selection: stage0-en-ecapa-xtts/selection.csv (sha256 5aa26f7b…), English PhonemeDF XTTS-only train (2,244 + 2,244 clips, 126 readers, 4 inferred voices). Chatterbox held out in select/calibrate/threshold/validate. Voice-, reader-, text- and parent-disjoint folds.
- Frozen treatment `norm` (pipeline_v3/en_treatment.py, frozen.json in stage0-en-ecapa-xtts-diag/). Its formal audit FAILED: stage0-en-ecapa-xtts-norm/prep-audit.json (upper_band 0.817, rolloff 0.779, tilt 0.758, joint 0.758, speech_ratio 0.659). The explanation (explain/explain.json) shows that both XTTS and Chatterbox fakes are band-limited and darker than LibriSpeech. Expect the teacher to exploit this; the baseline exists to measure what it learns, not to pass the gate.

## Code to review (commit f408e0d and the files it touches)
- pipeline_v3/exploratory.py: create() copies the failed selection/eligible/prep-audit into stage0-en-ecapa-xtts-norm-exploratory/ and writes exception.json. exception_for() validates the record against the exact failed report. refuse() blocks student/export.
- pipeline_v3/stage0.py: --exploratory (catalog/pack only), exploratory_ok() in catalog and pack, and audit.json["exception"].
- pipeline_v3/cli.py: bound_audit() accepts a failed audit only with a bound exception; student/export/evaluate/benchmark refuse exploratory runs.
- pipeline_v3/teacher_eval.py:
  - freeze: Platt on calibrate, thresholds on threshold, using the release rule from export.py, written once.
  - evaluate: validate fold × 8 simulated channels; LibriSpeech dev genuine panel and MLAAD matched panel, both preprocessed with fixed_crop + norm; counts and rates; an unmeasured list.
- pipeline_v3/run_exploratory.sh: the exact GPU chain. Also check en_treatment.py, audit.py (probe_scores), train.py, pilot.py, data.py, audio.py and export.py where the chain depends on them.

## Questions to answer, each with file:line evidence
1. Gate integrity.
   - Can any path other than stage0-en-ecapa-xtts-norm-exploratory with a matching exception.json get past a failed audit?
   - Is the default gate behaviour byte-for-byte unchanged for every other run, including error paths and messages that tests rely on?
   - Can the exploratory run reach student, export, release, sealed.py or Stage 5? Is the exception carried into audit.json, pilot.json, the teacher checkpoints' provenance, teacher-frozen.json and teacher-eval.json? If it isn't in one of these, propose the minimal fix.
2. Binding and order.
   - Does stage/pilot/teacher bind to this pack's index?
   - Is the threshold frozen before any validate or panel scoring? Can evaluate run without freeze, or freeze run twice?
   - Does evaluate refuse a changed teacher or pack?
   - Are calibrate and threshold rows disjoint from validate, and from the panels (readers and utterances)?
3. Preprocessing parity.
   - Do training (stored bytes = norm(fixed_crop)) and every evaluation input (validate payloads, LibriSpeech dev, MLAAD/M-AILABS through fixed_crop → apply('norm') → decode → channel → model_view with pair='phonemedf') reach the model identically?
   - Are there decode, resampling, length or normalisation differences, e.g. model_view's std normalisation after norm, panel WAV/FLAC sample rates, or the phonemedf branch for non-PhonemeDF rows?
4. Pilot and teacher compatibility.
   - Does pilot.py run on an XTTS-only train cell with Chatterbox as the unseen domain, and are the go criteria unchanged?
   - Does train.py's early stopping use select domain/unseen = Chatterbox?
   - Will the cell weights (pair, language, label) behave sensibly with one fake family?
   - Does staging train, select, calibrate, threshold and validate break anything in pilot.py or train.py, which read only some splits?
5. Evaluation correctness in teacher_eval.py.
   - Are the verdict bands, false-alarm and detection definitions, reader and voice counting, separability AUCs, and deterministic channel RNG right?
   - Is the "at least 50 per class" guard right?
   - Does MLAAD reader parsing work on real member paths?
   - Does panel_rows read the right files (stage0/inventory.csv groups dev-clean/dev-other; stage0-en-ecapa-xtts/mlaad-diagnostic.csv), and can it touch a sealed path (data.DENY)?
   - Runtime: estimate the clip × channel teacher passes for validate (about 1,136 clips × 8 channels) and the panels (about 1,460 LibriSpeech dev × 3 and about 2,000 MLAAD × 3) on an A100 and on an L4, and flag anything that would dominate.
6. Robustness of run_exploratory.sh.
   - Resume after a disconnect: which steps are idempotent? A rerun hits "exception already recorded" because of the guard; check that the guard works. Check whether pack refuses an existing packed/ dir and whether stage/teacher resume from the last checkpoint.
   - Drive I/O hotspots: catalog writes about 10,400 small files to Drive.
   - Check that it stops on any failure and never runs teacher on a no-go pilot.
7. Reporting.
   - Does teacher-eval.json answer the owner's request (false alarms on genuine speech; detection by generator and channel; clip, reader and voice counts; unmeasured items listed)?
   - What is missing?
8. Tests.
   - Add the smallest tests for any real defect, plus an end-to-end tmp-dir test of catalog → pack → bound_audit for the exploratory path, and for a non-exploratory failed audit being refused.

## Rules
- Report findings ranked blocker / major / minor, each with a failure scenario. No style nits.
- You may commit fixes with tests on pipeline-v3-scratch for blocker and major findings. Don't change:
  - thresholds, the 0.60 gate or the pilot go criteria
  - the Teacher V1 recipe (steps, epochs, lr, augmentation, early stopping)
  - folds, the voice rule, the frozen treatment or the selection
- Don't start Colab, download data, or open archives. Don't touch sealed sets, deepfake_model/ or detector/. Never commit audio, data, keys or papers. Never push.
- All tests must pass: venv/bin/python -m pytest -q pipeline_v3/tests

## Final reply (short)
- findings by severity, each with file:line, scenario and fix (or "not fixed: why")
- go / no-go for running pipeline_v3/run_exploratory.sh as is, with the commit to deploy
- expected GPU runtime by step, A100 vs L4
- tests added and the pytest result
````
