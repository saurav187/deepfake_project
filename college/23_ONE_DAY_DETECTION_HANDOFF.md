# Claude assignment — 24-hour English detection experiment

## Owner update — modern TTS first, 26 November

Prioritize clean modern TTS. The V2 target is at least 70% detection on each of Gemini, ElevenLabs, FishTTS and Chatterbox at one shared cutoff, with at most 15% of genuine clips flagged. Report pooled detection and raw counts per family. Run `python3 -m pipeline_v3.modern_tts_target --data "$D"` for a diagnostic at the 15% limit and the false-alarm cost of 70% detection on the existing panel. Apply the same analysis to the fallback when its scores are complete. Choose a deployment candidate's cutoff on separate calibration data and freeze it before testing on independent speakers and prompts. Treat the earlier 2% and 5% results as historical comparisons.

Give training-eligible matched modern TTS and genuine controls first claim on new training time. Reserve modern generator families, or when deliberately training on familiar families, reserve speakers, prompts and versions and label familiarity. Focus on Fish and Chatterbox while tracking regressions on Gemini and ElevenLabs. Defer call, Hindi and vocoder expansion until after this milestone. Continue already-running jobs and preserve their outputs.


Own execution through **26 November 2026, 23:20 IST**. Use the updated modern TTS target above. Report clean first, with Opus and simulated phone as secondary results. This is a development benchmark on three genuine source readers. Independent confirmation and actual calls are separate outcomes. A missed target is a valid result. Report the achieved operating point and raw counts by the deadline.

## Start now

Deploy the current repository's pipeline_v3 source in its own directory on the existing mounted GPU session, if available. Reuse the V1 artifacts and verified feature cache. Coordinate with the existing session owner before releasing a shared VM. Work only under audio-v3/v2/day-target-1 for this run. Preserve the paused V2-L checkpoint and existing V1 pack, pilot and thresholds.

From the deployed repository root, run these commands in order. Check each exit status. Keep one writer, save the logs to the experiment directory and record measured time and CU.

```bash
set -euo pipefail
D=/content/drive/MyDrive/audio-v3
W="$D/v2/day-target-1"
mkdir -p "$W"
for phase in baseline fit evaluate; do
  python3 -u -m pipeline_v3.day_target "$phase" --data "$D" 2>&1 | tee "$W/$phase.log"
done
```

The fit phase regenerates the exact original seed-42 pilot selection and uses the existing WavLM feature files. It reproduces the best full-layer linear clean pilot and checks its selection EER against pilot.json. It also fits one head on the original and Opus views of the same training clips. The calibrate fold supplies provisional cutoffs; select chooses between the two candidates using the worst generator/channel performance and false alarms. Threshold-fold genuine recordings set one final cutoff per candidate across clean, Opus and simulated phone. The primary choice is frozen before outside scoring. The teacher keeps its original cutoff. Comparing observed false alarms is essential because the models' realised rates on outside data can differ.

The evaluate phase aligns each candidate with the same V1 trial IDs and saves per-recording scores incrementally. result.json contains the primary candidate, original pilot and V1 teacher. Scores of the new heads are decision logits. Their cutoffs are in those same units. If a cache path is missing, inspect pilot.cache_directory and the original encoder revision path, pack index and cache signature. Recover the exact features or rebuild missing features from the bound pack in isolated scratch. Preserve the EER reproduction check.

## Time allocation and fallback

- **First 2 hours:** verify the baseline, reproduce the pilot, freeze candidate choice and thresholds. If waiting on a GPU slot, complete the CPU baseline and cache inventory first.
- **By hour 6:** score the exact outside panel, publish detections and false alarms with denominators, and record which generators fail. The strict objective requires all generator/channel cells. Include clean-only success or failure separately.
- **Hours 6–18:** if the primary candidate misses, test the existing released three-head detector as a separately identified fallback on the same development recordings. Verify its training provenance and potential overlap first. Freeze its existing weights and operating point, log exclusions and report its genuine false alarms alongside detection. If a separately audited broader training subset becomes ready in this window, prioritize a single bounded training comparison over another threshold sweep. Use separate outputs and development-only selection. Leave A08 with unresolved lineage excluded.
- **Hours 18–24:** reserve time for independent checking and a written result. Before evaluating a new independent panel, freeze one candidate, the cutoff and the panel membership. Use only permitted development audio with verified labels, both genuine and fake controls and documented overlap checks. Report small or unrepresentative panels descriptively. Preserve the sealed files. If independent data or real calls are unavailable, record those results as unmeasured. End expensive work by the deadline and report the best defensible result, even if below target.

Send the owner an initial acknowledgement with the active session, a progress update at candidate freeze, and the final per-generator and per-channel table. Record elapsed time, CU, failed attempts, checkpoint locations and any remaining blocker in the Claude handoff.
