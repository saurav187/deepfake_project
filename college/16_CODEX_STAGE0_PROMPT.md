# Codex prompt, stage 0: add PhonemeDF and LibriSpeech to pipeline v3, with voice-safe splits

Run on a high-reasoning model from the repository root (`~/deepfake_project`, branch `pipeline-v3-scratch`). Paste everything inside the box.

````text
You are continuing pipeline v3 (pipeline_v3/, docs college/8–15, handoff college/13_HANDOFF_TO_CLAUDE.md). This is STAGE 0 of the agreed plan:
0 code → 1 preprocess + shortcut audit → 2 pilot (go/no-go) → 3 teacher → 4 two students → 5 final training on all development data with the frozen recipe → 6 sealed test once.
Stage 0 is code only: make the pipeline able to use the new data correctly, prove it on the real file listings, and leave exact Colab commands for stage 1. You have freedom to improve the design, but every change must keep the evaluation honest.

## What is now on the owner's Drive (MyDrive/audio-v3/development/), all checksum-verified
- Previous: mailabs/en_US.tgz; mlaad/fake/en/{Chatterbox,ElevenLabs-v3,FishTTS,Gemini-3.1-Flash-TTS} (1,000 WAV + meta.csv each); kathbath/hindi (32 shards); indicsynth/Hindi (14 shards); indicvoices/hindi (5 shards); models/{xlsr,wavlm-base}; inventory.jsonl; selection_v0_english_thin.csv.
- NEW librispeech/: dev-clean.tar.gz, dev-other.tar.gz (OpenSLR 12, sha256 as pinned in pipeline_v3/fetch.py LIBRI; 40 + 33 readers).
- NEW phonemedf/: PhonemeDF v1.0.0 (Zenodo 19362791, CC BY 4.0, md5-verified; record.json and files.tsv saved alongside):
  - PhonemeDF_real.tar.gz (9.42 GB): real/wav (16 kHz) + real/transcripts, 28,539 utterances from LibriSpeech train-clean-100
  - PhonemeDF_XTTSv2.tar.gz (8.19 GB), PhonemeDF_ChatterboxTTS.tar.gz (7.85 GB), PhonemeDF_MeloTTS.tar.gz (7.70 GB): one synthetic rendition per real utterance each
  - PhonemeDF_references.tar.gz: 10 VCTK reference voices used for synthesis (p228_023, p230_023, p233_023, p236_023, p243_023, p244_023, p254_023, p258_023, p259_023, p273_023)
  - Documentation_PhonemeDF.pdf: average duration real 12.69 s, MeloTTS 10.15 s, XTTS v2 10.47 s, Chatterbox TTS 10.15 s; all audio 16 kHz; file naming per the GitHub README is <utteranceID>-<system>.wav, e.g. 19-198-0003-Real.wav / 19-198-0003-MeloTTS.wav
  - **No per-clip target-voice metadata is published.** Which of the 10 VCTK voices (or a built-in voice) each fake uses is unknown.

Local, git-ignored, metadata only (no audio), prepared for you:
- detector/runs/v3_prepare/inventory.jsonl, selection.csv, prepare.log (the earlier run)
- detector/runs/v3_prepare/phonemedf_listing/: `tar -tzvf` listings of PhonemeDF XTTSv2, ChatterboxTTS, MeloTTS and references, LibriSpeech dev-clean and dev-other, and files.tsv (Zenodo md5s). Fake members are named by LibriSpeech utterance ID, e.g. `XTTSv2/103-1240-0000.wav` (28,539 WAVs per system). **PhonemeDF_real.listing.txt and real_transcripts/ are being added shortly by Claude.** If they're not there yet when you need them, build against the fake listings (same utterance IDs), keep the real-side parsing configurable, and say which check to rerun once they arrive. Use them to build and test adapters against the real member names and counts. If something you need is missing, say exactly which command the owner should run in Colab to produce it.

## Stage 0 tasks
1. **Adapters and inventory.** Add PhonemeDF (real + XTTSv2 + ChatterboxTTS + MeloTTS) and LibriSpeech dev (real) to inventory/catalog/pack. Read tar.gz members without full extraction where practical, and keep foreign keys (utterance ID → reader, chapter, transcript). Add `phonemedf` (and `libri-dev`, already present) to fetch.py with the md5s from files.tsv, dry-run sizes, and resume.

2. **Matching and splits for English.** Pair each PhonemeDF fake with its real utterance by utterance ID. Assign whole LibriSpeech readers to train/select/calibrate/threshold/validate (reader-disjoint), and keep text/parent disjointness. Verify that LibriSpeech train-clean-100 (PhonemeDF) and dev-clean/dev-other readers are disjoint, and keep LibriSpeech dev readers out of PhonemeDF-based folds or assign them explicitly.

3. **Voice identity: the key risk.** With only 10 cloning reference voices, the model could learn those voices instead of synthesis traces, and every fold would share them. Solve it:
   - Infer each fake's target voice: compare speaker embeddings with the 10 references (e.g. an ECAPA or WavLM speaker-verification model available on Hugging Face), with a confidence margin and an "unknown" class. Detect whether MeloTTS (and any system) uses built-in voices instead, by clustering.
   - Write this as a Colab job, CPU or a cheap GPU, resumable, writing a voice-label CSV to Drive. Give the expected runtime and compute units.
   - Then make splits voice-disjoint for fakes, in addition to reader-disjoint for real speech. Report the trade-off in pairs per fold, choose an allocation (e.g. most voices in train, at least 1 unseen voice per evaluation fold), and handle low-confidence clips (exclude or own fold).
   - Measure on the listing/metadata what you can; mark the rest as Colab measurements.

4. **Duration and other shortcuts.** PhonemeDF fakes are ~2.5 s shorter on average than reals, a label shortcut. Make sure preprocessing and windowing remove it: equal-length crops or windows, no padding cues, matched trimming. Extend the audit with duration, speech ratio, edge silence, loudness, spectral tilt/bandwidth and corpus identity, per language and per source pair. Training must be blocked if any metadata-only probe exceeds AUC 0.60 on selection.

5. **Pilot configuration.**
   - English: PhonemeDF XTTSv2 + MeloTTS in train; **Chatterbox TTS fully held out** as the unseen family. MLAAD (ElevenLabs-v3, Gemini, Chatterbox, Fish) stays a separate diagnostic panel, reported apart.
   - Hindi: unchanged (Kathbath + IndicSynth XTTS-v2, FreeVC held out). Address the one-reader FreeVC selection limitation: rebalance, or report it clearly with denominators.
   - Keep the planned Gemini/Microsoft generation on LibriSpeech dev texts (English) and Kathbath texts (Hindi) as an add-on that joins at stage 3. Make the jobs ready, with --dry-run cost. Microsoft can use either Azure F0 or edge-tts; Gemini via API. The owner decides later.
   - Final-release variant (stage 5): the same recipe with held-out families (Chatterbox, Gemini, FreeVC) added back into training; the probe-transfer numbers are reported separately.

6. **Tests and proof.**
   - Unit and regression tests for the new adapters, voice-disjoint splitting, duration control and audit gates.
   - Run the new inventory/match logic locally on the listings and report a counts table: language × split × generator × label, with distinct readers and distinct voices per cell. Mark which numbers are measured and which depend on the Colab voice-labelling job.

## Constraints
- Don't download data, generate audio, call paid APIs or start Colab sessions yourself. Write commands; the owner (or Claude) runs them.
- Keep v1, deepfake_model/ and the sealed sets untouched: In-the-Wild, VA, VoxENES, Deepfake-Eval, app_testing_dataset/, and future owner recordings.
- Never commit audio, data, listings, inventories, keys or papers. Never push. Commit code and docs on pipeline-v3-scratch only.
- Keep Colab Python ≥ 3.11 and its preinstalled torch compatible. Don't pin transformers<5 or huggingface-hub<1 (Colab has newer; fix pipeline_v3/requirements-colab.txt accordingly).

## Deliverables
- Code and tests on pipeline-v3-scratch, with all tests passing.
- `college/17_STAGE0_REPORT.md`:
  - what changed
  - the counts tables
  - the voice-labelling method and its Colab command
  - the audit additions
  - the exact stage-1 Colab command sequence, in order: inventory → voice labels → match → audit → catalog → pack, with expected runtimes and compute units
- An update to college/13_HANDOFF_TO_CLAUDE.md.

## Final reply (short)
- what was added
- the English and Hindi pairs per split (measured vs projected)
- how voice leakage is handled
- the stage-1 commands in order
- anything the owner must decide
````
