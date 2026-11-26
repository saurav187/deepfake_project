# Codex prompt: find new datasets for pipeline v3 (research and shortlist only)

Run on a high-reasoning model with web access, from the repository root (`~/deepfake_project`, branch `pipeline-v3-scratch`). Paste everything inside the box.

````text
You are helping pipeline v3, a Hindi/English audio deepfake detector (pipeline_v3/, docs college/8–18, handoff college/13_HANDOFF_TO_CLAUDE.md, read its top sections first). This task is RESEARCH ONLY: find and rank datasets the owner could download next, and prepare exact, dry-run-able fetch entries. Do not download data, start Colab, call paid APIs, or buy anything.

## Why we need new data (measured, 25 Nov 2026)
- English pilot data is PhonemeDF (Zenodo 19362791): LibriSpeech train-clean-100 real + XTTS-v2 / MeloTTS / ChatterboxTTS renditions of the same utterances.
- Our shortcut audit fits simple probes on nuisance descriptors of the exact 4 s model-input crops (duration, rms, edge silence, energy above 4 kHz, 95% spectral rolloff, spectral tilt = energy below 1 kHz, speech ratio, codec, sample rate, and a joint probe). Training is blocked if any probe's AUC is above 0.60 in either direction on the select fold.
- Untreated PhonemeDF crops failed (rms 0.886, upper_band 0.817, rolloff 0.779, tilt 0.758, speech_ratio 0.659, joint 0.887). A frozen level normalisation fixed rms (0.493) but the spectral fields and speech ratio still fail. The cause is that real and fake audio come from different recording/synthesis chains (LibriSpeech MP3-origin recordings vs TTS output), not the voices.
- Hindi (Kathbath real + IndicSynth XTTS-v2 fake) fails mainly on speech ratio (0.724). Matching on speech density fixes the probes but leaves only 4 select readers (minimum 5).
- A per-generator analysis (XTTS vs Chatterbox, on shared parents) is running now; criteria below may be sharpened by its result.

So the most valuable data is data where **genuine and fake audio share the same recording chain and conditions**, or where both are drawn from the same source so the nuisance descriptors match.

## Already on Drive (don't propose again unless a different subset helps)
PhonemeDF (real, XTTSv2, ChatterboxTTS, MeloTTS, references); LibriSpeech dev-clean, dev-other; M-AILABS en_US; MLAAD fake/en {Chatterbox, ElevenLabs-v3, FishTTS, Gemini-3.1-Flash-TTS}; Kathbath Hindi (32 shards); IndicSynth Hindi (14 shards); IndicVoices Hindi (5 shards); models xlsr, wavlm-base, ECAPA VoxCeleb.

## SEALED: never propose these or anything derived from them as training/design data
In-the-Wild, VA, VoxENES, Deepfake-Eval, app_testing_dataset/, and future owner recordings. Flag any candidate that overlaps one of these (same source clips, same speakers, re-release), and say how you checked.

## What to look for, ranked by priority
1. **Matched-chain pairs (highest).** Real and fake share text and speaker or recording source, and the fakes keep the real chain's sample rate and bandwidth. Examples to verify, not to assume: vocoder resynthesis of a real corpus (e.g. LibriTTS/LibriSeVoc-style, WaveFake-style), ASVspoof 2019 LA / 2021 / ASVspoof 5 (licence, access, how bona fide and spoof were produced), codec-resynthesis sets (CodecFake-style), MLAAD's paired real sources. For each, state from the paper or data card how real and fake were produced, at what sample rate, and whether real audio went through the same post-processing.
2. **Hindi and other Indic languages.** Hindi real + fake with at least 5 readers in each evaluation fold after reader-disjoint splitting; any Indic deepfake corpus with documented generation; real Hindi speech from the same source as an existing Hindi fake set.
3. **Modern TTS and voice-conversion families** not yet covered (diffusion and flow TTS, codec-LM TTS, commercial APIs, recent open models), ideally rendered from real corpora we can pair. Fake-only sets go to the diagnostic panel, like MLAAD, unless a matched real source exists.
4. **Genuine phone or WhatsApp-like recordings** (narrowband, VoIP or Opus, noisy), real speech only, licensed for research. They're needed for false-alarm analysis on phone audio. They must not overlap the sealed sets.

## Paid options (the owner will spend a small amount if justified)
- List anything that needs payment: dataset licences, LDC-style access, cloud egress, or TTS API credits to render matched fakes from texts and speakers we already hold (LibriSpeech dev, Kathbath).
- For TTS generation, give the provider, per-character or per-second price with source link and date checked, the minimal job (number of clips, characters, seconds), the total cost in USD and INR, and the terms of use for training a detector. Keep the existing rule: generation never starts without the owner's explicit approval.
- Put cost into tiers: free; under USD 10; USD 10–50; over USD 50.

## For every candidate, report (verify each fact from a primary source and link it)
- name, version, URL, DOI or record ID, licence (commercial or research, redistribution), access method (direct, form, gated HF, LDC), size in GB, file count
- languages, number of real speakers or readers, number of fake systems with their names, sample rates, codecs, clip durations
- how real and fake were produced, and whether they are matched (text, speaker, source recording, sample rate, post-processing)
- expected shortcut risk against our audit fields (rms, upper_band, rolloff, tilt, speech_ratio, edge, duration, codec, rate), with your reasoning, and whether a published paper reports shortcut or bias issues
- overlap or contamination risk with the sealed sets and with data we already hold
- what fold role it could take (train, held-out family, diagnostic panel, phone-genuine false-alarm set)
- checksums if published; download command; cost

## Deliverables
1. `college/20_DATASET_CANDIDATES.md`: a ranked shortlist (top 5–8, plus a "rejected and why" list). Lead with a one-paragraph recommendation of what to download first, given the audit failure. Include a summary table and a per-candidate section with the fields above.
2. Dry-run fetch entries only: extend `pipeline_v3/fetch.py` (or a new `pipeline_v3/fetch_candidates.py`) with pinned URLs, sizes, published checksums and resume, defaulting to `--dry-run`, for the top candidates that are freely downloadable. Add tests that the dry run lists sizes and never downloads. Don't add gated or paid sources to code; describe them in the doc.
3. A short update under a new heading at the top of `college/13_HANDOFF_TO_CLAUDE.md`, pointing to the shortlist.

## Rules
- Don't download, purchase, register for gated access, or generate audio. Write the commands; the owner decides.
- Don't modify `deepfake_model/` or `detector/`, and don't touch sealed sets. Never commit audio, data, listings, keys or papers. Never push. Commit only docs and code on `pipeline-v3-scratch`; all tests must pass (`venv/bin/python -m pytest -q pipeline_v3/tests`).
- Don't change audit thresholds, gates, folds or the frozen ECAPA voice rule.
- Don't put the owner's name, email or project details in any search query or request to an outside service.
- Mark every unverified fact as "unverified". Prefer primary sources (paper, data card, Zenodo/HF/OpenSLR page) over blog posts.

## Final reply (short)
- the top 3 datasets to download first and why, each with size, licence and cost
- which candidates are most likely to pass our audit, and why
- paid options with total cost by tier
- anything the owner must approve or register for
````
