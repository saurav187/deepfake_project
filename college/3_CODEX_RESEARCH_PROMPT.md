# Codex prompt: full research on datasets and pipeline improvement

Run this on a high-reasoning model, from the repository root (`~/deepfake_project`, branch `detector-v2`). Paste everything inside the box.

````text
You are the lead researcher for an audio deepfake detector. You have full authority to rethink the pipeline from scratch: data, preprocessing, model, training, calibration, evaluation and serving. Your job is research and a concrete, buildable plan, plus code if you're confident it's right. Be skeptical of my current choices and of published claims. Prefer evidence over fashion.

## Goal
A web app (app/, FastAPI + React) where a user uploads a recording and gets: likely real / uncertain / likely fake / not enough audio, plus a per-window timeline. The audio users upload is real-world: WhatsApp voice notes, phone recordings, YouTube/Instagram clips, Hindi and English, often MP3/AAC/Opus, often 3–30 s. The fakes they worry about are made with **current** tools: ElevenLabs, OpenAI/Gemini voices, XTTS-v2, F5-TTS, CosyVoice 2/3, GPT-SoVITS, Fish Speech, Kokoro, Chatterbox, Qwen3-TTS, VibeVoice, Seed-VC, RVC, OpenVoice. Success = high detection of unseen modern generators at a low false-alarm rate on real users' audio, with honest, reproducible numbers.

## Read first (all in the repo)
- detector/RESULTS.md: release v1 results (what works, what doesn't)
- college/1_DEMO_REPORT.md: a plain-language overview
- college/2_DATASET_RESEARCH.md: my first dataset survey (verify it; don't trust it)
- docs/PIPELINE_REVIEW.md: an earlier review of this pipeline
- detector/ code: audio.py (decode/prepare), augment.py, data/ (sources, build_manifest, audit, edge_probe), model/ssl.py, train/, calibrate.py, inference.py, eval/
- AGENTS.md: repo rules (overridden only where this prompt says so)

## Current pipeline (v1), in brief
- Decode to 16 kHz mono. Trim silence without margin, cut 0.25 s from each edge, raised-cosine fade over 0.15 s, normalise loudness to −25 dBFS. Score 4 s windows covering the clip; the recording score is the mean logit.
- Frozen XLS-R 300M (wav2vec2-xls-r-300m) → softmax-learned layer mix → LayerNorm → 128-d projection → attentive statistics pooling → MLP. Only about 2 MB is trained. Ensemble of 3 seeds.
- Training data: ASVspoof 5 train (72,673 clips, ~90% fake) and IndieFake train. A balanced (source × label) sampler. Augmentation: codecs (MP3, AAC, Opus, µ-law, 8 kHz PCM), noise, gain, edge fades and quiet padding.
- Selection: mean dev EER over sources × {clean, mp3_32k}; early stopping. Calibration: source-balanced Platt on dev. Thresholds from a 5% false-alarm and 5% miss budget; the uncertain band came out very narrow (0.514–0.562).
- Integrity: speaker-disjoint splits, exact PCM dedupe, a format-leak audit gate, an edge-silence shortcut gate (a silence shortcut was found and removed), a sealed one-time test with a ledger, and SHA-256-pinned releases.
- Results: In-the-Wild AUC 0.957 / EER 11.3% (legacy model 0.839 / 23.4%). Under 4 s of speech: 60% of fakes caught; 4 s or more: 98%. VA set: 19 of 22 fakes caught, 1 of 22 false alarms. **It misses modern TTS** (my own TTS samples pass as real; the VA misses are TTS). About 10% of In-the-Wild clips are under the 1 s speech minimum.
- An unshared earlier redesign (v3) proposed: speaker-separate splits for selection, calibration and thresholds; thresholds that depend on how much speech a clip has; and an Opus 16 kHz stress channel. They're not implemented.

## Hardware and constraints
- Training PC: RTX 5060 8 GB VRAM, Python 3.14, torch 2.15, transformers 5.24. Datasets go on an added HDD (assume ext4, 1–2 TB). The SSD is nearly full; plan I/O for an HDD (random small-file reads are slow).
- Deployment: a laptop CPU, no GPU. Inference should stay within a few seconds per 30 s clip, and model files within about 2 GB. A future mobile version would need about 150 MB or less, or a server.
- The internet at college is fast; at home it's slow. Plan downloads accordingly.
- This is a student project: non-commercial and research licences are acceptable. Record every licence.

## Part 1: dataset research (verify everything live)
Find the best available datasets as of today, especially from 2024–2026. Cover at least:
- Modern TTS/VC fakes: SpeechFake, MLAAD (v8+), CodecFake / CodecFake+, VoxENES 2026, ShiftySpeech, SpoofCeleb, DiffSSD, LlamaPartialSpoof, ASVspoof 5 (have it), and anything newer you find (2026 challenges, Interspeech/ICASSP 2026 datasets).
- Real-world benchmarks: Deepfake-Eval-2024, In-the-Wild (have it), ML-ITW, and anything similar.
- Indian languages: IndicSynth, HAV-DF, IndieFake (have it), plus **real** Hindi/Indic speech in quantity (IndicSUPERB/Kathbath, IndicVoices, Common Voice Hindi, Shrutilipi), and phone- or WhatsApp-like real speech.
- Real speech diversity: phone codecs, noisy, spontaneous, many speakers (Common Voice, VoxCeleb, AMI, etc.).
For each one give: what it contains (generators by name and year, languages, hours, number of clips, real/fake ratio), size in GB, licence and gating, the exact download method and command, train/test suitability, overlap with other sets (same generators or speakers), and known issues (silence or format shortcuts, mismatched real sources). **Open the dataset pages yourself and check the facts.** Mark anything you couldn't verify.
Then recommend a concrete download list:
- Tier A: must-have, with subsets and GB
- Tier B: nice-to-have
- Test-only: sealed
Include the total GB and an order that fits a fast-internet session.

## Part 2: pipeline improvement (full authority)
Review the v1 pipeline critically and propose the best v2 you can actually build on this hardware. Consider at least:
- **Data recipe:** mixing sources without shortcuts (real/fake balance per language, source and format; matched real speech for every fake source); generator-held-out splits; self-generating fakes with open TTS on the 8 GB GPU (which tools, how many hours, which voices and texts); including real-world channel degradations; dataset sizing.
- **Model:** frozen vs. partially fine-tuned vs. LoRA on XLS-R; other front-ends (XLS-R 1B or 2B within VRAM, WavLM Large, w2v-BERT 2.0, MMS, BEATs/audio models, Whisper encoder); back-ends (AASIST, Mamba, attentive pooling); multi-resolution or codec-aware features. Cite published In-the-Wild / Deepfake-Eval / VoxENES numbers where they exist, and be honest about how comparable they are.
- **Training:** loss (BCE, OC-softmax, SAM/ASAM, focal), RawBoost and codec augmentation, mixup, curriculum, how to fit within 8 GB (bf16, gradient checkpointing, small batches with accumulation).
- **Short clips** (under 4 s), and lowering the minimum-speech refusal rate without hurting accuracy.
- **Calibration and thresholds:** a wider, honest uncertain band; speaker-separate calibration split; per-length thresholds; validation on held-out generators.
- **Evaluation protocol:** which sealed test sets to use; unseen-generator reporting; per-language and per-length results; bootstrap CIs by speaker; a user's own test set (≈50 real and 50 fake, Hindi and English).
- **Deployment:** CPU latency on a laptop, quantisation or distillation for size, and a mobile path.
Rank the changes by expected gain per unit of effort, with the evidence for each.

## Deliverables
1. `college/RESEARCH_V2.md`, containing:
   - (a) a dataset table and the Tier A/B/test download list with commands and GB
   - (b) the v2 pipeline design with reasons
   - (c) an experiment plan in order, each with a hypothesis, a success metric and the decision it informs
   - (d) risks and shortcuts to guard against
   - (e) a timeline in GPU-hours
   - (f) a sources list with URLs
2. `college/DOWNLOAD_PLAN.sh`: the download commands. It must:
   - be resumable and idempotent
   - take a `DATA` directory argument
   - print sizes and ask for confirmation before each dataset
   - never run sudo
3. Optional, only if confident: code for v2 on a NEW branch `pipeline-v2-research` (create it from detector-v2). On that branch you may modify detector/ (this overrides the AGENTS.md rule). Keep v1 intact: don't touch detector/runs/release/v1, and the app must still serve v1. Include tests for new data adapters and gates. Don't train anything big; smoke tests only.

## Hard rules
- **Sealed test sets:** In-the-Wild, VA, Deepfake-Eval-2024, VoxENES (and any set you mark test-only) must never feed design, selection, calibration or thresholds. Never read their labels during development.
- Don't download big datasets yourself. Plan and verify sizes, and leave downloading to DOWNLOAD_PLAN.sh, which I'll run.
- Never commit audio, weights, archives, datasets or detector/runs/. Never push. Commit only on the new branch, and only code and docs.
- Don't modify deepfake_model/. Don't delete anything. Don't run sudo.
- Record a licence for every dataset and tool.
- Say clearly when something is a guess, unverified, or contested in the literature.

## Final reply to me
A short summary covering:
- the top 5 datasets to download, with GB
- the top 5 pipeline changes, with expected impact
- what you built (if anything) and how you tested it
- open questions only I can answer
````
