# Codex prompt: full research, diagnosis, datasets, and a new pipeline built from scratch

Run on a high-reasoning model from the repository root (`~/deepfake_project`, branch `detector-v2`).

**Before you start:**
- Copy the `papers/` folder (git-ignored) to the machine Codex runs on, and delete the duplicate files 25–30.
- Paste everything inside the box.

````text
You are the lead researcher and engineer for a student project: an app that tells whether an uploaded voice recording is real or AI-generated. You have FULL FREEDOM: think from the beginning, brainstorm new ideas, challenge every earlier decision (mine, earlier AI-written reports in this repo, and published papers), and design the best system that fits the scope below. Verify facts, measure instead of assuming, and label anything that is a guess.

======================================================================
THE OWNER'S GOALS (in priority order)
======================================================================
1. The model must catch DIFFERENT modern TTS and voice-cloning deepfakes, not just the generators in old benchmarks. The owner tests with Google Gemini TTS and Microsoft TTS (Azure / Speech playground voices).
2. The final model must be SMALL enough to run locally in a future phone/laptop app (target ≤ 90 MB after 8-bit quantisation, a few seconds per 30 s clip on a laptop CPU; say what is realistic below that).
3. All training happens on Google Colab Pro.

The owner's own first ideas (address them explicitly):
(a) Diagnose WHY the current model cannot detect even simple TTS fakes, let alone voice-cloning deepfakes.
(b) Build a SMALL model / quick pilot first and test whether it gives reliable results, before training on the entire dataset.

======================================================================
SCOPE (fixed)
======================================================================
- Input: a recording or audio/video file uploaded to the web app (app/: FastAPI + React, already built). Output: likely real / uncertain / likely fake / not enough audio, a probability, and a per-segment timeline (docs/PRD_deepfake_audio_app.md §7 is the interface the app calls).
- Audio only. Hindi and English, including Indian-accented English. Real-world uploads: WhatsApp voice notes (Opus), phone recordings, MP3/AAC/M4A, social-media clips, often 3–30 s.
- Threats: current TTS and voice cloning, e.g. Gemini TTS, Microsoft neural/HD voices, ElevenLabs, OpenAI voices, XTTS-v2, F5-TTS, CosyVoice 2/3, Chatterbox, Fish Speech, Kokoro, GPT-SoVITS, RVC, OpenVoice, Seed-VC.
- Out of scope: video/face deepfakes, live call interception, speaker identification, watermarking.

======================================================================
COMPUTE AND STORAGE
======================================================================
- Training: Google Colab Pro. The GPU varies (T4 / L4 / A100). Compute units are limited: about 100 units/month, roughly 8 A100-hours or about 20 L4-hours. Use cheap GPUs for data work and the pilot, and the A100 only for the expensive model.
- Storage: the owner's Google Drive has 5 TB. Download every dataset ONCE, inside Colab (Hugging Face / Zenodo / Kaggle, tokens in Colab Secrets), straight to Drive; never upload large data from home (slow internet). Drive's mount is slow for many small files:
  - Pack each dataset once into large shards (~0.5–1 GB, e.g. tar/WebDataset or Parquet) plus an index, on Drive.
  - At each session start, copy the needed shards to the Colab VM's local disk and train from there.
  - Checkpoint to Drive every epoch, and resume automatically after disconnects.
- The owner has the **Google Colab CLI** installed locally (`colab`, google-colab-cli 0.8.0; `colab skill` prints its guide). Design every step so it runs BOTH from the notebooks and from the terminal:
  - `colab new -s NAME --gpu L4` → `colab exec -s NAME -f script.py` → `colab download` → `colab stop -s NAME`
  - or one-shot `colab run --gpu T4 script.py ARGS`
  - Scripts must take all paths as arguments, write results to Drive or /content, and exit non-zero on failure.
  - Starting a VM costs compute units: never run `colab new` or `colab run` yourself; print the exact commands for the owner.
- Code must run on Colab's Python and preinstalled torch (stay compatible with Python ≥ 3.11; do not force a torch/CUDA reinstall). Every data path is configurable through one DATA root.
- A local RTX 5060 (8 GB, Python 3.14, torch 2.15) and an HDD with already-downloaded data exist as a fallback. The HDD inventory script is college/DOWNLOAD_STATUS.py.

======================================================================
READ FIRST
======================================================================
- papers/: 54 unique papers (file-number prefixes, e.g. 13th-…). Skip duplicates 25–30.
- college/5_LITERATURE_MATRIX.md (an earlier paper classification; verify and correct it).
- college/6_PROJECT_PLAN_FROM_LITERATURE.pdf and college/4_HOW_THE_PIPELINE_WORKS.pdf (use pdftotext).
- college/RESEARCH_V2.md, college/2_DATASET_RESEARCH.md, college/DOWNLOAD_PLAN.sh (earlier dataset research; verify, it may overreach).
- detector/RESULTS.md, detector/ code, docs/PIPELINE_REVIEW.md, AGENTS.md.
- If present: detector/datagen/tts_dataset.py (a draft Microsoft/Gemini generation script), detector/eval/compare_dev.py and the x1 changes (--rate-masked audit, rate round trip and low-pass augmentation).

======================================================================
PROJECT HISTORY: FAILURES TO LEARN FROM
======================================================================
1. Legacy model (deepfake_model/): a CNN-LSTM trained from scratch on spectrograms. Great in the lab, In-the-Wild AUC 0.839 / EER 23.4 %. File format and sample rate predicted the label; an 8 kHz phone channel flipped most real audio to FAKE.
2. Silence shortcut: in ASVspoof 5 train, real files have edge silence and natural fade-outs, fakes stop abruptly (metadata alone: AUC 0.93). It was fixed with no-margin trimming, a 0.25 s edge cut, 0.15 s fades, edge-silence augmentation on both labels, and an edge-probe gate.
3. v1 (current release): frozen XLS-R 300M + learned layer mix + attentive statistics pooling + MLP (183K trained parameters per head), 3 seeds, trained on ASVspoof 5 + IndieFake (English only, ~8.7k real / 67k fake), Platt calibration, 5 %/5 % thresholds. Results:
   - In-the-Wild AUC 0.957 / EER 11.3 %
   - VA 19/22 fakes caught
   - under 4 s of speech: ~60 % of fakes caught
   - "uncertain" band only 0.514–0.562
   - total model 316M parameters, 1.2 GB
4. **v1 fails on modern TTS.** The owner's Gemini TTS clips (24 kHz WAV, 5–9 s of speech) score 0.7 % and 6.4 % fake, LOWER than the owner's real WhatsApp note (10.4 %). Opus/MP3 re-encoding changes nothing, and every 4 s window is confidently "real". Working hypotheses to test, not facts:
   - the generator families are unseen
   - "real" in training = clean read speech, so clean fluent TTS without known artefacts maps to real
   - a frozen encoder with a tiny head memorises the training generators' artefacts
5. Experiment x1:
   - Recovered ~5,800 IndieFake fakes (24/44.1 kHz) that format balancing had excluded. Masked the stored rate, with rate round trips and low-pass augmentation on both labels.
   - IndieFake dev EER improved from 7.4 % to 4.1 %, ASVspoof 5 was unchanged, and the shortcut controls passed.
   - **The Gemini clips were still not detected.** v1 already caught IndieFake's 24 kHz fakes 98 % of the time, so they were not the blind spot.
   - Lesson: you reliably catch the generator families you train on, and they must come with matched real speech.
6. Spectral-balance confounds: band energies differ strongly between recording setups, speakers and labels (AUC up to 0.9 between real 16 kHz and real 44.1 kHz IndieFake speakers). Any new data mix needs gates for spectral tilt, bandwidth, sample-rate history, duration, silence, loudness and dataset identity.
7. IndieFake dev has only 4 speakers, so Indian-speech dev estimates are weak.

======================================================================
TASKS
======================================================================
1. DIAGNOSE the current failure (owner idea a)
   Using v1 (detector/runs/release/v1 if available, otherwise its documented design) and dev/self-generated data only, test the hypotheses in history item 4 and any you add. Ideas:
   - score clean read speech vs spontaneous/phone real speech
   - score small sets of Gemini/Microsoft clips generated for development (never the owner's test clips)
   - check per-layer separability of frozen XLS-R features for these fakes vs matched real (linear probe)
   - compare against a handcrafted baseline
   Conclude with the main cause(s) and what that implies for the design.

2. DEFINE THE RESEARCH PROBLEM
   A one-page problem statement (context, gap with evidence, aim, scope), objectives, research questions and hypotheses. Cite paper file numbers and the project's own evidence.

3. ANALYSE EVERY PAPER (full text, not just abstracts)
   For each of the 54 papers record:
   - title, venue, year, task (audio/video/multimodal) and relevance
   - **pipeline:** preprocessing (resampling, trimming/VAD, input length), features or front-end (MFCC/LFCC/CQCC, spectrogram, raw waveform, SSL model, and whether it is frozen, fine-tuned or adapted), back-end, loss, augmentation, training data, input length, evaluation protocol
   - **results:** each test set, metric and number, in-domain vs cross-dataset; especially In-the-Wild, ASVspoof 2021 DF, ASVspoof 5, and any modern-TTS test
   - model size and speed if reported; code/weights/data availability
   - limitations and validity threats: random in-domain splits, speaker overlap, accuracy-only reporting, untested channels, silence/format shortcuts
   Then synthesise:
   - the common pipeline patterns
   - which choices correlate with cross-dataset success
   - what small or efficient models achieve
   - a cross-dataset comparison table with caveats
   Mark unverifiable numbers.

4. RESEARCH GAPS
   A gap table: gap → evidence (paper numbers, numbers or quotes) → how this project addresses it → how it will be measured. Verify or reject these, and add your own:
   - lab vs real-world generalisation
   - modern commercial TTS untested
   - compression/AI-processing false alarms
   - no Hindi detector
   - small, generalising on-device models missing
   - few calibrated, deployable systems
   - weak evaluation hygiene

5. DATASETS: WHAT TO DOWNLOAD, GENERATE AND COLLECT
   Verify every dataset live: licence, gating, exact files, GB, generators by name and version, languages, real/fake counts. Pair each fake source with a matched REAL source (same language and recording style).
   - Public: evaluate Codex's earlier Tier A (MLAAD v11 English folders incl. Gemini-3.1-Flash-TTS, ElevenLabs-v3, Chatterbox, FishTTS + M-AILABS; IndicSynth Hindi + Kathbath; IndicVoices) and alternatives (SpeechFake, CodecFake/CodecFake+, DiffSSD, SpoofCeleb, ShiftySpeech, LlamaPartialSpoof, newer 2025–26 sets). Say what is worth it and why.
   - Self-generated:
     - Microsoft voices: edge-tts is free, with hi-IN, en-IN and multilingual voices; the Azure free tier covers 0.5M characters/month.
     - Gemini TTS: free tier, or paid at about $20 per 1M audio tokens; verify the current price.
     - 2–3 open voice-cloning models (e.g. XTTS-v2, F5-TTS, Chatterbox, OpenVoice/Seed-VC) cloning speakers of the matched real corpora, within licences and consent.
     - Specify texts (Hindi + English), voices, counts and held-out voices/sentences/generators.
   - Owner-collected: a protocol for ~50–100 real Hindi/English clips on phones and through WhatsApp, with consent, kept test-only or development-only as you justify.
   - Sealed tests: In-the-Wild, VA, VoxENES 2026, Deepfake-Eval-2024, the owner's private set, and the owner's clips in app_testing_dataset/.
   Produce:
   - (i) a MINIMUM high-value set first (≈ 40 GB, ~1–3 GPU-hours per training run)
   - (ii) extras up to ~150 GB in total, each justified by the new generators, languages or channels it adds and its expected gain. No dataset is included just for size.
   - (iii) sealed tests
   Each with Colab download cells (to Drive), GB and cost.

6. BRAINSTORM, THEN BUILD A NEW PIPELINE FROM SCRATCH
   First brainstorm widely: list at least 10 genuinely different ideas (front-ends, training signals, data strategies, augmentations, one-class or anomaly framing, generator-agnostic cues, multi-view features, codec-aware modelling, self-generation loops, distillation, test-time aggregation, calibration/abstention, anything new). For each idea give the evidence for and against and the cost, then choose and justify.
   The chosen design must cover:
   - Data: matched real/fake per language and channel; speaker-, voice- and generator-held-out splits; leakage/shortcut gates (format, sample rate, bandwidth/spectral tilt, duration, silence, loudness, dataset identity); Drive packing.
   - Preprocessing: one shared path for training, evaluation and serving; short clips and masks.
   - **Teacher → small student (goal 2):** a strong teacher (e.g. XLS-R/WavLM/w2v-BERT with LoRA or partial fine-tuning on an A100), then a small student (e.g. the first 4–6 encoder layers, or a compact speech model) trained by distillation, quantised to 8-bit, target ≤ 90 MB. Report teacher vs student accuracy, size and laptop-CPU latency. If you find a better route to a small generalising model, argue for it.
   - Training: losses, augmentation (real codecs incl. WhatsApp-style Opus, phone band, playback, RawBoost, rate round trips, low-pass, all on both labels), sampling, early stopping on held-out generators.
   - Calibration and decisions: a separate speaker split; false-alarm budget around 2 %; a meaningful uncertain band; length-aware abstention.
   - Evaluation: seen / related / unseen generator, language, length and channel slices; speaker-clustered bootstrap CIs; each sealed set used once per release.

7. PILOT FIRST (owner idea b)
   A fast, cheap pilot that runs before any full training (minutes to about 1 hour on a T4/L4):
   - About 1–3k clips per class.
   - Cache per-layer pooled SSL statistics once.
   - Train linear probes and a small MLP; compare full XLS-R vs its first 4/6 layers vs a small encoder vs a handcrafted baseline (e.g. LFCC/MFCC + logistic regression or a small CNN).
   - Report EER on held-out voices, speakers and generators, the effect of WhatsApp-style Opus, and which layers carry the signal.
   - Define explicit go/no-go criteria for scaling up.

8. HOW WE DIFFER FROM THE PAPERS
   A table, us vs typical papers:
   - generators (modern commercial, self-generated), Hindi
   - preprocessing (silence/edge handling, shared serving path)
   - encoder use, teacher→student and size
   - augmentation with real channels
   - shortcut gates
   - calibration and abstention
   - evaluation (sealed, held-out generators, speaker CIs, slices)
   - deployment (local app)
   Be honest where papers are stronger.

======================================================================
DELIVERABLES
======================================================================
Docs (in college/):
1. 8_DIAGNOSIS_AND_PROBLEM.md: diagnosis of the current failure, problem statement, objectives, RQs, hypotheses.
2. 9_PAPER_ANALYSIS.md + 9_paper_analysis.csv: one row per paper (pipeline, data, results, size, limitations, relevance), then the synthesis and cross-dataset table.
3. 10_RESEARCH_GAPS.md: the gap table.
4. 11_DATASETS_V3.md: minimum list, extras and sealed sets; matched real pairs; self-generation and owner-collection protocols; Colab download cells; GB and costs.
5. 12_PIPELINE_V3_DESIGN.md: the brainstorm table, chosen design with reasons, pilot go/no-go criteria, ordered experiment plan (hypothesis, metric, decision, GPU type and hours, compute units), expected results as honest ranges, and the "how we differ" table.

Code on a NEW branch `pipeline-v3-scratch` (create it from detector-v2):
- data adapters, packing to Drive shards, leakage/shortcut gates (with tests)
- self-generation scripts (Microsoft, Gemini, open cloning models) with --dry-run cost and plan output
- the pilot (feature caching, probes, baseline, report)
- teacher training, distillation to the student, 8-bit export, calibration, dev evaluation
- an exporter that produces a release the existing app loads through PRD §7, plus a CPU latency benchmark
- Colab notebooks colab/01_setup_and_data.ipynb, 02_pilot.ipynb, 03_teacher.ipynb, 04_student_export.ipynb, 05_evaluate.ipynb, keeping the logic in repo modules
- a README with the exact commands and Colab cells, in order

======================================================================
HARD RULES
======================================================================
- PERMISSION AND COST: never call paid APIs, generate data with external services, or start downloads or training runs yourself. Write the scripts with --dry-run, show the plan and estimated cost, and let the owner run them. Ask before anything that costs money or uses more than 5 GB.
- Never use sealed sets for design, selection, calibration or thresholds: In-the-Wild, VA, VoxENES, Deepfake-Eval-2024, the owner's private set, and app_testing_dataset/ (the owner's Gemini clips fake1, fakev and the real clip). Never read their labels during development.
- Keep release v1 (detector/runs/release/v1) intact and the app able to serve it.
- Never commit audio, weights, datasets, archives, papers/, API keys or detector/runs/. Never push. Commit only code and docs, on the new branch.
- Don't modify deepfake_model/. Don't delete data. Don't run sudo.
- Cite paper file numbers for every literature claim; mark unverified facts; give expected results as ranges with reasons, never as promises.

======================================================================
FINAL REPLY TO ME (short)
======================================================================
- why the current model fails (the diagnosis, in 3–5 lines)
- the problem statement in 3 sentences
- the top 5 research gaps
- the minimum dataset list with GB and cost
- the chosen pipeline in 10 lines, including teacher → student and the final size
- the pilot's go/no-go criteria
- expected improvement as honest ranges
- the first 3 things I should run in Colab
- questions only I can answer
````
