You are reviewing an audio-deepfake detection project and deciding, from first principles, what the best detection pipeline is for its goal. Think independently. The current design is one candidate, not the answer. You may conclude it should be kept, changed or replaced.

## The goal
A web app (already built in `app/`) where a user uploads a recording — often a WhatsApp voice note, MP3, M4A, or a clip taken from social media — and gets a calibrated verdict (likely real / uncertain / likely fake), a probability and a per-segment timeline. The owner is a student; this is an academic project that may become public. Users may be in India, so Hindi and Indian-accented English matter. What matters most is reliability on real-world, compressed, unseen-generator audio, not benchmark scores.

## Hard constraints
- Hardware: one RTX 5060 (8 GB VRAM), 12 CPU cores, 14 GB RAM, of which about 8 GB is usually in use by other programs. About 20 GB of disk is free.
- Internet: Zenodo gives about 2 MB/s in total; Hugging Face is faster. Large downloads take hours.
- Software: Python 3.14 in `venv/` with torch 2.15 (CUDA 13.2), transformers 5.24, librosa, soundfile, soxr, scikit-learn, ffmpeg 9.
- Licences: an academic project is fine with research licences, but flag anything that would block a public or commercial app.

## What exists — read it; don't trust this summary
Read these before forming opinions. Verify every claim below against the code and result files.
- `deepfake_model/REPORT.md` and `deepfake_model/docs/CALL_CHANNEL_PROBE.md`: the legacy log-Mel CNN-LSTM. An audit found its historical 0.963 AUC was invalid because of test leakage and duplicates. On reserved IndieFake speakers it scored AUC 0.994, but it does not transfer.
- `deepfake_model/research/asvspoof_la_v1/README.md`: on ASVspoof 2021 LA it scored AUC about 0.62–0.70. A format audit found that **sample rate and channel count alone predict the IndieFake label at AUC 0.95**.
- `deepfake_model/research/training_formats_v1/README.md` and `deepfake_model/research/channel_augmentation_v1/README.md`: native-format training, and 8 kHz channel augmentation. The augmentation fixed channel false alarms but did not improve ASVspoof transfer (AUC 0.61–0.65).
- `detector/runs/baselines/legacy_itw/summary.csv` (if present) and `detector/README.md`: on In-the-Wild, the legacy audited model scores AUC 0.839 / EER 23.4%, while the native-format models are near chance (0.38–0.55).
- `detector/` (the new v2 pipeline):
  - `audio.py`: canonical decode, silence trim, loudness normalisation, repeat-padded 4 s windows.
  - `data/sources.py`, `data/build_manifest.py`, `data/audit.py`: dedupe, speaker splits, format-stratum balancing and a leakage gate. `detector/runs/audit/report.json` shows the current audit result.
  - `augment.py`: codec, noise and gain augmentation.
  - `model/ssl.py`: frozen WavLM Base+ or XLS-R 300M, a learned layer mix, attentive pooling and an MLP head.
  - `train/data.py`, `train/train.py`: balanced source × label sampling, with dev-only selection on clean and MP3 dev.
  - `calibrate.py`: Platt scaling and budget-based thresholds.
- Data on disk, or downloading (see `detector/datasets/`):
  - ASVspoof 5: train tars `flac_T_aa` and `flac_T_ab` (of 5), dev `flac_D_aa` (of 3), and the protocols.
  - In-the-Wild: complete; used as a test set.
  - IndieFake: in `deepfake_model/datasets/`.
  - VA: 44 files; test only.
- `docs/PRD_deepfake_audio_app.md`, section 7: the detector interface the app expects.

## Rules for this task
- **Read-only, except for one output file.** Write your analysis to `docs/PIPELINE_REVIEW.md`. Do not modify any other file.
- Background jobs are running: dataset downloads (`pdownload.py`), and `detector/run_model_a.sh`, which will train once the downloads finish. Don't stop, restart or interfere with them, and don't delete or move data.
- You may run short, read-only analysis commands: reading manifests and CSVs, computing statistics, and inspecting protocol files or audio metadata. Don't train models, don't download large files, and don't use the GPU for more than a few minutes.
- Treat In-the-Wild and VA as sealed test sets. Don't propose any design choice that is tuned on them, and point out anywhere the current code risks that.
- Where you rely on published results, say which paper or benchmark the claim comes from and how confident you are. Separate what you verified in this repository from what you believe from the literature. If you are unsure whether a model, dataset or result exists, say so instead of guessing.

## Questions to answer
Answer from the evidence. Argue against the current design wherever it's weak.

1. **Problem framing.** Is "utterance-level binary classification with segment scores" the right framing for uploads? Consider partially spoofed audio (a real recording with spliced synthetic words), music or background noise, multiple speakers, and very short clips. What should the app say in each case?
2. **Data.** Is ASVspoof 5 (2 of 5 train tars, 1 of 3 dev tars) plus format-balanced IndieFake a sound training set for this goal? What is missing, especially modern zero-shot TTS and voice-conversion systems, neural-codec and audio-LLM generators, Indian languages, and real-world compression? Rank concrete additions (for example MLAAD, ASVspoof 2019 LA, SpoofCeleb, CodecFake, Famous Figures, or self-generated fakes made with open TTS on genuine speech), considering licence, size and download time on this connection. Check how the partial ASVspoof 5 tars are composed (labels, attacks and speakers in each tar) using the protocol files, and whether two of five train tars are representative.
3. **Shortcuts and leakage.** Beyond file format, which shortcuts could this pipeline still learn? Consider silence and trimming artefacts, loudness, bandwidth (a fake stored at 24 kHz versus genuine at 16 kHz), codec history, duration, speaker identity, language and corpus identity. Is the audit's format-stratum balancing correct and sufficient? Is the In-the-Wild duration difference (fakes are longer) a risk for the evaluation or for the model? Propose concrete, cheap tests for each shortcut, such as metadata-only baselines, band-limited controls and per-corpus probes.
4. **Front end and model.** Frozen SSL with a light head, versus partially or fully fine-tuned SSL (for example XLS-R + AASIST, or SLS-style multi-layer heads), versus a raw-waveform model (AASIST, RawNet2), versus spectrogram models, versus an ensemble. What is realistically trainable on 8 GB of VRAM (gradient checkpointing, LoRA or adapters, freezing lower layers, bf16, smaller batches)? Is WavLM Base+ or XLS-R 300M the better encoder here, and is there a better encoder we should be using? Justify with expected gains versus cost.
5. **Augmentation.** Is the current chain (codecs, gain, noise) right? Should it add RawBoost, room impulse responses, MUSAN-style noise and music, speed or pitch changes, tandem codecs (re-encoding twice), or band-limiting? Which augmentations could *hurt*, by destroying the artefacts that reveal fakes?
6. **Windows and aggregation.** Is 4 s right? Mean logit versus max versus top-k versus attention over segments? How should the score for a 10-minute file be formed, and how should "uncertain" work? Note that dev calibration uses at most 4 windows per recording while inference uses all windows: does that matter?
7. **Selection and evaluation protocol.** Is "mean EER across sources × {clean, MP3}" a good selection metric? Is calibrating on the same dev set used for epoch selection a problem? What should the reported test results contain (per-source results, per-attack results, bootstrap confidence intervals by speaker, results at fixed false-alarm rates, calibration error)? What would a credible final confirmation set look like, for example consenting speakers cloned with current tools and sent through WhatsApp?
8. **Calibration and thresholds.** Is source-balanced Platt scaling with 5% false-alarm and 5% miss budgets appropriate? Should the base rate (the prior) be handled differently for a public app? How should probabilities behave on out-of-distribution input?
9. **If you started from scratch** with these constraints, what pipeline would you build? Give the full design: data, preprocessing, model, training, selection, calibration, evaluation and serving. Then give the smallest set of changes that moves the current code toward it.

## Required output (`docs/PIPELINE_REVIEW.md`)
1. **Verdict in one paragraph:** keep, modify or replace the current pipeline, and why.
2. **Verified findings:** problems or risks you confirmed in this repository, each with the file and line, or the command and output, that shows it. Order them by severity.
3. **Recommended pipeline:** the design from question 9, with the reasoning for each choice.
4. **Ranked experiment plan:** at most 8 experiments, in order. For each, give the hypothesis, exactly what to change, the data and compute needed (GPU-hours and download GB on this machine), the success criterion measured on dev only, and what you'd do if it fails. Put the experiments most likely to improve real-world reliability first.
5. **Data acquisition plan:** what to download next, sizes, licences, and the order to download them in given about 2 MB/s.
6. **Risks and open questions:** what could make the app give confidently wrong answers, and what the owner must decide.
7. **Answers to questions 1–8:** briefly, referring back to the sections above instead of repeating them.

Be concrete and quantitative where possible, and state your confidence. Prefer a short, correct, well-argued recommendation over a long list of options.
