You have full freedom to redesign this audio-deepfake detection pipeline, including radical changes, and then build the redesign. A previous review (`docs/PIPELINE_REVIEW.md`) recommended modifying the current design. This time, assume nothing is fixed. Problem framing, data, preprocessing, model, training objective, evaluation and serving are all open. The goal is the most reliable detector this project can realistically achieve, not the smallest change.

## The goal
A web app (already built in `app/`) takes an uploaded recording — often a WhatsApp voice note, MP3, M4A, or audio from a social-media video — and returns a verdict (likely real / uncertain / likely fake), a probability and a per-segment timeline. The detector plugs into the app through the interface in `docs/PRD_deepfake_audio_app.md`, section 7. You may propose changing that interface, but say exactly what the app would need to change. Users may be in India, so Hindi, Indian-accented English and code-switching matter. What counts is reliability on real-world, compressed audio from generators the model never saw — especially not falsely accusing genuine speakers — not benchmark scores.

## Constraints
- Hardware: one RTX 5060 (8 GB VRAM), 12 CPU cores, 14 GB RAM (about 6 GB usually free), about 17 GB of free disk.
- Network: Zenodo gives about 2 MB/s in total; Hugging Face is faster. 1 GB takes about 8 minutes from Zenodo.
- Software: `venv/` (Python 3.14, torch 2.15 + CUDA 13.2, transformers 5.24, librosa, soundfile, soxr, scikit-learn), and system ffmpeg 9.
- An academic project, which may become public. Flag any licence that blocks public or commercial use.

## Read first, and verify rather than trust
- `docs/PIPELINE_REVIEW.md`: the previous review. Challenge it as well; it is not ground truth.
- `deepfake_model/REPORT.md`, `deepfake_model/docs/CALL_CHANNEL_PROBE.md`, and `deepfake_model/research/*/README.md`: legacy findings.
  - Leaked historical metrics.
  - File format alone predicted the IndieFake label at AUC 0.95.
  - Channel augmentation fixed 8 kHz false alarms but not transfer.
  - ASVspoof 2021 LA AUC was about 0.62.
  - In-the-Wild AUC was 0.839 for the audited model, and near chance for the native-format models.
- `detector/`: the current v2 pipeline (see `detector/README.md`).
  - Shared decode, trim and loudness normalisation.
  - Manifest and audit, with dedupe, speaker splits, format-stratum balancing and gates.
  - A frozen WavLM / XLS-R encoder, a learned layer mix and an attentive-pooling head.
  - Codec, gain and noise augmentation.
  - A balanced source × label sampler.
  - Dev-only selection.
  - Platt calibration with budget thresholds (`detector/calibrate.py`).
- Changes made after the previous review, in response to it:
  - The ASVspoof 5 adapter skips the 1,184 dev enrollment-only files.
  - The audit gate checks format leakage per source and in the pooled mixture weighted like the sampler, and fails on a missing label.
  - Failed training loads are excluded from the loss.
  - Dev windows cover clips up to 16 s, with the codec applied before trimming and normalisation.
  - Runs record hashes of their inventory.
- Data on disk (all under the main checkout, git-ignored):
  - ASVspoof 5: train `flac_T_aa`, `flac_T_ab`, dev `flac_D_aa`, protocols. Some may still be downloading; trust only files with a `.ok` marker.
  - In-the-Wild: sealed test set.
  - IndieFake: `deepfake_model/datasets/indiefake`.
  - VA: sealed test set, 44 files.
  - Encoders in `detector/pretrained/`: WavLM Base+ and XLS-R 300M.

## Ground rules
- **Work in a separate git worktree on a new branch** so nothing running in the main checkout is disturbed:
  ```sh
  git worktree add ../deepfake_project_v3 -b pipeline-v3 detector-v2
  ```
  Build everything inside `../deepfake_project_v3`. In the main checkout (`~/deepfake_project`), you may only *read* data, manifests and pretrained weights by absolute path. Do not modify, move or delete anything there, and do not commit to `detector-v2` or `main`. If the main checkout has uncommitted changes, they won't be in your worktree: stop and tell me rather than copying them over by hand.
- **Background jobs are running in the main checkout:** dataset downloads (`pdownload.py`), and `detector/run_model_a.sh`, which trains the v2 model on the GPU. Do not stop or interfere with them.
  - Before using the GPU, check `nvidia-smi`. If a training process is running, use the GPU only for short smoke tests under 5 minutes and about 2 GB of VRAM, or use the CPU.
  - Keep RAM use under about 4 GB, and use at most 4 data-loader workers.
  - Do not download more than 3 GB in total without asking me first.
- **In-the-Wild and VA are sealed.** Never use their labels for any design choice, threshold, selection or "quick check". Code may read them only inside a final evaluation entry point that refuses to run unless a release is frozen and hashed.
- **Be honest about evidence.** Separate what you verified in this repository (give the file and line, or the command and its output) from what you believe from the literature (name the paper or benchmark and give your confidence). If you are unsure whether a model, dataset or result exists, say so; don't guess. A radical change needs a reason and a dev-only test that could prove it wrong. Novelty is not a reason.

## What to do
1. **Red-team the current system.** Before designing anything, list concretely how v2, as built, would give confidently wrong answers on real uploads. Order the list by likely harm, and back each item with evidence or a cheap test you actually ran on train or dev data.
2. **Redesign from first principles.** For each part, decide whether to keep, change or replace it, and why. Consider at least:
   - **Framing:** utterance classification versus localisation of synthetic regions, versus one-class or anomaly detection of genuine speech, versus abstention-first (selective prediction).
   - **Data:** what to train on, given the size and download limits; whether to generate fakes locally from genuine speech with open TTS and VC models (what counts as legally and ethically acceptable, and with whose consent); and how to group splits by speaker, session, source content and generator family.
   - **Model:** frozen SSL with a light head, SSL with adapters or LoRA, top-layer fine-tuning, an AASIST-style back end, a multi-layer (SLS-style) head, multiple encoders, or something else — under 8 GB of VRAM.
   - **Training objective:** BCE, one-class or OC-softmax, contrastive or supervised contrastive losses, generator-adversarial or domain-adversarial training against corpus identity, and mixup-style regularisation.
   - **Augmentation**, including what could destroy the evidence the model needs.
   - **Aggregation and windowing**, and one shared window iterator for calibration, evaluation and serving.
   - **Selection, calibration and abstention:** an independent calibration split, the prior, and how to behave on out-of-distribution input.
   - **Evaluation:** leave-one-generator-family-out, codec and channel stress tests, shortcut probes, speaker-clustered confidence intervals, and a protocol for a fresh confirmation set.
3. **Build it in the worktree.** Implement the redesigned pipeline end to end, as far as the constraints allow:
   - data → audit → training → selection → calibration → frozen release → sealed evaluation entry point → `inference.py` implementing the app interface (or your proposed interface).
   - Reuse v2 code where it is good; replace it where it isn't. Write tests for the parts most likely to be wrong silently: decode parity, window coverage, leakage gates, split grouping, calibration budgets, and release freezing and hash checks.
4. **Smoke-run it.** Run every stage on a small train/dev subset, within the GPU and RAM limits above, and show that each stage works. Don't run full training while the v2 training job is using the GPU. Instead, give the exact commands and the expected time for a full run.
5. **Design a head-to-head comparison with v2.** It must use dev data only, with fixed splits and seeds, and a decision rule fixed *before* any results exist. State the metrics, how significance is judged (paired, speaker-clustered bootstrap), and what result would make you recommend keeping v2 instead.

## Deliverables
- In the worktree: the code, tests and a `PIPELINE_V3.md` covering:
  1. a one-paragraph verdict: what changed, and why it should be more reliable
  2. the red-team findings, with evidence
  3. the design, part by part, marked kept, changed or replaced, with reasons
  4. what you built, and what you didn't build and why
  5. smoke-run outputs (commands and results)
  6. the comparison protocol versus v2, with its decision rule
  7. the data-acquisition plan (sizes, licences, order at about 2 MB/s, consent requirements for any locally generated or recorded data)
  8. the commands for full runs, with time estimates
  9. risks, open questions, and decisions I must make
- In your final message: a summary of the above, the test results, the branch name, and the worktree path.

Prefer a design that is simpler but well evidenced over one that is complicated but speculative. Where you're uncertain, build the cheap experiment that resolves it, rather than guessing.
