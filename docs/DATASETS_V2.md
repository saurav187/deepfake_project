# Datasets for detector v2: modern TTS (research, 2026-11-23)

**Why:** release v1 misses modern TTS. It was trained on ASVspoof 5 and IndieFake, whose generators are older. The VoxENES 2026 paper shows the same problem across the field: 8 public detectors trained on older data score **29–56% EER** on LLM-era TTS (Qwen3-TTS, CosyVoice 3, VibeVoice, Chatterbox, Seed-VC, …). Training data from newer generators is the main fix.

## Storage first

| Dataset | Full size | What to take |
|---|---:|---|
| SpeechFake | not stated (3M+ clips, ~3,000 h) | English/bilingual subset, ~30–60 GB |
| MLAAD | 183 GB | English plus a few languages, ~20–40 GB |
| IndicSynth | 845 GB (Hindi ≈ 206k clips) | a Hindi slice, ~10–20 GB |
| Real Hindi speech (IndicSUPERB / Kathbath) | varies | a matching amount, ~10–20 GB |
| VoxENES 2026 (test only) | 53,628 clips | all of it |
| Deepfake-Eval-2024 (test only) | 56.5 h, 1,820 clips | all of the audio |

**Plan: an added HDD holds the new datasets.** The training PC's Linux partition has only about 5 GB free, so nothing large goes there.
- Format the HDD as **ext4** if only Linux uses it. It's faster and safer for millions of small files than exFAT or NTFS.
- An HDD is slow at reading many small files in random order, which is what training does. To keep training fast, the training run should read the audio packed into a few large files, or with a fixed order per epoch. Packing is part of the v2 training work, not the downloads.

## The datasets

### For training (generators the model has never seen)

**1. SpeechFake (ACL 2025): top priority**
- 3M+ fake clips, about 3,000 h, 46 languages, 40 tools. The public release has 30 open-source tools, including **CosyVoice, GPT-SoVITS, Fish Speech, Seed-VC, OpenVoice, StyleTTS2, ChatTTS, MeloTTS, Parler-TTS, FireRedTTS, EdgeTTS and TorToiSe**, plus vocoders. The 10 API-service tools are not released.
- License CC BY 4.0. Real speech comes from VCTK, LibriTTS, AISHELL and CommonVoice; check whether the real audio is in the release.
- Download from ModelScope, no sign-up: `pip install modelscope`, then `modelscope download --dataset inclusionAI/SPEECHFAKE --local_dir <drive>/speechfake`. Use `--include` to download only some folders; list the file tree first.
- <https://github.com/YMLLG/SpeechFake> · <https://arxiv.org/abs/2507.21463>

**2. MLAAD (v8/v9): many TTS architectures and languages**
- About 140 TTS models, 51 languages, 687 h, 183 GB. Fakes are in `fake/<language>/<model>/` with a `meta.csv` per model.
- License CC BY-NC 4.0 (non-commercial; fine for a college project). Gated: log in on Hugging Face and accept the terms.
- The real audio is **M-AILABS** (downloaded separately); the authors recommend pairing them.
- Download: `pip install -U huggingface_hub`, `hf auth login`, then `hf download mueller91/MLAAD --repo-type dataset --include "fake/en/*" --local-dir <drive>/mlaad`. Check the exact language folder names in the Files tab first.
- <https://huggingface.co/datasets/mueller91/MLAAD> · <https://arxiv.org/abs/2401.09512>

**3. IndicSynth: Hindi and 11 other Indian languages**
- 4,000+ h of synthetic speech, made with XTTS-v2, VITS and FreeVC24. About 206k Hindi clips.
- License CC BY-NC 4.0, academic use. The real speech is from **IndicSUPERB** and must be downloaded separately.
- Download: `load_dataset("vdivyasharma/IndicSynth", name="Hindi", split="train", streaming=True)`. Stream and save only part of it, or use `hf download` with `--include` on the Hindi folder.
- <https://huggingface.co/datasets/vdivyasharma/IndicSynth>

**Important: add real Hindi speech too** (IndicSUPERB / Kathbath, from AI4Bharat). If every Hindi clip in training is fake, the model learns "Hindi = fake" and will flag every real Hindi speaker. Real and fake must be balanced within each language. The audit gate checks the same kind of shortcut for formats today.

**Optional: CodecFake** (Interspeech 2024, VCTK, CC BY 4.0, `rogertseng/CodecFake` on Hugging Face). It covers neural-codec speech, the technology behind many LLM-era TTS systems. It's large (708k clips), so take a subset.

### For testing only: never train on these

**4. VoxENES 2026**
- 53,628 clips in English and Spanish. TTS: VoxCPM 1.5, Qwen3-TTS, GLM-TTS, Chroma, VibeVoice, CosyVoice 3 and Chatterbox ML. VC: Seed-VC, OpenVoice v2 and RVC v2. It also has 10 post-processing conditions.
- Download from Kaggle (free account plus API token): `kaggle datasets download interspeech2712/voxenes-2026`.
- CosyVoice, Seed-VC and OpenVoice also appear in SpeechFake, so report results separately for those three and for the generators that are **unseen** in training. The unseen group is the honest number.
- <https://arxiv.org/abs/2607.11706>

**5. Deepfake-Eval-2024**
- Real deepfakes that circulated online in 2024: 1,820 audio clips (1,110 real, 710 fake), 56.5 h, 42 languages. This is the closest public match to what an app user would upload.
- <https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024> (accept the terms) · <https://github.com/nuriachandra/Deepfake-Eval-2024>

**6. Your own set** (made by you): ~50 TTS fakes and ~50 real clips from the tools and languages you care about. Keep it out of training.

## Rules that keep the results honest

- Split by **speaker** and, where possible, hold out whole **generators** for testing, so the model is tested on tools it never saw.
- Balance real and fake within each language and source; the audit gate must pass.
- Choose the model and thresholds on dev data only, then evaluate once on VoxENES, Deepfake-Eval-2024, In-the-Wild and your own set, as release v2.
- All of these are for research use only (some are non-commercial). Don't commit audio to Git.

## Codex prompt (for downloading at college)

```text
Download training and test datasets for an audio deepfake detector. Read docs/DATASETS_V2.md first; it lists each dataset, its licence and its role.

Rules:
- Ask me for the destination drive/folder (call it $DATA) before downloading anything, and check its free space with df -h. Never fill a disk past 90%.
- Download only the subsets in docs/DATASETS_V2.md, with the size targets there. Before each download, list the remote file tree (hf api / modelscope / kaggle), compute the size of what you'll fetch, show me, and wait for OK.
- Gated datasets (MLAAD, Deepfake-Eval-2024) and Kaggle need my login; tell me exactly what to run (hf auth login, ~/.kaggle/kaggle.json).
- Resumable downloads only (hf download, modelscope download, kaggle). On a stall, retry; don't restart from zero.
- Don't modify detector/, deepfake_model/ or app/. Don't commit anything. Don't delete anything outside $DATA.
- Don't convert or re-encode audio; keep the original files.

Order: 1) SpeechFake (English/bilingual subset), 2) MLAAD fake/en plus 2-3 more languages, plus the matching M-AILABS real audio, 3) IndicSynth Hindi slice (~10-20 GB), plus real Hindi speech from IndicSUPERB/Kathbath of similar size, 4) VoxENES 2026 (all), 5) Deepfake-Eval-2024 audio (all), 6) optional CodecFake subset.

For each dataset write $DATA/<name>/DOWNLOAD.md with: source URL, licence, exact command used, file count, total size, real/fake counts if metadata says so, and a sha256 of each archive. Finally print one table of everything downloaded and anything that failed or needed a decision.
```
