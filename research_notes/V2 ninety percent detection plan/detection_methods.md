# Generalizable speech deepfake detection of modern TTS (2024-2026): methods, checkpoints and a 14-hour Colab recipe

Scope: what can be trained or adopted in about 14 h on a Colab T4/L4/A100 to reach >=90% detection per family (Chatterbox, ElevenLabs v3, Fish Speech, Gemini TTS) at <=15% false alarms on independent English genuine speech. Research date: 2026-11-26. All EERs below are self-reported by the authors unless stated otherwise. None of the sources found test ElevenLabs v3, Fish Speech or Gemini TTS directly. The only direct hits are an older ElevenLabs (LibriFake) and Chatterbox ML (VoxENES 2026).

## 1. Architectures and back-ends: cross-dataset generalization

### Takeaway
Three things move cross-dataset EER far more than the choice of back-end: the size of the SSL front-end (300M -> 1B/2B), the breadth and diversity of training data, and fine-tuning the front-end. Among back-ends on XLS-R-300M, Nes2Net-X, XLSR-Mamba and SLS sit within about 2 EER points of each other on In-the-Wild (5.5-7.5%). Data diversity alone (AUDETER) drops XLS-R+SLS from 7.46% to 1.87% on In-the-Wild. Post-training a 1B/2B model on 74k hours (AntiDeepfake) reaches 1.2% on In-the-Wild, yet it still scores 25-28% EER zero-shot on Deepfake-Eval-2024. On 2025 LLM-era TTS, published pretrained detectors collapse to 29-58% EER.

### Cited Findings
- **Nes2Net / Nes2Net-X** (Liu et al., 2025): a lightweight nested back-end on SSL features with no dimensionality-reduction layer, 511k params, 58-91 MMACs, against AASIST's 447k params and 708 MMACs. Best (mean of 3 seeds) EERs: In-the-Wild 5.80 (7.06) for Nes2Net and 5.52 (6.60) for Nes2Net-X; ASVspoof 2021 DF 1.65 (1.91) for Nes2Net-X; ASVspoof 5 5.92% EER and minDCF 0.1535 for Nes2Net-X, against re-implemented AASIST at 6.08% EER. Code and pretrained models: https://github.com/Liu-Tianchi/Nes2Net — [Nes2Net arXiv 2504.05657](https://arxiv.org/html/2504.05657v2)
- Same paper, comparison on In-the-Wild: Mamba back-end 6.71%, SLS 7.46% (mean 8.87); SLS has about 23.4M back-end params — [Nes2Net arXiv 2504.05657](https://arxiv.org/html/2504.05657v2)
- **XLSR-Mamba** (Xiao & Das, Nov 2024): dual-column bidirectional Mamba on wav2vec2/XLS-R. In-the-Wild EER 6.71%, against XLSR+SLS 7.46% (reported by the SLS authors), XLSR+Conformer+TCM 7.79% and XLSR+Conformer 8.42% — [arXiv 2411.10027](https://arxiv.org/abs/2411.10027)
- **Data beats back-end, AUDETER** (Wang et al., Sep 2025): 4,682 h of synthetic audio from 21 synthesis systems, 10 of them recent TTS. XLS-R+SLS trained on AUDETER (CrowdSource + US Congress subsets) reaches **1.87% EER on In-the-Wild**, against 7.46% for the official SLS weights; XLS-R+RawNet/AASIST ("XLR+R+A") goes from 10.46% to 2.03%. Without augmentation, the paper's text gives 5.05% (XLR+SLS) and 4.17% (XLR+R+A), but its Table 5 swaps the labels (XLR+R+A 5.05, XLR+SLS 4.17); check the table before quoting either. RawNet2 trained on the same data only reaches 27.13%, which the authors read as "sufficient learning capacity from large backbones is essential". Generating the dataset took about 2,000 GPU hours. Code: https://github.com/mike-qz-wang/AUDETER — [AUDETER arXiv 2509.04345](https://arxiv.org/abs/2509.04345)
- AUDETER also reports a LoRA variant: XLS-R frozen except LoRA, AASIST body at LR 1e-6, LoRA and classifier at 1e-5, 20 epochs, batch 128, balanced real/fake sampling, RawBoost, plus KL distillation to a stage-1 teacher (T=2.0, λ=0.3) — [AUDETER arXiv 2509.04345](https://arxiv.org/abs/2509.04345)
- **Scale of the SSL model, AntiDeepfake** (NII Yamagishi Lab, Jun 2025): post-training wav2vec2, HuBERT, MMS and XLS-R models (95M to 2.2B) on 56.4k h genuine and 18.3k h fake audio from 27 datasets, MLAAD and ASVspoof5 among them, on 8x H100. Zero-shot EER for XLS-R-2B: In-the-Wild 1.23% (with RawBoost) / 1.31% (without), ADD2023 4.67%, DEEP-VOICE 2.23%, **Deepfake-Eval-2024 27.76% / 25.76%**. After fine-tuning on Deepfake-Eval-2024 (6k steps, AdamW, LR 1e-6, RawBoost), XLS-R-1B reaches 8.28% EER on 50 s segments and 11.86% on 4 s, against AASIST 16.99% at 4 s — [arXiv 2506.21090](https://arxiv.org/html/2506.21090v4)
- **Modern LLM-era TTS (VoxENES 2026**, Sharma & Wang, Jul 2026): 7 TTS systems released or updated in 2025 (VoxCPM 1.5, Qwen3-TTS, GLM-TTS, FlashLabs Chroma, VibeVoice, CosyVoice 3, **Chatterbox ML**) and 3 VC systems (Seed-VC, OpenVoice v2, RVC v2). Real speech is LibriSpeech and VoxPopuli, 4 s clips at 16 kHz. Eight pretrained detectors, no fine-tuning. Overall EER: AST-ASVspoof5 **28.98% (best)**, Wav2Vec2-AASIST 39.16%, Wav2Vec2-Large 44.38%, RawNet2 47.03%, Wav2Vec2-ASVspoof5 51.53%, Wav2Vec2-DF 55.51%, AASIST2 57.86% (inverted predictions). Seed-VC: no detector below 41% EER. Dataset: kaggle.com/datasets/interspeech2712/voxenes-2026 — [arXiv 2607.11706](https://arxiv.org/abs/2607.11706)
- **Newer TTS, Jan 2026** (Singh et al.): EERs on Dia2 / Maya1 / MeloTTS. XLS-R-SLS 7.07 / 17.60 / 27.10; SSL-AASIST 9.18 / 19.57 / 23.25; Whisper(tiny)-MesoNet 27.20 / 35.95 / 17.05. A proprietary detector (UncovAI, from one of the authors) reports F1 above 0.98 on all three, which cannot be verified. Conclusion: no single paradigm works across TTS families, so the authors recommend multi-view or ensemble detection — [arXiv 2601.20510](https://arxiv.org/html/2601.20510v1)
- **Layer choice**: a layer-wise study of frozen Wav2Vec2, HuBERT and WavLM finds lower and intermediate layers more informative than top layers for deepfake detection — [arXiv 2502.03559](https://arxiv.org/pdf/2502.03559); a second study finds intermediate layers most useful out of domain and uses layer selection and fusion with MHFA pooling — [arXiv 2509.12003](https://arxiv.org/html/2509.12003v1)
- **Deepfake-Eval-2024** (Chandra et al., Mar 2025): 56.5 h of in-the-wild audio from 88 websites in 52 languages. Off-the-shelf open-source audio detectors lose about 48% AUC relative to older benchmarks. The best commercial audio detector scores accuracy 0.89, AUC 0.93, F1 0.87; the authors attribute the commercial advantage mainly to training-data curation — [arXiv 2503.02857](https://arxiv.org/abs/2503.02857)

### Inferences
- The team's back-end (XLS-R-300m fine-tuned) is near the SOTA architecture class for its size. Swapping to Nes2Net, Mamba or SLS would plausibly gain only 1-2 EER points on old benchmarks, which will not close a gap from 55% to 90% detection on ElevenLabs and Gemini. What changes results in the sources is training data that includes the target generation families, plus a larger front-end (1B/2B).
- VoxENES shows that models trained only on ASVspoof-era data fail on 2025 TTS (Chatterbox ML included). The team's ~90% on Chatterbox therefore probably comes from IndieFake or in-family data, not from general generalization.
- WavLM, Whisper-encoder and codec-based detectors had no strong 2025-2026 cross-dataset evidence in what I found. Whisper-tiny features underperformed XLS-R on two of the three new TTS systems in arXiv 2601.20510.

### Gaps
- I found no numeric In-the-Wild, MLAAD or ASVspoof5 results for MoE-of-SSL-layers, WavLM+AASIST, deepfake-whisper-features or codec-based detectors in this pass; I did not fetch those papers.
- No source reports results on ElevenLabs v3, Fish Speech or Gemini TTS.
- Per-system EERs for Chatterbox ML in VoxENES appear only in a figure; the exact numbers could not be extracted.

## 2. Open pretrained checkpoints downloadable now

### Takeaway
Two families of strong, broadly trained checkpoints are downloadable now: **AntiDeepfake** (NII; wav2vec2, MMS, XLS-R and HuBERT from 95M to 2B; CC BY-NC-SA 4.0) and **DF Arena 1B** (Speech-Arena-2025; non-commercial; a transformers pipeline). Both take 16 kHz mono input, and both are non-commercial. Neither has published results on ElevenLabs v3, Fish or Gemini. Use them as zero-shot baselines and as initialisation or teachers, but measure them on the team's own dev panel first.

### Cited Findings
- **AntiDeepfake** checkpoints (W2V-Small 95M, W2V-Large 317M, MMS-300M, MMS-1B, XLS-R-1B, HuBERT-XL, XLS-R-2B): Hugging Face org `nii-yamagishilab` (for example `nii-yamagishilab/wav2vec-large-anti-deepfake-nda`), plus Zenodo doi 10.5281/zenodo.15580542 and code at github.com/nii-yamagishilab/AntiDeepfake. Input is 16 kHz mono, normalised; training crops were 10-13 s and testing used full duration. Trained on MLAAD (378 h fake), ASVspoof5, SpoofCeleb, CodecFake, CVoiceFake, vocoded VoxCeleb2 and LibriTTS, and genuine MLS, VoxCeleb2, FLEURS and LibriTTS — [arXiv 2506.21090](https://arxiv.org/html/2506.21090v4)
- AntiDeepfake licence: model weights are CC BY-NC-SA 4.0, research and education use only — [HF model card](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake-nda). Third-party mirrors: [SpeechAntiSpoofingBenchmarks/MMS-1B-AntiDeepfake](https://huggingface.co/SpeechAntiSpoofingBenchmarks/MMS-1B-AntiDeepfake) and [Wav2Vec2-Large-AntiDeepfake](https://huggingface.co/SpeechAntiSpoofingBenchmarks/Wav2Vec2-Large-AntiDeepfake). The "NDA" variants appear to be versions without data augmentation, which is my reading of the naming and was not verified.
- **DF Arena 1B v1** (`Speech-Arena-2025/DF_Arena_1B_V_1`): trained on ASVspoof 2019 and 2024, Codecfake, LibriSeVoc, DFADD, CtrSVDD, SpoofCeleb, MLAAD and EnvSDD. Non-commercial licence. Usage: `pipeline("antispoofing", model=..., trust_remote_code=True)` on 16 kHz audio, with labels `spoof` and `bonafide`. Reported EERs: In-the-Wild 0.91%, ASVspoof2021 DF 1.75%, ASVspoof 2024 (ASVspoof5) 17.25%, Codecfake 8.37%, FakeOrReal 2.92%, ADD2022 track 1 22.21%, average 5.92%, pooled 9.52%. Related paper: arXiv 2603.06164 (Mar 2026, compact SSL backbones, RAPTOR) — [HF card](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1)
- **VeriSpoof-Base** (an audio-LLM detector trained on 9 corpora including MLAAD): pooled EER 4.43% and mean 4.28% across 14 Speech-DF-Arena test sets (self-reported) — [HF card](https://huggingface.co/AustinXiao/VeriSpoof-Base)
- **Nes2Net** pretrained models: github.com/Liu-Tianchi/Nes2Net — [arXiv 2504.05657](https://arxiv.org/html/2504.05657v2)
- **DeepFense** (2026): a modular framework that releases many recipes and checkpoints on Hugging Face — [arXiv 2604.08450](https://arxiv.org/pdf/2604.08450)
- Speech DF Arena is a leaderboard that reports EER, pooled EER, accuracy and F1 across test sets, and notes high out-of-domain EER for many systems — [arXiv 2509.02859](https://arxiv.org/html/2509.02859v1)

### Inferences
- The cheapest high-value experiment (under 1 h on a T4) is to run AntiDeepfake MMS-1B or XLS-R-1B and DF Arena 1B zero-shot on the team's dev panel, then report per-family detection at 15% false alarms on the genuine readers. If either already beats 55-58% on ElevenLabs and Gemini, it is the best starting point for fine-tuning.
- The licences are non-commercial. That matters if the app is ever deployed commercially; check against the PRD.

### Gaps
- No checkpoint card or paper reports scores on ElevenLabs v3, Gemini TTS, Fish Speech or Chatterbox.
- The licence of the AntiDeepfake GitHub code and the exact architecture of DF Arena 1B were not confirmed.
- SpeechFake-trained checkpoints: the dataset is released at github.com/YMLLG/SpeechFake, but I did not confirm released model weights.

## 3. Multi-system training data vs. held-out systems; how much in-family data; augmentation

### Takeaway
Training on many generators (MLAAD, SpeechFake, AUDETER) clearly improves held-out generalization, but held-out modern families stay well below 90% without in-family data. Small amounts of in-family data are very effective. On ElevenLabs (LibriFake), 50-100 fake clips bring EER from about 40% to 1-5% with suitable adaptation (ADD-GP), and plain SSL-AASIST fine-tuning reaches about 11% with 100 clips. Augmentation helps robustness but has inconsistent effects on raw EER.

### Cited Findings
- **ADD-GP** (Glazer et al., Interspeech 2025; LibriFake = YourTTS, WhisperSpeech, VALL-E X and F5-TTS as seen systems, **ElevenLabs held out**). Front-end: XLS-R frozen except its last block. Before adaptation, ElevenLabs EER is 40.31% for ADD-GP and 42.48% for fine-tuned SSL-AASIST. Few-shot EER on ElevenLabs:
  - ADD-GP+MixPro: 7.89 / 4.86 / 3.31 / 1.14 / 0.54 at 5 / 10 / 20 / 50 / 100 shots.
  - SSL-AASIST full fine-tuning: 16.17 / 15.55 / 12.56 / 11.73 / 10.97.
  - SSL-AASIST LoRA fine-tuning: about 20% throughout.
  - After 100 ElevenLabs shots, ADD-GP keeps 0.1% EER on the seen TTS; full fine-tuning rises to 2.3%.
  - Code and data: github.com/NetaGlazer/ADD-GP.
  — [arXiv 2505.23619](https://arxiv.org/html/2505.23619v1)
- Meta-learning: about 96 samples from an unseen dataset cut In-the-Wild EER from about 21.67% to 10.42% — [arXiv 2410.20578](https://arxiv.org/html/2410.20578v1)
- Prototypical-network few-shot adaptation with 10 samples gives up to 32% relative EER reduction — [arXiv 2508.13320](https://arxiv.org/html/2508.13320v1)
- **SpeechFake** (ACL 2025): over 3M fake samples and over 3,000 h from 40 tools. Models trained on it "significantly outperform" those trained on existing datasets on unseen deepfakes. Generation method and language diversity matter most; speaker variation has a minimal effect. Baselines are AASIST and W2V+AASIST — [arXiv 2507.21463](https://arxiv.org/html/2507.21463v1). A later paper describes the training set as 30 methods and 1,163 h, which conflicts with these figures — [arXiv 2606.08038](https://arxiv.org/pdf/2606.08038.pdf)
- **AUDETER** within-dataset held-out test: training on 5 TTS + 5 vocoders and testing on the others gives low EERs (Table 5 "Unseen" column, XLR+SLS 0.010-0.026 as fractions). Cross-domain In-the-Wild EER rises with less diverse training: TTS-only 11.59% and vocoder-only 6.09% (XLR-SLS), against 1.87% for all subsets with augmentation — [arXiv 2509.04345](https://arxiv.org/abs/2509.04345)
- **Frozen-probe audit of MLAAD v9 English** (Pagon et al., Jun 2026): a logistic-regression probe on frozen XLS-R (best at layer 18) trained on ASVspoof2019 LA gets 88.84% macro-average per-generator spoof recall on 84 MLAAD English generators. Generator difficulty is predicted by the relative Fréchet margin between source-real and source-fake regions. Some "unseen" test generators sit close to already-seen spoof structure, so held-out results may overstate novelty. MLAAD evaluation there is spoof-only (no false-alarm side) — [arXiv 2606.26384](https://arxiv.org/abs/2606.26384)
- **Augmentation**: AntiDeepfake with and without RawBoost gives In-the-Wild 1.23 vs 1.31 (small gain), ADD2023 4.67 vs 6.84 (large gain) and Deepfake-Eval-2024 27.76 vs 25.76 (RawBoost worse) — [arXiv 2506.21090](https://arxiv.org/html/2506.21090v4). AUDETER: augmentation cuts In-the-Wild EER from about 4-5% to about 2% and models are more robust to white noise, telephone and background noise — [arXiv 2509.04345](https://arxiv.org/abs/2509.04345). VoxENES: post-processing (MP3, AAC, noise, 8 kHz resampling, speed, peak normalisation) changes EER "in a highly non-uniform and occasionally counter-intuitive manner"; for example, white noise lowered EER for one detector — [arXiv 2607.11706](https://arxiv.org/abs/2607.11706). Nes2Net used RawBoost on ASVspoof 2019/2021 and MUSAN+RIR on ASVspoof5 — [arXiv 2504.05657](https://arxiv.org/html/2504.05657v2)

### Inferences
- To reach 90% at 15% false alarms on ElevenLabs v3 and Gemini, the most evidence-backed lever is in-family training data. A few hundred to a few thousand clips per family, from many speakers and texts, is plausibly enough when combined with a strong pretrained front-end. ADD-GP shows that even 50-100 ElevenLabs clips can give EER near 1-5% on a matched LibriSpeech-style test.
- That ADD-GP result is on matched LibriSpeech-derived genuine speech. The team's gap between readers suggests genuine-side diversity is the binding constraint as much as fake-side coverage.
- Hold out speakers and texts, not just clips, when measuring in-family detection; the Fréchet-margin finding means held-out generators near seen ones flatter results.
- RawBoost: keep it but don't expect it to fix unseen-family generalization; codec and MP3 augmentation matters if the genuine test audio is compressed.

### Gaps
- I found no paper that gives a dose-response curve (number of clips per family against detection at a fixed false-alarm rate) for commercial LLM-era TTS beyond ADD-GP's ElevenLabs (older model version).
- The original MLAAD paper's cross-training numbers were not fetched in this pass.

## 4. Calibration across speakers and recording conditions; setting thresholds at a target false-alarm rate

### Takeaway
EER hides deployment failure: thresholds do not transfer across domains. A threshold calibrated on ASVspoof 2019 LA rejects 78.7% of genuine In-the-Wild speech. Monotonic score normalisations (z-norm, temperature scaling) cannot change EER or within-domain ranking, and AS-norm with an unlabeled target cohort made things worse. The evidence-backed fixes are more diverse bona fide training data and setting the threshold on a held-out genuine set that matches the deployment conditions, reported with HTER or false alarms at a transferred threshold.

### Cited Findings
- Zhou & Wang (Jun 2026) audit a frozen SSL-AASIST detector trained on ASVspoof 2019 LA. In-domain EER is 0.21%. With the LA threshold transferred to In-the-Wild, HTER is 39.5% and **78.7% of bona fide speech is rejected**, while the oracle In-the-Wild EER is 11.2%. They prove that any strictly increasing transform (z-norm, temperature or shift calibration, mean alignment of embeddings under a frozen linear head) leaves EER unchanged. AS-norm with an unlabeled cohort raised In-the-Wild EER from 11.2% to 60.2%. Pseudo-label calibration cut In-the-Wild HTER by 38% relative but degraded to 50% HTER on ASVspoof 2021 DF, where the spoof prior is 96%. None of the seven unlabeled corrections improved EER by more than 1% relative. Recommendation: report HTER at a transferred threshold alongside EER — [arXiv 2606.21584](https://arxiv.org/abs/2606.21584)
- Good discrimination can coexist with poor calibration, so report actDCF at a fixed threshold and ECE alongside EER — [arXiv 2610.16390](https://arxiv.org/pdf/2610.16390)
- Speaker verification analogue: irrelevant calibration causes serious issues under unknown acoustic conditions; compares logistic regression, magnitude estimation networks and adaptive s-norm — [arXiv 2203.15106](https://arxiv.org/pdf/2203.15106.pdf)
- Spoof-score normalisation to counter intra- and inter-speaker variation, tuning to a single threshold (2022) — [hrcak.srce.hr/412491](https://hrcak.srce.hr/file/412491)
- Calibrate before fusing scores (Interspeech 2024) — [arXiv 2406.10836](https://arxiv.org/html/2406.10836v2)
- **Diverse bona fide data**: AntiDeepfake uses 56k h of genuine speech (MLS in 8 languages, VoxCeleb2, CNCeleb2, FLEURS, LibriTTS) against 18k h fake — [arXiv 2506.21090](https://arxiv.org/html/2506.21090v4). AUDETER attributes cross-domain gains to "diverse real and deepfake patterns" — [arXiv 2509.04345](https://arxiv.org/abs/2509.04345). SpeechFake finds speaker variation has minimal effect on the fake side — [arXiv 2507.21463](https://arxiv.org/html/2507.21463v1)
- Personalised or speaker-conditioned detection: ADD-GP gets 2.59% EER out of domain and 0.39% with 5 shots per speaker on LibriSpeech, but needs deepfakes of the target speaker — [arXiv 2505.23619](https://arxiv.org/html/2505.23619v1)

### Inferences
- The 0-53% false-alarm spread across readers comes from the genuine distribution shifting. Fix it in the order the evidence supports:
  1. Add many genuine sources to training (thousands of speakers, varied microphones, rooms and codecs, read and conversational speech). Common Voice, MLS, VoxCeleb and FLEURS-type data are all used by AntiDeepfake.
  2. Pick the 15% false-alarm threshold on a held-out, speaker-disjoint genuine pool that is as diverse as the deployment population.
  3. Report the worst-reader false-alarm rate, not only the pooled rate.
- Per-speaker or per-recording score normalisation without labels is not supported by the 2026 audit (AS-norm hurt). Only try it if validated on reader-disjoint data.
- Fixed crop length and loudness normalisation at inference (both classes treated the same) should reduce condition-driven variance; this is an inference, not a measured result.

### Gaps
- I found no paper that targets per-speaker genuine score variance in anti-spoofing with a measured fix (for example, domain-adversarial speaker removal) and reports worst-speaker false alarms.

## 5. Fine-tuning time and compute; frozen-feature fast alternative

### Takeaway
Published training-cost numbers are sparse. Large post-training runs use 8x H100 (AntiDeepfake); typical recipes fine-tune XLS-R-300M at a small LR (1e-6 to 1e-5) with 4 s crops for up to 75-100 epochs on ASVspoof-sized data. A frozen-feature linear probe on intermediate XLS-R layers is a validated fast baseline: 88.8% per-generator spoof recall on MLAAD English from ASVspoof2019 training alone. That makes feature caching plus a linear, MLP or GP head a realistic sub-hour loop on Colab for sweeping data mixes.

### Cited Findings
- AntiDeepfake post-training: 8 NVIDIA H100s, 10-13 s crops, maximum batch duration 100 s (50 s for 2B); no GPU-hour totals given. Fine-tuning on Deepfake-Eval-2024: 6k steps at LR 1e-6 — [arXiv 2506.21090](https://arxiv.org/html/2506.21090v4)
- Nes2Net recipes: ASVspoof 2019/2021 at about 4 s, Adam, LR 2.5e-7 to 1e-6, batch 12, 100 epochs; CtrSVDD with WavLM, batch 34, 75 epochs; ASVspoof5 with early stopping after 5 epochs without dev improvement. Training GPU not stated — [arXiv 2504.05657](https://arxiv.org/html/2504.05657v2)
- AUDETER LoRA recipe: 20 epochs, batch 128, Adam, LoRA LR 1e-5, AASIST LR 1e-6 — [arXiv 2509.04345](https://arxiv.org/abs/2509.04345)
- Frozen probe: logistic regression on frozen XLS-R-300M layer 18 (64,000 samples = 4 s at 16 kHz) gets 88.84% macro per-generator spoof recall on MLAAD v9 English (84 generators) — [arXiv 2606.26384](https://arxiv.org/abs/2606.26384). Linear probing of frozen SSL models (with WavLM ensembles) was also used in a 2024 challenge system — [arXiv 2408.07414](https://arxiv.org/pdf/2408.07414). A truncated frozen wav2vec2 plus linear classifier is used for an in-browser detector — [arXiv 2606.30780](https://arxiv.org/pdf/2606.30780)
- ADD-GP: frozen XLS-R except the last block, GP head, batch 80, very fast few-shot updates (compute not reported) — [arXiv 2505.23619](https://arxiv.org/html/2505.23619v1)

### Inferences (rough estimates, not measured)
- On an A100, an XLS-R-300M forward and backward pass on 4 s clips plausibly handles a few hundred clips per second for inference. So caching layer-wise pooled features for about 100k clips is plausibly about 15-30 min for 300M and 1-1.5 h for 1B models. Full fine-tuning of 300M on about 100k 4 s clips is plausibly about 1-2 h per epoch on an A100 and 3-5x slower on a T4. These are my estimates; no source measured them.
- Recommended 14 h plan, inference:
  1. **About 1 h**: zero-shot AntiDeepfake (MMS-1B, XLS-R-1B, maybe XLS-R-2B on A100) and DF Arena 1B on the dev panel. Report per-family detection at 15% false alarms and the worst reader's false-alarm rate.
  2. **About 2 h**: cache multi-layer features (mean and std pooling per layer) from the best front-end over training genuine data (diverse sources) and fakes (MLAAD English, ASVspoof5, IndieFake and every in-family clip available). Sweep logistic, MLP and GP heads and layer subsets, with splits that hold out speakers and texts.
  3. **About 6-8 h**: fine-tune the best pretrained model end to end (or with LoRA or last-block tuning), with a balanced sampler across families and genuine sources, RawBoost plus codec augmentation, small LR (1e-6 to 1e-5), 4-10 s crops. Optionally distil from an ensemble.
  4. **About 1-2 h**: calibrate the threshold on held-out, reader-disjoint genuine audio; ensemble two or three diverse models; final per-family report.

### Gaps
- No source gives measured wall-clock times for XLS-R-300M+AASIST fine-tuning on Colab T4, L4 or A100.

## 6. Shortcut pitfalls (silence, duration, loudness, sample rate, corpus or channel) and controls

### Takeaway
Shortcuts are documented. ASVspoof2019 leaks labels through silence duration and bitrate, audio-visual sets through leading silence (a silence-only classifier is over 98% accurate), and a detector trained on ASVspoof-era data produced inverted predictions on modern TTS, which the authors attribute to learned silence or channel artefacts. For MLAAD vs M-AILABS specifically, I found no published shortcut audit; it needs measuring locally. MLAAD fakes come from TTS models, while M-AILABS genuine clips are audiobook recordings, so channel, silence and duration may differ systematically.

### Cited Findings
- Leading-silence shortcut in FakeAVCeleb and AV-Deepfake1M: a silence-only classifier is over 98% accurate. The same paper notes prior work showing ASVspoof19 leaks labels through silence duration and bitrate (CVPR 2025) — [arXiv 2412.00175](https://arxiv.org/html/2412.00175v3)
- VoxENES 2026: AASIST2 gives inverted predictions (57.86% EER), attributed to "dataset-specific artifacts, such as silence patterns or channel characteristics, that are reversed or absent in the target domain". Control used: every clip, genuine and fake, standardised to 16 kHz mono with a 4 s truncate or zero-pad, so padding cues are shared across classes — [arXiv 2607.11706](https://arxiv.org/abs/2607.11706)
- MLAAD is built on the M-AILABS corpus: 687.4 h synthesized by 140 TTS models in v9. The frozen-probe audit evaluates MLAAD as spoof-only because pairing with genuine audio from another source confounds the result; audio is resampled to 16 kHz and cropped or padded to 4 s — [arXiv 2606.26384](https://arxiv.org/abs/2606.26384)
- AntiDeepfake controls: resample to 16 kHz mono and normalise amplitude, use random 10-13 s crops, and evaluate FakeOrReal in both original and "norm" (loudness- and duration-normalised) versions. EER is 0.67 on original and 0.97 on norm, so normalisation raises EER slightly — [arXiv 2506.21090](https://arxiv.org/html/2506.21090v4)
- Frozen-probe audit: general-purpose SSL linear probes approach bespoke detectors, so benchmark scores may reflect general modality differences, for example corpus or recording differences, rather than forensic cues — [arXiv 2606.26384](https://arxiv.org/abs/2606.26384)

### Inferences
- Before training, run trivial-feature baselines (logistic regression on leading and trailing silence, duration, RMS or LUFS, spectral rolloff and bandwidth, original sample rate) on every real and fake pair (MLAAD vs M-AILABS, IndieFake, in-family sets vs their genuine sources). An AUC well above 0.5 means a confound to remove.
- Controls the evidence supports:
  - Use the same pipeline for both classes: resample to 16 kHz, convert to mono, normalise loudness, and trim silence with VAD the same way on both sides.
  - Use fixed random crops of the same length.
  - Apply codec and bandwidth augmentation to both classes.
  - Take genuine data from many corpora, so that no single corpus equals the "real" label.
- Where possible, pair fakes with genuine speech from the same speaker and text, as in ADD-GP's LibriFake (LibriSpeech-derived) design.

### Gaps
- I found no published quantitative audit of silence, duration or loudness shortcuts in MLAAD vs M-AILABS.
- Sample-rate leakage in commercial TTS (for example 22.05 or 24 kHz output against 16 or 44.1 kHz genuine recordings) is not documented in what I found.
