# Literature matrix: the 60 downloaded files in `papers/`

**60 files = 54 unique papers.** Six were downloaded twice; delete one copy of each:

| Keep | Delete (same paper) |
|---|---|
| 1st (`1-s2.0-S2214212626003017`) | 25th |
| 2nd | 26th |
| 3rd | 27th |
| 4th | 28th |
| 5th | 29TH |
| 6th | 30th |

(`33rd-getPDF.jsp` and `40th-getPDF.jsp` are real PDFs; rename them to `.pdf`. Note that two different papers are both numbered "40th".)

**Relevance** to this project (audio-only deepfake detection app, real-world uploads, Hindi/English):
- **H** = core: cite and discuss in depth
- **M** = supporting: cite briefly
- **L** = background only, or exclude (video-only or off-topic)

## A. Reviews and surveys (background and gaps)

| # | Paper (short) | Venue, year | Rel. | Use it for |
|---|---|---|---|---|
| 6 | Kanwal et al., *Deepfake detection: review of audio, visual and multimodal techniques* (>120 studies, 2021–25) | Applied Soft Computing 2026 | H | Big-picture background; real-world deployment challenges |
| 50 | Dixit & Kaur, *Review of audio deepfake detection techniques: issues and prospects* | Expert Systems 2023 | H | History of features and classifiers; open issues |
| 45 | *Audio deepfakes detection: literature review of voice-cloning fraud detection* | iSSSC 2025 | M | Fraud and voice-cloning motivation (problem statement) |
| 8 | Ul Ain et al., *Adversarial and generative AI-based anti-forensics in AV deepfake detection* | Information Fusion 2026 | M | Adversarial attacks on detectors (limitations section) |
| 52 | Mallinson et al., *A place for (socio)linguistics in audio deepfake detection* | Language & Linguistics Compass 2024 | M | Accent and language matter; argument for Hindi/Indian English |
| 51 | Chen & Goh, *Understanding deepfake research through topic modelling* | ASIS&T 2023 | L | One-line intro citation |

## B. Self-supervised (SSL) models: the family our model belongs to

| # | Paper (short) | Venue, year | Rel. | Key result / idea |
|---|---|---|---|---|
| 13 | Meng et al., *MSLA-XLS-R: enhancing hierarchical SSL representations* | Computer Speech & Language 2027 | **H** | Fine-tuned XLS-R; **ITW EER 6.83 %**. Lower-to-middle layers carry the most deepfake information, which matches our learned layer weights (layers 4, 6, 10) |
| 55 | Wang et al., *Gaussian attention for adapter-based fine-tuning of WavLM* | Int. J. Intelligent Systems 2026 | **H** | Adapters (parameter-efficient fine-tuning); **ITW 6.11 %**. Robust to codec/noise. A v2 option |
| 15 | *LHCC: cross-domain detection via acoustic + semantic consistency fusion* | Information Fusion 2026 | **H** | **ITW 7.1 %** cross-domain with a **frozen** XLS-R backbone plus ~40M task parameters, but **0.03 %** in-domain. In-domain numbers are inflated |
| 5 | Wang et al., *MWAADD: multi-feature wavelet attention* (XLS-R + emotion features) | Knowledge-Based Systems 2026 | H | **ITW 9.08 %**; plain SSL baselines about 24–30 % on ITW |
| 12 | *NSF-PINN: physics-informed source-filter decomposition + sharpness-aware training* (WavLM) | CVIU 2026 | H | **ITW 19.4 %**. Sharpness (SAM) linked to poor generalisation |
| 41 | Jin et al., *Wave-spectrogram cross-modal aggregation* | conference 2025/26 | H | **ITW 6.58 %**; combines waveform and spectrogram views |
| 37 | *HuBERT SSL features + ECAPA-TDNN with OC-softmax* | IEEE MLSP 2025 | M | OC-softmax loss (one-class) for unseen attacks; a v2 option |
| 16 | *Hybrid acoustic features + PCA-compressed WavLM* | Procedia CS 2026 | M | Small 106-D representation; reports in-domain ITW (EER 0.029), not cross-domain |
| 22 | *Transfer learning from a pretrained ASR model* | Procedia CS (KES) 2026 | M | ASR encoder as front-end (Whisper-style idea) |
| 35 | *Fooling the forgers: GAN augmentation + contrastive learning* | ICASSP 2025 | M | Data augmentation with HiFi-GAN fakes |

## C. Robustness and real-world conditions (closest to the app's problem)

| # | Paper (short) | Venue, year | Rel. | Why it matters |
|---|---|---|---|---|
| 42 | *Auxiliary multi-label training for robustness on AI-processed data* | ICASSP 2026 | **H** | Speech enhancement or neural codecs make **real** audio look fake: a false-alarm risk for WhatsApp-style uploads |
| 32 | *How to label resynthesized audio: dual role of neural audio codecs* | ICASSP 2026 | **H** | Codec-resynthesised speech: real or fake? Directly relevant to modern codec-based TTS and compressed uploads |
| 9 | Wu et al., *CLAD: robust detection against manipulation attacks with contrastive learning* | Knowledge-Based Systems 2026 | H | Noise, resampling and compression attacks; contrastive training |
| 18 | Febrinanto et al., *SIGNL: label-efficient detection via spectral-temporal graph learning* | Expert Systems w/ Applications 2026 | H | ITW 10.16 % (trained on CFAD); **also tests CosyVoice 2.0 (modern TTS)** |
| 3 | *Multi-cepstral feature integration (MFCC+LFCC+CQCC+GFCC) under domain shift* | CVIU 2026 | H | Lightweight, interpretable; cross-domain to ITW. A good small-model baseline |
| 36 | *BTS-E: breathing-talking-silence encoder* | ICASSP 2023 | H (critical) | Uses silence and breathing as evidence. **Contrast with our finding** that silence is a dataset shortcut |
| 7 | Zhan et al., *Spectral fusion + dual-branch Conformer* | Applied Soft Computing 2027 | M | Evaluates on ASVspoof 5 and ITW |
| 4 | *Splicing detection and localization for speech deepfakes* | CVIU 2026 | M | Partially fake audio; relates to our per-window timeline |
| 34 | *VeriVoice: depthwise frequency convolutions (GEAT)* | DECoN 2025 | M | **Built its own dataset with OpenVoice clones** (32 speakers); 99.66 % accuracy on its own data (the 24.32 % in the paper is a cited RawNet2 baseline on ASVspoof 2021, not an In-the-Wild result). A template for our own-dataset step |

## D. Handcrafted features and CNN/RNN classifiers (classic approach)

| # | Paper (short) | Venue, year | Rel. | Note |
|---|---|---|---|---|
| 23 | Tahaoglu et al., *Spectral features (LFCC/MFCC/CQCC) + ResNeXt* | Knowledge-Based Systems 2025 | M | Strong in-domain ASVspoof |
| 11 | Yang et al., *DAMRN: attention residual network, multi-scale cepstral features* | Signal Processing 2026 | M | Cepstral + attention |
| 46 | Huang et al., *ResNet with hybrid features + self-attention* | Expert Systems 2025 | M | |
| 1 | Mulani et al., *Hybrid learning for audio deepfakes and manipulation attacks* (PCEN spectrograms, hybrid loss) | J. Information Security & Applications 2027 | M | Indian authors (NIT Warangal) |
| 2 | Liu et al., *SGCADD: simplifying graph convolution* (RawGAT-ST) | Knowledge-Based Systems 2026 | M | Efficiency of graph models |
| 21 | Ouyang et al., *Adaptive reverse perturbation network* | Neurocomputing 2025 | M | Uses ASVspoof 5 |
| 20 | Celik & Akbal, *FakeSleuthNeXt: handcrafted features, no neural network* | Expert Systems w/ Applications 2026 | M | Fully interpretable, tiny |
| 14 | Yang et al., *Forensic detection using segmental speech features* (speaker-specific) | Forensic Science International 2026 | M | Forensic-phonetics angle |
| 54 | Shaaban & Yildirim, *Siamese CNN with StacLoss* | Engineering Reports 2025 | M | |
| 58 | Chaudhury et al., *Ensemble of MFCC/statistical/STFT models* | Scientific Reports 2026 | M | |
| 57 | Jellali et al., *MFCC + spectral contrast* on ITW | Multimedia Tools & Applications 2025 | M (caution) | 98.4 % accuracy **within** ITW (random split, likely speaker overlap) |
| 43 | *Acoustic features + Conformer/LSTM* on Fake-or-Real | ICACC 2024 | L | Accuracy only, one dataset |
| 40 | Chiddarwar et al., *Real-time detection of AI-generated audio* | ICTBIG 2024 | L | Real-time claim, weak evaluation |

## E. Low-resource languages, apps and efficiency (your app and mobile goals)

| # | Paper (short) | Venue, year | Rel. | Why |
|---|---|---|---|---|
| 44 | *RealACaller: real-time Arabic deepfake detection app for smartphones* (wav2vec2) | IEEE ICMI 2025 | **H** | **Closest to this project**: an SSL model, a non-English language, a real app |
| 17 | Singha et al., *Bangla deepfake audio detection with prosodic + spectral features* | Procedia CS 2026 | **H** | Indian-subcontinent language gap; **Hindi is still missing**, which is your gap |
| 48 | Falcón-López et al., *Optimal number of MFCCs for voice deepfake detection* | Expert Systems 2026 | H | Explicitly targets mobile and edge cost |
| 59 | *Fragment feature overlaid quantization model* | Int. J. Machine Learning & Cybernetics 2025 | H | Quantisation for efficiency, relevant to the ~50 MB goal |
| 31 | *DeepShield: multimodal detection + audio watermark verification* | IGNITE 2026 | L–M | System design; watermarking as a complementary idea |
| 56 | *Quantum gate optimisation in hybrid NNs for audio deepfakes* | Scientific Reports 2026 | L | Exploratory; not practical here |

## F. Video or multimodal: background only, or exclude from an audio-only review

| # | Paper (short) | Venue, year |
|---|---|---|
| 10 | Explainable multimodal framework (image/audio/video, SHAP/LIME/Grad-CAM) | Ain Shams Eng. J. 2026 (XAI ideas only) |
| 19 | AV asynchrony detection | Information Fusion 2026 |
| 24 | ART-AVDF articulatory AV detection | CVIU 2024 |
| 33 | Lightweight multimodal late-fusion 4-class | ICSCCC 2026 |
| 38 | AV feature fusion + temporal attention | ICBAIE 2025 |
| 39 | Attention multimodal spatial/frequency + audio | ICSCAI 2026 |
| 40 (getPDF) | Muppalla, Jia, Lyu, *Integrating audio-visual features* | IEEE URTC 2023 |
| 47 | Quantum-based deepfake **video** detection (without audio) | Int. J. Intelligent Systems 2025 |
| 49 | Statistical analysis of deepfake video forgery traces | J. Forensic Sciences 2024 |
| 53 | ResNet-Swish-BiLSTM visual forensics | Security & Privacy 2025 |

## The comparison that matters: In-the-Wild (unseen, real-world) EER

| Method | Encoder | Trained on | ITW EER |
|---|---|---|---:|
| WavLM + Gaussian-attention adapters (#55) | WavLM, fine-tuned (adapters) | ASVspoof 2019 LA | 6.11 % |
| Wave-spectrogram aggregation (#41) | – | ASVspoof 2019 LA | 6.58 % |
| MSLA-XLS-R (#13) | XLS-R, fine-tuned | ASVspoof 2019 LA | 6.83 % |
| LHCC (#15) | XLS-R **frozen** + ~40M-parameter fusion head | ASVspoof 2019 LA | 7.1 % |
| MWAADD (#5) | XLS-R + emotion | ASVspoof 2019 LA | 9.08 % |
| SIGNL (#18) | graph | CFAD | 10.16 % |
| **This project, v1** | **XLS-R, frozen; 3-seed ensemble** | **ASVspoof 5 + IndieFake** | **11.3 %** (95 % CI 8.9–13.5) |
| NSF-PINN (#12) | WavLM | ASVspoof 2019 LA | 19.4 % |
| Plain SSL baselines (cited in #5) | various | ASVspoof 2019 LA | ≈ 24–30 % |
| Legacy model of this project | CNN-LSTM, from scratch | ASVspoof 2019 | 23.4 % |

**How to read it honestly:**
1. **The rows are not perfectly comparable.** Others usually train on ASVspoof 2019 LA and report EER over all ITW clips. v1 trains on ASVspoof 5 + IndieFake, excludes clips with under 1 s of speech, and reports speaker-clustered confidence intervals.
2. **Most of the best systems adapt the SSL encoder** (#13 fine-tuned, #55 adapters), but #15 reaches 7.1 % with a **frozen** XLS-R and a large (~40M-parameter) head. So both 'adapt the encoder' and 'a much bigger head on a frozen encoder' are supported; v1's head is tiny (183K). Compare both in the pilot/teacher experiments.
3. **In-domain ITW results** (#15 at 0.03 %, #16 at 0.029, #57 at 98.4 % accuracy) come from random splits of ITW itself and **don't** show generalisation.

## Gaps across the literature (use them for the problem statement)

1. **Evaluation is mostly in-domain** (ASVspoof 2019 LA, near 0–1 % EER). Few papers test on unseen real-world audio; those that do report 6–30 % EER.
2. **Modern TTS (2024–26) is rarely tested.** Only #18 (CosyVoice 2) and #34 (OpenVoice) touch it.
3. **Compression and AI-processing** (codecs, speech enhancement) cause false alarms (#42, #32, #9). This is exactly what WhatsApp and phone uploads look like.
4. **Indian languages are largely missing.** There's Bangla (#17), Arabic (#44) and Indian-English IndieFake, but **no Hindi** detector in this set.
5. **Deployment is rare.** Only #44 builds an app; #48 and #59 study efficiency. Calibrated, user-facing decisions (real / uncertain / fake) are almost absent.
6. **Shortcuts and evaluation hygiene are rarely checked.** Silence (#36 even uses it as a feature), format and speaker leakage go untested, and random in-domain splits inflate results.

## Important papers that are missing from the folder (download these)

These are the foundations reviewers and mentors expect to see cited:
- Müller et al., *Does Audio Deepfake Detection Generalize?* (Interspeech 2022): introduces the **In-the-Wild** dataset.
- Müller et al., *Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?* (ASVspoof 2021 workshop): **the silence shortcut**, which our pipeline guards against.
- Wang et al., *ASVspoof 5* challenge/database papers (2024–25), plus ASVspoof 2019 and 2021 summary papers.
- Tak et al., *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation* (Odyssey 2022): the standard XLS-R baseline.
- Jung et al., *AASIST* (ICASSP 2022) and Tak et al., *RawBoost* (ICASSP 2022).
- Babu et al., *XLS-R* (Interspeech 2022): the encoder we use.
- Dataset papers: **MLAAD**, **SpeechFake** (ACL 2025), **Deepfake-Eval-2024**, **VoxENES 2026**, **IndieFake**, **IndicSynth** (see `2_DATASET_RESEARCH.md`).

*Corrected 25 Nov 2026 after Codex's full-text check (college/9_PAPER_ANALYSIS.md): #34's 24.32 % and the claim that all top systems fine-tune the encoder.*
