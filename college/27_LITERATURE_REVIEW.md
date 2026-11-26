# Literature Review: Audio Deepfake Detection

*Based on the 54 distinct papers in `papers/`. Six files are duplicates: 25–30 repeat 1–6. Two `getPDF.jsp` files (33 and 40b) are PDFs. Citations use the file number, for example [P13]. The full reference list is in Section 12. Per-paper extraction tables are in `9_PAPER_ANALYSIS.md` and `9_paper_analysis.csv`.*

---

## 1. Introduction and scope

Modern text-to-speech (TTS), voice conversion (VC), and codec-based language-model synthesis can now clone a voice from a few seconds of reference audio. Several of the reviewed papers cite real fraud cases: the 2019 UK CEO voice fraud ($243k) [P14, P44], the 2020/21 UAE bank heist ($35M) [P40b, P45], and the fabricated Zelensky surrender clip [P40b, P54]. These cases explain why detection research has grown so quickly. Audio deepfake detection (ADD) is usually framed as a binary countermeasure (CM) task: decide whether an utterance is *bona fide* or *spoof*. Some papers extend it to localizing spliced regions [P4] or to verifying audio-visual consistency [P19, P24].

The corpus breaks down as follows:

| Group | Papers | Count |
|---|---|---:|
| Audio-only detection methods | 1, 2, 3, 4, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 32, 34, 35, 36, 37, 40, 41, 42, 43, 44, 46, 48, 54, 55, 56, 57, 58, 59 | 37 |
| Multimodal / video deepfake detection | 10, 19, 24, 31, 33, 38, 39, 40b, 47, 49, 53 | 11 |
| Surveys, bibliometrics, position papers | 6, 8, 45, 50, 51, 52 | 6 |

Most papers were published in 2025–2026. The oldest are P36 (ICASSP 2023), P50 (2023) and P51 (2023). The review is organized by theme (Sections 3–8), not paper by paper, because the papers mainly differ along a few design axes: input representation, how pretrained models are used, robustness strategy, and evaluation protocol.

---

## 2. Background: threats, benchmarks and metrics

### 2.1 Attack types
- **TTS and VC (logical access, LA).** This is the main threat in most papers. VC "transforms source speaker vocal attributes to match target speaker characteristics without altering linguistic content" [P37].
- **Replay (physical access, PA).** A recording of real speech is played back to a sensor. P23 and P46 show that replay behaves very differently from TTS. P46's EER rises to 25.14% under real replay.
- **Partial spoofing / splicing.** Synthetic segments are inserted into genuine speech to change the message [P4].
- **Neural codec resynthesis and AI processing.** Codec reconstruction, speech enhancement, and vocal separation do not change the words, but they push genuine audio through a neural network. The labels then become ambiguous [P32, P42].
- **Adversarial and anti-forensic manipulation.** These are perturbations designed to evade detectors. They range from simple volume and fade changes [P9] to generative anti-forensics [P8].

### 2.2 Benchmarks used across the corpus
| Dataset | Role in the literature | Papers |
|---|---|---|
| ASVspoof 2019 LA / PA | The default training set. Also used for in-domain testing | 1, 2, 3, 5, 11, 12, 13, 15, 16, 23, 36, 37, 41, 46, 48, 54, 55 |
| ASVspoof 2021 LA / DF / PA | Tests unseen codecs and channels (DF is the main generalization test) | 5, 11, 12, 13, 15, 18, 21, 22, 23, 36, 41, 46, 55 |
| ASVspoof 5 | Newest challenge data; crowdsourced, more diverse | 18, 21, 32 |
| In-the-Wild (ITW / IWA) | Real public-figure audio scraped from the web; the main **out-of-domain** test | 3, 5, 10, 11, 12, 13, 15, 16, 18, 41, 55, 57 |
| Fake-or-Real (FoR) | English TTS corpus, often 2-second clips | 1, 15, 20, 43, 58 |
| WaveFake, FakeAVCeleb, SceneFake, CFAD (Chinese), DECRO (Chinese/English), BanglaFake, ArAD (Arabic), DEEP-VOICE, CodecDeepfakeDetection | Specialized or language-specific | 7, 12, 15, 17, 18, 21, 32, 35, 44, 55, 56 |
| PartialSpoof, SpeechSplice | Splicing localization | 4 |
| Custom corpora (OpenVoice clones, ElevenLabs, RVC) | Modern commercial and open-source generators | 14, 34, 59 |

### 2.3 Metrics
- **EER (equal error rate).** The operating point where the false-acceptance rate equals the false-rejection rate. It is the standard ASVspoof metric and does not depend on a chosen threshold.
- **min t-DCF.** A cost function that combines the CM with a downstream speaker-verification system. It is reported for LA [P1, P13, P20, P23, P37, P54].
- **Accuracy, F1, AUC.** These dominate the less specialized venues [P10, P17, P40, P43, P57, P58]. They are sensitive to class imbalance. P56 is the clearest warning: zero-shot accuracy of 0.892 sits alongside *balanced* accuracy of 0.499 and AUC of 0.412, which is no better than chance.
- **Calibration and likelihood ratio (Cllr).** These are rarely reported. Exceptions are temperature scaling [P33], calibration comparisons [P1], and forensic likelihood-ratio scoring [P14].

---

## 3. Input representations

### 3.1 Hand-crafted cepstral and spectral features
Cepstral features are still widely used, especially in lightweight and interpretable systems.

- **Multi-cepstral fusion.** Several papers combine MFCC (perceptual mel scale), LFCC (linear, so it keeps the high-frequency artifacts that vocoders leave), CQCC (constant-Q, with fine low-frequency resolution) and GFCC (gammatone, a cochlear model). Alfano et al. [P3] feed utterance statistics of all four into kNN/RF/XGBoost/MLP under speaker-independent folds. In-domain EER is 0.30% (FakeAVCelebV2) and 1.98% (ITW). Training on 2019 LA and testing on ITW gives **28.55%**. Their *domain-stability analysis* ranks which descriptors survive dataset shift, which is unusual and useful. P7 uses the same four cepstra but sends them into two Conformer branches, one for "envelope" and one for "resonance". The branches are fused with bidirectional cross-attention and the hyperparameters are tuned with a Grey Wolf Optimizer. On CFAD it reduces pooled EER by 57.1% relative to ResNeXt-SF across clean, noisy (0–20 dB) and codec conditions. P11 (DAMRN: MFCC+LFCC+GFCC with channel-excitation bottleneck, multi-scale dilated attention and multi-scale fusion) and P23 (MFCC/LFCC/CQCC into improved ResNeXt branches with MLP fusion: 1.05% on 2019 LA, 1.14% on PA, 7.44% on 2021 LA) follow the same pattern.
- **How many coefficients?** Falcón-López et al. [P48] treat MFCC dimensionality as a design variable for edge deployment. They sweep it across five classical and five deep models on 2019 LA. Deep models reach near-peak performance with few coefficients. Classical models need more, and Linear SVC consistently underperforms. They recommend **32 coefficients** as the operating point for hybrid systems.
- **Wavelets and engineered pipelines.** FakeSleuthNeXt [P20] uses no neural network at all. It applies histogram-driven binary patterns to 7-level DWT sub-bands, computes 15 statistics per level, and selects features with an INCA/IChi2/IReliefF ensemble before kNN or cubic SVM. Across six datasets (over 72k segments) it reports 89–99% accuracy, EER of 0.97–10.85%, and min t-DCF of 0.124 (2019 LA). The cost is about 7,700 s of feature processing for 19.8 GB of audio, and evaluation uses 10-fold cross-validation inside each dataset.
- **Simple feature + classifier studies.** P17 (Bangla, prosodic+spectral, six classical models, SVM/GB at 0.99–1.00 accuracy), P57 (MFCC ΔΔ + spectral contrast → ANN, 98.42% within ITW), P43 (MFCC-Conformer/LSTM, about 87.6% on 2 s FoR), P58 (stacked CNN on spectrograms + LSTM on MFCC + MLP on statistics, 97.85% on FoR, deployed through Gradio), and P40 (spectral features + CNN, 92.5% accuracy, 130 ms inference). These show that within-corpus accuracy on easier datasets is close to saturated. None of them tests on a different corpus.

**Takeaway.** Cepstral systems can be extremely accurate *in domain*: DAMRN reaches 0.10% on 2019 LA. They degrade sharply *across domains*: DAMRN is 26.14% on ITW, and P3 is 28.55% from LA to ITW. P11's authors acknowledge this gap themselves ("real-world robustness remains an open problem").

### 3.2 Learnable front-ends and raw waveforms
- **Trainable PCEN.** P1 replaces static log compression with trainable Per-Channel Energy Normalization (learnable s, α, δ, r) in front of a Residual Cascaded Dilated Conv Net (RCDLNet). The model has only **1.26M parameters and 4.6 GFLOPs**. PCEN alone cuts 2019 LA EER from 5.37% (log-mel) to 2.81%. A Hybrid Focal-BEC loss (dynamic-gamma focal loss + margin-based consistency) handles the roughly 1:9 class imbalance. P1 also reports a cross-corpus matrix: LA→FoR 7.32%, LA→ODSS 7.14%, FoR→LA 5.61%.
- **Graph networks on raw audio.** P2 (SGC-ADD) replaces RawGAT-ST's iterative nonlinear graph attention with Simplifying Graph Convolution, a single linear aggregation step. This lowers compute (about 0.274 ms/sample) and EER on 2019 LA, and the model converges in fewer epochs.
- **Reverse perturbation.** ARPNet [P21] partially reverses speech segments in time ("prime-window reversal") and uses trainable asymmetric differential convolutions to expose phase discontinuities. It reports 1.98% EER on 2021 LA. The *same architecture* scores 29.56% on ASVspoof 5 and 28.27% on CFAD, and 39.56% at 0 dB SNR. P21 also finds an inverse correlation (r = −0.51) between language-specific weight similarity and accuracy.
- **Waveform + spectrogram jointly.** WaveSpec [P41] aligns a wav2vec2 waveform branch with a spectrogram branch through multi-scale cross-modal fusion. It adds a *single-center loss* that pulls bona fide embeddings together and leaves spoof embeddings unconstrained. Trained on 2019 LA it reaches **1.90% (21DF) and 6.58% (ITW)**, among the best transfer results in the corpus.
- **Other architectures.** P46 (dual-path CNN + STFT-mel, max-pool fusion, self-attention, ResNet classifier: 9.67% PA, 8.94% DF on 2021). P34 (VeriVoice/GEAT: depthwise frequency convolutions + graph-enhanced transformer, 99.66% on a custom OpenVoice-clone corpus of 32 speakers). P54 (Siamese CNN with residual and self-attention blocks and a contrastive "StacLoss": 2.95% EER on 2019, 3.6M parameters, 5 ms per pair. It needs a genuine reference recording for each comparison).

### 3.3 Self-supervised (SSL) and pretrained speech representations
The strongest cross-domain results in the corpus nearly all use large pretrained speech encoders: wav2vec2/XLS-R, WavLM, HuBERT, and Whisper.

| Paper | Backbone and adaptation | Novel idea | 21DF EER | ITW EER |
|---|---|---|---:|---:|
| P55 | WavLM-Large + Gaussian-attention adapters | Parameter-efficient fine-tuning; SHAP shows focus on voiced frames rather than silence | 3.56% | **6.11%** |
| P41 | wav2vec2 + spectrogram | Cross-modal multi-scale alignment, single-center loss | **1.90%** | 6.58% |
| P13 | XLS-R (conv frozen, transformer trainable) | Multi-Scale Attention on *every* layer, then weighted layer aggregation ("enhance-then-aggregate") | 2.11% | 6.83% |
| P15 | Frozen XLS-R split into low/high blocks (via L2-norm and CKA) | Checks *consistency* between low-level acoustic and high-level semantic views (Asymmetric Fusion + Dependency Capture) | 4.06% | 7.10% |
| P5 | XLS-R + speech-emotion embeddings | Wavelet attention + LSTM-attention | 2.55% | 9.08–9.10% |
| P18 | wav2vec2 → spectral/temporal graphs | Non-contrastive graph self-supervised pretraining; **5% labels** | 7.88% | 10.16% (CFAD-trained) |
| P12 | WavLM-Base+ (6 frozen / 6 adapted) | Physics-informed source–filter decomposition + Sharpness-Aware Minimization | 12.38% | 19.4% |

Other SSL studies:
- **P37.** HuBERT Base/Large/XLarge + ECAPA-TDNN with OC-Softmax. Score fusion of Large and XLarge gives **0.20% EER and min t-DCF 0.006** on 2019 LA. No cross-corpus test.
- **P16.** 106-D fusion of handcrafted features and PCA-compressed WavLM, fed to an MLP. 0.063 EER on 2019 LA, 0.029 on an internal ITW split, about 120 ms per 3 s clip.
- **P22.** Whisper-Small encoder with a classification head, transfer-learned on 2021 DF subsets (EER 0.061, AUC ≈ 0.9). Larger Whisper models gave diminishing returns.
- **P35.** Compares XLS-R, Whisper, UniSpeech-SAT, WavLM and wav2vec2 with HiFi-GAN-based GAN augmentation and contrastive learning on DECRO. Cross-lingual EER is 1.5% (Chinese→English) and 2.0% (English→Chinese) with XLS-R.
- **P44.** Fine-tunes wav2vec2 on Arabic (ArAD) and deploys it in an Android app (≈98% accuracy).

**Synthesis.** The SSL papers suggest that layer selection and fusion are where much of the gain comes from. P13 finds that per-layer enhancement helps four different back-ends (MLP, AASIST, Conformer, Mamba) and even Whisper. P15 shows that *mismatch* between shallow and deep layers is itself a forgery cue. P55 and P12 show that parameter-efficient adaptation (adapters, partial freezing) can match full fine-tuning. The trade-off is model size: P15 has 357.7M parameters in total (40.3M trainable), and P55 still needs WavLM-Large at inference. A small trainable head does not make deployment small.

---

## 4. Robustness to real-world processing

This is the most practically important theme in the corpus. It also has the fewest papers.

- **Simple manipulations defeat state-of-the-art detectors.** CLAD [P9] systematically tested widely used detectors (AASIST, RawNet2, and others) on 2019 LA under transformations that do not affect human perception. False-acceptance rates rose to **36.69% (volume), 31.23% (fade) and 51.28% (noise)**. CLAD pretrains a contrastive encoder so that a clip and its manipulated version map to the same embedding. It adds a *length loss* that clusters real audio tightly, bringing FAR below 1.63% in all tests (0.81% under noise). This costs about 14 h of extra pretraining. Code is public.
- **Neural codecs have a dual role.** P32 (ICASSP 2026) asks how codec-resynthesized genuine speech (CoRS) should be labelled. The same codecs (EnCodec, DAC, Mimi, XCodec2, S3Tokenizer) are used both for transmitting real speech and inside LM-based TTS (CosyVoice2, Llasa, MARS5, etc.). On their CodecDeepfakeDetection extension of ASVspoof 5, an ASVspoof5-trained X-AASIST collapses to 33–45% EER on codec data. CDD-trained models fail the other way (51–68% on ASVspoof5). Their guidance: for compression-oriented codecs, label resynthesis as bona fide to learn codec-invariant cues. For synthesis-oriented codecs, label it spoof. Codecs that are good at both remain unsolved.
- **AI-processed genuine audio.** P42 shows that neural enhancement, vocal separation and codecs move *genuine* audio toward the spoof region of the embedding space. Auxiliary Multi-Label Training adds labels for "AI-processed" next to bona fide and spoof. It raises SSL-AASIST from 65.89% to 72.28% and SSL-Conformer from 71.21% to 76.63%, outperforming plain augmentation.
- **Noise and codec stress.** P7 (CFAD clean, noisy and codec conditions), P21 (EER degrades at low SNR), and P55 (Gaussian attention claimed to ignore codec and noise distortions) evaluate under these conditions. RawBoost augmentation is used in P1, P13 and P15.
- **Adversarial anti-forensics.** The P8 review (Information Fusion 2026) catalogues traditional and generative-AI adversarial attacks on audio-visual detectors. It also covers defences: model fusion, decoys, knowledge distillation and generative defence. It notes the lack of quantitative comparison between defences and the need for unified CMs that resist attacks.

**Implication.** A detector that is not trained or tested against benign processing (WhatsApp/Opus compression, volume normalization, noise suppression) will produce false alarms on genuine audio *and* miss lightly processed fakes. P9 and P42 approach the problem from opposite directions, and together they cover both failure modes.

---

## 5. Interpretable, physiological and phonetic cues

A smaller group of papers looks for cues grounded in how humans produce speech, on the expectation that these are harder for generators to fake.

- **Breathing, talking and silence.** BTS-E [P36] segments clips into breathing, talking and silence. It encodes the sequence with an RNN, transformer or CNN and fuses it with RawNet2, improving the baseline by up to 46% (eval EER 9.79% → 8.75%). The authors argue that TTS rarely synthesizes realistic breaths. However, P55's SHAP analysis warns that silent segments are often a *shortcut* that reflects how a corpus was prepared rather than real forgery evidence. Models that rely on them generalize worse.
- **Source–filter physics.** NSF-PINN [P12] decomposes WavLM features into glottal-source and vocal-tract components under reconstruction, smoothness and orthogonality losses. The aim is to penalize patterns that real vocal tracts cannot physically produce. Combined with SAM it cuts EER by 44.9% relative to the WavLM baseline. Absolute cross-domain numbers are still modest (19.4% ITW).
- **Forensic segmental phonetics.** Yang, Sun, Lyu and Rose [P14] borrow from forensic voice comparison. They use vowel formants and segmental features in a **speaker-specific** likelihood-ratio framework and test on ElevenLabs clones of YouTube talkers. Segmental features (vowel EER ≈ 11–15%) beat MFCC (34–37%), while global features like mean F0 add little. The paper argues that speaker-specific, interpretable evidence fits courts better than speaker-independent black boxes.
- **Emotion and prosody.** P5 adds speech-emotion embeddings. P21 links artifacts to phonology, and P17 uses prosody for Bangla.
- **Sociolinguistics.** Mallinson et al. [P52] argue that linguists can contribute expert phonetic knowledge, accent- and dialect-aware evaluation, and training for human listeners. They also point out that detectors may perform unevenly across accents and social groups.
- **Explainability tooling.** SHAP, LIME, LRP and Grad-CAM are used in P10 and P55. t-SNE plots appear in P1, P15 and P42. Feature importances are reported in P16 and P20.

---

## 6. Partial spoofing and localization
Only P4 addresses *where* in a clip the manipulation is. It slides an embedding extractor (4 s down to 0.5 s windows) over the audio, builds a self-similarity matrix, and computes **audio novelty**, a technique from music segmentation, to find splice points. The model is trained only on unspliced audio. The authors release **SpeechSplice**, a corpus built from ASVspoof 2019 without splicing artifacts, so that detectors cannot rely on boundary glitches. Balanced accuracy is about 0.66–0.75, which shows the task is far from solved. P4 also argues that file-level EER says nothing about localization ability.

---

## 7. Data and label efficiency, language coverage and deployment

### 7.1 Label efficiency and augmentation
- **SIGNL [P18].** Pretrains spectral and temporal graph encoders with non-contrastive self-supervision on unlabelled audio, then fine-tunes on 5% of the labels. It reaches 3.95% EER on ASVspoof 5 and 7.88% on 21DF, and 10.16% on ITW (CFAD-trained) against 7.69% with full labels. It is also one of the few papers that tests on **CosyVoice 2.0**.
- **GAN augmentation.** P35 generates extra spoofs with HiFi-GAN. P32 and P42 show that the *labelling* of augmented or processed data matters as much as the amount.

### 7.2 Language coverage
Most research is English-centric, a bias that P45 and P52 point out explicitly. Non-English work in this corpus covers: **Chinese** (CFAD in P7, P18 and P21; DECRO in P35), **Bangla** (P17), **Arabic** with dialects (P44), and multilingual or cross-lingual tests (P1 on ODSS and GGMDDC; P35 C↔E transfer). No paper studies Hindi or other Indian languages, or code-switched speech.

### 7.3 Efficiency and on-device deployment
| Paper | Efficiency claim |
|---|---|
| P1 | 1.26M params, 4.6 GFLOPs, competitive cross-corpus |
| P2 | Linearized graph conv, ≈0.27 ms/sample |
| P54 | 3.6M params, 5 ms per pair |
| P16 | 106-D features, ≈120 ms per 3 s clip (but needs WavLM extraction) |
| P3 | 70 ms features + negligible MLP time |
| P48 | 32 MFCCs as the compact operating point |
| P59 | Weight quantization (uniform, non-uniform, PTQ, QAT) of a DenseNet/attention model on RVC voices; non-uniform gives 2.84% error, ROC-AUC 0.998 |
| P44 | Android app (Arabic, wav2vec2) |
| P58, P31 | Gradio and web demos |
| P40 | "Real-time", 130 ms |

Few papers report the numbers a deployment actually needs: model size on disk, end-to-end CPU latency for a realistic clip length, and whether inference runs fully offline. P44, for example, shows the app workflow but does not establish standalone on-device inference.

---

## 8. Multimodal and video deepfake detection

About a fifth of the corpus deals with video, usually with audio as one branch.

- **Audio-visual consistency.** ART-AVDF [P24] learns articulatory representations from audio and matches them to lip motion through self-supervision (DFDC, FakeAVCeleb, DefakeAVMiT; within-dataset AUC 97–99%). P19 targets *asynchrony* caused by network jitter in video calls. It frames alignment as integer linear programming solved with the Hungarian algorithm and adds masked feature reconstruction, reaching an average AUC of 90.3%.
- **Fusion classifiers.** P38 (ViT with temporal attention + mel audio encoder: 99.34% fused against 79.78% audio-only), P39 (ResNet18 + FFT + MFCC: audio 88.6%, video 91.3%, fusion 95.2%), P40b (four-class fine-grained labels from per-modality real/fake, FakeAVCeleb and TMC), P33 (lightweight late fusion, four classes, temperature-scaled calibration, 98.75%), and P10 (InceptionV3 / DenseNet169-BiLSTM / CNN-BiLSTM-XGBoost with cross-attention and a combined SHAP/LIME/LRP/Grad-CAM XAI layer, 96–99%).
- **Proactive verification.** DeepShield [P31] combines ResNeXt-LSTM video detection, a pretrained audio detector, and **DWT audio watermarking** to detect trimming or re-encoding. The authors argue that passive detection and proactive provenance complement each other.
- **Video only.** P47 (ResNet + quantum LSTM, fewer parameters), P49 (pixel-level statistical traces + fine-tuned InceptionResNetV2, 99.8% FF++), P53 (ResNet-Swish-BiLSTM, 96% FF++ but **78%** on combined FF++/DFDC).

In the papers that report ablations, the fusion gains come largely from the visual stream (P38, P39). These results do not tell us how audio-only detectors perform. They matter for audio work mainly as evidence that cross-modal consistency, calibration and provenance are useful extra signals.

### 8.1 Quantum approaches
P56 carefully evaluates a five-qubit hybrid quantum-classical network, screening 31 gate combinations (CNOT+S was best). It is strong in domain on DEEP-VOICE, but zero-shot transfer to ASVspoof 2019 is at chance (balanced accuracy 0.499). Scaling to ten qubits did not help. The authors themselves limit their causal claims. P47 is the video counterpart. Neither paper shows any practical advantage over classical models.

---

## 9. Surveys and meta-research
- **P50 (Dixit and Kaur, 2023).** An early full taxonomy of audio deepfake generation (TTS, VC, GAN and VAE vocoders) and detection features. It predates the current commercial and codec-based generators.
- **P45 (2025).** Reviews detection for voice-cloning fraud. Hybrid spectrogram/pitch/attention models perform best. The open problems it lists are cross-dataset generalization, adversarial robustness, low-resource languages, real-time limits, privacy, and the need for explainability and human-perception-like standards.
- **P6 (Kanwal et al., 2026).** A large review of image, video, audio and multimodal detection from 2021–2025. Its recurring gaps are generalization, degradation under compression and noise, compute cost and low interpretability. It proposes a reliability-weighted, uncertainty-aware, cost-sensitive multimodal framework.
- **P8.** Adversarial and generative anti-forensics (Section 4).
- **P51 (Chen, 2023).** Topic modelling with BERTopic on ACM, IEEE and Scopus records from 2018–23. Detection dominates the field. Privacy, law and social impact are under-researched.
- **P52.** The sociolinguistics position paper (Section 5).

---

## 10. Critical analysis

### 10.1 The generalization gap is the central problem
The same pattern appears throughout the corpus: models are near-perfect in domain and much worse out of domain.

| System | In-domain (2019 LA) | Out-of-domain (ITW) |
|---|---:|---:|
| DAMRN [P11] | 0.10% | 26.14% |
| Multi-cepstral MLP [P3] | — | 28.55% |
| NSF-PINN [P12] | 4.49% | 19.4% |
| MWAADD [P5] | 0.18% | 9.10% |
| MSLA-XLS-R [P13] | 0.14% | 6.83% |
| Gaussian-adapter WavLM [P55] | 0.42% | 6.11% |

In-domain EER on 2019 LA is essentially saturated and no longer tells systems apart. ITW, 2021 DF and ASVspoof 5 are now the informative tests. Large SSL front-ends combined with careful layer use (P13, P15, P41, P55) give the best transfer seen so far, around 6–7% on ITW, but none of them solves the problem.

### 10.2 Methodological weaknesses seen across papers
1. **Within-corpus testing reported as robustness.** P16, P57 and P20 report high ITW or multi-dataset numbers from *internal* splits or 10-fold cross-validation. These cannot be compared with cross-corpus EER. P7 says openly that its extra-dataset results need retraining.
2. **Accuracy on imbalanced data.** ASVspoof 2019 LA is about 1:9 bona fide to spoof, so accuracy can hide failure (P56).
3. **Small or custom corpora.** P34 (32 speakers, a single generator), P59 (8 RVC identities) and P17 (one language) report about 99% or higher. Speaker overlap between training and test splits is often not ruled out. P3 is a positive exception because it uses speaker-independent folds.
4. **Shortcut cues.** Silence and boundary cues [P36, P55] and corpus-specific recording channels inflate results without any real detection of synthesis.
5. **Replay mixed with synthesis.** PA and LA are different threats, yet papers often pool them [P23, P46].
6. **Internal inconsistencies.** Values differ between abstract and tables: P5 (ITW 9.08% vs 9.10%), P57 (98.52% vs 98.42%), P53 (96% vs 98.5%).
7. **Reproducibility.** Most papers state "data available on request". Public code is available only for P9 (CLAD), P13, P18 (SIGNL), P32 and P37.
8. **Missing operating points.** Almost no paper reports the false-alarm rate at a fixed, low threshold, or calibration. Both are what a consumer app or a forensic report actually needs (P14 and P33 are partial exceptions).

### 10.3 Research gaps
1. **Modern generators.** Only P14 (ElevenLabs), P18 and P32 (CosyVoice 2, codec-LM TTS), P34 (OpenVoice) and P59 (RVC) test 2024–26 systems. Benchmarks lag behind generators.
2. **Benign processing and channels.** Phone and messaging codecs, enhancement, and re-recording are covered only by P9, P32 and P42. There is no standard false-alarm benchmark for processed genuine speech.
3. **Low-resource and Indian languages, code-switching.** No paper covers these.
4. **Partial spoof localization.** Only one paper (P4), with balanced accuracy around 0.7.
5. **Calibration, uncertainty and abstention.** Rarely studied.
6. **Deployment evidence.** There are few measurements of end-to-end on-device latency, memory and model bytes for large SSL models. Compression of SSL front-ends (distillation, quantization as in P59, pruning layers as P13 suggests) is under-explored.
7. **Fairness across accents, dialects and demographics** [P52].
8. **Adversarial and anti-forensic robustness evaluated under a common protocol** [P8].
9. **Combining passive detection with provenance and watermarking** [P31].

---

## 11. Implications for this project
For an audio-only, upload-based detector that must keep false alarms on genuine voice notes low, the literature points to the following:

- **Front-end.** Use a pretrained SSL encoder (XLS-R or WavLM) with multi-layer aggregation [P13, P55], not a single last layer. Report how much a frozen encoder plus a light head costs [P15, P55].
- **Training data.** Pretrained features alone are not enough. Include modern generators (codec-LM TTS, commercial cloning, RVC/VC) [P14, P18, P32, P59]. Apply codec and enhancement augmentation with deliberate labels: AMLT-style auxiliary labels [P42] and P32's codec-labelling rules.
- **Robustness training.** Train for invariance to benign manipulations through contrastive pairs and a compact bona fide class [P9, P41].
- **Shortcut control.** Trim or neutralize leading and trailing silence, and check attributions (SHAP) to confirm the model uses voiced frames [P36, P55].
- **Evaluation.** Always report cross-corpus EER (ITW at minimum), the false-alarm rate at the chosen threshold, and calibration. Use speaker-disjoint splits [P3] and confidence intervals across seeds [P12, P18].
- **Deployment.** Measure size, latency and memory on the target CPU. Quantization [P59] and compact front-ends [P1, P48] are the levers available in the literature.

---

## 12. Reference list (local corpus)

| # | Title | Venue, year |
|---|---|---|
| P1 | Hybrid learning for detection of audio deepfakes and manipulation attacks (PCEN + RCDLNet) | J. Information Security & Applications, 2027 |
| P2 | Audio deepfake detection based on simplifying graph convolution (SGC-ADD) — Liu et al. | Knowledge-Based Systems, 2026 |
| P3 | Multi-cepstral feature integration for audio deepfake detection under domain shift — Alfano, Bisogni, Nappi, Narducci, Pero | Computer Vision and Image Understanding, 2026 |
| P4 | Splicing detection and localization for speech deepfakes using audio novelty | Computer Vision and Image Understanding, 2026 |
| P5 | Multi-feature wavelet attention network for audio deepfake detection (MWAADD) | Knowledge-Based Systems, 2026 |
| P6 | Deepfake detection: A review of audio, visual and multimodal techniques — Kanwal, Ashraf, Jhanjhi et al. | Applied Soft Computing, 2026 |
| P7 | Deepfake audio detection with spectral fusion and dual-branch Conformer architecture — Zhan, Wang et al. | Applied Soft Computing, 2027 |
| P8 | Adversarial and generative AI-based anti-forensics in audio-visual deepfake detection: a comprehensive review | Information Fusion, 2026 |
| P9 | CLAD: Robust audio deepfake detection against manipulation attacks with contrastive learning | Knowledge-Based Systems, 2026 |
| P10 | An integrated explainable framework for multimodal deepfake detection across image, audio, and video | Ain Shams Engineering Journal, 2026 |
| P11 | Deep attention mechanism residual network (DAMRN) using multi-scale cepstral features | Signal Processing, 2026 |
| P12 | NSF-PINN: Physics-informed source-filter decomposition with SAM for generalizable ADD | Computer Vision and Image Understanding, 2026 |
| P13 | MSLA-XLS-R: Enhancing hierarchical SSL representations for audio deepfake detection | Computer Speech & Language, 2027 |
| P14 | Forensic deepfake audio detection using segmental speech features — Yang, Sun, Lyu, Rose | Forensic Science International, 2026 |
| P15 | LHCC: Cross-domain ADD via consistency-aware fusion of acoustic and semantic representations | Information Fusion, 2026 |
| P16 | Hybrid acoustic feature and self-supervised WavLM embeddings for robust ADD — Verma et al. | Procedia Computer Science, 2026 |
| P17 | Bangla deepfake audio detection using prosodic and spectral features | Procedia Computer Science, 2026 |
| P18 | SIGNL: Label-efficient ADD via spectral-temporal graph non-contrastive learning | Expert Systems with Applications, 2026 |
| P19 | Predictive inter-modal alignment and feature reconstruction in audio–visual asynchrony scenarios | Information Fusion, 2026 |
| P20 | FakeSleuthNeXt: multi-level iterative feature engineering for deepfake audio detection — Celik, Akbal | Expert Systems with Applications, 2026 |
| P21 | Adaptive reverse perturbation network (ARPNet) for audio deepfake detection | Neurocomputing, 2025 |
| P22 | Transfer learning in audio deepfake detection – adaptation of an ASR model (Whisper) — Stasiak et al. | Procedia Computer Science (KES), 2026 |
| P23 | Deepfake audio detection with spectral features and ResNeXt-based architecture — Tahaoglu et al. | Knowledge-Based Systems, 2025 |
| P24 | Audio–visual deepfake detection using articulatory representation learning (ART-AVDF) | Computer Vision and Image Understanding, 2024 |
| P31 | DeepShield: Multi-modal deepfake detection with audio watermark verification | IGNITE, 2026 |
| P32 | How to label resynthesized audio: the dual role of neural audio codecs in ADD | ICASSP, 2026 |
| P33 | A lightweight audio-visual framework for deepfake detection | ICSCCC, 2026 |
| P34 | VeriVoice: Enhancing ADD with depthwise frequency convolutions (GEAT) | DECoN, 2025 |
| P35 | Fooling the forgers: a multi-stage framework for audio deepfake detection | ICASSP, 2025 |
| P36 | BTS-E: Audio deepfake detection using breathing-talking-silence encoder | ICASSP, 2023 |
| P37 | HuBERT-derived SSL features and ECAPA-TDNN matching for robust ADD | IEEE MLSP, 2025 |
| P38 | Multimodal deepfake detection via audio-visual feature fusion and temporal attention | ICBAIE, 2025 |
| P39 | Attention-based multimodal deepfake detection using spatial, frequency and audio features | ICSCAI, 2026 |
| P40 | Real-time detection of AI-generated deepfake audio: a novel approach | ICTBIG, 2024 |
| P40b | Integrating audio-visual features for multimodal deepfake detection | IEEE URTC, 2023 |
| P41 | Wave-spectrogram cross-modal aggregation for audio deepfake detection (WaveSpec) | ICASSP, 2025 |
| P42 | Auxiliary multi-label training for robust ADD on AI-processed data | ICASSP, 2026 |
| P43 | Leveraging acoustic features and deep neural architectures for ADD | ICACC, 2024 |
| P44 | RealACaller: real-time Arabic audio deepfake detection app for smartphones | IEEE ICMI, 2025 |
| P45 | Audio deepfakes detection: a comprehensive literature review of voice cloning fraud detection techniques | iSSSC, 2025 |
| P46 | Enhancing deepfake audio detection: a ResNet framework based on hybrid features and self-attention — Huang et al. | Expert Systems, 2025 |
| P47 | Framework for quantum-based deepfake video detection (without audio) — Pandey, Rudra, Krishnan | Int. J. Intelligent Systems, 2025 |
| P48 | On the optimal selection of MFCCs for voice deepfake detection — Falcón-López et al. | Expert Systems, 2026 |
| P49 | A statistical analysis for deepfake video forgery traces + fine-tuned InceptionResNetV2 — Sandhya, Kashyap | J. Forensic Sciences, 2025 |
| P50 | Review of audio deepfake detection techniques: issues and prospects — Dixit, Kaur | Expert Systems, 2023 |
| P51 | Understanding deepfake research and trends through topic modelling — Chen et al. | Proc. ASIS&T, 2023 |
| P52 | A place for (socio)linguistics in audio deepfake detection and discernment — Mallinson et al. | Language and Linguistics Compass, 2024 |
| P53 | Next-gen deepfake detection: ResNet-Swish-BiLSTM for visual forensics — Palani et al. | Security and Privacy, 2025 |
| P54 | Audio deepfake detection using deep learning (Siamese CNN + StacLoss) — Shaaban et al. | Engineering Reports, 2025 |
| P55 | Leveraging Gaussian attention for adapter-based fine-tuning of WavLM in ADD — Wang et al. | Int. J. Intelligent Systems, 2026 |
| P56 | Quantum gate optimization in hybrid neural networks for deepfake audio detection | Scientific Reports, 2026 |
| P57 | Pushing the boundaries of deepfake audio detection with hybrid MFCC and spectral contrast — Jellali, Ben Fredj, Ouni | Multimedia Tools and Applications, 2025 |
| P58 | Ensemble learning model for deepfake audio detection using multi-feature extraction | Scientific Reports, 2026 |
| P59 | Optimizing deepfake audio detection: fragment feature overlaid quantization model — Hassan et al. | Int. J. Machine Learning & Cybernetics, 2025 |

*Numbers in this review are as reported by each paper's authors. They come from different training data, test protocols and operating points, so small differences between papers should not be read as rankings.*
