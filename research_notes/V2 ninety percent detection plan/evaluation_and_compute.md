# Evaluation design and Colab compute plan for a 90%-per-family / 15%-FA speech-deepfake milestone (as of 2026-11-26)

Milestone under study: >=90% detection separately for each of Chatterbox, ElevenLabs v3, Fish Speech and Gemini TTS, at <=15% false alarms on independent English genuine recordings, within about 14 hours on Google Colab.

## 1. Sample sizes and statistics (Wilson intervals, power, speaker clustering)

### Takeaway
If you want the 95% Wilson lower bound on detection to be >=85% when the true rate is about 93%, plan for about 200 independent fake clips per family. That gives about 96% power, and 100 clips gives only about 73%. For false alarms, about 300 independent genuine recordings give about 96% power for an upper bound <=15% when the true FA is 8%. Speaker and voice clustering shrinks the effective n through the design effect 1+(m-1)·ICC, so spread clips across many voices and speakers (for example, 200 fakes as 40 voices x 5) and report a cluster bootstrap next to Wilson.

### Cited Findings
- The Wilson score interval is the recommended interval for a binomial proportion. Wald intervals behave erratically and have poor coverage near p=0.9, even at moderate n. Brown, Cai and DasGupta recommend Wilson or Jeffreys for n<=40 and Wilson, Jeffreys or Agresti-Coull above that — [Brown, Cai & DasGupta 2001, Statistical Science](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full)
- Cluster sampling inflates variance by the Kish design effect DEFF ≈ 1+(m-1)·ICC, where m is the cluster size and ICC is the intra-cluster correlation. The effective sample size is n/DEFF — [Design effect, Wikipedia (Kish)](https://en.wikipedia.org/wiki/Design_effect)
- ASVspoof 5 is an example of scale: the eval set has 737 speakers with 138,688 bona fide and 542,086 spoofed utterances, and train, dev and eval speakers are disjoint — [ASVspoof 5 database paper, arXiv 2408.08739](https://arxiv.org/html/2408.08739v1)

### Computed numbers (my own calculation, z=1.96, exact binomial; script logic: Wilson bound and summed binomial pmf)
These numbers assume independent clips (ICC=0). Use the n_eff after the clustering correction below.

**Detection (fakes per family). The smallest n for which the Wilson 95% lower bound reaches the target at a given observed rate:**

| Observed detection | LB >= 85% | LB >= 87% |
|---|---|---|
| 90% | 184 | 455 |
| 92% | 92 | 156 |
| 93% | 64 | 107 |
| 95% | 45 | 64 |

**Power. Probability that the Wilson LB is >=85%, given the true rate:**

| n | true 93% | true 95% | P(observed >=90% \| true 93%) | P(observed >=90% \| true 90%) |
|---|---|---|---|---|
| 100 | 0.73 | 0.94 | 0.91 | 0.58 |
| 150 | 0.83 | 0.98 | 0.94 | 0.57 |
| 200 | 0.96 | 1.00 | 0.96 | 0.56 |
| 300 | 0.99 | 1.00 | 0.98 | 0.55 |

A model whose true rate is exactly 90% passes the "observed >=90%" point check only about 55-58% of the time at any n. To pass the point target reliably, the model needs a true margin of about 93% or more.

**Example intervals:**
- n=100: 93% gives [86.3, 96.6] and 90% gives [82.6, 94.5].
- n=200: 93% gives [88.6, 95.8] and 90% gives [85.1, 93.4].
- n=300: 93% gives [89.5, 95.4].
- n=500: 93% gives [90.4, 94.9].

**False alarms (genuine recordings). The smallest n for which the Wilson 95% upper bound stays under the target at a given observed FA:**

| Observed FA | UB <= 15% | UB <= 20% |
|---|---|---|
| 5% | 45 | 25 |
| 8% | 92 | 40 |
| 10% | 184 | 54 |

**FA power, P(UB <= 15% \| true FA 8%):** n=100 gives 0.59, n=200 gives 0.88, n=300 gives 0.96 and n=400 gives 0.99. Observed FA of 10% with n=200 gives the interval [6.6, 14.9].

**Design effect examples:**

| m (clips per cluster) | ICC 0.1 | ICC 0.2 | ICC 0.3 |
|---|---|---|---|
| 5 | 1.4 | 1.8 | 2.2 |
| 10 | 1.9 | 2.8 | 3.7 |
| 20 | 2.9 | 4.8 | 6.7 |

For example, 200 fakes from 10 voices x 20 clips at ICC 0.2 gives n_eff of about 42. That is too small.

### Inferences
- **Recommended per-family fake set:** at least 200 clips, from at least 40 distinct voices or speaker prompts (5 clips each) and diverse text. If each voice is a cluster with a modest ICC of about 0.1, DEFF ≈ 1.4 and n_eff ≈ 143. Lower bound at an observed 93% is then about 87%. 300 clips per family is safer if generation is cheap.
- **Recommended genuine set:** at least 300 recordings from at least 60-100 distinct speakers (3-5 each), across multiple sources or devices. FA errors cluster heavily by speaker and channel; one noisy microphone can produce many FAs. Cap the clips per speaker.
- **Primary statistic:** report per-family detection and pooled genuine FA at a threshold frozen beforehand, with the Wilson 95% CI. Also report a cluster (speaker or voice-level) bootstrap 95% CI (resample clusters with replacement, 2,000-10,000 reps). Where they disagree, treat the cluster bootstrap as authoritative.
- **Multiple comparisons:** there are four family tests plus one FA test. Requiring all of them to pass is a conjunction ("intersection-union"), so no alpha correction is needed for the pass/fail decision. The probability of passing all five, though, is the product of the individual powers (for example 0.96^4 x 0.96 ≈ 0.82). Size each family for >=95% power.
- **Pre-register:** the threshold, the n, the cluster definition, the pass rule ("point >=90% and Wilson LB >=85%", or similar) and the stopping rule. Do not keep generating clips until a family passes.

### Gaps
- No published ICC values for "clips from the same TTS voice" or "recordings from the same genuine speaker" in deepfake detection were found. Estimate the ICC from the data with a pilot or the bootstrap.
- No literature source gives a standard pass rule such as "LB>=85%". That rule is a planning choice.

## 2. Literature best practices for independent test sets (ASVspoof 5, Deepfake-Eval-2024, SpeechFake, MLAAD, In-the-Wild)

### Takeaway
Credible benchmarks make the test set disjoint from training in speakers, attacks/generators and source. Ideally, genuine and fake share a source corpus, so the label cannot be predicted from the corpus. They strip leading and trailing non-speech and include codec and channel conditions. The threshold should be fixed on separate dev data, and the realised (actual) error should be reported along with the oracle metrics. In-the-wild evaluations show large degradations, so a test set drawn only from the developer's own pipeline will overstate performance.

### Cited Findings
- **ASVspoof 5, disjointness and source:** train, dev and eval speakers are disjoint, and attacks are disjoint too (A01-A08 train, A09-A16 dev, A17-A32 eval). Bona fide and spoofed audio both come from MLS English: training uses partition A, and dev/eval spoofs are cloned from the same partition C speakers whose bona fide audio is used. Eval speakers who also appear in LibriSpeech were removed because open-condition systems may use LibriSpeech-pretrained models — [arXiv 2408.08739](https://arxiv.org/html/2408.08739v1)
- **ASVspoof 5, artifacts and conditions:** leading and trailing non-speech segments were removed from evaluation utterances. Codec conditions are C00 (none), C01-C07 at 16 kHz (opus, amr, speex, Encodec, mp3, m4a, mp3+Encodec) and C08-C11 narrowband at 8 kHz (downsampled, encoded, then upsampled) — [arXiv 2408.08739](https://arxiv.org/html/2408.08739v1)
- **ASVspoof 5, calibration:** minDCF uses an oracle threshold, while actDCF uses a fixed Bayes threshold and assumes the scores are LLRs. Many top systems had actDCF near 1.0 because their scores were not calibrated. C_llr and EER are reported too, and the Track 1 costs are C_miss=1, C_fa=10 and π_spf=0.05 — [arXiv 2408.08739](https://arxiv.org/html/2408.08739v1); [ASVspoof 5 evaluation summary, arXiv 2601.03944](https://arxiv.org/html/2601.03944)
- **Does audio deepfake detection generalize? (Müller et al. 2022):** the authors released an In-the-Wild set of 37.9 h of celebrity and politician audio, 17.2 h of it fake. Models from related work degrade sharply on it ("up to one thousand percent"), which suggests the field over-fits ASVspoof. Using cqtspec/logspec features instead of melspec improved EER by 37% on average — [arXiv 2203.16263](https://arxiv.org/abs/2203.16263)
- **Deepfake-Eval-2024:** the audio part has 1,820 clips (1,110 real and 710 fake, 56.5 h, 42 languages) circulated in 2024. Open-source audio detectors lose about 48% AUC compared with earlier benchmarks. Commercial and fine-tuned models do better but still fall short of forensic analysts — [arXiv 2503.02857](https://arxiv.org/abs/2503.02857); [HTML v2](https://arxiv.org/html/2503.02857v2)
- **SpeechFake (ACL 2025):** this is a large multilingual dataset built with recent generation methods. Detectors trained on it do well on its own test set and several unseen sets, but performance degrades on unseen deepfakes. A 2026 paper argues that diversity of generation methods matters more than raw scale — [arXiv 2507.21463](https://arxiv.org/html/2507.21463v1); [ACL anthology PDF](https://preview.aclanthology.org/landing_page/2025.acl-long.493.pdf)
- **MLAAD:** 687.4 h of synthetic speech from 140 TTS models, with the genuine side taken from M-AILABS (audiobooks and speeches). Held-out generator subsets support open-set evaluation — [arXiv 2401.09512](https://arxiv.org/html/2401.09512v9); [Forensic Similarity, arXiv 2510.02864](https://arxiv.org/pdf/2510.02864)

### Inferences
- **Independence checklist for the milestone test set.** (a) Fake voices, speaker prompts and reference clips are not used in training. (b) Text and prompts are disjoint from training text. (c) Genuine speakers are disjoint from training genuine speakers and from pretraining corpora where known (XLS-R saw LibriSpeech, MLS, CommonVoice, VoxPopuli and BABEL, so prefer genuine audio from other sources or note the overlap). (d) Generation is done separately from the training-data generation, ideally by a different person or script version, with different sampling settings.
- **Matched genuine controls.** Where possible, make the source genuine recordings for voice cloning (the reference speakers) part of the genuine test set. That matches the ASVspoof 5 "same speakers, same corpus" design. Then a fake and its genuine counterpart differ only in synthesis, not in corpus.
- **Report two FA numbers:** FA on matched genuine audio (the same corpus as the cloning references) and FA on independent "wild" genuine audio (phone or laptop recordings, podcasts). This exposes corpus shortcuts.
- **Threshold.** Choose it on a separate calibration or dev split. This can be held-out training-distribution data or a small separate set; it must not be the test set. Freeze it before scoring, and report the realised detection and FA at that frozen threshold, which is the analogue of actDCF. Report EER and the oracle-threshold result only as secondary numbers. Optionally fit logistic calibration on dev to get LLR-like scores.
- **Reporting.** Give per-family detection with CIs, pooled and per-source FA with CIs, EER and AUC, a DET curve, score histograms per family, and the counts of clips, speakers and voices. Break results down by duration bins and codec condition.

### Gaps
- No public papers specifically evaluate Chatterbox, ElevenLabs v3, Fish Speech and Gemini TTS together. Recent dataset coverage of these exact generators was not verified in this pass.
- The Deepfake-Eval-2024 per-model audio numbers were not extracted, only the 48% AUC drop.

## 3. Avoiding shortcut learning (silence, duration, loudness, bandwidth, codec)

### Takeaway
Silence is the best-documented shortcut. A detector that uses only leading-silence duration reached 15.1% EER on ASVspoof 2019, and VAD trimming roughly doubles the EER of standard models. Bitrate/bandwidth leakage is also documented. Run "trivial-feature" probes such as duration, leading and trailing silence, RMS loudness, spectral rolloff/bandwidth and sample rate before trusting any result. Then remove the confounds: trim silence, resample everything to 16 kHz, loudness-normalize, and match codec and bandwidth between genuine and fake.

### Cited Findings
- On ASVspoof 2019 LA, bona fide audio has much longer leading and trailing silence than spoofs. A model trained only on leading-silence duration reached 15.1% EER, and trimming silence drastically reduced established models' performance — [Müller et al., ASVspoof 2021 Workshop, arXiv 2106.12914](https://arxiv.org/abs/2106.12914v3); [ISCA archive](https://www.isca-archive.org/asvspoof_2021/muller21_asvspoof.html)
- A 2023 follow-up found that VAD removal of silence, especially at the start and end, roughly doubles the EER, and that concatenating silence onto spoofed speech can act as an attack — [arXiv 2309.11827](https://arxiv.org/pdf/2309.11827)
- In FakeAVCeleb and AV-Deepfake1M, fakes start with a brief silence. A silence-only classifier gets over 98%, and supervised models drop once it is removed. The same paper notes that ASVspoof19 leaks labels through silence duration and through bitrate — [Smeu et al., CVPR 2025, arXiv 2412.00175](https://arxiv.org/html/2412.00175v1)
- An intervention-based shortcut-diagnosis framework for spoofing CMs finds that non-speech interventions cause the largest performance shifts — [Rubio et al., arXiv 2607.03150](https://arxiv.org/pdf/2607.03150)
- ASVspoof 5 itself strips leading and trailing non-speech from eval utterances and adds 8 kHz narrowband and multi-codec conditions — [arXiv 2408.08739](https://arxiv.org/html/2408.08739v1)

### Inferences (recommended checks for the planner)
1. **Trivial-feature probes.** Train logistic regression or a small gradient-boosted classifier on simple features only. Use duration, leading and trailing silence (ms), the speech/non-speech ratio, RMS/LUFS loudness, peak level, clipping rate, spectral rolloff at 95% energy (an effective bandwidth proxy), the original sample rate or codec, and DC offset. If any probe gets AUC well above 0.6-0.7 on train or test, that is a confound to fix. Run it separately for each family against genuine audio.
2. **Preprocessing parity.** Apply the same pipeline to every clip: decode, convert to mono, resample to 16 kHz, VAD-trim to about 0.1-0.2 s padding, loudness-normalize (for example to -23 LUFS or a fixed RMS), and crop or pad to a fixed window such as 4 s, with random crops in training.
3. **Codec and bandwidth matching.** Fakes from APIs (ElevenLabs and Gemini usually return MP3/PCM at 22-44 kHz) and genuine recordings (phone, Opus, mic) have different spectra. Augment both classes with the same random codec chain (MP3, Opus, AAC at random bitrates, plus an 8 kHz narrowband round-trip, the ASVspoof 5 C08-C11 style) so that codec presence carries no label. Optionally low-pass both classes at a random cutoff.
4. **Counterfactual tests on the final model.** Re-score the test set with (a) silence padded onto fakes, (b) genuine audio passed through the fake-generation codec, and (c) loudness shifted by ±6 dB. Large score shifts mean shortcut reliance.
5. **Duration matching.** Match duration distributions between genuine and fake in train and test, or report results by duration bin.

### Gaps
- No quantitative study of shortcuts specific to the four target generators (for example, whether ElevenLabs v3 or Gemini outputs have distinctive leading silence or bandwidth) was found. The planner should measure this with the probes above.

## 4. Google Colab compute (Nov 2026): CU rates, prices, session limits, XLS-R-300m throughput

### Takeaway
Consumer Colab sells compute units at about $0.10 each (Pay As You Go $9.99 per 100 or $49.99 per 500; Pro $9.99/month for 100 CU; Pro+ $49.99/month). Rates measured in March 2026 were T4 1.19, L4 1.71, A100-40GB 5.40, A100-80GB 7.52 and G4 (RTX PRO 6000) 8.71 CU/h. The H100 rate is unverified. Pro and Pay As You Go sessions are capped at about 12 h and Pro+ at 24 h, so a 14 h plan needs checkpoint/resume or Pro+. A full 14 h on an A100-40GB costs about 76 CU (~$8). Cost is not the constraint; availability and session limits are.

### Cited Findings
- **Measured CU burn on 2026-03-10 (southern California):** T4 1.19 CU/h ($0.12), L4 1.71 ($0.17, 22.5 GB), A100 40GB 5.40 ($0.54), A100 80GB 7.52 ($0.75), G4/RTX PRO 6000 Blackwell 8.71 ($0.87, ~96 GB). H100 is listed as an option, but the author could not get one to measure. Rates "fluctuate a little" and may vary by demand and location. Units cost $10 per 100 with no bulk discount — [McCormick, Colab GPUs Features & Pricing (updated Mar 2026)](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
- **Conflicting reports:** another source puts the T4 at about 1.76 CU/h and the A100 at about 15 CU/h (June 2026 guide), and Colab publishes no official per-GPU table. Treat the A100 at 5.4-15 CU/h as uncertain and check the in-session usage meter — [search summary of DEV Community / gpuperhour guides](https://gpuperhour.com/blog/google-colab-alternatives); [McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
- **Throughput ratios from McCormick's LLM fine-tune benchmark:** the L4 is about 2.2-2.8x and the A100 about 6-13x faster than the T4. Peak FP16 TFLOPS are T4 65, L4 121, A100 312 and H100 990. Time per $10 is T4 84 h, L4 58.5 h, A100-40 18.5 h, A100-80 13.3 h and G4 11.5 h — [McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
- **Pricing:** Pay As You Go is $9.99 for 100 CU or $49.99 for 500 CU. Pro is $9.99/month with 100 CU. Pro+ is about $49.99/month with 500 CU (some sources say 600) — [TechCrunch/AOL on PAYG launch](https://www.aol.com/news/google-colaboratory-launches-pay-option-233525876.html); [Google Workspace Updates, Colab Pro/Pro+](https://workspaceupdates.googleblog.com/2024/06/google-workspace-colab-pro-and-colab-pro-plus.html). The official signup/pricing page needs a login and could not be read: [colab.research.google.com/signup](https://colab.research.google.com/signup)
- **Session limits:** free, Pro and Pay As You Go notebooks run for at most 12 h, depending on availability. Pro+ supports up to 24 h of continuous execution when there are enough CU. Idle timeouts are unpublished. When CU run out, paid users fall back to free-tier limits. Background execution is described as a higher-plan feature — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **Colab Enterprise (a separate, dollar-billed GCP product):** an Iowa list price from Sept 2026 shows a T4 at about $0.42/h and an H100 configuration at up to about $11.76/h — [Colab Enterprise pricing](https://cloud.google.com/colab/pricing)
- **XLS-R-300m:** published W2V2-AASIST recipes fine-tune the 300M XLS-R front end end to end. One 2026 variant uses 5 epochs of end-to-end refinement at batch 32. A single RTX 3090 is a typical reported setup. Freezing XLS-R removes over 300M parameters from optimisation and speeds convergence. No paper found reports GPU-hours directly — [W2V2-AASIST model card](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST); [Tak et al. 2022, arXiv 2202.12233](https://arxiv.org/pdf/2202.12233.pdf); [SpAArSIST, arXiv 2606.11674](https://arxiv.org/pdf/2606.11674); [arXiv 2402.17127](https://arxiv.org/pdf/2402.17127)

### Inferences (cost table computed from McCormick's rates; flagged estimates)
**CU and $ for 14 h:**

| GPU | CU for 14 h | Approximate cost |
|---|---|---|
| T4 | ~17 | ~$1.7 |
| L4 | ~24 | ~$2.4 |
| A100-40 | ~76 | ~$7.6 (up to ~210 CU / $21 if the 15 CU/h report is right) |
| A100-80 | ~105 | ~$10.5 |
| G4 | ~122 | ~$12 |

The Pro tier's 100 CU/month covers one A100-40 run only on the lower rate estimate. A 500 CU top-up ($49.99) covers several runs on any GPU.

**XLS-R-300m throughput (UNVERIFIED engineering estimate; no source found):** at 16 kHz, a 4 s crop gives about 200 transformer frames. The 300M model at about 2·params·frames is roughly 0.12 TFLOP forward per clip, plus the CNN feature encoder, so a training step costs about 3x forward. That suggests the following rough sustained training rates with bf16/fp16, AMP and batch 32-64:

| GPU | Clips/s (4 s crops) | Clip-epochs/hour |
|---|---|---|
| T4 | ~15-30 | ~60-100k |
| L4 | ~30-60 | ~100-200k |
| A100 | ~100-200 | ~400-700k |
| H100 | ~2-3x A100 | |

Inference runs about 3-5x faster than training, so a 2,000-clip test set scores in well under 2 minutes on any GPU. Confirm with a 200-step timing probe in the first 10 minutes of the session.

**Fit to 14 h:** an A100 (or G4) with a 4 s crop, AMP and gradient checkpointing turned off could do on the order of 5-8 million clip-presentations in about 12 h. That is far more than the epochs typically needed (a few epochs over 50-200k clips). The realistic budget is therefore wall-clock, data I/O and session limits, not FLOPs.

**Session strategy:**
- Copy the dataset to the local VM disk (`/content`) as a few tar/npz shards rather than reading many small files from Drive.
- Save a checkpoint (model, optimizer, scheduler and RNG state) every 20-30 min to Drive.
- Write the training loop to resume from the latest checkpoint.
- Budget for one disconnect. Pro+ (24 h) removes the 12 h hard cap.
- Keep the evaluation as a separate short run that loads the frozen checkpoint and the frozen threshold.

**GPU choice:** the A100-40GB is the best cost/throughput trade-off if available. The L4 is a fallback (about 1/3-1/4 of A100 speed, 22.5 GB fits batch 16-32 at 4 s). The T4 is too slow for end-to-end XLS-R fine-tuning on a large set within 14 h unless the front end is frozen or partially frozen.

### Gaps
- No official Colab per-GPU CU table exists. The H100 CU rate on consumer Colab is unknown. A100 rates conflict (5.4 vs ~15 CU/h).
- Whether H100 is reliably allocatable on Pay As You Go or Pro in Nov 2026 was not confirmed.
- CU expiry (often quoted as 90 days) was not confirmed from an official source.
- No measured XLS-R-300m training or inference throughput on Colab GPUs was found. The throughput figures above are analytic estimates and must be checked with a timing probe.
- The exact list of Colab tiers that include background execution in Nov 2026 is unclear from the FAQ wording.
