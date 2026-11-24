# Bilingual detector and pilot decision

## Candidates considered before allocating compute

| Idea | Reason to test | Limitation or counterevidence | Incremental cost and decision |
|---|---|---|---|
| A. Matched real/synthetic utterances | The dev rate experiment and paper 32 expose label-channel confounding | Few audiobook readers may dominate | CPU matching first, include |
| B. Generate complementary commercial voices | Owner misses and paper 14 provide direct motivation | Service versions change, rights and prompt style matter | Begin with 240 clips/service, use development only |
| C. Multi-layer frozen SSL probes | Papers 13 and 37 find depth matters | Good results on a 2019 task can disappear on new families | One cache per encoder/view, include |
| D. Classical LFCC-style probe | Papers 3 and 48 support compact cepstra | Paper 3 external ITW EER 28.55% | CPU baseline, include |
| E. Adapt final two XLS-R blocks | Papers 13 and 55 motivate limited adaptation | Paper 15 gets 7.10% ITW with frozen backbone and a larger head | 1–3 A100 h if the pilot passes |
| F. Distil to a compact acoustic classifier | Teacher soft targets can supervise a cheaper representation | Student may discard indispensable speech context | 1–3 L4 h, compare same validation IDs |
| G. Deploy the first four or six SSL blocks | Early-layer probes can expose shallow signal | A truncated model retains convolution/positional weights and runtime. Probe size is misleading | Compare representations in pilot; export a shallow encoder in a separate experiment if the acoustic student fails |
| H. Genuine-only density or OC-softmax | Papers 37 and 41 model genuine structure | Novel genuine phone conditions can look anomalous | Deferred one-head ablation |
| I. Paired codec and enhancement auxiliary targets | Papers 32 and 42 show effects on genuine recordings | An ordinary Opus encoder differs from a neural codec and true phone capture | Real codec views first; defer auxiliary prediction |
| J. Raw and spectrogram fusion | Paper 41 reports 6.58% external ITW EER | Two backbones increase memory and latency | Defer until the single-view teacher justifies the expense |
| K. Temporal novelty for partial fakes | Paper 4 directly evaluates annotated splice times | Few-word edits can vanish in the recording mean | Keep timestamps; defer partial-edit claims until event evaluation |
| L. Calibrated abstention | V1 thresholds spanned only .048, despite a long tail of short files | With very few genuine speakers a 2% estimate is unstable | Two separate held-out folds, CPU fits |
| M. Iterative data acquisition by failure slice | One vendor can hide an unseen-family miss in a pooled metric | Reusing a final failure as validation contaminates confirmation | Use development failures only, one new family at a time |

## Chosen pipeline in ten steps

1. Download paired English MLAAD/M-AILABS and Hindi IndicSynth/Kathbath samples once to Drive. IndicVoices supplies a separate genuine control. Add explicitly consented phone and cloning pairs.
2. Preserve source and target identities, prompt families and parents. Hold out speakers/voices across all five folds and reserve complete generator families. Keep opaque versions labelled as related or unknown in manual additions.
3. Audit original encoding, sample-rate history, bandwidth/tilt, duration, edge silence, loudness, source and paired provenance. Pack original encoded bytes into indexed ~750 MB tar shards. Copy the selected shards to local storage.
4. Decode one audio stream to mono 16 kHz, trim exterior low-energy frames, cut/fade exposed edges, scale level and retain original timestamps. Clip scoring uses up to 4 s of valid samples, with no padded frames in either encoder. The energy gate measures an active span and can accept nonspeech.
5. Pilot two frozen backbones, every layer, pooled shallow/full representations, linear probes, an MLP and a linear-frequency cepstral control. Report paired Opus degradation on the same scorable recordings.
6. Only after the development gate passes, train XLS-R with the last two transformer blocks and statistics head trainable. Use source/language/class-balanced sampling, BCE and lowest matched unseen-family dev EER. Stop after three stale epochs.
7. Train a small frame MLP on 64 log linear-frequency bands and context from adjacent frames. Distil the teacher’s window logit at temperature 2 with a supervised BCE term. This pluggable student removes the SSL backbone from serving. A training view is drawn independently of label from clean, real Opus/MP3/AAC, 8 kHz round trip, 24 kHz round trip, simulated playback, noise or RawBoost impulsive signal-dependent noise. [RawBoost authors](https://github.com/TakHemlata/RawBoost-antispoofing)
   **Owner requirement (25 Nov): train and compare TWO students on the same validation IDs before choosing.** (a) the compact acoustic student above (~0.15 MB); (b) an SSL student: the first 4–6 XLS-R transformer blocks (CNN front-end and positional convolution included) with the same statistics head, distilled from the same teacher with the same views and steps, int8-quantized (expected ~50–90 MB). Report for both: held-family EER and recall at the common genuine FPR, Hindi/English EER, Opus penalty, short-clip (<4 s) coverage, teacher–student gap, model bytes and warm/cold CPU latency on the target laptop. Choose the smaller student only if it meets the E4 promotion criteria; otherwise ship the SSL student. Literature reason: compact cepstral models transfer poorly across corpora (papers 3 and 11: 28.55 % and 26.14 % external ITW EER).
8. Quantize the student linear weights to int8 on CPU. Fit separate positive-slope Platt mappings for window and file logits after quantization on calibration speakers. Whole-synthetic/full-genuine training means window probabilities have that reference population. Average the logits across the recording.
9. Choose thresholds on different speakers. Start with the empirical 98th percentile of genuine and 2nd percentile of fake probabilities and require an uncertain interval at least [.40,.60]. Abstain under 1 s; 1–4 s yields a probability and uncertain verdict. Report achieved budgets and determinate coverage by class.
10. Measure teacher/student EER, accuracy and score coverage on validation; record student bytes and CPU latency. Freeze the release and scoring code before the custodian's one-time evaluations. Use PRD §7 and the opt-in v3 backend for inference.

## Resource estimates and decisions

These are planning ranges, using the owner's approximations of 5 units/L4-hour and 12.5 units/A100-hour. Actual Colab charges vary by allocation. CPU data preparation does not require an accelerator. Measure elapsed time on the first 100 clips before attempting a complete run.

| Experiment and hypothesis | Resource estimate | Metric and decision |
|---|---|---|
| E0. Check lineage and nuisance-only discrimination | CPU, no paid API | Unique hashes/identities across folds. Symmetric nuisance AUC ≤.60. If source alone predicts label, verify explicit original-parent joins and report the structural confound. Rematch acoustic conditions if raw signal probes fail. |
| E1. Compare old v1 to fresh matched modern development audio | CPU or T4, ~0.1–0.5 h; T4 units unverified | Cross clean/phone recording style with original/Opus channel. Compare median score, EER and per-family misses. |
| E2. Compare small, frozen, and handcrafted probes | L4 .25–1 h target, ~1.25–5 units; T4 alternative | Use 1,000–3,000 train clips/class. At least one SSL representation must reach **unseen-family EER ≤20%**, beat LFCC by **≥3 percentage points**, lose **≤5 points** under Opus, keep both Hindi and English selection EER **≤25%**, and score **≥95%** of selected clips. Report speaker CIs. Insufficient coverage or missing language → no-go. |
| E3. Partial teacher adaptation adds transfer | A100 1–3 h, ~12.5–37.5 units | Compare held-family EER and recall at common FPR with best frozen representation. If no reproducible benefit, investigate data before a second A100 run. |
| E4. Student retains transfer after quantization (two students: compact acoustic vs 4–6-block XLS-R, same teacher and validation IDs) | L4 2–5 h for both | Promotion target ≤3-point EER loss vs teacher, ≤90 decimal MB model, ≤3 s warm median per 30 s on target laptop. Benchmark the actual release. |
| E5. Decisions stay useful on validation | CPU or L4 .25–1 h, ~1.25–5 units if L4 | Aim for ≤2% false accusations and ≥50% determinate coverage per supported language. Report numerator/denominator, uncertainty and false reassurance. Failure → keep experimental. |
| E6. Sealed confirmation | Evaluator only; timing depends on sets | Freeze source revision, serialized model, calibration/threshold file, manifest hashes and generator taxonomy. Evaluate once per release, each set. New design requires new confirmation data for claims about generalization. |

E2–E5 total about 20–63 units on the given assumptions. The 35 GB transfer and encoder downloads are separate from per-run compute. Optional repeated seeds cost more. On a T4, paired clean/Opus and two SSL encoders may exceed one hour; cache reuse makes an interrupted run resumable. First test a small number of development clips and record actual rate.

## Expected outcomes and measurements available now

The 90 MB target is generous for this student. An **untrained architecture fixture** serialized to **151,661 bytes (~0.152 MB)** of model weights and TorchScript structure. On an AMD Ryzen 5 7600X desktop CPU with two threads, the complete bytes→verdict path on synthetic 30 s PCM took **0.045 s warm median** across six timed calls and **0.067 s first call after model loading**. Python, PyTorch, SciPy and FFmpeg runtime sizes are additional. A phone runtime port and a real laptop benchmark remain open. Current PyTorch produces deprecation warnings for eager dynamic quantization and TorchScript on Python 3.14. The tested deployment is Linux CPU.

For planning, allow **−5 to +10 points** of absolute held-family recall change at a fixed genuine FPR versus a matched v1 baseline. This is a sensitivity range: the lower end allows degradation under new channel mixtures; the upper end motivates testing family coverage. An improvement outside this interval is possible and there is no statistical estimate yet for Gemini, Microsoft, Hindi or the trained student. The 3.35-point paired IndieFake EER gain from x1 concerns *known* generators. A plausible student tradeoff to investigate is **0–10 points higher EER**; stop at the stricter 3-point promotion criterion. Record teacher and student measured values after owner-run training.

## Differences from published work

| Dimension | Project | Paper precedent or stronger alternative |
|---|---|---|
| Modern generators and Hindi | Consent-reviewed bilingual matched pairs; vendor, version and family holdouts | 14 uses ElevenLabs, 18 tests CosyVoice 2, 34 builds OpenVoice data, 59 uses RVC. 17 evaluates Bangla. IndicSynth has broader Indic-language coverage. |
| Preprocessing | Shared decoder, energy cut, fades, valid-frame pooling and timeline offsets | 36 explicitly models breathing and silence; 4 handles precise splice localization. |
| Encoder and size | Frozen probes first, partially adapted teacher and separately exported int8 acoustic student | 13/41/55 are stronger published external ITW results. 1/54 report compact architectures. |
| Channels | Paired Opus and genuine phone validation, symmetric augmentation | 32/42 distinguish neural processing from synthesis more extensively. |
| Shortcut checks | Parent/voice/text isolation plus metadata and spectrum controls | 3 has speaker-independent evaluations; 7 has multiple perturbation conditions. |
| Calibration | Separate scoring, threshold and validation speakers plus explicit abstention | 1 compares calibration losses; 14 uses calibrated likelihood-ratio cost; 33 includes calibration. |
| Evaluation | One ledger entry per frozen release/set and speaker-clustered intervals | 4 has richer partial-edit annotations, 18 has label-efficiency and controlled attack comparisons. |
| Deployment | Existing local FastAPI/React contract, model size and per-file CPU time | 44 demonstrates an Android app, 58 an accessible hosted demo. A future mobile port here remains an engineering goal. |
