# Train on target families or miss 90%

No existing model, combination or threshold in this project can reach 90% on all four families at 15% false alarms. The plan below is the best attempt the 14 hours allow, but the honest expected outcome is weaker than the target. The best measured operating point today is the frozen release v1 × Teacher V1 z-mean combo. At 15.4% genuine false alarms (FA) it detects **86.7% of Chatterbox, 77.9% of ElevenLabs v3, 98.8% of FishTTS and 82.9% of Gemini**, on a development panel that has been inspected many times. Even choosing the cutoff on that same panel, reaching 90% on every family costs **25.5% FA** for the combo and **37–49% FA** for release v1 alone. Threshold tuning therefore cannot close the gap. The only lever with evidence behind it is training on audio from the target families, plus far more diverse genuine speech to control the reader-to-reader shift in false alarms (0–53% FA depending on the reader). In the 14 hours from 18:45 IST 26 Nov to 08:45 IST 27 Nov, the realistic plan is as follows:

- Screen two broadly trained open checkpoints zero-shot.
- Train a release-style detector on all ten MLAAD English folders from the target vendors. These are labelled "familiar family", with voices and readers held out.
- Pre-render a telephone/VoIP codec chain applied identically to both labels.
- Freeze the cutoff on separate genuine calibration speakers.
- Run a single, newly signed-off independent evaluation.

My judgment is that FishTTS and Chatterbox will very likely pass. ElevenLabs v3 and Gemini are roughly a coin flip on unseen voices. The binding risk is genuine FA on new speakers. Overall, the chance that all four families plus the FA bound pass a prespecified independent test by 08:45 is about one in three. A real phone-call claim is out of reach tonight: only simulated channels can be measured. Three steps need a separate owner decision before compute starts:

- spending on paid synthesis (ElevenLabs/Gemini, about $25–40);
- accepting gated datasets and licences (MLAAD if not already accepted, RTCFake, the non-commercial checkpoint licences);
- signing off a new evaluation protocol, because the 90% claim replaces the 70% protocol 27.

## Threshold tuning is exhausted at 25.5% false alarms

Every number the project has is zero-shot transfer. Release v1 is a frozen XLS-R-300M encoder with three small heads, trained on ASVspoof 5 and IndieFake. None of the four target generators is in its training data. Its dev results are strong on old benchmarks: pooled AUC 0.990, EER 5.4% ([release.json](file:///home/saurav/deepfake_project/detector/runs/release/v1/release.json)). At its shipped cutoff it catches **89.6% of FishTTS but only 6.1% of ElevenLabs v3 and 7.2% of Gemini**, at 1/697 FA ([13_HANDOFF, day-target-1 branch](file:///home/saurav/deepfake_project/.claude/worktrees/v2-overnight)). Teacher V1 fine-tunes the last two XLS-R layers on PhonemeDF XTTS-v2 against LibriSpeech. It runs only under the `exploratory-failed-audit-v1` exception, after its nuisance audit failed with upper-band AUC 0.817. The student, export, evaluate and sealed steps all refuse it ([27 protocol §8](file:///home/saurav/deepfake_project/.claude/worktrees/v2-overnight/college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md)). The combo therefore inherits a component that cannot be released, and the 90% plan should not depend on it.

The in-sample cost of reaching 90% was recomputed from local score files, and it settles the threshold question:

| Model | FA needed for all families ≥70% | FA needed for all families ≥90% |
|---|---|---|
| Release v1, single 4 s crop | 22.5% | **48.5%** |
| Release v1, full recording | 19.4% | **37.0%** |
| Release × Teacher combo | 8.5% | **25.5%** |
| Teacher V1 alone | 82.2% | 93.5% |

These figures are optimistic, because each cutoff was chosen on the panel it is scored on ([internal audit, fa15-descriptive.json and calc.py recomputation](file:///home/saurav/.claude/jobs/63d812f9/tmp/data/v2/day-target-1/fa15-descriptive.json)). The genuine side makes things worse. Across six reader-disjoint rotations, realised FA ranged from **0/215 to 136/255**. At the frozen cutoff, one reader (elliot_miller) alone produced 77/255 false alarms, or 30.2% ([26_MODERN_TTS_70_AT_15.md, worktree-modern-tts-70-at-15](file:///home/saurav/deepfake_project/.claude/worktrees/v2-overnight)). The literature explains why calibration tricks will not fix this. Any strictly increasing transform (z-norm, temperature scaling, shift) leaves the ranking unchanged. A threshold carried over from ASVspoof 2019 to In-the-Wild rejected 78.7% of genuine speech. AS-norm with an unlabelled cohort raised In-the-Wild EER from 11.2% to 60.2% ([Zhou & Wang 2026](https://arxiv.org/abs/2606.21584)). The channels are no better. At the clean cutoff, combo FA is **26.3% on simulated phone audio and 27.7% on Opus** ([26_MODERN_TTS_70_AT_15.md](file:///home/saurav/deepfake_project/.claude/worktrees/v2-overnight)).

Three facts constrain what can be rebuilt tonight:

- **The independent evaluation has never run.** Protocol 27 (commit `4dfda40`) and its scoring script (`c1fccda`) exist. The report step is untested, and there is no sign-off file. All workers stopped at about 02:10 IST 26 Nov on usage limits.
- **Part of the intake was lost with the dead intake VM**, as the lead claimed at 18:01 IST; this is not yet confirmed on Drive. The loss covers the second relay of 60 files (SpeechFake shards 29–49, AMI, SCDF and the RVCBench `pairs.jsonl`/lineage files), the 2/3/4 s trim report and the 5-fold acoustic check. Both VMs (`v3-t4p`, `intake-cpu`) are gone. `pairs.jsonl` can be regenerated from pinned metadata in 15–30 CPU-minutes. The frozen candidate, the release v1 fallback scores and the first 59-file intake are reported safe on Drive ([JOBS/8820e109 timeline](file:///home/saurav/.claude/jobs/8820e109/timeline.jsonl)).
- **The last recorded Colab balance was 148.83 CU**, at 01:28 IST 26 Nov. The current balance is unknown.

## In-family data is the only lever with evidence behind it

Published detectors collapse on LLM-era TTS. Eight pretrained detectors tested without fine-tuning on ten 2025 systems, Chatterbox ML among them, scored **29–58% EER** ([VoxENES 2026](https://arxiv.org/abs/2607.11706)). Back-end choice matters little: Nes2Net-X, XLSR-Mamba and SLS on XLS-R-300M sit within about two EER points of each other on In-the-Wild ([Nes2Net](https://arxiv.org/html/2504.05657v2)). Data matters far more. Training XLS-R+SLS on the diverse AUDETER corpus cut In-the-Wild EER from 7.46% to **1.87%** ([AUDETER](https://arxiv.org/abs/2509.04345)). Small amounts of target-family audio are very effective. With ElevenLabs held out of training, 50–100 ElevenLabs clips brought ADD-GP from 40% EER to **0.5–1.1%**, and plain full fine-tuning of SSL-AASIST to about 11% ([ADD-GP, Interspeech 2025](https://arxiv.org/html/2505.23619v1)). That ElevenLabs was an older model, tested against matched LibriSpeech-style genuine audio. It shows the fake side can be learned quickly. It does not show the genuine side will hold up on new speakers.

The target-family data available is larger than the project has been using. MLAAD's current revision (`30c3dec7…`, the same one pinned on Drive) has **1,000 clips in each of ten relevant English folders**: Chatterbox, Chatterbox-Turbo, Resemble.ai, ElevenLabs-v3, ElevenLabs-Turbo-v2.5, ElevenLabs-v2-Multilingual, FishTTS, Fish-S2-Pro, Openaudio-S1-Mini and Gemini-3.1-Flash-TTS. That is about 10,000 clips and 3.1 GB, under CC BY-NC 4.0 and click-through gating ([HF tree API](https://huggingface.co/api/datasets/mueller91/MLAAD/tree/main/fake/en); [MLAAD card](https://huggingface.co/datasets/mueller91/MLAAD)). Locally, the four core folders split across three US M-AILABS readers and one UK reader (Elizabeth Klett, 320–345 clips per family). The dev panel used 250 per family from en_US. Two structural traps follow:

- **ElevenLabs v3 and Gemini use the same 26 and 30 stock voices in en_US and en_UK.** A US-train/UK-test split therefore does not hold voices out. Any voice-disjoint split has to be cut within those voice sets, and anything less must be labelled "familiar voice" ([internal audit, MLAAD CSV recount](file:///home/saurav/.claude/jobs/435e59bf/tmp/meta/mlaad)).
- **The MLAAD folder names may not match the owner's targets.** MLAAD's Gemini is the 3.1-flash preview, while the current models are `gemini-3.8-flash-tts` and `-lite-tts` ([Gemini docs](https://aistudio.google.com/docs/speech-generation)). Its "FishTTS" folder is probably an older Fish Speech, separate from Fish-S2-Pro. The owner must say which versions define each family.

Two broadly trained open checkpoints can be tested in the first hour. Neither has published results on the four targets.

- **AntiDeepfake** (NII; XLS-R-1B/2B, MMS-1B). It was trained on 74k hours including MLAAD and ASVspoof5 and reports 1.23% In-the-Wild EER, but still 25.8–27.8% on Deepfake-Eval-2024 ([arXiv 2506.21090](https://arxiv.org/html/2506.21090v4)).
- **DF Arena 1B**, which also includes MLAAD and reports 0.91% In-the-Wild EER and 17.25% on ASVspoof5 ([HF card](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1)).

Both are non-commercial (CC BY-NC-SA 4.0 for AntiDeepfake; [model card](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake-nda)), so the owner must accept that before they become a release candidate. Their MLAAD training also creates a contamination risk. If they saw MLAAD clips that are on the dev panel, their dev numbers are inflated. Treat them as front-ends or teachers to measure, not as ready answers. A cheaper fallback also has evidence: a logistic probe on frozen XLS-R layer 18 recalled 88.8% of spoofs per generator across 84 MLAAD English generators, trained on ASVspoof2019 alone ([Pagon et al. 2026](https://arxiv.org/abs/2606.26384)). Feature caching plus a small head is therefore a credible sub-hour training loop.

The genuine side must become far more diverse, or the reader shift will sink the FA bound. Ungated sources with speaker labels can be downloaded tonight ([HF API listings](https://huggingface.co/api/datasets/mythicinfinity/libritts_r)):

- LibriTTS-R and VCTK (CC BY 4.0);
- GLOBE V2 (CC0, accented speech);
- a streamed slice of People's Speech;
- the 95 AMI segments already held.

AntiDeepfake used three times as much genuine as fake audio, for this reason ([arXiv 2506.21090](https://arxiv.org/html/2506.21090v4)). The project's own failure mode argues for a specific safeguard: the 0.60 shortcut audit. MLAAD fakes paired against LibriSpeech or other corpora risk the same corpus and band shortcut that sank Teacher V1. Matched M-AILABS en_US parents are on Drive, but en_UK parents are not. Before training, the owner must decide whether the audit gate is binding, or whether a failed audit becomes a reported caveat. Reporting it as a caveat would let the run proceed, with a weaker claim.

## Credible 90% claims need 200 fakes per family

The independent test has to be large enough to tell 90% from 85%. With Wilson 95% intervals, an observed 93% needs about 64 clips for a lower bound of 85%. A model whose true rate is exactly 90% passes an "observed ≥90%" check only **55–58% of the time at any sample size**, so a reliable pass needs a true rate near 93% ([Brown, Cai & DasGupta 2001](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full); power computed in the evaluation notes). Voices cluster errors. With the Kish design effect 1+(m−1)·ICC, 200 fakes from 10 voices at an intra-voice correlation (ICC) of 0.2 shrink to about 42 effective clips ([design effect](https://en.wikipedia.org/wiki/Design_effect)). The protocol should therefore fix these minimums:

- at least 200 fakes per family, from at least 40 distinct voices or prompts;
- at least 300 genuine recordings from 60–100 speakers;
- per-speaker caps;
- a cluster bootstrap reported next to Wilson intervals.

Protocol 27 already specifies 2,000 bootstrap repetitions with seed 20261126. Requiring all five tests (four families plus FA) to pass at about 0.96 power each gives about 0.82 overall, and that holds only if the model has real margin.

This is where paid synthesis matters. The planned MLAAD en_UK test set (1,333 clips) has one reader and reuses the ElevenLabs and Gemini voices. Fresh test audio would be independent in voice, text and generation script, which is the ASVspoof 5 standard ([ASVspoof 5](https://arxiv.org/html/2408.08739v1)). The 2,000-clip estimates below come from vendor pricing and docs:

| Family | Source | Estimated cost and time for 2,000 clips |
|---|---|---|
| ElevenLabs v3 | Paid API | About $14–18 ([pricing](https://elevenlabs.io/pricing/api)); Voice Library voices need a paid plan |
| Gemini 3.8 Flash TTS | Paid API | About $3 ([pricing](https://aistudio.google.com/docs/pricing)); 30 prebuilt voices plus an Extended Voice Library with hundreds more |
| Fish S2-Pro | Open weights | About 1–2 A100-hours (RTF 0.195 on an H200; [fish-speech](https://github.com/fishaudio/fish-speech)) |
| Chatterbox | Open weights (MIT) | Unmeasured; every output carries a Perth watermark that a detector could learn as a shortcut ([chatterbox](https://github.com/resemble-ai/chatterbox)) |

The repo already has a Gemini path (`pipeline_v3/generate.py`, with a cost dry-run) but no ElevenLabs path. A 300-clip independent test per family is about a fifth of these costs. Genuine test speakers must be disjoint from training and calibration speakers. Note also that XLS-R was pretrained on LibriSpeech and MLS, so LibriTTS-R test speakers carry a pretraining overlap that should be disclosed ([evaluation notes, ASVspoof 5 practice](https://arxiv.org/html/2408.08739v1)).

## Call channels can be simulated tonight but not proven

There is no public real-call dataset containing ElevenLabs v3, Gemini or Fish Speech. RTCFake is the only large set with real VoIP transport: about 600 hours across Zoom, Telegram, WeChat and other apps. It includes OpenAudio-S1 and ChatterboxVC, but none of the four target TTS families. It is gated and non-commercial ([RTCFake](https://arxiv.org/html/2604.23742)). Microsoft's set of real fraud calls with ElevenLabs is private ([Delgado et al.](https://arxiv.org/html/2509.26471v2)). Real transport costs a lot. On RTCFake, an XLS-R+AASIST trained offline goes from 5.42% EER on clean audio to **13.79%** on transmitted audio. Training on transmitted audio recovers to 8.35% ([RTCFake](https://arxiv.org/html/2604.23742)).

Simulation recovers most of that loss if it is done in the right order. Pindrop sent MLAAD audio over real Twilio VoIP/PSTN. For the narrowband chain, ordering mattered:

| Narrowband chain | EERc on real telephone audio |
|---|---|
| Downsample to 8 kHz only | 24.7% |
| RawBoost, then 8 kHz | 23.3% |
| 8 kHz, then RawBoost | 19.5% |
| Simulated G.711 + quantize + RawBoost | **15.1%** (beat training on real calls, 18.9%) |

([Klein et al., Odyssey 2026](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf)). Codecs, not bandwidth, cause most of the degradation, and an unseen codec at test time is the main failure. RawBoost alone took XLS-R+AASIST from 7.65% to 0.82% EER on ASVspoof 2021 LA's real VoIP/PSTN conditions ([Tak et al. 2022](https://arxiv.org/pdf/2202.12233)).

The local ffmpeg has the needed encoders: G.711 µ/A-law, GSM, AMR-NB, Opus, Speex, G.722 and G.726. torchaudio's codec API was removed in 2.9, so pre-rendering with ffmpeg subprocesses is the practical route ([torchaudio 2.8 docs](https://docs.pytorch.org/audio/2.8/generated/torchaudio.io.AudioEffector.html)). The recipe:

- Draw the condition from the same distribution for both labels.
- Pre-render each clip two or three times.
- Apply RawBoost after the codec.
- Hold GSM or AMR-NB out entirely, as the unseen codec.
- Choose one cutoff so that FA stays at or below 15% on every channel stratum (the maximum over strata, not the pooled figure).

A real-call test set needs about 3–4 hours of real-time transmission: Twilio playback, a carrier call and WhatsApp/Zoom, with the same clips through each route. It also needs consent and some spending ([Klein et al.](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf); [Twilio guidance](https://support.twilio.com/hc/en-us/articles/360011435554-Best-Practices-for-Using-Twilio-to-Manage-and-Record-Communications-Between-Users)). It does not fit alongside training tonight. Tonight's claim should be labelled "simulated channel", with real calls as tomorrow's owner-approved task.

## Hour-by-hour plan, 18:45 IST 26 Nov to 08:45 IST 27 Nov

The plan uses one A100 session for GPU work (measured at 8.16 CU/h on this account), a CPU or L4 side session for downloads, codec rendering and open-weight generation, and the local RTX 5060 for scoring. Colab Pro and pay-as-you-go sessions cap at 12 hours ([Colab FAQ](https://research.google.com/colaboratory/faq.html)). The training loop must therefore save a checkpoint every 30 minutes to Drive and resume. Budget about 10 A100-hours (about 82 CU) plus about 5 CU of CPU/T4 time. If the balance is now below about 120 CU, top up 500 CU ($49.99) ([McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)). New code goes under `pipeline_v3/` or a copy, because AGENTS.md forbids edits to `detector/` and `deepfake_model/`. No audio or weights go into git.

| Time (IST) | Work | Owner decision / checkpoint |
|---|---|---|
| 18:45–19:15 | Owner answers decisions D1–D6 (below). Start the A100 session and record the CU balance. List Drive to confirm what survived (`modern-intake/v1` 59 files; the MLAAD folders present). | **D1–D6 required** |
| 19:15–20:15 | **GPU:** zero-shot AntiDeepfake XLS-R-1B / MMS-1B and DF Arena 1B on the 1,398-recording dev panel plus libri-dev. Report per-family detection at the libri-dev 15% cutoff and worst-reader FA. **CPU:** download the ten MLAAD English folders (3.1 GB) and LibriTTS-R, VCTK and GLOBE subsets. Regenerate `pairs.jsonl`. **API (if D2):** launch Gemini 3.8 and ElevenLabs v3 test generation from non-M-AILABS texts and voices not used in MLAAD. | **G1 (20:15):** pick the front-end. If a 1B model beats release v1 by ≥10 points on ElevenLabs/Gemini, use it. Otherwise stay on XLS-R-300M. |
| 20:15–21:30 | Write protocol 28 and freeze the splits. **Train:** MLAAD en_US, all ten vendor folders, with about a third of ElevenLabs/Gemini voices held out entirely; genuine from M-AILABS en_US, LibriTTS-R train and VCTK/GLOBE train speakers. **Calibration:** held-out voices plus disjoint genuine speakers. **Test:** en_UK MLAAD plus fresh generated audio, against ≥300 genuine recordings from ≥60 unseen speakers. Write the MLAAD→`fixed4s-rmsnorm-v1` cache intake. Run the trivial-feature audit on train. Start ffmpeg codec pre-rendering. | **G2 (21:30):** audit outcome under the D5 rule. **D3 sign-off written (one writer, write-once)** before any test score exists. |
| 21:30–23:00 | Cache layer-wise features. Train a release-style head (softmax layer mix plus MLP) on clean and codec features, about 1 h. Score calibration voices and speakers. Meanwhile, generate Fish S2-Pro and Chatterbox test clips on the side GPU from LibriTTS-R test-speaker references. | **G3 (23:00):** on held-out calibration voices, is each family ≥90% at ≤15% FA? If ElevenLabs/Gemini are <80%, the fine-tune is mandatory. If FA by speaker swings >25 points, add more genuine sources before fine-tuning. |
| 23:00–03:30 | Fine-tune the last 2–4 encoder layers (or LoRA) at lr 1e-5. Use a sampler balanced across families and genuine sources, and identical codec/RawBoost augmentation with GSM held out. Checkpoint every 30 min, and resume across any disconnect. | **G4 (01:30):** mid-run calibration score. If it is no better than the G3 head, stop and keep the head. |
| 03:30–04:30 | Choose the cutoff on the calibration genuine speakers as the maximum over clean and simulated-codec strata. Freeze the model hash and cutoff, and write the sign-off binding. | **G5 (04:30):** frozen, with no further changes. |
| 04:30–06:30 | Single-pass independent evaluation: clean, the seen-codec stratum and the held-out GSM stratum. Report per-family Wilson and voice-cluster bootstrap intervals, plus pooled and worst-speaker FA. Add RTCFake English online audio as an external channel check only if D4 access has arrived. | Read the results once; nothing is retuned. |
| 06:30–08:00 | Write the report, including counterfactual checks (silence padding, ±6 dB loudness, genuine audio through the fake codec). Keep 45 minutes of buffer for one session loss. | |
| 08:00–08:45 | Owner readout: pass/miss/insufficient per family, the familiarity labels, and the next-day real-call collection proposal. | |

The owner decisions are distinct and cannot be inferred from "use more CU and download more data":

- **D2 (paid TTS synthesis):** about $25–40 total for ElevenLabs v3 and Gemini 3.8 test sets, possibly an ElevenLabs Creator plan ($22).
- **D1 (gated datasets and licences):** MLAAD click-through if this account has not already accepted it, the CC BY-NC-SA terms for AntiDeepfake and DF Arena, RTCFake registration, and SpoofCeleb (2–3 day approval, so it is out of tonight's scope).
- **D3 (new evaluation protocol):** sign-off for a 90% claim with the splits, cutoff rule and pass rule. Protocol 27 states that an adaptation is not evaluated under it without a new sign-off.
- **D4:** whether RTCFake is requested at all.
- **D5:** whether a failed 0.60 shortcut audit blocks or merely annotates.
- **D6:** which model versions define "FishTTS" and "Gemini TTS", and whether a "familiar family, unseen voice" result counts toward the goal.

When behind schedule, cut in this order:

1. The real-call work and RTCFake check.
2. The 1B front-end (stay on XLS-R-300M).
3. The fine-tune (ship the G3 head).
4. Opus as a separate stratum.
5. Shrink fresh test sets to 200 per family.

Never cut freezing before scoring, per-family reporting or speaker-disjoint genuine audio.

## Conclusion

The project has spent its effort on calibration and protocol rigour around models that never saw the target generators. That rigour is worth keeping, but it has hit a ceiling no cutoff can lift. Tonight's plan turns the problem into supervised in-family learning against a much wider genuine population. That changes what a pass would mean. A 90% result tomorrow morning would be a "familiar family, unseen voice and text" claim on simulated channels. That is honest and useful for an app, but it does not show the detector can handle a new generator, and it says nothing yet about real phone calls.

The most likely morning outcome is mixed. FishTTS and Chatterbox will probably clear 90%, ElevenLabs v3 and Gemini will land in the 80s or low 90s, and genuine FA will hinge on how diverse the calibration speakers were. If that happens, the next investment is not more model tuning. It is about 2,000 freshly generated clips per commercial family from hundreds of voices, and a consented real-call recording session, both of which wait on owner decisions.
