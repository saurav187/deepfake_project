# Call-channel (telephony / VoIP) robustness for TTS and voice-clone detection, as of Nov 2026

Scope: how to train for and credibly evaluate phone-call and VoIP robustness for the team's XLS-R-300m detector (16 kHz, 4 s crops; targets Chatterbox, ElevenLabs v3, Fish Speech, Gemini TTS; goal is >=90% detection per family at <=15% genuine false alarms), given ~14 h and paid Colab.
Terms used throughout: **SIM** = codec applied in software; **VoIP** = real transport through a VoIP app or SIP/PBX; **REPLAY** = loudspeaker into a phone microphone in a room; **PSTN** = real carrier or mobile network call.

## Datasets with real or realistically transmitted calls (access, size, licence, modern TTS coverage)

### Takeaway
Only RTCFake (VoIP apps, English and Chinese, about 600 h) is public and large with real transport. It includes ChatterboxVC but none of the four target TTS families. ASVspoof 2021 LA has real VoIP and PSTN transport, but its attacks are old (2019-era). The most realistic phone-fraud set (Microsoft's "Fraud Academy", with ElevenLabs and other engines over real phone calls) is private. No public dataset found has ElevenLabs v3, Gemini TTS or Fish Speech sent over a real call, so the team must make its own small real-call test set.

### Cited Findings
- **RTCFake** (Xue, Yi et al., ACL 2026 Findings; arXiv 2604.23742):
  - Size: about 600 h, 307 speakers, paired offline (clean) and online (transmitted) subsets. Samples: real 23,863 offline and 31,024 online; fake 112,290 offline and 149,899 online — [arXiv 2604.23742](https://arxiv.org/html/2604.23742); [ACL Anthology](https://aclanthology.org/2026.findings-acl.285/)
  - Generators: TTS G01–G07 are F5-TTS, OpenAudio-S1, VoxCPM, LLaSA, IndexTTS2, Doubao and SparkTTS. VC G08–G10 are CosyVoice, SeedVC and ChatterboxVC. Real speech comes from LibriHeavy (English) and Chinese-Lips — [arXiv 2604.23742](https://arxiv.org/html/2604.23742)
  - Platforms: P01–P07 are Zoom, QQ, WeChat, DingTalk, Lark, VooV and Telegram. Train uses P01–P02 only; Eval adds unseen platforms (P03–P07), unseen generators and unseen noise (office, coffee, echo, rain, footsteps, keyboard) — [arXiv 2604.23742](https://arxiv.org/html/2604.23742)
  - Recording method: two PCs; audio is played through a virtual audio device into the app and recorded at the receiver with OBS; Whisper is used to discard mismatched transcripts. This is VoIP transport with no microphone or room — [arXiv 2604.23742](https://arxiv.org/html/2604.23742)
  - Hosted at HF `JunXueTech/RTCFake`. The paper text does not state a licence; the team brief says registration is required and use is noncommercial — [arXiv 2604.23742](https://arxiv.org/html/2604.23742)
  - Models trained on other public corpora generalise badly to RTCFake (XLS-R+AASIST with RawBoost; EER on RTCFake overall): ASVspoof2019 50.28%, ASVspoof5 43.71%, CodecFake 41.90%, SpoofCeleb 34.06%, CD-ADD 33.95% — [arXiv 2604.23742](https://arxiv.org/html/2604.23742)
- **ASVspoof 2021 LA**:
  - Real transport, not simulation. C2 and C4–C7 went through an Asterisk PBX, one SIP call per utterance, on a LAN or France to Italy or Singapore. C3 was a real PSTN call from a mobile phone in Spain, recorded at a SIP endpoint — [ASVspoof 2021 paper](https://arxiv.org/html/2210.02437v3)
  - Conditions: C1 none (16 kHz); C2 a-law (8 kHz, VoIP, 64 kbps); C3 unknown + µ-law (PSTN+VoIP); C4 G.722 (16 kHz, VoIP); C5 µ-law (8 kHz, eval only); C6 GSM (8 kHz, 13 kbps, eval only); C7 Opus (16 kHz, VBR 16 kbps, eval only) — [ASVspoof 2021 paper](https://arxiv.org/html/2210.02437v3)
  - Narrowband conditions were harder than wideband ones, with GSM and the PSTN route hardest. The transmission route itself had little effect — [ASVspoof 2021 paper](https://arxiv.org/html/2210.02437v3)
  - Attacks are ASVspoof 2019-era TTS and VC, so no modern commercial TTS (inference from the corpus design; see the paper above).
- **ASVspoof 2021 DF**: codecs are software only (mp3, m4a, ogg, including dual coding) via ffmpeg and sox; no network transport. The best challenge EER was 15.63% — [ASVspoof 2021 paper](https://arxiv.org/html/2210.02437v3)
- **ASVspoof 5**: codec conditions C1–C7 are at 16 kHz and C8–C11 are 8 kHz narrowband (downsample to 8 kHz, encode, upsample to 16 kHz), stored as 16 kHz FLAC. C0 is clean — [ASVspoof 5 paper](https://arxiv.org/pdf/2408.08739)
  - One team saw the largest degradation at 8 kHz Opus (C08), 8 kHz Speex (C10) and neural codecs — [Vicomtech-UGR](https://arxiv.org/pdf/2408.10361)
  - All ASVspoof 5 codecs are SIM, not transport.
- **Delgado et al. (Microsoft), "On Deepfake Voice Detection – It's All in the Presentation"** (arXiv 2509.26471):
  - "Presented" set (about 70K, training only): deepfakes injected into real phone calls (Bluetooth, or a wired analog feed into the phone mic input; Samsung Galaxy A12 and Redmi Note 8 Pro) or played back through an ESI Aktiv monitor or a JBL Bluetooth speaker in an untreated room — [arXiv 2509.26471](https://arxiv.org/html/2509.26471v2)
  - TTS used: ElevenLabs, Play.ht, OpenAI Voice Engine, Mars5, YourTTS — [arXiv 2509.26471](https://arxiv.org/html/2509.26471v2)
  - "Fraud Academy" real-world set: 80 participants role-playing fraud calls, 10 TTS engines, at least 16 calling devices, 5 loudspeaker types; 2,263 segments cut into 20 s clips. It is **private** — [arXiv 2509.26471](https://arxiv.org/html/2509.26471v2)
  - The paper does not say whether calls went over PSTN or VoIP — [arXiv 2509.26471](https://arxiv.org/html/2509.26471v2)
- **SpoofCeleb**: over 2.5M utterances (2,687,292 in the v2 table), 1,251 VoxCeleb1 speakers, 23 attacks generated from TITW-Easy. Gated HF repo `jungjee/spoofceleb`, approval in 2–3 days with an affiliation email. The community mirror `DeepFense/SpoofCeleb` says apache-2.0, which conflicts with the official gated terms — [HF jungjee/spoofceleb](https://huggingface.co/datasets/jungjee/spoofceleb); [arXiv 2409.17285](https://arxiv.org/html/2409.17285v2); [DeepFense mirror](https://huggingface.co/datasets/DeepFense/SpoofCeleb/blob/main/README.md). Its audio is in-the-wild but not phone calls.
- **ADD-C benchmark** (arXiv 2504.12423): SIM codecs AMR-WB, EVS, IVAS, Opus, Speex-WB and SILK × packet-loss rates 0/1/5/10/20% = 30 conditions. No download link was found in the paper — [arXiv 2504.12423](https://arxiv.org/html/2504.12423v2)
- **Challenge-response live-call dataset** (arXiv 2402.18085): contributions from 100 smartphone and desktop users, aimed at real-time voice clones on calls — [arXiv 2402.18085](https://arxiv.org/pdf/2402.18085v1.pdf)

### Inferences
- For the four target families, no public real-call data exists. RTCFake is the best public proxy for *transport* robustness: VoIP noise suppression, AGC and codecs on the genuine and fake sides alike. Use RTCFake **English** online samples (LibriHeavy real versus TTS/VC) as an external channel benchmark, and possibly as training data. Check its licence against the project's use: it is noncommercial, so it is fine for research evaluation but not for a shipped commercial model.
- ASVspoof 2021 LA C3 (real PSTN) and C6 (GSM) are the only public real-PSTN spoof data. They are useful to check that *genuine* false alarms do not explode on 8 kHz transport, not to test modern-TTS detection.
- RTCFake is large. Download only the English online eval split, or a subset, to stay within the 14 h.

### Gaps
- No public "fraud call" or PSTN dataset with ElevenLabs v3, Gemini TTS or Fish Speech was found. I did not check CodecFake, CodecFake+, In-the-Wild, or "VoIP replay" sets in detail; they are codec or in-the-wild data, not calls.
- RTCFake's licence text and the size of its English-only subset were not confirmed from the HF card.
- The release status of Delgado's GitHub test protocol was not checked.

## Channel simulation best practice and tools (applied identically to genuine and fake)

### Takeaway
Build a stochastic chain applied with the same probabilities to genuine and fake: optional room/device effect, then RawBoost, then resample to 8 kHz (or stay at 16 kHz for wideband), then a codec drawn at random from a pool (G.711 µ/A-law, GSM-FR, AMR-NB, Opus 6–24 kbps, Speex, G.722), then optional packet loss, then back to 16 kHz. Pindrop's Odyssey 2026 study shows that applying RawBoost *after* downsampling matters a lot, and that simulated G.711 plus augmentation beat training on real Twilio telephony.

### Cited Findings
- Pindrop (Klein, Tak et al., Odyssey 2026): their real-call path used the Twilio Programmable Voice API to send MLAAD audio over **VoIP and PSTN** ("Tel."). The SIM codecs were Opus via ffmpeg (`libopus`, ogg container, bitrate drawn from {24, 32, 48, 64} kbps) and G.711 A-law and µ-law via ffmpeg — [Klein et al. 2026](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf)
- Same paper, on ordering of the narrowband chain (EERc on real Tel.):
  - 8k only: 24.7%
  - RawBoost→8k: 23.3%
  - **8k→RawBoost: 19.5%**
  - 8k→Quantize: 19.2%

  The authors attribute the benefit to RawBoost masking resampling and roll-off artifacts that the model would otherwise overfit to — [Klein et al. 2026](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf)
- Same paper, quantization augmentation: requantise int16 audio by a factor Q, with Q drawn so that SNR is 35–60 dB. For narrowband RawBoost they set 8 kHz sampling and a 4 kHz maximum frequency. When combined with quantization, RawBoost is applied with p=0.5 — [Klein et al. 2026](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf)
- Same paper: adding augmentation *on top of* G.711 or real Tel. training data hurt performance on Tel., because G.711 is already close to VoIP/PSTN — [Klein et al. 2026](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf)
- RawBoost has three noise types: linear and non-linear convolutive, impulsive signal-dependent, and stationary signal-independent. Code: [github.com/TakHemlata/RawBoost-antispoofing](https://github.com/TakHemlata/RawBoost-antispoofing) — [Klein et al. 2026](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf)
- XLS-R (wav2vec 2.0) + AASIST with RawBoost reached 0.82% pooled EER on ASVspoof 2021 LA, which is real VoIP and PSTN transport, and 2.85% on DF. Without augmentation the figures were 7.65% (LA) and 7.69% (DF). The authors call this an almost 90% relative improvement on LA — [Tak et al. 2022, arXiv 2202.12233](https://arxiv.org/pdf/2202.12233)
- UR-AIR's ASVspoof 2021 system grouped codec augmentation into cellular (GSM, AMR-NB, AMR-WB) and landline (G.711, G.726) channels — [arXiv 2107.12018](https://arxiv.org/pdf/2107.12018)
- A study of data augmentation with G.711 and AMR telephony codecs reports that it "improves EER and min t-DCF significantly" — [arXiv 2110.10491](https://arxiv.org/pdf/2110.10491)
- ASVspoof 5's narrowband recipe is: downsample to 8 kHz, apply the codec, upsample to 16 kHz — [ASVspoof 5](https://arxiv.org/pdf/2408.08739)
- Packet-loss simulation at 1–20% across AMR-WB, EVS, Opus and SILK is used in ADD-C. Training on its augmented set (1.83M utterances, about 5× the original) kept EER stable across conditions — [arXiv 2504.12423](https://arxiv.org/html/2504.12423v2)
- The torchaudio codec tooling is gone. `torchaudio.functional.apply_codec` and `torchaudio.io.AudioEffector` were deprecated in 2.8 and removed in 2.9, and I/O moved to TorchCodec — [torchaudio 2.8 AudioEffector docs](https://docs.pytorch.org/audio/2.8/generated/torchaudio.io.AudioEffector.html); [torchaudio stable](https://docs.pytorch.org/audio/stable/torchaudio.html); [pytorch/audio#3902](https://github.com/pytorch/audio/issues/3902)
- Codec references: G.729 is 8 kbps narrowband and degrades under repeated transcoding. AMR-NB at 12.2 kbps is roughly on par with G.729 — [VoIPmonitor codec guide](https://www.voipmonitor.org/doc/Audio_Codecs_-_Comprehensive_Guide)
- A commercial vendor states that its detector added support for G.711, G.729, AMR-WB and Opus (vendor claim, not independently verified) — [Resemble AI](https://www.resemble.ai/introducing-telephony-optimized-deepfake-detection-model)

### Inferences
- **Local tool check (this machine, verified by me with `ffmpeg -encoders`):**
  - Available encoders: `pcm_mulaw`, `pcm_alaw`, `libgsm`, `libopencore_amrnb`, `libopus` and native `opus`, `libspeex`, `g722`, `g726`, `g723_1`.
  - No AMR-WB, G.729 or EVS encoders. `sox` and `torchaudio` are not installed in `venv/`.
  - Colab's apt ffmpeg should be checked the same way. Do not upgrade the pinned packages; run ffmpeg as a subprocess.
- **Offline pre-rendering beats on-the-fly in a 14 h budget.** Codec-render each training clip K=2–3 times with random conditions using parallel ffmpeg subprocesses (CPU), save as 16 kHz FLAC, and mix with clean at roughly 50/50.
- Suggested condition pool, all ending at 16 kHz:
  - G.711 µ-law and A-law at 8 kHz (`-ar 8000 -c:a pcm_mulaw`)
  - GSM-FR at 8 kHz (`-c:a libgsm -ar 8000 -ac 1`)
  - AMR-NB at 4.75–12.2 kbps (`-c:a libopencore_amrnb -ar 8000 -b:a 12.2k`)
  - Opus NB/WB at 6–24 kbps (`-c:a libopus -b:a 6k..24k -application voip`)
  - Speex
  - G.722 (wideband VoIP)
  - Double-transcode, e.g. AMR-NB→G.711 or Opus→G.711, which mimics mobile-to-PSTN-to-VoIP routes
  - Bandpass 300–3400 Hz plus mild level and AGC changes
  - Packet loss: zero out 20 ms frames at 1–10% with short bursts (crude, but approximates concealment gaps)
- Apply RawBoost **after** the downsampling and codec step, following Pindrop's ordering result. Do not stack heavy RawBoost on clips that are already strongly coded.
- **Use identical randomisation for genuine and fake:** draw the condition from the same distribution regardless of label, and log the condition ID per clip so that evaluation can be stratified by condition. Otherwise the detector can learn "codec = genuine".
- Hold out at least one codec family entirely from training (for example GSM or AMR-NB) to measure unseen-codec generalisation. Pindrop shows that the unseen codec is the main failure mode.

### Gaps
- No open, maintained Python library was found that bundles telephony codecs with packet loss for anti-spoofing. Real-WebRTC options (Asterisk, pjsip, a WebRTC loopback) were not researched in depth and are probably too slow to set up in 14 h.
- No source quantifies the benefit of G.729, EVS or AMR-WB specifically for SSL detectors.

## Degradation under telephony codecs for SSL detectors, and what recovers it

### Takeaway
Real transport is clearly worse than simulated codecs. On RTCFake, an XLS-R+AASIST trained offline goes from 5.42% EER clean to 13.79% on real-app transmissions. Training on transmitted data cuts that to 8.35%, at a cost of 9.57% on clean. Mixing data or consistency learning balances the two. Pindrop's chain degrades step by step: clean, then codec or bandwidth, then G.711, then real Twilio. SIM G.711 plus augmentation recovered more than training on real calls.

### Cited Findings
- RTCFake (XLS-R+AASIST with RawBoost):

  | Training data | EER offline (clean) | EER online (average) |
  |---|---|---|
  | Offline only | 5.42% | 13.79% |
  | Online only | 9.57% | 8.35% |
  | Mixed | 7.33% overall | |
  | Phoneme-guided consistency learning (PCL) | 5.81% overall | 6.77% |

  Unseen noise with PCL gave 9.53–13.40% EER — [arXiv 2604.23742](https://arxiv.org/html/2604.23742)
- Pindrop (source tracing, i.e. *which generator*, not binary detection; ResNet-type model trained from scratch, 4 s crops; metric EERc):
  - In-condition EERc: clean 16k 10.6%; Opus 12.8%; clean 8k 12.9%; G.711 14.8%; real Tel. 18.9%
  - Cross-condition: clean-16k model on Opus 30.2%; clean-8k model on Tel. 24.7%; G.711-trained model on Tel. 19.4%
  - Best result: 8k→Quantize→RawBoost + G.711 gave **15.1% on Tel., better than training on Tel. itself (18.9%)**
  - Wideband: RawBoost cut the Opus EERc from 30.2% to 16.1%, and Quantize→RawBoost to 14.9%

  — [Klein et al. 2026](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf)
- Pindrop: "codecs are responsible for the majority of degradation", more than bandwidth, and the unseen codec at test time is the biggest challenge — [Klein et al. 2026](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf)
- Delgado et al. (frozen WavLM + LLGF backend):
  - Adding "Presented" (real-call injection and replay) plus augmented data lowered average EER on the base test sets from 7.1% to 4.8%, and MDR at 1% FAR from 24.1% to 15.7%
  - Best detection on the private real-world set: 88.2% for injection calls and 76.3% for loudspeaker playback calls (at 1% FAR)
  - Realistic presentation data helped more than vocoder or codec augmentation; using both was best
  - The abstract claims +39% (lab) and +57% (real-world) accuracy, without defining the metric

  — [arXiv 2509.26471](https://arxiv.org/html/2509.26471v2)
- XLS-R + AASIST on ASVspoof 2021 LA (real VoIP/PSTN): 7.65% EER without augmentation, 0.82% with RawBoost — [arXiv 2202.12233](https://arxiv.org/pdf/2202.12233)
- ASVspoof 5 participants found 8 kHz Opus and Speex the hardest codec conditions — [arXiv 2408.10361](https://arxiv.org/pdf/2408.10361)

### Inferences
- **Expected size of the effect for this team:** a detector tuned on clean 16 kHz audio will likely see roughly 2–3× higher EER on real VoIP/PSTN audio (RTCFake: 5.4% to 13.8%). The 90%-at-15%-FA goal is unlikely to hold on real calls without channel training.
- **Recommended recipe:** keep the 16 kHz XLS-R model (narrowband is upsampled to 16 kHz as in ASVspoof 5) and train multi-condition rather than building a separate 8 kHz model. That gives one model and one input pipeline, and Pindrop's and RTCFake's results show multi-condition training recovers most of the loss.
- **Replay** (loudspeaker into a phone) is the hardest scenario (76% versus 88% detection in Delgado). Treat it as a separate reported condition, not part of the "call" claim.

### Gaps
- No published per-codec EER table was found for XLS-R-300m fine-tuned with versus without G.711/AMR/GSM augmentation on *modern* commercial TTS. The per-codec table in Tak et al. (Table 5) could not be extracted from the PDF.
- Pindrop's figures are for source tracing, not bona fide versus spoof detection, so the magnitudes may not transfer.

## Collecting a small real-call test set quickly and ethically

### Takeaway
Make a small, labelled, consented real-call set:
- **Content:** about 200–300 genuine and 200–300 fake clips, balanced per family.
- **Routes:**
  - Twilio PSTN/VoIP: a scripted `<Play>` of WAVs to a number that records them
  - Phone to phone over the carrier network, with a digital or line-in feed, logged as PSTN
  - WhatsApp or Zoom between two devices, logged as VoIP
- **Replay:** optionally, one loudspeaker-to-phone route kept as its own stratum.
- **Pairing:** send the same genuine and fake clips through every route.
- **Consent:** only use voices of consenting team members, or public-licence corpora, and record with consent.

### Cited Findings
- Twilio's Programmable Voice API has been used to send dataset audio over VoIP and PSTN for exactly this purpose. The authors note that real telephony collection is costly because it runs only in real time, and that its codecs and processing are not controllable — [Klein et al. 2026](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf)
- RTCFake's method for VoIP apps: a virtual audio cable into the app on PC A, OBS recording on PC B. Clips were concatenated before sending and re-segmented by timestamp afterwards, with Whisper used to verify the segments — [arXiv 2604.23742](https://arxiv.org/html/2604.23742)
- Delgado's methods: injection via Bluetooth or a wired analog feed into the phone mic input, versus playback from a studio monitor or Bluetooth speaker in an untreated room. Clips were 20 s and VAD was used to strip silence — [arXiv 2509.26471](https://arxiv.org/html/2509.26471v2)
- Consent:
  - Twilio's guidance: disclose recording and obtain consent from all participants — [Twilio support](https://support.twilio.com/hc/en-us/articles/360011435554-Best-Practices-for-Using-Twilio-to-Manage-and-Record-Communications-Between-Users)
  - US all-party consent states number about 11–12 depending on the source (California, Florida, Illinois, Washington and others), and sources disagree on the list — [HubSpot](https://knowledge.hubspot.com/calling/what-are-the-call-recording-laws); [Telnyx](https://telnyx.com/resources/call-recording-what-is-isnt-legal)

### Inferences
- **Fast protocol (about 3–4 h of wall time, mostly real-time transmission):**
  1. Choose about 60–100 genuine utterances from consenting speakers (team members reading prompts) plus a licensed corpus such as LibriSpeech.
  2. Make matched fakes per family (Chatterbox, ElevenLabs v3, Fish, Gemini) of the same texts, cloning only consenting speakers' voices.
  3. Concatenate clips into about 5-minute "tapes" with 1 s silence and a beep marker.
  4. Send each tape through 3 routes:
     - (i) Twilio outbound call that `<Play>`s the tape to a second number, recording the far end
     - (ii) mobile-to-mobile carrier call with a line-in or Bluetooth feed, far-end recorded
     - (iii) WhatsApp or Zoom between two devices, received audio captured with OBS or a virtual cable
  5. Re-segment the recordings by the markers, verify with ASR, and resample to 16 kHz.
  6. Log the route (SIM / VoIP / REPLAY / PSTN), the devices and the date for every clip.
- **Real-time cost:** 300 clips × about 6 s ≈ 30 min of audio per route, so 3 routes ≈ 1.5 h of transmission plus setup.
- **Sample size:**
  - At about 100 fakes per family, the 95% binomial CI half-width around 90% detection is roughly ±6 points.
  - At about 300 genuine clips, the CI on a 15% FA rate is roughly ±4 points.
  - Report CIs rather than point claims. Treat the set as a sanity check and stratified estimate, not as proof of a 90% target.
- **Ethics and engineering:** state clearly that "fake" voices are clones of consenting team members. Never call third parties. Keep test audio out of git per AGENTS.md (audio must not be committed), and store recordings outside the repo.
- **Same treatment for both labels:** genuine and fake must go through the same route, in the same session, interleaved in the same tapes. That way route and day effects cannot correlate with the label.

### Gaps
- No public sample-size guidance specific to call-channel spoof evaluation was found. The figures above are standard binomial approximations, not taken from a source.
- Twilio pricing and the exact TwiML `<Record>` and `<Play>` usage were not fetched. Verify current per-minute costs and trial-account restrictions; trial accounts may play a preamble message that must be trimmed.
- Non-US consent law (for example India's IT rules) was not researched.

## Calibration across clean and call channels

### Takeaway
No source found gives per-channel threshold recipes for this setting. The evidence (RTCFake clean versus online EER gaps; Delgado's operating point at a fixed 1% FAR) implies that genuine-audio scores shift under codecs. A single cutoff tuned on clean audio can either flood call audio with false alarms or miss fakes. Recommendation: set the cutoff on a pooled, channel-stratified genuine dev set, and report per-channel FA and detection at that one cutoff.

### Cited Findings
- Delgado et al. report detection (MDR) at a fixed 1% FAR, and show that the operating point found on lab data does not transfer to real-world calls. The drop from the base test set to the real-world set is "severe", shown in charts without exact figures — [arXiv 2509.26471](https://arxiv.org/html/2509.26471v2)
- In RTCFake, clean-trained models lose about 8 EER points online, while online-trained models lose about 4 points on clean audio. The error in each direction depends on the training channel mix — [arXiv 2604.23742](https://arxiv.org/html/2604.23742)
- Pindrop reports FPR95 rising sharply off-condition: the clean-8k model gives 18.5% on clean 8k versus 60.9% on Tel. — [Klein et al. 2026](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.pdf)
- ASVspoof 2021 notes that source corpus (genuine speech domain) can move EER from 0.33% to 23.39% for the same top system, which is larger than the codec effects — [ASVspoof 2021 paper](https://arxiv.org/html/2210.02437v3)

### Inferences
- **Practical approach:**
  1. **Prefer one cutoff.** The app cannot reliably know the channel, and per-channel cutoffs need a channel classifier, which adds a failure mode. Choose the threshold so that genuine FA stays at or below 15% on *every* channel stratum: clean, SIM-codec and real-call genuine. That means using the maximum over strata, not the pooled FA, then report per-family detection per channel at that cutoff.
  2. If one cutoff cannot meet both targets, a fallback is channel-conditional cutoffs keyed by a simple, robust bandwidth detector. For example, energy above 4 kHz below some threshold marks the clip as narrowband/telephone. Report it explicitly as a two-cutoff system.
  3. **Fix the threshold before scoring the real-call set**, matching the team's prespecified evaluation protocol. Use SIM-channel dev data plus a small dev split of the real-call recordings (for example a third of the clips) for calibration, and the rest for testing.
  4. **Watch for genuine call audio drifting toward "fake".** Noise suppression and codecs make genuine speech smoother and more "synthetic-like" (inferred from RTCFake's degradation and Delgado's results; not directly measured). This is the main risk to the 15% FA constraint.
  5. Score-level aggregation over multiple 4 s crops (mean of logits) per call should reduce variance on longer call recordings. This is an untested assumption here.

### Gaps
- No paper found measures genuine-only false-alarm shifts (FA at a fixed clean threshold) for SSL detectors on real calls. No paper compares a single cutoff with per-channel cutoffs.
- No calibration method (for example Platt or temperature scaling per condition) was found validated for codec shift in anti-spoofing.
