# Training and evaluation data sources for modern English TTS (Chatterbox, ElevenLabs v3, Fish/OpenAudio, Gemini TTS), as of 26 Nov 2026

Method: Hugging Face Hub API queries (file trees, card metadata) run on 2026-11-26, official vendor pages, and web search. MLAAD is gated, so per-clip metadata (`meta.csv`) could not be read without a token. Counts below come from the public file tree, which lists names and sizes even without access.

## 1. MLAAD: current revision, clips per family, voices, licence, matched genuine audio

### Takeaway
MLAAD's current `main` (commit `30c3dec7…`, last modified 2026-08-28) has exactly 1,000 clips per system folder, in 54 language folders. In English it already has more than the four folders the team uses: Chatterbox, Chatterbox-Turbo, ElevenLabs-v3, ElevenLabs-Turbo-v2.5, ElevenLabs-v2-Multilingual, FishTTS, Fish-S2-Pro, Openaudio-S1-Mini, Gemini-3.1-Flash-TTS and Resemble.ai (Apr 2025). That is about 10,000 English clips (about 3.1 GB) for these families. Another 25 folders in de/es/fr/it/ko/ru/ar/ja/pt/zh-cn add 25,000 more. It is CC BY-NC 4.0, for research only, and has no genuine audio of its own.

### Cited Findings
- HF API: `mueller91/MLAAD` sha `30c3dec763fa4803f11c2df1557d3b0828395366`, lastModified 2026-08-28T12:53:44Z. It is gated ("auto", meaning click-through acceptance) and licensed cc-by-nc-4.0. — [HF API](https://huggingface.co/api/datasets/mueller91/MLAAD)
- The card says: "published strictly for non-commercial academic research use, under the CC-BY-NC 4.0 license. Commercial use is not permitted". Total size is about 183 GB. The structure is `fake/<lang>/<model>/` with a `meta.csv` (it includes `architecture` and `transcript`, and a `reference_speaker` column since v8). It contains no genuine audio, and the authors recommend pairing it with M-AILABS. The paper is arXiv 2401.09512. The card has no changelog and no per-system counts. — [MLAAD card](https://huggingface.co/datasets/mueller91/MLAAD)
- `fake/en` has 143 system folders. The file tree (queried 2026-11-26) gives these counts for the target families: — [HF tree API](https://huggingface.co/api/datasets/mueller91/MLAAD/tree/main/fake/en)

| Folder (en) | wav clips | size |
|---|---|---|
| Chatterbox | 1000 | 282 MB |
| Chatterbox-Turbo | 1000 | 293 MB |
| Resemble.ai (April 12th, 2025) (commercial Resemble API; same vendor) | 1000 | 302 MB |
| ElevenLabs-v3 | 1000 | 330 MB |
| ElevenLabs-Turbo-v2.5 | 1000 | 315 MB |
| ElevenLabs-v2-Multilingual | 1000 | 335 MB |
| FishTTS | 1000 | 356 MB |
| Fish-S2-Pro | 1000 | 279 MB |
| Openaudio-S1-Mini | 1000 | 265 MB |
| Gemini-3.1-Flash-TTS | 1000 | 390 MB |

- Folders in other languages (1,000 clips each, 228–415 MB): — [HF tree API](https://huggingface.co/api/datasets/mueller91/MLAAD/tree/main/fake)
  - Chatterbox Multilingual: de, es, fr, it, ko, ru
  - ElevenLabs-v3: de only
  - ElevenLabs-v2-Multilingual: de
  - FishTTS: de, fr
  - Fish-S2-Pro: ar, de, es, fr, ja, ko, pt, ru, zh-cn
  - Openaudio-S1-Mini: fr, it
  - Resemble.ai: de, es, fr, it
  - Gemini TTS: **English only** (no other language has a Gemini folder)
- Other modern commercial or LLM-TTS English folders that could serve as related negatives or positives include Cartesia Sonic-3, MiniMax Speech 2.6-HD / 2.8-Turbo, OpenAI TTS-1 HD, Inworld-TTS-2, Hume TADA, Rime-Coda, DeepGram, Qwen3-TTS, VibeVoice, Higgs-Audio V2/V3, Index-TTS 2.0, VoxCPM2, Dia2 and Kyutai. — [HF tree API](https://huggingface.co/api/datasets/mueller91/MLAAD/tree/main/fake/en)

### Inferences
- MLAAD offers only 1,000 clips per family. If the team uses only the four folders it was using, it misses about 6,000 English clips from the same vendors: Chatterbox-Turbo, Resemble API, two other ElevenLabs models, Fish-S2-Pro and OpenAudio-S1-Mini. "FishTTS" is probably an older Fish Speech (1.x) checkpoint, and Fish-S2-Pro / Openaudio-S1-Mini are the modern ones. This could not be confirmed without `meta.csv`.
- For Gemini, MLAAD has only 1,000 clips, from a single preview model. Clips from the newer GA models (`gemini-3.8-flash-tts`, `-lite-tts`) will have to be generated.
- MLAAD texts come from M-AILABS book readings, so all of them are read-speech style. Because the genuine M-AILABS audio shares those texts, a detector can learn content or style cues. Training needs genuine audio that is not M-AILABS (see section 4).
- The ~1,000 clips per folder are probably split into speaker-disjoint train/val/test by reference_speaker. Plan for about 700 train clips per folder per language.

### Gaps
- Could not read `meta.csv` (gated, needs HF login), so the following are unknown: voices used, reference speakers, number of distinct voices per folder, and durations. Fix: accept the conditions on HF, then `huggingface-cli download mueller91/MLAAD --repo-type dataset --include "fake/en/ElevenLabs-v3/meta.csv" ...` and count `reference_speaker`.
- Could not confirm a version number. The card mentions "v8" only for the `reference_speaker` column. There is no changelog, so it is unknown which commit added Fish-S2-Pro or Gemini.
- M-AILABS genuine availability: the card links it but does not host it. The original M-AILABS download site has historically been unreliable, and this was not verified today.

## 2. Other public datasets with ElevenLabs, Gemini, Fish/OpenAudio or Chatterbox audio

### Takeaway
No public benchmark found, other than MLAAD, contains ElevenLabs v3 or Gemini TTS at scale. A few newer sets contain Fish or Chatterbox: BabelFake, VoxENES 2026 (whose HF shards include a Chatterbox split), and a small Fish S1-mini set. Older sets (DiffSSD, Deepfake-Eval-2024, SpeechFake, ShiftySpeech) carry older ElevenLabs models or in-the-wild audio. They are useful as auxiliary or OOD evaluation data, not as per-family training data. Several LAION / TTS-AGI dumps of Gemini and ElevenLabs output on HF are a fast, unofficial source of extra positives.

### Cited Findings
- **BabelFake** (arXiv 2610.06339, Oct 2026) is a multilingual audio-visual benchmark across 5 languages with consenting speakers. Its voice-cloning generators include Qwen3-TTS, FishAudio2, OmniVoice and ChatterBox. — [arXiv](https://arxiv.org/html/2610.06339v2)
- **VoxENES 2026** (arXiv 2607.11706) is an English/Spanish benchmark of 53,628 samples from 10 recent LLM-based TTS and VC systems, and shows a large generalisation gap for 8 pretrained detectors. — [arXiv](https://arxiv.org/pdf/2607.11706). An HF dataset `34data/voxenes-tts-chatterbox` exists, which suggests a per-system Chatterbox split. Licence and size were not checked. — [HF search API](https://huggingface.co/api/datasets?search=chatterbox)
- **hakamdzakwan/Audio_DeepFake** is English, real plus Fish Audio S1-mini fakes, CC BY-NC 4.0, gated (manual approval), last modified 2026-06-09. — [HF](https://huggingface.co/datasets/hakamdzakwan/Audio_DeepFake); [HF API](https://huggingface.co/api/datasets/hakamdzakwan/Audio_DeepFake)
- **DiffSSD** (Purdue, arXiv 2409.13049) has about 200 h of labelled speech from 8 open-source diffusion generators and 2 commercial ones. — [arXiv](https://arxiv.org/abs/2409.13049). Another paper describes a Purdue dataset with 25,000 synthetic tracks from 5 diffusion voice-cloning methods, including ElevenLabs. — [arXiv 2507.08626](https://arxiv.org/pdf/2507.08626). This is 2024-era ElevenLabs, not v3.
- **Deepfake-Eval-2024** covers in-the-wild audio from 2024: 56.5 h of audio, 45 h of video and 1,975 images from 88 websites in 52 languages. — [arXiv 2503.02857](https://arxiv.org/pdf/2503.02857). HF `nuriachandra/Deepfake-Eval-2024` is CC BY-SA 4.0, gated (manual), updated 2026-08-11. — [HF API](https://huggingface.co/api/datasets/nuriachandra/Deepfake-Eval-2024). It has no per-generator labels, so it is useful as a realistic OOD test only.
- **ShiftySpeech** HF `ash56/ShiftySpeech` is Apache-2.0, updated 2026-09-02, with folders `TTS/`, `Vocoders/`, `real-data/`, `real_data_flac/` and `tts.yaml`. — [HF tree API](https://huggingface.co/api/datasets/ash56/ShiftySpeech/tree/main). Its generator list was not verified.
- **SpeechFake** is on HF as `DeepFense/SpeechFake`. — [HF search API](https://huggingface.co/api/datasets?search=speechfake). The team already has it. Whether it contains any of the four families is unknown.
- **ReplayDF** `mueller91/ReplayDF` is CC BY-NC 4.0, ungated, 100K–1M clips (re-recorded or replayed deepfakes), updated 2025-06-10. — [HF API](https://huggingface.co/api/datasets/mueller91/ReplayDF). It is useful for channel robustness.
- **CodecFake** `ajaykarthick/codecfake-audio` is CC BY-NC-ND 4.0 and 100K–1M clips. — [HF API](https://huggingface.co/api/datasets/ajaykarthick/codecfake-audio). It covers codec resynthesis, not the target families.
- **Small ElevenLabs / Gemini dumps on HF**, unofficial and unvetted: — [HF search API](https://huggingface.co/api/datasets?search=elevenlabs); [HF search API](https://huggingface.co/api/datasets?search=gemini%20tts)
  - `laion/elevenlabs-v4-five-language-voice-acting` (652 downloads)
  - `guey-khala-mari/ElevenLabs_v4_VoiceActing`
  - `WpythonW/elevenlabs_multilingual_v2-technical-speech` (Apache-2.0, fewer than 1K clips)
  - `behavioralsignals/deepfake-detection-demo` (fewer than 1K clips; its ElevenLabs clips are mostly non-English)
  - `laion/gemini-2.5-pro-tts-voice-profiles` (451 downloads) and `laion/gemini_pro_25_tts_voice_acting_vocal_burst_tags`
  - `TTS-AGI/gemini-adult-voices`
  - `akuzdeuov/gemini_flash_tts`
  - `Trelis/tricky-tts-gemini-pro-tts`
- Other 2026 benchmarks that may be relevant but whose generator lists were not confirmed: MADBench (arXiv 2608.09593), AT-ADD (2608.23437), ML-ITW (2603.05852), TwinShift (2510.23096), LJ-Spoof (2601.07958). — [search results](https://arxiv.org/pdf/2608.23437)

### Inferences
- The LAION Gemini-2.5-Pro-TTS and ElevenLabs-v4 sets are the fastest way to add Gemini and ElevenLabs model versions that MLAAD lacks. They are expressive voice acting rather than read speech, which helps break the M-AILABS style shortcut. Their licences, sizes, English share and speaker counts must be checked first. As vendor outputs, their redistribution terms are uncertain.
- For a strict per-family ≥90% target, the only scaled, labelled source is MLAAD plus self-generated data (section 3).

### Gaps
- Not verified for any of these: SONAR, FakeOrReal, In-the-Wild, ASVspoof 5, LlamaPartialSpoof, or TTS-Arena dumps containing the four families. Based on their publication dates (2019–2025), FoR, ITW and ASVspoof 5 predate ElevenLabs v3, Gemini TTS, Chatterbox and OpenAudio S1. This is an inference, not checked.
- Not checked: the VoxENES system list and download, the BabelFake download or licence, and the sizes and licences of the LAION and TTS-AGI dumps.

## 3. Generating data: open weights and paid APIs, with cost and time for 2,000 × 6 s clips per family

### Takeaway
All four families can be generated within the 14 h window. 2,000 × 6 s is 12,000 s (3.3 h) of audio per family:
- ElevenLabs v3: about 180k characters, roughly $15 by API, or one Creator plan ($22, 275k characters).
- Gemini 3.8 Flash TTS: about 300k audio tokens, roughly $3 (about $6 on 3.1-flash-preview).
- Fish S2-Pro: well under 1 GPU-hour on an A100/H100-class GPU.
- Chatterbox (MIT): GPU-hours of the same order. Its output carries a Perth watermark.

### Cited Findings
**Chatterbox (Resemble AI)**
- MIT licence. Since August 2026 it is "retired research and is no longer maintained".
- Variants:
  - Chatterbox-Turbo: 350M, English, paralinguistic tags, one-step mel decoder
  - Chatterbox-Nano: 110M, about 3× real time on 8 CPU cores
  - Multilingual V3: 500M, 23+ languages
  - Original Chatterbox: 500M, English
- Install with `pip install chatterbox-tts` (tested on Python 3.11). Cloning takes a reference clip (the example is about 10 s).
- **Every output carries the Resemble Perth watermark**, and the README describes no opt-out.
- No GPU RTF or VRAM numbers are given.
- — [GitHub](https://github.com/resemble-ai/chatterbox)

**Fish Speech / OpenAudio**
- OpenAudio S1-mini: 0.5B, distilled from the proprietary 4B S1, CC BY-NC-SA 4.0, gated on HF. — [HF](https://huggingface.co/fishaudio/openaudio-s1-mini)
- The current fish-speech repo lists Drama 3 (35B-A3B) as flagship and S2-Pro (4B) as previous flagship.
- S2-Pro reaches RTF 0.195 on one H200 with about 100 ms to first audio, needs 10–30 s of reference audio, and is "small enough to deploy on a consumer GPU".
- Licences: the code and Drama 3 use the Fish Audio Community License, and the S2-Pro weights use the Fish Audio Research License. Weights are at `fishaudio/s2-pro` and `fishaudio/drama-3`.
- — [GitHub](https://github.com/fishaudio/fish-speech)

**ElevenLabs**
- Official API page prices per 1k characters:
  - Eleven v3: $0.08
  - Multilingual v2: $0.08
  - Flash/Turbo: $0.04
- Eleven v4 is presented as the newest model.
- Plans:
  - Free: 10k characters on v3/v4/v2, API allowed, but no Voice Library voices via API
  - Starter: $6/mo, 75k characters
  - Creator: $22/mo, 275k characters
  - Pro: $99/mo, 1.238M characters
- Concurrency tables list 4/6/10/20/30/40 concurrent calls by tier. Mapping these to Free/Starter/Creator/Pro is an inference.
- — [ElevenLabs API pricing](https://elevenlabs.io/pricing/api)
- Third-party pages also quote $0.10 per 1k characters for v3, and one calls v3 "legacy" (Oct 2026). — [ModelsLab](https://modelslab.com/elevenlabs-api-pricing); [Developers Digest](https://www.developersdigest.tech/models/speech/elevenlabs)

**Gemini TTS**
- Models: `gemini-3.8-flash-tts` and `gemini-3.8-flash-lite-tts` (current), plus `gemini-3.1-flash-tts-preview` and `gemini-2.5-pro-preview-tts` (earlier).
- 30 prebuilt voices: Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe, Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi, Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird, Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat.
- An Extended Voice Library with "hundreds more" voices is available via `client.voices.list()`.
- Voice replication (needs reference plus consent audio), voice design and remix are available on the 3.8 models.
- At most 2 speakers per request.
- Output is 24 kHz mono 16-bit PCM WAV, with a configurable sample rate (16k or 8k).
- — [Gemini speech generation docs](https://aistudio.google.com/docs/speech-generation)
- Pricing per 1M tokens (text input / audio output):

| Model | Standard | Batch/Flex |
|---|---|---|
| gemini-3.8-flash-tts | $0.50 / $9.00 until 31 Dec 2026, then $1.00 / $18.00 | half of Standard |
| gemini-3.8-flash-lite-tts | $0.50 / $6.00 | half of Standard |
| gemini-3.1-flash-tts-preview | $1 / $20 | $0.50 / $10 |

  - Free tier: yes for Standard, no for Batch.
  - Audio is 25 tokens/s, so 10 s costs about $0.00225 on 3.8 Flash.
  - — [Gemini pricing](https://aistudio.google.com/docs/pricing)

**Cost and time estimates for 2,000 × 6 s per family** (arithmetic from the figures above; about 15 characters per second of speech is an assumption)
- ElevenLabs v3: 2,000 × about 90 characters = 180k characters.
  - Cost: $14.40 at $0.08/1k, or $18 at $0.10. One Creator month ($22, 275k characters) covers it.
  - Time: at 10 concurrent calls and about 3–5 s per request, roughly 10–20 min.
- Gemini 3.8 Flash TTS: 12,000 s × 25 = 300k output tokens.
  - Cost: about $2.70 standard, $1.35 batch, $6 on 3.1-flash-preview. Input tokens are negligible.
  - The free tier may suffice but has unpublished per-minute and per-day limits.
- Fish S2-Pro: 12,000 s × 0.195 RTF is about 39 min on an H200. Expect roughly 1–2 h on an A100 and longer on a T4, which is unverified. S1-mini (0.5B) should be faster.
- Chatterbox: no published GPU RTF. Budget roughly 1–2 h per 3.3 h of audio on an A100 (unverified), or use Turbo for speed.

### Inferences
- The Perth watermark in Chatterbox output is a shortcut risk. If MLAAD's Chatterbox clips are watermarked and wild Chatterbox clips are too, a detector may learn the watermark. That would be "correct" in deployment, but it would fail on watermark-stripped audio. Consider adding clips that are watermark-free (edited source) or passed through codecs.
- To get speaker diversity for cloning-capable systems (Chatterbox, Fish), use reference clips from speaker-disjoint genuine corpora (LibriTTS-R or VCTK train speakers). Keep evaluation speakers out of the reference set.
- For Gemini, cycle through all 30 prebuilt voices plus Extended Library voices, with style prompts, and use 16 kHz output to match the detector front end.
- ElevenLabs v3 should use many Voice Library voices, which requires a paid plan for API access, and texts outside M-AILABS (conversational and news).

### Gaps
- Not found: Gemini TTS rate limits (RPM/RPD) and free-tier quotas; ElevenLabs commercial-use terms; measured Chatterbox or S2-Pro speed on T4 or an RTX 5060 8 GB, and whether S2-Pro (4B) fits in 8 GB.
- SynthID watermarking of Gemini TTS output is not stated in the docs page.

## 4. Diverse genuine English speech for false-alarm control

### Takeaway
LibriTTS-R and VCTK (CC BY 4.0) plus GLOBE V2 (CC0, accented) are fast, ungated and speaker-labelled. Expresso is high quality but CC BY-NC. People's Speech is huge and CC BY family, so stream a subset. Common Voice v17 is no longer cleanly hosted on HF. The VoxCeleb mirror on HF has an "unknown" licence and is gated.

### Cited Findings

| Dataset (HF id) | Licence | Size / gating | Updated |
|---|---|---|---|
| `mythicinfinity/libritts_r` | CC BY 4.0 | 10K–100K rows, ungated | 2024-02 |
| `CSTR-Edinburgh/vctk` | CC BY 4.0 | 10K–100K, ungated | 2024-08 |
| `MushanW/GLOBE_V2` | CC0 1.0 | ungated | 2024-11 |
| `ylacombe/expresso` | CC BY-NC 4.0 | ungated | 2024-04 |
| `MLCommons/peoples_speech` | CC BY / CC BY-SA mix | more than 1T bytes, ungated | 2024-11 |
| `mozilla-foundation/common_voice_17_0` | none listed in metadata | — | 2025-10 |
| `ProgramComputer/voxceleb` | "unknown" | gated | 2026-10 |

— [HF API](https://huggingface.co/api/datasets/mythicinfinity/libritts_r); [HF API](https://huggingface.co/api/datasets/CSTR-Edinburgh/vctk); [HF API](https://huggingface.co/api/datasets/MushanW/GLOBE_V2); [HF API](https://huggingface.co/api/datasets/ylacombe/expresso); [HF API](https://huggingface.co/api/datasets/MLCommons/peoples_speech); [HF API](https://huggingface.co/api/datasets/mozilla-foundation/common_voice_17_0); [HF API](https://huggingface.co/api/datasets/ProgramComputer/voxceleb)

### Inferences
- How to get speaker-disjoint subsets quickly:
  - LibriTTS-R: use the official speaker-disjoint splits (train-clean-100/360 for training, dev-clean/test-clean for evaluation). Because the team already evaluates on LibriSpeech dev, keep LibriTTS-R dev/test out of training.
  - VCTK: split by speaker ID, about 110 speakers.
  - GLOBE: split by its speaker or accent IDs.
  - People's Speech: stream it with `datasets` (`streaming=True`) and take a few thousand clips. It has no reliable speaker labels, so split by source recording or file prefix.
- Mix read speech (LibriTTS-R), studio speech (VCTK, Expresso), accented speech (GLOBE) and spontaneous or far-field speech (People's Speech, AMI). This lowers false alarms on "clean studio" genuine audio, which modern TTS imitates.

### Gaps
- Not verified: EARS hosting (`fixie-ai/ears` and `sdialog/EARS` returned 401, gated or nonexistent; the original is on GitHub at facebookresearch/ears, CC BY-NC 4.0 per prior knowledge, unverified today), the current hosting location of Common Voice (Mozilla moved distribution off HF in 2025, unverified), and the exact VoxCeleb licence (historically CC BY-SA 4.0 for metadata with audio from YouTube, unverified).
- Hour counts per corpus were not fetched in this session.
