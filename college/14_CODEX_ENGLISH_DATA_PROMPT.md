# Codex prompt: why English matching collapsed, and how to fix it

Run on a high-reasoning model from the repository root (`~/deepfake_project`, branch `pipeline-v3-scratch`). Paste everything inside the box.

````text
You are the lead researcher on this project's pipeline v3 (pipeline_v3/, docs in college/8–13). The first real data run exposed a problem in the English half of the dataset. Analyse it yourself, research solutions and workarounds, choose one with clear reasoning, and implement it. You have full freedom to redesign the English data recipe and the matching/splitting rules, as long as the result stays honest (no speaker, voice, text or parent leakage; no label shortcuts).

## Goal of the project (unchanged)
A small Hindi + English audio deepfake detector (teacher → ≤90 MB student, trained on Colab Pro) that catches MODERN TTS and voice cloning: above all Google Gemini TTS and Microsoft TTS, plus ElevenLabs, Chatterbox, Fish, XTTS etc., on real-world uploads (phone/WhatsApp, 3–30 s). The pilot (college/12_PIPELINE_V3_DESIGN.md, E2) must measure held-out-family and held-out-speaker performance for BOTH languages.

## What happened (measured on the real files, 25 Nov 2026)
Downloaded and SHA-verified on the owner's Drive (MyDrive/audio-v3/development/): M-AILABS en_US.tgz 8.02 GB, MLAAD English (Chatterbox, ElevenLabs-v3, FishTTS, Gemini-3.1-Flash-TTS; 1,000 WAV + meta.csv each), Kathbath Hindi (32 shards), IndicSynth Hindi (14 of 107 shards), IndicVoices Hindi (5 shards), XLS-R 300M and WavLM-base.

`pipeline_v3.cli inventory` + `match` ran on Colab. Local copies of the outputs, metadata only and git-ignored:
- detector/runs/v3_prepare/inventory.jsonl (196,137 records: kathbath 91,752; indicsynth 26,945; indicvoices 27,144; mlaad 4,000; mailabs 46,296)
- detector/runs/v3_prepare/selection.csv (16,201 selected)
- detector/runs/v3_prepare/prepare.log

Match result:
- Hindi: 5,594 Kathbath real + 10,075 IndicSynth fake (xtts_v2 9,281; freevc24 794, held out). All five splits are filled. Usable, but only older generators.
- **English: 266 M-AILABS real + 266 MLAAD fake (ElevenLabs-v3 208, Gemini 58). Chatterbox 0, FishTTS 0. Only train/calibrate/threshold exist; select and validate have NO English.**
- Exclusions: reserved speaker or family 13,466; unresolved parent or voice 3,245; missing or crossing prompt 1,978.

Root causes I confirmed (verify them yourself):
1. **M-AILABS en_US has only 3 readers** (elliot_miller 17,207 clips, judy_bieber 15,720, mary_ann 13,369; 14 books). match.py assigns splits by hashing the reader, so English can form at most 3 speaker groups and can never fill five splits.
2. **MLAAD v11 meta.csv** columns are path|original_file|language|is_original_language|duration|training_data|model_name|architecture|transcript|speaker_similarity. ElevenLabs-v3 and Gemini rows have reference_speaker; **Chatterbox and FishTTS have none**, so match.py rejects all 2,000 as "unresolved parent or voice".
3. **~1/3 of every MLAAD folder's original_file is en_UK** (e.g. en_UK/by_book/female/elizabeth_klett/...). Only en_US was downloaded, so ~1,333 fakes have no real parent.
4. match.py also requires assign(reader) == assign(voice). For stock commercial voices the "voice" is not the reader, so independent hashes discard most remaining pairs.
5. Gemini-3.1-Flash-TTS is in --heldout by default, so it never enters train. That's correct for measuring transfer, but the final release must also be trained on Gemini (the owner's main target).

## Your tasks
1. **Verify and quantify.** Reproduce the counts from inventory.jsonl/selection.csv with a small script. For every English fake give its fate under the current rules. Then compute what each candidate rule change would yield: per language × split × generator × label, plus the number of independent speakers and voices per split. Write the script into pipeline_v3/ (e.g. a `match --report` or a separate diagnostics module) so it can be rerun.

2. **Analyse the leakage trade-offs, then research options.**
   Leakage trade-offs:
   - When is splitting by book or text instead of reader acceptable?
   - What leaks if the same human reader appears in train and test as REAL while the fakes are stock voices? (The model may learn "these 3 voices = real".)
   - How should stock voices with unknown identity be handled: voice = generator family or voice = model_name/voice name; group by family?
   - Can held-out-family evaluation still be meaningful with 3–5 readers?

   Options to research, live with sources:
   - (a) M-AILABS en_UK (size, readers): does it add enough speakers?
   - (b) Multi-speaker real English to pair with self-generated fakes on the same texts: LibriTTS-R, LibriSpeech, VCTK, Common Voice English, Indian-English sources such as IndieFake (local, 50 speakers) and Svarah or NPTEL-type Indian English if licensed. Give speakers, hours, GB, licence and HF ids.
   - (c) ASVspoof 5 (already local on the owner's PC, LibriSpeech/MLS-derived, hundreds of speakers): can its real speech serve as speaker-diverse English genuine data, and is reuploading 21 GB from a slow home line avoidable by downloading from Zenodo inside Colab?
   - (d) Other public English deepfake sets with many speakers AND matched real (SpoofCeleb, DiffSSD, LlamaPartialSpoof, CodecFake+, MLAAD's other languages): pros, cons, licences.
   - (e) Self-generated Gemini + Microsoft (edge-tts/Azure) + open cloning (XTTS-v2, Chatterbox) English clips reading the texts of the chosen real corpus. Many voices, held-out voices and texts. Use detector/datagen/tts_dataset.py (resumable, --dry-run cost) or pipeline_v3/generate.py; reconcile the two.

3. **Choose a recipe.** Pick the English recipe with the best expected held-out validity per GB and per hour of owner effort, and say what you rejected and why. Targets:
   - at least ~1,000–3,000 English pairs
   - English in all five splits
   - ≥ 20 independent real speakers in train and ≥ 5 in each evaluation split, or justify a lower number
   - Gemini and Microsoft represented in development with held-out voices
   - one modern family fully held out for the pilot
   Then specify the final-release variant that trains on all families including Gemini.

4. **Implement.**
   - Change match.py and adapters as needed. Possible changes: configurable split unit (reader | book | voice | text); voice fallback for stock-voice families; optional en_UK; new real-corpus adapters; self-generated clip adapter reading tts_dataset manifest.csv.
   - Keep all leakage checks; add new ones where the rule changes create risk.
   - Add tests.
   - Add or extend fetch keys (dry-run sizes; ask before > 5 GB).
   - Update pipeline_v3/README.md and the Colab notebooks with exact commands.

5. **Prove it on the real metadata.** Run the new match logic locally against detector/runs/v3_prepare/inventory.jsonl (no audio needed) and report the new counts table. For data not yet downloaded, report projected counts from published metadata, labelled as projections.

## Extra data is allowed
Nothing has been packed or trained yet. **The owner is happy to download extra datasets, or generate extra clips, before preprocessing and training start**, if that is what makes English (and modern Hindi) valid. Don't settle for a weak workaround just to avoid a download. Recommend the best data, and list each addition with dataset id, exact files, GB, licence/gating, what it fixes and expected pairs per split. Keep the total within the agreed bound (minimum set first, extras up to ~150 GB in total, each justified by the new speakers, generators, languages or channels it adds). The owner runs the downloads in Colab straight to Drive (MyDrive/audio-v3/), then reruns inventory + match.

## Constraints
- Don't download datasets, call paid APIs, generate audio, or start Colab sessions yourself. Write the commands and dry-runs; the owner runs them. pyarrow may be missing locally, so pip-install it into the venv only if needed for reading metadata.
- Keep v1, deepfake_model/ and sealed sets untouched (In-the-Wild, VA, VoxENES, Deepfake-Eval, app_testing_dataset/).
- Never commit audio, data, inventory files, keys or papers. Never push. Commit code and docs on pipeline-v3-scratch only.
- Cite sources (URLs) for every dataset fact; mark unverified numbers.

## Deliverables
- `college/15_ENGLISH_DATA_FIX.md`, containing:
  - the diagnosis with verified numbers
  - an options table (speakers, GB, licence, effort, leakage risk, expected pairs per split)
  - the chosen recipe and the final-release variant
  - the before/after counts table
  - exact Colab/CLI commands in order
- Code and tests on pipeline-v3-scratch, with all tests passing.
- An update to college/13_HANDOFF_TO_CLAUDE.md.

## Final reply (short)
- the root cause in 3 lines
- the chosen fix and why
- the before/after English counts per split
- what the owner must download or generate, with GB and cost
- the next 3 commands
````
