# English data for the bilingual pilot

Use the two LibriSpeech development archives and synthesize one stock-voice rendition per selected transcript. This costs about 0.65 GB of additional downloads and targets 1,440 matched pairs from 60 human readers. Keep the four MLAAD folders as a separate diagnostic panel. They cannot supply English speaker-held-out selection and validation for this pilot. The numbers below distinguish the observed Drive metadata from the jobs that the owner still needs to generate. [OpenSLR files and licence](https://www.openslr.org/12/) · [LibriSpeech paper, Table 1](https://www.clsp.jhu.edu/wp-content/uploads/2015/11/icassp2015_librispeech.pdf)

## What happened to the 4,000 English fakes

The local replay reproduced every field of all 16,201 saved selection rows. The inventory contains 196,137 rows. English has 46,296 real utterances across 14 books, read by Elliot Miller (17,207), Judy Bieber (15,720) and Mary Ann (13,369). The other 1,333 parent references all name Elizabeth Klett in en_UK. Each folder has 1,000 fakes. Chatterbox and Fish lack a reference voice in every row. ElevenLabs and Gemini use named stock voices. The ledger assigns the first applicable reason to each row, checking parent availability before voice identity and counting a held family before a simultaneous voice mismatch. [Publisher schema](https://huggingface.co/datasets/mueller91/MLAAD) · [Replay code](../pipeline_v3/english_report.py)

| English family | Unavailable parent | Missing voice after parent join | Voice assigned elsewhere | Held out of train | Kept |
|---|---:|---:|---:|---:|---:|
| Chatterbox | 320 | 680 | 0 | 0 | 0 |
| ElevenLabs-v3 | 334 | 0 | 458 | 0 | 208 |
| FishTTS | 345 | 655 | 0 | 0 | 0 |
| Gemini 3.1 Flash TTS | 334 | 0 | 385 | 223 | 58 |

Each entry below gives **distinct real recordings / fake recordings**. A real parent can serve multiple fakes. The named-voice variant assigns each named voice to the eligible reader fold with the most matching rows. The cohort variant also assigns each unknown-voice family wholly to one fold. Neither variant can create another human reader. The UK projection assumes every referenced UK member exists and resolves uniquely. Only the three US rows of comparisons were measured against available parent metadata.

| English recipe | Train | Select | Calibrate | Threshold | Validate |
|---|---:|---:|---:|---:|---:|
| Original, replayed | 131/131 | 0/0 | 62/62 | 73/73 | 0/0 |
| Named voices allocated to reader folds | 139/139 | 0/0 | 157/157 | 259/260 | 0/0 |
| Plus unknown family cohorts | 370/371 | 0/0 | 157/157 | 484/488 | 0/0 |
| Named voices plus UK parents, projected | 596/596 | 0/0 | 117/117 | 149/149 | 0/0 |
| Primary English recipe on current inventory | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| Primary recipe after successful synthesis, projected | **768/768** | **192/192** | **192/192** | **144/144** | **144/144** |
| Projected distinct real readers | **32** | **8** | **8** | **6** | **6** |
| Projected Azure / Gemini fakes | 768/0 | 96/96 | 96/96 | 72/72 | 72/72 |
| Projected distinct Azure / Gemini stock voices | 2/0 | 2/2 | 2/2 | 2/2 | 2/2 |

Run the metadata replay from the repository root. It emits language × split × generator × label counts with reader and named voice cardinalities for each candidate. The CSV contains the original reference and all three measured fates per fake. Keep it outside Git. Choose a fresh output filename on each run.

```bash
P=detector/runs/v3_prepare
venv/bin/python -m pipeline_v3.english_report --inventory "$P/inventory.jsonl" --selection "$P/selection.csv" --fates /tmp/english-fates.csv
```

Hindi retains 5,594 real and 10,075 fake recordings. Available real/XTTS-v2/FreeVC24 counts are 1,759/2,500/0 in train, 811/1,386/210 in select, 1,404/2,403/100 in calibrate, 510/699/262 in threshold and 1,110/2,293/222 in validate. There are 8/4/4/7/6 distinct real speakers respectively. **Only one source reader contributes the 210 FreeVC24 selection rows** (ten target IDs). Report that denominator alongside the Hindi held-family EER. Those clips cannot establish generalization across Hindi readers by themselves.

## Alternatives and leakage

The last column gives the decision for this pilot. Every deferred option contributes zero pairs to every fold. A genuine-only release contributes pairs only after transcript-matched synthesis. A speaker count for an entire corpus is an upper bound on the independent people a chosen subset can supply.

| Source and access | Scale, licence and effort | Expected coverage and decision |
|---|---|---|
| [M-AILABS UK](https://github.com/i-celeste-aurora/m-ailabs-dataset) + [MLAAD](https://huggingface.co/datasets/mueller91/MLAAD) | en_UK.tgz, 3.5 GiB compressed, 45 h 34 m. Publisher-specific M-AILABS licence. Complete UK reader count unverified. | One additional reader in these 1,333 references. Projected split counts above. Other MLAAD languages change the language under study. Defer. |
| [LibriSpeech](https://www.openslr.org/12/) ([HF openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr)) | CC BY 4.0, open. dev-clean.tar.gz and dev-other.tar.gz, approximately 337 MB and 314 MB, 40 + 33 readers, 5.4 + 5.3 hours. Source speaker IDs, FLAC and transcripts. The 100 h training alternative alone is 6.3 GB. [Subset table](https://www.clsp.jhu.edu/wp-content/uploads/2015/11/icassp2015_librispeech.pdf) | Choose both development archives exclusively for this project's development. The original ASR development score is no longer an independent result. No reference audio goes to a provider. |
| [LibriTTS-R](https://www.openslr.org/141/) | 24 kHz restored speech, roughly 585 h across the corpus. train_clean_100.tar.gz alone is 8.1 GB, CC BY 4.0. Exact eligible speaker count and official HF ID unverified. | Good sentence boundaries. Restoration can give genuine speech its own processing signature. Same audiobook lineage. Defer. |
| [VCTK 0.92](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full) | About 110 readers, ~44 h and 10.94 GB, CC BY 4.0. Use Edinburgh distribution. Official HF mirror unverified. | Accent diversity, but shared passages and multiple microphones need utterance and text grouping. More transfer. |
| [Mozilla Common Voice English](https://commonvoice.mozilla.org/en/datasets) | Scripted v25 English was 87.84 GB, CC0, Mozilla Data Collective access. Release-specific English speaker/hour counts unverified here. | Crowdsourced mics help phone diversity. Many speakers read identical sentences. More substantial curation and no confirmed current official HF route. |
| [IndieFake](https://indie-fake-dataset.netlify.app/) | 50 Indian English speakers, 27.17 h including fakes, locally available. Colab bytes, redistribution terms and official HF ID unverified. | Useful next accent panel using authorized original speaker splits. Existing v1 exposure and synthetic vendor overlap complicate novelty claims. |
| [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah/blob/main/README.md) / [ai4bharat/NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL/blob/main/README.md) | Svarah 117 speakers, 9.6 h, 1.095 GB, CC BY 4.0, access terms, published as test. Card schema lacks an explicit speaker ID. NPTEL card lists CC BY 4.0. English-specific lecturer count, hours, bytes and reliable ID map unverified. | Indian accents are valuable. Svarah needs a reliable reader crosswalk. Lecture/channel identities and transcript quality need verification for NPTEL. Defer both. |
| [ASVspoof 5](https://zenodo.org/records/14498691) | MLS-derived, 400 training speakers, about 18k bona fide. Use archive LICENSE.txt. Five train tar parts at 7.5 GB each plus 20.7 MB protocols, around 37.5 GB for the full train transfer. [Organizer counts](https://www.asvspoof.org/file/ASVspoof_Workshop_2024_Slides.pdf) | Colab can fetch any chosen train parts directly from Zenodo, avoiding home upload. Useful diverse genuine controls after transcript/speaker linkage. Adding them as genuine only introduces corpus identity as a label. The owner's local 21 GB is not the whole Zenodo train release. |
| [jungjee/spoofceleb](https://huggingface.co/datasets/jungjee/spoofceleb) / [purdueviperlab/diffssd](https://huggingface.co/datasets/purdueviperlab/diffssd) | SpoofCeleb 1,251 celebrity identities, ~134 GB, CC BY 4.0 plus approved access. DiffSSD 196 h, ~18.1 GB, BY-NC-ND plus upstream terms. Exact matched speaker counts unverified. | SpoofCeleb exceeds the aggregate budget with the existing material. DiffSSD adds diffusion models but needs original linkage and usage review. |
| [LlamaPartialSpoof](https://zenodo.org/records/14214149) / [CodecFake+](https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset) | LlamaPartialSpoof v1.0.b ~28.22 GB, CC BY 4.0, LibriTTS-derived, ~130 h in the paper protocol. CodecFake+ ~101 GB, 110 VCTK readers across splits, MIT-tagged with upstream terms. | Partial splices test localization. Codec resynthesis tests codec artifacts. Neither fills the missing paired modern stock-voice folds with less work. [Dataset provenance and version notes](11_DATASETS_V3.md) |

A book or text split answers a narrower question about **new material from familiar readers**. When the same three human voices always mean genuine and stock voices always mean synthetic, the detector can succeed by recognizing readers. A whole-family holdout still tests a specific generator on those readers, but additional clips cannot identify a speaker population effect. For named stock voices reserve the provider and voice identity across model versions. For clones keep the human reference identity in the same namespace as the genuine reader. An unknown family-wide voice block prevents *within-family* cross-fold reuse. Its size in independent voices is unknown, and it cannot rule out reused references across families. These are reasons to keep the MLAAD diagnostic panel separate.

## Pilot and final release

The planner takes 30 readers from each archive, orders readers deterministically, and assigns entire readers to folds. Each supplies 24 globally distinct normalized transcripts of 6–45 words. Both labels share parent, text, channel, reader condition and fold. The plan stops before generation if either archive lacks enough eligible speakers or an audio member is missing. The published 73-reader population supports the projection. Eligibility of 60 readers under the transcript filter remains a Colab measurement. Synthesis changes voice identity and prosody. Even when the text is identical, audiobook microphone noise and delivery can still predict the label. Catalog converts both sides to mono 16 kHz PCM. English and Hindi each have to pass the nuisance audit on selection. Inspect source rates, original durations, endings, voice counts and phone/Opus results before scaling. If an English nuisance probe separates the labels, repair the paired conditions and rerun the audit.

Pilot training uses Microsoft Azure stock voices. Eight additional Azure and eight Gemini voice names are reserved across the four evaluation folds. Azure and Gemini selection trials measure unseen voices and only Gemini measures a previously unseen English family. Pilot probes score language-specific held-family EER against matching genuine parents. Two named voices per family per evaluation fold limit vendor-wide claims. The English source does not establish Indian English phone performance. Retain the Indian English matched panel and genuine phone checks as follow-up conditions.

The release version replaces 384 Azure training renditions with 384 Gemini renditions on the same train readers/texts, leaving the English evaluation jobs identical. This trains on **both primary English families**. For Hindi, rebuild the base using an empty family holdout to admit FreeVC24 alongside XTTS-v2 where lineage permits. Probe fitting reserves Gemini and FreeVC24 even on the release pack. Report those as probe transfer results. The final teacher's Gemini result measures held-out **voices**, since the teacher sees the family. To extend beyond these families, collect audited matched ElevenLabs and Fish stock voices or identifiable licensed recordings, and consented XTTS/Chatterbox references. Unknown MLAAD reference voices cannot be counted as independent release evaluation identities. Additional modern Hindi Azure/Gemini development pairs should use Kathbath text/reader lineage and reserve multilingual stock identities globally across English and Hindi. No modern-Hindi gain has been measured.

Use pipeline_v3.generate for these paired jobs. It uses the same content-derived filename for generation, resume and assembly. The older detector/datagen/tts_dataset.py produces its own prompts and train/dev/test partitions. Its manifest lacks the required LibriSpeech parent join. Treat its existing outputs as a separate diagnostic until each has a verified parent, text and global voice partition. Open cloning is an optional further condition using the v3 generator's xtts or chatterbox jobs with a consent ID and development reference. [XTTS model](https://huggingface.co/coqui/XTTS-v2) · [Chatterbox model](https://huggingface.co/ResembleAI/chatterbox)

The pilot adds 1,104 Azure and 336 Gemini requests. At 6–12 s per Gemini output the paid output estimate is **$1.01–2.02**, plus input at $1/M tokens and retries. Azure F0 includes 500,000 neural characters/month. Check the actual job character count and available quota before execution. Release adds 384 new Gemini requests, about $1.15–2.30 output at the same durations. Model availability and realized durations depend on the account. At eight seconds × 24 kHz mono PCM16, 1,440 generated WAVs take roughly 0.55 GB before catalog and packing. Original planned downloads total about 36.06 GB including the 35.409 GB already budgeted. [Azure pricing](https://azure.microsoft.com/en-us/pricing/details/speech/) · [Google pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-tts-preview)

## Owner-run sequence

Use the updated code archive on a mounted CPU Colab session. From home, the [runbook](../pipeline_v3/README.md) has the git archive, colab upload, drivemount and bootstrap commands. The updated [setup notebook](../colab/01_setup_and_data.ipynb) runs the same stages and loads credentials from Colab Secrets. For a Colab terminal, run the following in order. The final pack path must be unused. On this Drive the base selection and Hindi inventory already exist.

```bash
cd /content/v3-src
export DATA=/content/drive/MyDrive/audio-v3
python -m pipeline_v3.fetch --data "$DATA" --sources libri-dev --dry-run
python -m pipeline_v3.fetch --data "$DATA" --sources libri-dev --execute
python -m pipeline_v3.english plan --data "$DATA"
python -m pipeline_v3.generate --data "$DATA" --jobs english-pilot-jobs.csv --dry-run
# With SPEECH_KEY, SPEECH_REGION and GEMINI_API_KEY in this process environment:
python -m pip install azure-cognitiveservices-speech google-genai
python -m pipeline_v3.generate --data "$DATA" --jobs english-pilot-jobs.csv --execute
python -m pipeline_v3.english assemble --data "$DATA"
python -m pipeline_v3.cli catalog --data "$DATA" --selection selection-pilot.csv
python -m pipeline_v3.cli pack --data "$DATA"
```

From a terminal on the owner's machine the same remote commands can run inside a CLI kernel. After bootstrap, for example:

```bash
colab exec -s v3-data --timeout 0 <<'PY'
import subprocess, sys
data = "/content/drive/MyDrive/audio-v3"
for module, args in [("fetch", ["--sources", "libri-dev", "--dry-run"])]:
    subprocess.run([sys.executable, "-m", "pipeline_v3."+module, "--data", data, *args], cwd="/content/v3-src", check=True)
PY
```

Use the pilot notebook to stage train/select, run the nuisance audit and run E2. Preserve its pack, feature directory, audit and pilot reports. For a release rerun, use a new experiment root with the same development inputs and model files accessible. Keep the pilot artifacts available. In that root run the following, then stage, audit and run the pilot probes again for the new pack digest before training.

```bash
python -m pipeline_v3.cli match --data "$DATA" --heldout --selection base-release.csv
python -m pipeline_v3.english plan --data "$DATA" --base base-release.csv --release
python -m pipeline_v3.generate --data "$DATA" --jobs english-release-jobs.csv --dry-run
python -m pipeline_v3.generate --data "$DATA" --jobs english-release-jobs.csv --execute
python -m pipeline_v3.english assemble --data "$DATA" --base base-release.csv --release
python -m pipeline_v3.cli catalog --data "$DATA" --selection selection-release.csv
python -m pipeline_v3.cli pack --data "$DATA"
```
