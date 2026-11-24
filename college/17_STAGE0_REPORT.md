# Stage 0 — paired English speech and a strict shortcut gate

**Decision for the current Drive snapshot.** The English archives justify voice labelling. Bilingual training is already a **no-go** under the requested provenance gate. In the saved Hindi selection all 811 genuine recordings come from Kathbath and all 1,596 fakes come from IndicSynth. Ranking by that single source field gives selection AUC **1.00** (also 1.00 on train). Cropping or re-encoding cannot change a provenance field. Adding English alone cannot change the Hindi result. Address this before buying GPU time if the immediate goal is a passing bilingual pilot. A corpus that supplies both labels under genuine shared provenance can help; measure the resulting gate again. Any decision to treat archive identity as a descriptive control instead changes the research protocol.

## Local proof and what the counts mean

The three full fake member lists agree on **28,539** distinct utterance IDs each. Those IDs identify **251** readers. The actual members follow patterns such as `XTTSv2/103-1240-0000.wav`. LibriSpeech dev-clean has **2,703** members and **40** readers, dev-other **2,864** and **33**. Both are reserved for the later commercial panel. The reader intersections between all three source partitions are empty. The ten reference filenames resolve to ten distinct VCTK IDs. No audio was required for these checks. Replaying the real listing and transcript directory verified 28,539 genuine WAV members and 28,539 nonempty transcripts. All 85,617 synthetic members have a unique genuine parent. There are 28,537 distinct normalized transcripts; zero repeats cross the assigned reader folds. [Zenodo release and checksums](https://zenodo.org/records/19362791) · [Publisher protocol](https://arxiv.org/html/2603.15037v1)

**English, before voice inference.** The second column is a measured candidate count **for each** of XTTS-v2, MeloTTS and Chatterbox TTS (label 1). It is also the measured count of unique genuine parents (label 0) in that fold before voice and crop checks. The last three columns illustrate a planning scenario in fake→real pairs. It assumes all ten VCTK targets are equally frequent and independent of readers, all identification and text checks pass, and every Melo clip qualifies for the train-only built-in cohort. True retention can be much lower, including zero. Each retained fake has one parent; multiple families can share that genuine recording.

| Fold | Candidate IDs per system / readers | Reserved VCTK targets | XTTS pairs, scenario | Melo pairs, scenario | Chatterbox pairs, scenario |
|---|---:|---:|---:|---:|---:|
| Train | 14,231 / 126 | 5 | 7,116 | 14,231 | 0 |
| Select | 4,795 / 44 | 2 | 959 | 0 | 959 |
| Calibrate | 3,930 / 33 | 1 | 393 | 0 | 393 |
| Threshold | 2,289 / 19 | 1 | 229 | 0 | 229 |
| Validate | 3,294 / 29 | 1 | 329 | 0 | 329 |

The replay accepted **zero English pairs** in every fold with the local inputs. Final retained readers, voices and genuine-parent counts require the voice labels and audio eligibility checks on Drive. The match prints those counts for every language/fold/generator/label cell. The audit prints the same counts after dropping short or orphaned members. The built-in cohort contributes zero to the identifiable-voice count. One target voice in a fold limits population-level claims regardless of clip count.

**Hindi, measured from the original selection.** Genuine cells give recordings/readers; fake cells give recordings/source readers/target IDs. The audit will verify bytes, cropping eligibility where applicable and the nuisance gates on Drive.

| Fold | Kathbath genuine | XTTS-v2 fake | FreeVC24 fake | Total fake→parent trials |
|---|---:|---:|---:|---:|
| Train | 1,759/8 | 2,500/8/8 | 0 | 2,500 |
| Select | 811/4 | 1,386/3/3 | 210/1/10 | 1,596 |
| Calibrate | 1,404/4 | 2,403/3/3 | 100/1/5 | 2,503 |
| Threshold | 510/7 | 699/3/3 | 262/4/11 | 961 |
| Validate | 1,110/6 | 2,293/3/3 | 222/4/7 | 2,515 |

Report the Hindi unseen-family EER as a result on **one source reader, ten targets and 210 fakes** before feature eligibility. Repeated trials cannot establish transfer across Hindi source speakers. The MLAAD panel is exported separately as parent/member pairs for later descriptive scoring. Local exact US-parent matches were 680 Chatterbox, 666 ElevenLabs-v3, 655 FishTTS and 666 Gemini. These records contribute zero primary training or decision trials.

To reproduce the local proof:

```bash
P=detector/runs/v3_prepare
venv/bin/python -m pipeline_v3.stage0 proof --data /tmp --base "$P/selection.csv" \
  --listings "$P/phonemedf_listing" --real-texts "$P/phonemedf_listing/real_transcripts"
```

The real-side proof used the actual `real/wav` members and `real/transcripts/<utteranceID>.txt` files. Stage 1 streams the original archives on Drive. No additional listings are needed.

Local verification passed all **53 backend and pipeline tests** (five warnings). The setup notebook parsed, Ruff passed, and the six pinned PhonemeDF byte counts and MD5 values matched the supplied files.tsv. Cloud inference throughput, inferred target accuracy and post-crop retention remain unmeasured.

## Inferred target voices

The CSV label is an estimate. The publisher gives references but no per-utterance target key. The separate speaker-verification checkpoint produces normalized embeddings from two nonoverlapping views of each fake, up to 3 s each. Both must select the same reference centroid with cosine ≥0.86 and a best-versus-runner-up margin ≥0.08 in each view. Clips shorter than 2 s abstain. The margin and thresholds are engineering choices awaiting validation on synthetic speech. Review representative high-confidence and marginal assignments by system before relying on a voice-disjoint result. [WavLM verification checkpoint](https://huggingface.co/microsoft/wavlm-base-plus-sv)

For each system, internally consistent rejections far from every reference (cosine <0.65) are clustered. The CSV records cluster IDs and reference scores. Clustering is exploratory; one acoustic cluster need not represent one person. Only MeloTTS has documented built-ins in this acquisition. Its compact clusters (≥25 members, within-centroid cosine ≥0.88) enter **one train-only cohort**; apparent VCTK matches and ambiguous examples are excluded. XTTS and Chatterbox use reference assignments only. Mislabelling a cloned target can still cause actual speaker overlap. The hard split guarantee applies to inferred IDs.

The same VCTK ID has the same fold in every family. Train gets p230/p258/p259/p243/p273, select p228/p233, calibrate p236, threshold p244 and validate p254. Only rows whose real reader and estimated target are assigned to the same fold survive. Texts and parents cannot cross folds. The 5/2/1/1/1 allocation reserves a majority of targets for training while retaining new target identities at every evaluation stage. Unknowns are excluded. Completed embedding batches are checkpointed in a Drive SQLite database tied to the inventory digest, checkpoint revision and view policy. Restart the same command on the same Drive with one writer. Estimates are independent of detector scores.

## Duration, metadata and training gates

For PhonemeDF, preprocessing finds the active span with the shared edge trim, takes its central 64,000 true samples, and encodes mono 16 kHz PCM with the same method on both labels. A span under four seconds is ineligible. A fake whose genuine parent is too short is dropped as well; a genuine parent with no surviving fake is removed. Pilot and training use that fixed window without trimming it a second time or padding. This controls the original duration cue (12.69 s real versus roughly 10–10.5 s synthetic). Evaluate short-upload behavior separately on the app's variable-length path. [Publisher duration table](https://arxiv.org/html/2603.15037v1)

Controls train on train rows and score selection. They measure duration, active-frame ratio, edge silence, RMS loudness, upper-band energy, spectral tilt, 95% energy rolloff, container codec and rate, source corpus, pair, and their combination. AUC above 0.60 in either orientation, an undefined result, or a missing matched parent fails the gate. Reports cover the full set, each language and each matched source pair. The audit reads the exact effective bytes before cataloging. The catalog rechecks their hashes and the packer verifies them again. Pilot and teacher require the passing strict-policy report bound to that pack index. Failures leave the audit report on Drive.

## Stage 1 on Colab

The owner reports that `v3-data` is running with Drive mounted. Upload the new Stage 0 commit before running commands in its terminal. From the local repository root:

```bash
git archive --format=tar --output=/tmp/v3-code.tar pipeline-v3-scratch pipeline_v3 colab college/DOWNLOAD_PLAN.sh
colab upload -s v3-data /tmp/v3-code.tar /content/drive/MyDrive/audio-v3/v3-code.tar
colab exec -s v3-data --timeout 0 -f colab/bootstrap.py
```

Use the updated code tar from this branch, mounted Drive, and the setup notebook or a Colab terminal. The current Drive already contains the audio. The separate verification checkpoint requires about 405 MB for inference. [Model files](https://huggingface.co/microsoft/wavlm-base-plus-sv/tree/main) This sequence stops on failure. The Hindi AUC above is already sufficient to predict a stop at audit if the Hindi recipe is kept. Running only the inventory and proof is sufficient to inspect the data issue without renting a GPU.

```bash
set -e
cd /content/v3-src
export DATA=/content/drive/MyDrive/audio-v3
python -m pip install -r pipeline_v3/requirements-colab.txt
python -m pipeline_v3.stage0 inventory --data "$DATA"
python -m pipeline_v3.fetch --data "$DATA" --sources wavlm-sv --dry-run
python -m pipeline_v3.fetch --data "$DATA" --sources wavlm-sv --execute
python -m pipeline_v3.voice_labels --data "$DATA" --batch 8
python -m pipeline_v3.stage0 match --data "$DATA"
python -m pipeline_v3.stage0 audit --data "$DATA"
python -m pipeline_v3.stage0 catalog --data "$DATA"
python -m pipeline_v3.stage0 pack --data "$DATA"
```

Planning estimates, **unmeasured** on the owner's Drive. Inventory 10–60 min CPU; voice inference 1–6 h on a T4/L4 for 85,617 fakes (roughly 4–24 clips/s overall), potentially 12–72 h on CPU; match 1–5 min CPU; audit 1–6 h CPU depending on retained clips, Drive throughput and FFmpeg process startup; catalog 0.5–4 h CPU; pack 10–90 min CPU. Use the first 256–1,024 embeddings and elapsed time to revise the estimate. Compute units are the displayed CU/h × elapsed GPU hours; at an illustrative 2 CU/h the voice job would use about 2–12 CU. Use a free CPU runtime for the other stages if available. [Colab resource policy](https://research.google.com/colaboratory/faq.html)

The optional recovery key `python -m pipeline_v3.fetch --data "$DATA" --sources phonemedf --dry-run` shows the six named files, bytes and MD5s, totalling about 33.18 GB. The executor resumes partial transfers and verifies the published byte count and checksum before promoting a file.

A passing pack lives under `$DATA/stage0/packed` with its own `audit.json`. Stage 2 can use `python -m pipeline_v3.cli stage --data "$DATA/stage0" --splits train select` followed by `python -m pipeline_v3.cli pilot --data "$DATA/stage0" --encoder "$DATA/models/xlsr" --small-encoder "$DATA/models/wavlm-base"`. Match requires ≥20 real English readers in train and ≥5 per other fold, both seen training families and held Chatterbox in every evaluation fold. In the pilot, Chatterbox, Gemini and FreeVC24 are always withheld from probe fitting.

## Stage 3 commercial panel and Stage 5 data

On the populated Drive these commands plan up to 24 jobs per language/fold/provider, retaining transcript, reader, parent and voice provenance. English uses reserved LibriSpeech dev readers. Hindi uses genuine rows of the authorized Kathbath selection, joined to original transcripts by archive and row. Azure Hindi has two reserved voices (train/select); Gemini Hindi voices are distinct from the English panel. Stage 3 reserves Gemini for evaluation. Use the same Microsoft choice across stages. Edge and Azure may expose the same named voices.

```bash
MS=azure  # or edge
python -m pipeline_v3.stage0 jobs --addons --data "$DATA" --microsoft "$MS"
python -m pipeline_v3.generate --data "$DATA" --jobs "stage3-$MS-jobs.csv" --dry-run
# After the owner runs approved generation with --execute:
python -m pipeline_v3.stage0 match --data "$DATA" --addons --microsoft "$MS"
# audit, catalog, pack in order using --addons on each; products live in stage3/
```

Azure requires the speech SDK, key and region; Edge requires edge-tts; Gemini requires google-genai and a key. The generation dry-run reports characters, pending requests and estimated Gemini cost. All synthesis remains an owner operation. To prepare the final family-inclusive variant after freezing the recipe:

```bash
python -m pipeline_v3.cli match --data "$DATA" --heldout --selection base-release.csv
python -m pipeline_v3.stage0 jobs --release --data "$DATA" --base base-release.csv --microsoft "$MS"
python -m pipeline_v3.generate --data "$DATA" --jobs "stage5-$MS-jobs.csv" --dry-run
# Once approved release jobs have been generated:
python -m pipeline_v3.stage0 match --data "$DATA" --base base-release.csv --release --addons --microsoft "$MS"
# audit, catalog, pack in order using --release on each; products live in stage5/
```

The final data builder requires Chatterbox, Gemini and FreeVC24 in train. The stage-specific release labels identify these as seen. Historical held-family probe results describe the restricted fit only. A release still requires the later frozen training, student comparison and sealed evaluation. Stage 0 produced no trained detector or cloud performance measurement.
