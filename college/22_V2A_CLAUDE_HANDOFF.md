# V2-A English teacher handoff

Build a multi-source baseline using MelGAN, Parallel WaveGAN, WaveGrad and WaveRNN from LibriSeVoc, plus ASVspoof 5 A08. Run CPU preparation first. Use the GPU for the pilot and teacher after the formal mixture audit passes. Keep the V1 experiment available for comparison. Use one writer per experiment and capture command output in the session log.

## Continue from a new VM

Mount the existing Drive. Set the working directory to the deployed repository root. A source-only copy of the pipeline and this handoff can also be extracted from the Drive bundle into a fresh local directory.

```bash
D=/content/drive/MyDrive/audio-v3
mkdir -p /content/v2a-code
tar -xzf "$D/v2/v2a-source.tar.gz" -C /content/v2a-code
cd /content/v2a-code
export V2A_DATA="$D"
export V2A_SCRATCH=/content
bash pipeline_v3/run_v2a.sh inventory
bash pipeline_v3/run_v2a.sh confirm
```

Use the existing CPU/T4 resources for these two steps. Inventory writes a CSV of publisher IDs and required metadata. Confirmation uses 40 ASV training speakers disjoint from the 40 used in the original diagnostic. It selects four genuine and four A08 clips per speaker, applies the frozen four-second crop and RMS normalisation, balances eligible clips within speaker, and runs five speaker-disjoint folds. It saves the worst oriented AUC and each probe score. Above 0.60 ends this recipe. Rerunning the command reads the saved decision.

## Resolve the metadata before the formal audit

The work directory is `$V2A_DATA/v2/v2a-norm-1`. Join publisher or independently verified metadata to `lineage-requests.csv` and write the resolved subset to `lineage.csv` in that directory. The required columns are `source,uid,speaker,transcript,evidence`. Keep an evidence reference identifying the transcript and person mapping for each admitted record. Omit unresolved rows and report counts. Do the joins by script.

- LibriSeVoc uses the LibriTTS utterance key and canonical `librispeech/<reader>`. The verified ZIP contains waveforms and no transcript table. Obtain the corresponding LibriTTS transcript metadata using the utterance key.
- ASV uses training utterance IDs. Resolve the actual spoken text for both labels and the target speaker's canonical identity. The public training protocol provides utterance and speaker IDs and attack labels. Those alone do not establish the spoken text or cross-corpus identity. Use a verified mapping to the LibriSpeech/MLS reader identity where available. A namespace prefix on an anonymised speaker is insufficient evidence of independence. Keep a speaker's genuine and cloned items under the same identity.
- The original and confirmation ASV speakers are excluded from ASV formal select. Preserve V1 speaker ownership, exclude conflicting text, and reserve its calibrate, threshold and validate readers. Check available counts by speaker, source, label and fold before launching the expensive extraction. If the A08 joins cannot be established, report which metadata is unavailable and how much of each source can be admitted. A LibriSeVoc-only run would be a separate experiment.

## Prepare, audit and train

```bash
bash pipeline_v3/run_v2a.sh prepare
# On the training VM, from the same source bundle and mounted Drive:
bash pipeline_v3/run_v2a.sh gpu
```

Preparation requires FFmpeg, ffprobe and the Python environment described in `pipeline_v3/requirements-colab.txt`. Set `V2A_SCRATCH` to a writable local disk with room for the treated crops. The upper bound for 100,000 four-second 16 kHz PCM crops is approximately 13 GB before filesystem overhead. Allow additional space on Drive for the pack and checkpoints. A fresh preparation attempt recreates scratch. Keep the input CSV, frozen recipe, decisions and pack together on Drive.

The formal audit fits probes on train and scores distinct select speakers. Each LibriSeVoc parent must retain all four reconstructions after cropping. A08 uses equal genuine and fake counts per speaker. Its utterance ID records the lineage of that clip; a recording-level genuine parent for a clone is unknown. For the pooled and LibriSeVoc aggregate probes, one fake is chosen deterministically per original to balance labels. The four individual vocoder panels each use all their eligible originals. Every panel, including A08, must have worst oriented AUC at or below 0.60. A failed mixture report ends the run. The published audit binds the decision to the pack, recipe, confirmation and input lineage.

The packed training fold contains the five approved fake families and their genuine controls. The existing teacher sampler gives equal expected weight to each source/label cell. All four complete LibriSeVoc variants are available across draws. The packed select fold for pilot and early stopping is the existing Chatterbox select panel with its genuine parents. Its known V1 acoustic shortcut still limits interpretation. The new V2 formal-audit select rows are used exclusively to check the mixture. Calibration, threshold and validation reuse the V1 comparison folds.

The GPU command prints the model and free memory and runs one disposable teacher forward/backward/optimizer step on a synthetic four-second input. The pilot then uses both speech encoders. A passing pilot starts the teacher, five epochs of 1,000 steps with early stopping. Rerunning uses the existing pilot, last completed training epoch and frozen teacher when available. Use a separate local staging directory if other work is active. Leave V1 sessions under their existing operator's control.

## Report to the owner

Report actual accelerator and memory, A08 confirmation, lineage coverage and exclusions, per-panel formal AUCs, pilot decision, teacher epochs, time and CU. Compare validation and external-panel counts with V1 before reporting differences. At frozen thresholds report genuine false alarms per 100 by channel and missed fakes per 100 by generator, with denominators. Simulated phone/Opus results and actual call recordings are separate measurements. The runner covers English development audio and simulated channels. Real calls require a separate evaluation panel.
