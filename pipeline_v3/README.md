# Train and evaluate the bilingual detector on Colab

The source in this branch is sufficient to build a student release after the pilot passes. The trained v1 release is available through `DETECTOR_BACKEND=detector`. A v3 release uses `DETECTOR_BACKEND=v3`. Large inputs and run products belong outside the checkout. Use Linux, Python ≥3.11, preinstalled PyTorch and FFmpeg. The model and APIs need no network during training once weights and shards are present.

## First three actions

1. From the local repository root, build the code-only archive below. Put it at `MyDrive/audio-v3/v3-code.tar`. It includes the data downloader and the new source. It fits an ordinary browser or CLI upload.
2. Open [01_setup_and_data](../colab/01_setup_and_data.ipynb) on a CPU VM. Run with `EXECUTE=False` to inspect the budget, then run the approved acquisition. Supply `HF_TOKEN` using Colab Secrets after accepting gated datasets. Review the join counts and `development/selection.csv` before packing.
3. Open [02_pilot](../colab/02_pilot.ipynb) on T4/L4. Enable execution to stage, audit and probe. Read `audit.json` and `pilot.json`. A failing audit exits nonzero. A completed pilot with `go: false` is a valid research outcome and blocks the teacher.

```bash
git archive --format=tar --output=/tmp/v3-code.tar pipeline-v3-scratch \
  pipeline_v3 colab college/DOWNLOAD_PLAN.sh
```

## Terminal workflow with google-colab-cli 0.8.0

Run these commands locally yourself. The named VM incurs compute charges. The interactive Drive mount needs your account. The named session preserves kernel state, and the Drive copy survives `stop`. Shell functions below take the session and stage explicitly. The default Drive root is `/content/drive/MyDrive/audio-v3`, code root `/content/v3-src`, and local shards `/content/v3-local`. Override with `--env V3_DATA=...`, `V3_REPO` or `V3_LOCAL` on each execution when using other directories.

```bash
colab new -s v3-data
colab drivemount -s v3-data
printf '%s\n' "from pathlib import Path" "Path('/content/drive/MyDrive/audio-v3').mkdir(parents=True, exist_ok=True)" | colab exec -s v3-data
colab upload -s v3-data /tmp/v3-code.tar /content/drive/MyDrive/audio-v3/v3-code.tar
colab exec -s v3-data --timeout 0 -f colab/bootstrap.py
colab exec -s v3-data --env V3_STAGE=setup -f colab/run_stage.py
```

Enable the next cell only for an approved data transfer. Where CLI kernels lack notebook-secret access, attach the browser to the **same** VM with `colab url -s v3-data`, then authorize the secret. No token belongs in a shell argument or manifest.

```bash
colab exec -s v3-data --timeout 0 <<'PY'
import os,sys,subprocess
from google.colab import userdata
os.environ["HF_TOKEN"]=userdata.get("HF_TOKEN")
subprocess.run([sys.executable,"-m","pipeline_v3.fetch","--data","/content/drive/MyDrive/audio-v3",
                "--sources","mailabs","mlaad","kathbath","indicsynth","indicvoices","xlsr","wavlm-base","--execute"],
               cwd="/content/v3-src",check=True)
PY

stage_v3() {
  colab exec -s "$1" --timeout 0 --env "V3_STAGE=$2" --env V3_EXECUTE=1 -f colab/run_stage.py
}
mount_v3() {
  colab drivemount -s "$1" && colab exec -s "$1" --timeout 0 -f colab/bootstrap.py
}
stage_v3 v3-data prepare
# Review selection.csv on Drive and resolve missing language, speaker or generator cells.
stage_v3 v3-data pack
colab stop -s v3-data

colab new -s v3-pilot --gpu L4
mount_v3 v3-pilot
stage_v3 v3-pilot pilot
colab download -s v3-pilot /content/drive/MyDrive/audio-v3/pilot.json ./pilot.json
colab stop -s v3-pilot
```

For a one-shot CLI syntax check the fetch planner is standalone. This command provisions a VM and stops it on exit. Its output is identical to running the script locally, so a VM is optional for this check.

```bash
colab run --gpu T4 pipeline_v3/fetch.py --data /content/v3 --dry-run
```

After a passing pilot:

```bash
colab new -s v3-teacher --gpu A100
mount_v3 v3-teacher
stage_v3 v3-teacher teacher
colab stop -s v3-teacher

colab new -s v3-student --gpu L4
mount_v3 v3-student
stage_v3 v3-student student
stage_v3 v3-student evaluate
colab download -s v3-student /content/drive/MyDrive/audio-v3/validation.json ./validation.json
colab stop -s v3-student
```

**Student comparison (owner requirement):** before freezing a release, train both the compact acoustic student and a 4–6-block XLS-R student from the same teacher, evaluate them on the same validation IDs, and record accuracy, Opus penalty, short-clip coverage, size and CPU latency for each (see college/12_PIPELINE_V3_DESIGN.md, step 7 and E4). The SSL student is not implemented in `pipeline_v3/` yet; add it as a second `student` model type before running E4.

The notebook equivalents are [03_teacher](../colab/03_teacher.ipynb), [04_student_export](../colab/04_student_export.ipynb) and [05_evaluate](../colab/05_evaluate.ipynb). Each starts in dry-run mode. If any CLI command fails, inspect its exit status and stop the session when done. To resume a disconnected run, mount Drive on a replacement VM, bootstrap the same source and rerun that stage. A completed selection or release is immutable. Start a fresh directory for a changed protocol. Stage only needs its listed split shards on each new VM.

## Manifests and pair verification

`inventory` visits the approved training directories, enumerating original tar members, MLAAD metadata, and Hindi Parquet training rows. `match` resolves exact original stems only when unique in the appropriate corpus. A stable pseudonymous speaker partition reserves train, selection, calibration, threshold and validation groups. Source/reference combinations that cross speaker partitions and texts that cross folds are excluded. Gemini-3.1-Flash-TTS and FreeVC24 are default withheld families. Use `--heldout` on the match command to change this *before fitting*. `selection.csv` stores the result. Compare publisher metadata and remove uncertain matches, especially ambiguous stock voices, unknown accents, missing Hindi reference IDs and cross-vendor families. Source annotations cannot establish independence of two vendor backends. Map known related versions to canonical families and mark partially known versions `related`.

A small number of US audiobook readers can leave entire English folds empty. Use additional, consented real English speakers or licensed matching originals. Never split an existing reader across folds to fill a cell. Unpaired IndicVoices audio is for a separate genuine dev stress report. Use paired self-generated fakes for a supervised source. For a fresh project, verify speaker and class counts in every language/fold before acquisition of a larger dataset.

| Columns | Meaning |
|---|---|
| archive, member | Relative to DATA/development. Exact tar member, zero-based Parquet row number or `-` for a loose file |
| id, split, label | Unique safe ID, one of train/select/calibrate/threshold/validate, 0 genuine or 1 synthetic |
| source, pair, language, channel | Original distribution, matched corpus/style, hi/en, observed channel |
| speaker, voice, parent, text | Namespaced real/reference speaker(s), synthesis voice, original utterance/family, canonical sentence family. Separate multiple identities with a vertical bar; keep translated or duplicated texts together |
| generator, domain | Canonical family and seen/related/unseen. Use `-` and `real` for genuine |
| path | Set by catalog adapter. A manually prepared catalog points to development-relative reviewed files |

Add self-generated or phone pairs by appending rows to the reviewed selection. `catalog` extracts selected members only, checks for conflicts and produces `catalog.csv`. `pack` writes original payloads with byte offsets, hashes and a manifest. Budget working space for temporary selected members, packed data and model checkpoints. `stage` verifies hashes and restores only missing or damaged shards. Use a separate pack directory for each study. This command sequence also works in a Colab terminal:

```bash
cd /content/v3-src
DATA=/content/drive/MyDrive/audio-v3
LOCAL=/content/v3-local
python -m pipeline_v3.fetch --data "$DATA" --dry-run
python -m pipeline_v3.cli inventory --data "$DATA"
python -m pipeline_v3.cli match --data "$DATA" --heldout Gemini-3.1-Flash-TTS freevc24
python -m pipeline_v3.cli catalog --data "$DATA"
python -m pipeline_v3.cli pack --data "$DATA"
python -m pipeline_v3.cli stage --data "$DATA" --local "$LOCAL"
python -m pipeline_v3.cli audit --data "$DATA" --local "$LOCAL"
python -m pipeline_v3.cli pilot --data "$DATA" --local "$LOCAL" --per-class 1000
python -m pipeline_v3.cli teacher --data "$DATA" --local "$LOCAL" --steps 1000 --epochs 5
python -m pipeline_v3.cli student --data "$DATA" --local "$LOCAL" --steps 1000 --epochs 5
python -m pipeline_v3.cli export --data "$DATA" --local "$LOCAL"
python -m pipeline_v3.cli evaluate --data "$DATA" --local "$LOCAL"
python -m pipeline_v3.cli benchmark --data "$DATA" --threads 2
```

## Optional development generation

Prepare two CSVs under DATA/development. `prompts.csv` has columns `text,text_family,seconds,language,split`. `voices.csv` has `engine,model,voice,speaker,language,split,consent_id,reference`. Use `edge`, `azure`, `gemini`, `xtts` or `chatterbox`. Set reference and consent ID for cloning. Review the account tier and record the actual provider model/voice; Edge voice IDs do not identify Azure HD revisions. Avoid unconsented reference voices. Prompt sentences belong to one fold. For example, a Hindi read prompt can be “कल शाम सात बजे हम पुस्तकालय के बाहर मिलेंगे।” and an English request can be “Can you send me the revised bus timetable?” Include scripts and conversational prompts in both languages and classes.

```bash
python -m pipeline_v3.plan_jobs --data "$DATA" --prompts prompts.csv --voices voices.csv --dry-run
python -m pipeline_v3.plan_jobs --data "$DATA" --prompts prompts.csv --voices voices.csv --out jobs.csv
python -m pipeline_v3.generate --data "$DATA" --jobs jobs.csv --dry-run
# Owner execution only, after reviewing costs and rights:
# python -m pipeline_v3.generate --data "$DATA" --jobs jobs.csv --execute
```

Install only the selected provider dependency in its own Colab environment. Edge uses `edge-tts`, Gemini `google-genai`, Azure `azure-cognitiveservices-speech`. Set the documented GEMINI_API_KEY or SPEECH_KEY/SPEECH_REGION via Colab Secrets. XTTS uses the `tts` executable from a compatible Coqui environment; Chatterbox uses `chatterbox.mtl_tts`. Use a separate synthesis environment for model-specific torch requirements. The classifier environment needs neither cloning stack. The open adapters execute their upstream model loaders only in execution mode. A failed call can incur upstream charges; check the provider bill before retry. Local output names hash the reviewed job; completed jobs are skipped.

## Release, CPU verification and final tests

Copy only `release-v3/student.ts` and `release-v3/release.json` to a local directory. Run from the repository root with the project Python environment:

```bash
V3_RELEASE=/absolute/path/to/release-v3 DETECTOR_BACKEND=v3 venv/bin/uvicorn app.backend.main:app
venv/bin/python -m pipeline_v3.cli benchmark --data /tmp --release /absolute/path/to/release-v3 --threads 2
venv/bin/python -m pytest -q pipeline_v3/tests app/backend/tests
```

A completed build requires training on approved data. The checked-in tests use random-weight fixtures and synthetic audio. The Linux v3 backend uses anonymous memory descriptors for seekable audio/video decoding and duration probing. It emits PRD §7 fields and uses no uploaded file as a development input. Use the CPU measurements in the design document as architecture checks.

Freeze the code commit, source taxonomy, split digest and SHA-256 of release.json before a custodian opens a final set. Under a separate sealed root the custodian prepares `SET/trials.csv` with `path,label,speaker`. Paths are relative to SET. Use source event IDs if speaker identity is unknown, document that substitution, and aggregate variants with their original. Sealed labels stay outside the development checkout and notebooks.

```bash
python -m pipeline_v3.sealed --release "$RELEASE" --sealed-root "$SEALED" --set voxenes \
  --ledger "$LEDGER" --expected-release-sha256 "$FROZEN_SHA" --dry-run
# Custodian removes --dry-run for the single confirmation run.
```

The ledger reserves a release/set before reading trials. A partial run leaves a marker for the custodian to investigate. Final reports contain aggregate outcomes. Preserve the ledger across sessions. Historical v1 source/weight dependencies for the optional `pipeline_v3.v1_dev` and `pipeline_v3.v1_channels` commands are available in the owner's existing local checkout.
