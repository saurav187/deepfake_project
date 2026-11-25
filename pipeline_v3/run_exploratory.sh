#!/usr/bin/env bash
# Owner-approved exploratory English Teacher V1 on the failed frozen-norm audit.
# Run with one writer on a GPU VM after mounting Drive and deploying the reviewed code.
set -euo pipefail
cd /content/v3-src
D=/content/drive/MyDrive/audio-v3
C=/content/crops
W=$D/stage0-en-ecapa-xtts-norm-exploratory
L=/content/v3-local-en-exploratory
X=$D/models/xlsr
log() { echo "$(TZ=Asia/Kolkata date +%T) IST $*"; }
run() {
  local n=$1 rc=0
  shift
  log "$n: start"
  local output=/content/x_$n.log
  "$@" > "$output" 2>&1 || rc=$?
  tail -30 "$output"
  cp "$output" "$W/logs/"
  log "$n: exit $rc"
  if ((rc != 0)); then log "CHAIN_STOPPED at $n"; exit "$rc"; fi
}
args=(--data "$D" --languages en --voices ecapa --xtts-only --treatment norm --crop-cache "$C" --exploratory)
for file in "$D/stage0-en-ecapa-xtts-norm/prep-audit.json" "$X/config.json" \
  "$D/models/wavlm-base/config.json" "$D/stage0/inventory.csv" "$D/stage0-en-ecapa-xtts/mlaad-diagnostic.csv"; do
  [[ -s "$file" ]] || { log "Missing input: $file"; exit 1; }
done
python3 -c 'import torch; import sys; sys.exit(0 if torch.cuda.is_available() else "CUDA unavailable")'
mkdir -p "$W/logs"
log "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"
[[ -f "$W/exception.json" ]] || run exception python3 -m pipeline_v3.exploratory create --data "$D"
if [[ ! -f "$W/audit.json" || ! -f "$W/packed/index.json" ]]; then
  run cache python3 -m pipeline_v3.en_treatment cache --data "$D" --cache "$C"
  run catalog python3 -m pipeline_v3.stage0 catalog "${args[@]}"
  run pack python3 -m pipeline_v3.stage0 pack "${args[@]}"
fi
run stage python3 -m pipeline_v3.cli stage --data "$W" --packed "$W/packed" --local "$L" \
  --splits train select calibrate threshold validate
if [[ ! -f "$W/pilot.json" ]]; then
  run pilot python3 -m pipeline_v3.cli pilot --data "$W" --local "$L" --encoder "$X" \
    --small-encoder "$D/models/wavlm-base" --languages en
fi
if ! python3 -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["go"] is True else 1)' "$W/pilot.json"; then
  log "CHAIN_STOPPED pilot no-go or unreadable report"; exit 1
fi
# Both teacher and evaluation independently require a bound passing pilot.
if [[ ! -f "$W/runs/teacher-frozen.json" ]]; then
  run teacher python3 -m pipeline_v3.cli teacher --data "$W" --local "$L" --encoder "$X" --languages en
  run freeze python3 -m pipeline_v3.teacher_eval freeze --data "$W" --local "$L" --encoder "$X"
fi
run evaluate python3 -m pipeline_v3.teacher_eval evaluate --data "$W" --local "$L" --encoder "$X" \
  --dev "$D" --panels libri-dev mlaad
log CHAIN_DONE
