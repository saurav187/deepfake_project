#!/usr/bin/env bash
# Overnight chain after the Model A encoder comparison (run_model_a.sh):
#   1. wait for MODEL_A_DONE
#   2. choose the encoder by a rule fixed before results: lowest best dev selection
#      EER; within 0.005 is a tie and goes to XLS-R (multilingual, for Hindi)
#   3. re-audit and edge-gate the full data (both ASVspoof 5 train tars)
#   4. train seeds 42, 43, 44 of the chosen encoder on the full data
#   5. calibrate them into detector/runs/release/v1 (dev only)
# It stops before any test-set evaluation. Log: detector/runs/final.log
# Usage (repo root):  setsid nohup bash detector/run_final.sh >/dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
exec >> detector/runs/final.log 2>&1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=venv/bin/python
NEED_MIB=6000

wait_for_gpu() {
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if (( free >= NEED_MIB )); then
            sleep 30
            free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
            (( free >= NEED_MIB )) && return
        fi
        echo "$(date '+%F %T') waiting for GPU: ${free} MiB free"
        sleep 300
    done
}

echo "$(date '+%F %T') waiting for the encoder comparison (MODEL_A_DONE)"
until grep -q MODEL_A_DONE detector/runs/model_a.log; do
    if ! pgrep -f 'bash detector/run_model_a.sh' >/dev/null && ! grep -q MODEL_A_DONE detector/runs/model_a.log; then
        echo "$(date '+%F %T') run_model_a.sh stopped without finishing; aborting"; exit 1
    fi
    sleep 60
done

$PY -m detector.calibrate --compare detector/runs/models/wav2vec2-xls-r-300m_s42 detector/runs/models/wavlm-base-plus_s42
ENCODER=$($PY - <<'EOF'
import json
eer = {e: json.load(open(f"detector/runs/models/{e}_s42/summary.json"))["best_selection_eer"]
       for e in ("wav2vec2-xls-r-300m", "wavlm-base-plus")}
x, w = eer["wav2vec2-xls-r-300m"], eer["wavlm-base-plus"]
print("wavlm-base-plus" if w < x - 0.005 else "wav2vec2-xls-r-300m")
EOF
)
echo "$(date '+%F %T') chosen encoder: $ENCODER (rule: lowest dev selection EER; ties within 0.005 go to XLS-R)"

echo "$(date '+%F %T') audit and edge gate on the full data"
$PY -m detector.data.audit
$PY -m detector.data.edge_probe

RUNS=()
for seed in 42 43 44; do
    out=detector/runs/models/final_${ENCODER}_s${seed}
    RUNS+=("$out")
    if [[ -f $out/summary.json ]]; then echo "already trained: $out"; continue; fi
    rm -rf "$out"
    wait_for_gpu
    echo "$(date '+%F %T') training final $ENCODER seed $seed"
    $PY -m detector.train.train --encoder "$ENCODER" --seed "$seed" --out "$out"
done

if [[ ! -d detector/runs/release/v1 ]]; then
    echo "$(date '+%F %T') calibrating release v1"
    $PY -m detector.calibrate --runs "${RUNS[@]}" --out detector/runs/release/v1
fi
echo "$(date '+%F %T') FINAL_DONE (test sets untouched)"
