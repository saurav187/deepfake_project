#!/usr/bin/env bash
# Model A, first comparison: waits for the ASVspoof 5 train (flac_T_aa) and dev
# (flac_D_aa) tars, rebuilds the manifest and audit, then trains one seed of
# each encoder. Safe to start before the downloads finish. Log: detector/runs/model_a.log
# Usage (repo root):  setsid nohup bash detector/run_model_a.sh >/dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p detector/runs
exec >> detector/runs/model_a.log 2>&1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=venv/bin/python
D=detector/datasets/asvspoof5

echo "$(date '+%F %T') waiting for $D/flac_T_aa.tar and $D/flac_D_aa.tar"
until [[ -f $D/flac_T_aa.tar.ok && -f $D/flac_D_aa.tar.ok ]]; do
    if grep -q 'FAILED' detector/datasets/fetch.log 2>/dev/null; then
        echo "$(date '+%F %T') a download failed its checksum; stopping"; exit 1
    fi
    sleep 60
done

# Wait until the GPU has room, so another job on the GPU (e.g. a smoke test) cannot
# make an unattended training run fail with out-of-memory. Needs NEED_MIB free on
# two checks 30 s apart; logs while waiting.
NEED_MIB=6000
wait_for_gpu() {
    local free waited=0
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if (( free >= NEED_MIB )); then
            sleep 30
            free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
            (( free >= NEED_MIB )) && return
        fi
        (( waited % 600 == 0 )) && echo "$(date '+%F %T') waiting for GPU: ${free} MiB free, need ${NEED_MIB} MiB"
        sleep 60
        waited=$((waited + 60))
    done
}

echo "$(date '+%F %T') building manifest"
$PY -m detector.data.build_manifest asvspoof5
echo "$(date '+%F %T') running audit"
$PY -m detector.data.audit

for encoder in wav2vec2-xls-r-300m wavlm-base-plus; do
    out=detector/runs/models/${encoder}_s42
    if [[ -f $out/summary.json ]]; then echo "already trained: $out"; continue; fi
    rm -rf "$out"   # an interrupted run is restarted from scratch
    wait_for_gpu
    echo "$(date '+%F %T') training $encoder seed 42 ($(nvidia-smi --query-gpu=memory.free --format=csv,noheader) free)"
    $PY -m detector.train.train --encoder "$encoder" --seed 42 --out "$out"
done
echo "$(date '+%F %T') MODEL_A_DONE"
