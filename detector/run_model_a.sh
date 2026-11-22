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

echo "$(date '+%F %T') building manifest"
$PY -m detector.data.build_manifest asvspoof5
echo "$(date '+%F %T') running audit"
$PY -m detector.data.audit

for encoder in wav2vec2-xls-r-300m wavlm-base-plus; do
    out=detector/runs/models/${encoder}_s42
    if [[ -f $out/summary.json ]]; then echo "already trained: $out"; continue; fi
    rm -rf "$out"   # an interrupted run is restarted from scratch
    echo "$(date '+%F %T') training $encoder seed 42"
    $PY -m detector.train.train --encoder "$encoder" --seed 42 --out "$out"
done
echo "$(date '+%F %T') MODEL_A_DONE"
