#!/usr/bin/env bash
# V2-90 GPU chain (protocol college/28_V2_90_PROTOCOL.md). Run from the deployed source root on the A100.
#   bash pipeline_v3/run_v2_90.sh prep|train|finish
# prep:   copy raw from Drive to local disk, preprocess, audit (outputs mirrored to Drive)
# train:  fine-tune with checkpoints every EVAL_EVERY steps (resumes from ckpt/last.pt)
# finish: calibrate -> signoff -> evaluate (one pass)
set -euo pipefail
D=/content/drive/MyDrive/audio-v3/v2/v2-90
W=/content/v290-work
R=/content/v290-raw
mkdir -p "$W" "$D/run"
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S) IST] $*" | tee -a "$D/run/chain.log"; }
case "${1:?phase}" in
  prep)
    log "copy raw"
    mkdir -p "$R"
    rsync -a --info=stats1 "$D/raw/" "$R/"
    cp /content/drive/MyDrive/audio-v3/development/mailabs/en_US.tgz "$R/mailabs_en_US.tgz"
    log "prep"
    python3 -u -m pipeline_v3.v2_90 prep --raw "$R" --work "$W" --mailabs "$R/mailabs_en_US.tgz" \
      --workers "$(($(nproc) - 1))" 2>&1 | tee "$D/run/prep.log"
    cp "$W/prep.json" "$D/run/prep.json"
    log "mirror shards"
    rsync -a "$W/shards/" "$D/shards/"
    log "audit"
    python3 -u -m pipeline_v3.v2_90 audit --work "$W" 2>&1 | tee "$D/run/audit.log"
    cp "$W/audit.json" "$D/run/audit.json"
    log "prep done"
    ;;
  train)
    # TAG, ENCODER, FREEZE_BELOW, BATCH, STEPS, LR_ENC via env
    TAG=${TAG:-c2}
    [ -d "$W/shards" ] || rsync -a "$D/shards/" "$W/shards/"
    if [ ! -f "$W/ckpt-$TAG/last.pt" ] && [ -f "$D/ckpt-$TAG/last.pt" ]; then mkdir -p "$W/ckpt-$TAG"; cp "$D/ckpt-$TAG/last.pt" "$W/ckpt-$TAG/"; fi
    log "train $TAG start (${ENCODER:-facebook/wav2vec2-xls-r-300m})"
    python3 -u -m pipeline_v3.v2_90 train --work "$W" --drive "$D" --tag "$TAG" \
      --encoder "${ENCODER:-facebook/wav2vec2-xls-r-300m}" --freeze-below "${FREEZE_BELOW:-0}" \
      --steps "${STEPS:-6000}" --batch "${BATCH:-32}" --lr-enc "${LR_ENC:-2e-5}" \
      --eval-every "${EVAL_EVERY:-1000}" --workers "$(($(nproc) - 1))" 2>&1 | tee -a "$D/run/train-$TAG.log"
    log "train $TAG done"
    ;;
  finish)
    # CANDS="c2=facebook/wav2vec2-xls-r-300m c3=nii-yamagishilab/xls-r-1b-anti-deepfake"
    [ -d "$W/shards" ] || rsync -a "$D/shards/" "$W/shards/"
    for c in $CANDS; do t=${c%%=*}; mkdir -p "$W/ckpt-$t"; [ -f "$W/ckpt-$t/best.pt" ] || cp "$D/ckpt-$t/best.pt" "$W/ckpt-$t/"; done
    log "select"
    python3 -u -m pipeline_v3.v2_90 select --work "$W" --cands $CANDS 2>&1 | tee "$D/run/select.log"
    cp "$W/selection.json" "$D/run/"
    log "calibrate"
    python3 -u -m pipeline_v3.v2_90 calibrate --work "$W" 2>&1 | tee "$D/run/calibrate.log"
    cp "$W/cutoff.json" "$D/run/"
    log "signoff"
    python3 -u -m pipeline_v3.v2_90 signoff --work "$W" --protocol college/28_V2_90_PROTOCOL.md \
      --commit "$(cat COMMIT)" 2>&1 | tee "$D/run/signoff.log"
    cp "$W/signoff.json" "$D/run/"
    log "evaluate"
    python3 -u -m pipeline_v3.v2_90 evaluate --work "$W" 2>&1 | tee "$D/run/evaluate.log"
    cp "$W/result.json" "$W/eval-scores.jsonl" "$D/run/"
    log "finish done"
    ;;
esac
