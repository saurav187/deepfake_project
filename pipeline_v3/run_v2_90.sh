#!/usr/bin/env bash
# V2-90 GPU chain (protocol college/28_V2_90_PROTOCOL.md). Run from the deployed source root.
#   bash pipeline_v3/run_v2_90.sh prep|train|finish
# prep:   copy raw from Drive, stage + verify + publish prepared shards (PREPARED.json last), preflight, audit
# train:  fetch prepared set if absent, verify its manifest, fine-tune (checkpoint bound to dataset+config)
# finish: select -> calibrate -> signoff -> evaluate (one pass; refuses on any binding mismatch)
set -euo pipefail
D=/content/drive/MyDrive/audio-v3/v2/v2-90
W=/content/v290-work
R=/content/v290-raw
mkdir -p "$W" "$D/run"
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S) IST] $*" | tee -a "$D/run/chain.log"; }
fetch_prepared() {
  if [ ! -f "$W/prepared/PREPARED.json" ]; then
    [ -f "$D/prepared/PREPARED.json" ] || { log "no published prepared set on Drive"; exit 2; }
    rm -rf "$W/prepared.copy"
    rsync -a "$D/prepared/" "$W/prepared.copy/"
    mv "$W/prepared.copy" "$W/prepared"
  fi
  python3 -u -m pipeline_v3.v2_90 verify --work "$W"
}
case "${1:?phase}" in
  prep)
    log "copy raw"
    mkdir -p "$R"
    rsync -a --info=stats1 --exclude librittsr --exclude globe --exclude voxenes --exclude '*.part' "$D/raw/" "$R/"
    [ -f "$R/mailabs_en_US.tgz" ] || cp /content/drive/MyDrive/audio-v3/development/mailabs/en_US.tgz "$R/mailabs_en_US.tgz"
    log "prep (staged; published only after verification)"
    python3 -u -m pipeline_v3.v2_90 prep --raw "$R" --work "$W" --mailabs "$R/mailabs_en_US.tgz" \
      --hindi /content/drive/MyDrive/audio-v3/development \
      --workers "$(($(nproc) - 1))" 2>&1 | tee "$D/run/prep.log"
    cp "$W/prepared/prep.json" "$D/run/prep.json"
    log "publish prepared set to Drive (staging dir, then rename)"
    rm -rf "$D/prepared.staging"
    rsync -a "$W/prepared/" "$D/prepared.staging/"
    if [ -d "$D/prepared" ]; then mv "$D/prepared" "$D/prepared-superseded-$(date -u +%Y%m%dT%H%M%S)"; fi
    mv "$D/prepared.staging" "$D/prepared"
    log "preflight"
    python3 -u -m pipeline_v3.v2_90 preflight --work "$W" 2>&1 | tee "$D/run/preflight.log"
    cp "$W/preflight.json" "$D/run/"
    log "audit"
    python3 -u -m pipeline_v3.v2_90 audit --work "$W" 2>&1 | tee "$D/run/audit.log"
    cp "$W/audit.json" "$D/run/audit.json"
    log "prep done"
    ;;
  train)
    # TAG, ENCODER, FREEZE_BELOW, BATCH, STEPS, LR_ENC, EVAL_EVERY, STOP_AFTER, EXTRA via env
    TAG=${TAG:-c2}
    fetch_prepared
    if [ ! -f "$W/ckpt-$TAG/last.pt" ] && [ -f "$D/ckpt-$TAG/last.pt" ]; then mkdir -p "$W/ckpt-$TAG"; cp "$D/ckpt-$TAG/last.pt" "$W/ckpt-$TAG/"; fi
    log "train $TAG start (${ENCODER:-facebook/wav2vec2-xls-r-300m})"
    python3 -u -m pipeline_v3.v2_90 train --work "$W" --drive "$D" --tag "$TAG" \
      --encoder "${ENCODER:-facebook/wav2vec2-xls-r-300m}" --freeze-below "${FREEZE_BELOW:-0}" \
      --steps "${STEPS:-6000}" --batch "${BATCH:-32}" --lr-enc "${LR_ENC:-2e-5}" \
      --eval-every "${EVAL_EVERY:-1000}" --workers "$(($(nproc) - 1))" \
      ${STOP_AFTER:+--stop-after "$STOP_AFTER"} ${EXTRA:-} 2>&1 | tee -a "$D/run/train-$TAG.log"
    log "train $TAG done"
    ;;
  finish)
    # CANDS="c2=facebook/wav2vec2-xls-r-300m c3=nii-yamagishilab/xls-r-1b-anti-deepfake"
    fetch_prepared
    for c in $CANDS; do t=${c%%=*}; mkdir -p "$W/ckpt-$t"
      for f in best.pt best.json; do [ -f "$W/ckpt-$t/$f" ] || cp "$D/ckpt-$t/$f" "$W/ckpt-$t/"; done; done
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
