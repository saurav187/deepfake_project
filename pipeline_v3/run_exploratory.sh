#!/usr/bin/env bash
# Exploratory English Teacher V1 baseline on the FAILED frozen-norm PhonemeDF audit
# (owner approval 25 Nov 2026; exception recorded by pipeline_v3/exploratory.py).
# One GPU VM with Drive mounted at /content/drive and this repo's pipeline_v3/ in /content/v3-src.
# Stops at the first failing step; Teacher V1 runs only if the pilot passes.
set -u
cd /content/v3-src
D=/content/drive/MyDrive/audio-v3
C=/content/crops
W=$D/stage0-en-ecapa-xtts-norm-exploratory
L=/content/v3-local-en-exploratory
X=$D/models/xlsr
log() { echo "$(TZ=Asia/Kolkata date +%T) IST $*"; }
save() { mkdir -p $W/logs; cp /content/x_*.log $W/logs/ 2>/dev/null; true; }
run() {
  local n=$1; shift
  log "$n: start"
  "$@" > /content/x_$n.log 2>&1; local rc=$?
  grep -v -i warn /content/x_$n.log | tail -30
  log "$n: exit $rc"; save
  [ $rc -eq 0 ] || { log "CHAIN_STOPPED at $n"; exit $rc; }
}
A="--data $D --languages en --voices ecapa --xtts-only --treatment norm --crop-cache $C --exploratory"
log "commit $(cat COMMIT 2>/dev/null) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
[ -f $W/exception.json ] || run exception python3 -m pipeline_v3.exploratory create --data $D
run cache python3 -m pipeline_v3.en_treatment cache --data $D --cache $C
run catalog python3 -m pipeline_v3.stage0 catalog $A
run pack python3 -m pipeline_v3.stage0 pack $A
run stage python3 -m pipeline_v3.cli stage --data $W --packed $W/packed --local $L \
  --splits train select calibrate threshold validate
run pilot python3 -m pipeline_v3.cli pilot --data $W --local $L --encoder $X \
  --small-encoder $D/models/wavlm-base --languages en
go=$(python3 -c "import json;print(json.load(open('$W/pilot.json')).get('go'))")
log "PILOT go=$go"
[ "$go" = "True" ] || { log "CHAIN_STOPPED pilot no-go"; exit 1; }
run teacher python3 -m pipeline_v3.cli teacher --data $W --local $L --encoder $X --languages en
run freeze python3 -m pipeline_v3.teacher_eval freeze --data $W --local $L --encoder $X
run evaluate python3 -m pipeline_v3.teacher_eval evaluate --data $W --local $L --encoder $X \
  --dev $D --panels libri-dev mlaad
log CHAIN_DONE
