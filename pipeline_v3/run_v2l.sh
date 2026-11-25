#!/usr/bin/env bash
# V2-L LibriSeVoc-only baseline. One writer. Mirrors run_v2a.sh with its own work dir.
#   bash pipeline_v3/run_v2l.sh prepare   # identity check, 1,000-clip benchmark, cache, audit, pack
#   bash pipeline_v3/run_v2l.sh gpu       # stage, smoke step, pilot, teacher, freeze, evaluate
set -euo pipefail
cd "$(dirname "$0")/.."
D=${V2L_DATA:-/content/drive/MyDrive/audio-v3}
W="$D/v2/v2l-lsv-norm-1"
S=${V2L_SCRATCH:-/content}
L="$S/v2l-staged"
X="$D/models/xlsr"
WORKERS=${V2L_WORKERS:-8}
log() { echo "$(TZ=Asia/Kolkata date +%T) IST $*"; }
case "${1:-}" in
  prepare)
    [[ -s "$W/lineage.csv" ]] || { echo "Missing $W/lineage.csv (run v2a_lineage --work v2l-lsv-norm-1)" >&2; exit 1; }
    log "prepare: start"
    python3 -m pipeline_v3.v2l prepare --data "$D" --scratch "$S" --workers "$WORKERS"
    log "prepare: exit 0"
    ;;
  gpu)
    [[ -s "$W/audit.json" ]] || { echo 'Complete the V2-L audit and pack first.' >&2; exit 1; }
    python3 -m pipeline_v3.v2l prepare --data "$D" --scratch "$S"   # verifies the published pack
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
    log "stage: start"
    python3 -m pipeline_v3.cli stage --data "$W" --packed "$W/packed" --local "$L"
    python3 - "$X" <<'PY'
import sys
import torch
from pipeline_v3.audio import WINDOW
from pipeline_v3.models import Teacher
model = Teacher(sys.argv[1]).cuda().train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
model(torch.zeros(WINDOW, device="cuda")).backward()
opt.step()
print("Teacher smoke peak allocated MiB", torch.cuda.max_memory_allocated() // 2**20)
PY
    args=(--data "$W" --local "$L" --encoder "$X" --languages en)
    if [[ ! -f "$W/pilot.json" ]]; then
      log "pilot: start"
      python3 -m pipeline_v3.cli pilot "${args[@]}" --small-encoder "$D/models/wavlm-base"
    fi
    python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["go"] is True else 1)' "$W/pilot.json" \
      || { log "CHAIN_STOPPED pilot no-go"; exit 1; }
    if [[ ! -f "$W/runs/teacher-frozen.json" ]]; then
      log "teacher: start"
      python3 -m pipeline_v3.cli teacher "${args[@]}"
      log "freeze: start"
      python3 -m pipeline_v3.teacher_eval freeze --data "$W" --local "$L" --encoder "$X"
    fi
    log "evaluate: start"
    python3 -m pipeline_v3.teacher_eval evaluate --data "$W" --local "$L" --encoder "$X" \
      --dev "$D" --panels libri-dev mlaad
    log CHAIN_DONE
    ;;
  *) echo "Usage: $0 prepare|gpu" >&2; exit 2 ;;
esac
