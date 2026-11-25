#!/usr/bin/env bash
# One writer per V2-A experiment. CPU preparation and GPU training can use separate VMs.
set -euo pipefail
cd "$(dirname "$0")/.."
D=${V2A_DATA:-/content/drive/MyDrive/audio-v3}
W="$D/v2/v2a-norm-1"
S=${V2A_SCRATCH:-/content}
L="$S/v2a-staged"
X="$D/models/xlsr"
case "${1:-}" in
  inventory|confirm|prepare)
    phase=$1
    [[ $phase != inventory ]] || phase=requests
    python3 -m pipeline_v3.v2a "$phase" --data "$D" --scratch "$S"
    ;;
  gpu)
    [[ -s "$W/audit.json" ]] || { echo 'Complete the V2-A audit and pack first.' >&2; exit 1; }
    python3 -m pipeline_v3.v2a prepare --data "$D" --scratch "$S"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
    python3 -m pipeline_v3.cli stage --data "$W" --local "$L"
    # Check the real teacher forward, backward and optimizer memory on this GPU.
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
      python3 -m pipeline_v3.cli pilot "${args[@]}" --small-encoder "$D/models/wavlm-base"
    fi
    if [[ ! -f "$W/runs/teacher-frozen.json" ]]; then
      python3 -m pipeline_v3.cli teacher "${args[@]}"
      python3 -m pipeline_v3.teacher_eval freeze --data "$W" --local "$L" --encoder "$X"
    fi
    python3 -m pipeline_v3.teacher_eval evaluate --data "$W" --local "$L" --encoder "$X" \
      --dev "$D" --panels libri-dev mlaad
    ;;
  *) echo "Usage: $0 inventory|confirm|prepare|gpu" >&2; exit 2 ;;
esac
