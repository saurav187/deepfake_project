#!/usr/bin/env bash
# Live training dashboard. Usage (repo root):  bash detector/watch.sh   (Ctrl+C to quit; training keeps running)
cd "$(dirname "$0")/.." || exit 1
LOG=detector/runs/model_a.log
PY=venv/bin/python
while true; do
  clear
  echo "Model A training  ($(date +%H:%M:%S))"
  if pgrep -f 'bash detector/run_model_a.sh' >/dev/null; then echo "  launcher: running"; else echo "  launcher: NOT running"; fi
  stage=$(grep -E 'waiting for|building manifest|running audit|running edge probe|training |MODEL_A_DONE|Traceback|early stop' "$LOG" | tail -1 | cut -c1-110)
  echo "  last stage: $stage"
  echo
  $PY - <<'EOF' 2>/dev/null
import csv, json, time
from pathlib import Path
for run in sorted(Path("detector/runs/models").glob("*_s*")):
    hist = run / "history.csv"
    if not hist.exists():
        print(f"{run.name}: starting (no finished epoch yet)\n"); continue
    rows = list(csv.DictReader(hist.open()))
    cfg = json.loads((run / "config.json").read_text())
    done = (run / "summary.json").exists()
    best = min(rows, key=lambda r: float(r["selection_eer"]))
    per_epoch = sum(float(r["train_s"]) + float(r["dev_s"]) for r in rows) / len(rows)
    left = 0 if done else (cfg["epochs"] - len(rows)) * per_epoch
    status = "DONE" if done else f"epoch {len(rows)}/{cfg['epochs']}, ~{per_epoch/60:.1f} min/epoch, at most ~{left/60:.0f} min left"
    print(f"{run.name}: {status}  (best epoch {best['epoch']}, mean dev EER {float(best['selection_eer']):.4f})")
    keys = [k for k in rows[0] if k.endswith("/eer")]
    print("  epoch  loss    mean EER  " + "  ".join(f"{k[:-4]:>17}" for k in keys))
    for r in rows:
        mark = " *" if r is best else "  "
        print(f"  {r['epoch']:>4}{mark}{float(r['train_loss']):.3f}   {float(r['selection_eer']):.4f}  "
              + "  ".join(f"{float(r[k]):>17.4f}" for k in keys))
    print()
EOF
  echo "  EER = dev equal error rate (lower is better); * = best epoch so far"
  echo "  GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader)"
  echo "  RAM available: $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo)"
  grep -q MODEL_A_DONE "$LOG" && echo "  All runs finished." && break
  sleep 15
done
