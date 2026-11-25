#!/usr/bin/env bash
# Show the English XTTS-only treated run's progress and compute units used. Read-only.
#   bash pipeline_v3/colab_progress.sh            # once
#   watch -n 120 bash pipeline_v3/colab_progress.sh
# Env: SESSION (default v3-a100), START_CU (balance before the A100 started, default 195.81)
set -u
COLAB="${COLAB:-$HOME/.local/bin/colab} --auth=oauth2"
SESSION="${SESSION:-v3-a100}"
START_CU="${START_CU:-195.81}"

echo "=== Compute ($(date '+%F %T %Z')) ==="
usage=$($COLAB usage 2>&1)
echo "$usage" | grep -E "balance|rate|assignments"
bal=$(echo "$usage" | grep -oE "balance: [0-9.]+" | grep -oE "[0-9.]+")
[ -n "$bal" ] && awk -v s="$START_CU" -v b="$bal" 'BEGIN{printf "CU used since run start: %.2f (start %.2f)\n", s-b, s}'

remote='
D=/content/drive/MyDrive/audio-v3; DG=$D/stage0-en-ecapa-xtts-diag; LOG=/content/treat_chain.log
echo; echo "=== Run on $(hostname) at $(date -u +%T) UTC ==="
[ -f $LOG ] || { echo "no chain log yet"; exit 0; }
echo "commit: $(cat /content/v3-src/COMMIT 2>/dev/null)"
echo "steps:"; grep -E "start$|exit [0-9]+|FROZEN|AUDIT|PILOT|CHAIN_|cache:" $LOG | sed "s/^/  /"
echo "cached crops: $(ls /content/crops 2>/dev/null | grep -c wav) of ~11,500 selection rows"
step=$(grep -E ": start$" $LOG | tail -1 | awk "{print \$2}" | tr -d :)
case "$step" in
  diagnose)
    [ -f $DG/protocol.json ] && echo "protocol.json written (rule recorded before scoring)"
    [ -f $DG/diagnostic.partial.json ] && python3 -c "import json;r=json.load(open(\"$DG/diagnostic.partial.json\"));[print(f\"  {t}: score {v[\"score\"]:.3f}\") for t,v in r.items()]";;
  catalog) T=$(python3 -c "import json;print(json.load(open(\"$DG/frozen.json\"))[\"treatment\"])"); echo "catalog bytes: $(ls $D/development/stage0-en-ecapa-xtts-$T/bytes 2>/dev/null | wc -l)";;
  pilot|teacher) T=$(python3 -c "import json;print(json.load(open(\"$DG/frozen.json\"))[\"treatment\"])"); tail -5 /content/t_$step.log | grep -v -i warn;;
esac
[ -f $DG/frozen.json ] && python3 -c "import json;f=json.load(open(\"$DG/frozen.json\"));print(\"frozen:\", f[\"treatment\"], {k:round(v,3) for k,v in f[\"scores\"].items()})"
echo "GPU: $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
echo "load: $(cut -d" " -f1-3 /proc/loadavg)"
'
for i in 1 2 3 4 5 6; do
  out=$(timeout 120 $COLAB ssh -s "$SESSION" -- "$remote" </dev/null 2>&1)
  echo "$out" | grep -q "HTTP 429" || break
  sleep 15
done
echo "$out" | grep -v "HTTP 429"
