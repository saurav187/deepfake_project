#!/usr/bin/env bash
# Live download progress. Usage: bash detector/datasets/progress.sh   (Ctrl+C to quit)
cd "$(dirname "$0")"
declare -A SIZE=(
  [in_the_wild/release_in_the_wild.zip]=8161489023
  [asvspoof5/flac_T_aa.tar]=7522498560
  [asvspoof5/flac_D_aa.tar]=6647265280
  [asvspoof5/flac_T_ab.tar]=7489587200
)
ORDER=(in_the_wild/release_in_the_wild.zip asvspoof5/flac_T_aa.tar asvspoof5/flac_D_aa.tar asvspoof5/flac_T_ab.tar)
prev_total=-1
while true; do
  clear
  echo "Dataset downloads  ($(date +%H:%M:%S))"
  total=0
  for f in "${ORDER[@]}"; do
    have=0; [[ -f $f ]] && have=$(stat -c %s "$f")
    total=$((total + have))
    state="waiting"
    [[ -f $f ]] && state="downloading"
    [[ -f $f.ok ]] && state="VERIFIED"
    grep -q "SKIPPED.*$f" fetch.log 2>/dev/null && state="SKIPPED (no disk space)"
    awk -v f="$f" -v h="$have" -v s="${SIZE[$f]}" -v st="$state" \
      'BEGIN{printf "  %-40s %6.2f / %5.2f GB  %5.1f%%  %s\n", f, h/1e9, s/1e9, 100*h/s, st}'
  done
  if (( prev_total >= 0 )); then
    awk -v a="$total" -v b="$prev_total" 'BEGIN{printf "\n  Speed: %.2f MB/s\n", (a-b)/5/1e6}'
  fi
  prev_total=$total
  printf "  Disk free: %s\n" "$(df -h --output=avail . | tail -1 | tr -d ' ')"
  grep -q FETCH_DONE fetch.log 2>/dev/null && echo "  All downloads finished." && break
  pgrep -f 'detector/datasets/fetch.sh|./fetch.sh' >/dev/null || echo "  WARNING: fetch.sh is not running (re-run: bash detector/datasets/fetch.sh)"
  sleep 5
done
