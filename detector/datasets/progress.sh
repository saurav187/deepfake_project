#!/usr/bin/env bash
# Live download progress. Usage: bash detector/datasets/progress.sh   (Ctrl+C to quit)
# Works for both downloaders: fetch.sh (sequential curl) and pdownload.py
# (parallel, sparse pre-sized file). Progress counts bytes actually on disk.
cd "$(dirname "$0")"
declare -A SIZE=(
  [in_the_wild/release_in_the_wild.zip]=8161489023
  [asvspoof5/flac_T_aa.tar]=7522498560
  [asvspoof5/flac_D_aa.tar]=6647265280
  [asvspoof5/flac_T_ab.tar]=7489587200
)
ORDER=(in_the_wild/release_in_the_wild.zip asvspoof5/flac_T_aa.tar asvspoof5/flac_D_aa.tar asvspoof5/flac_T_ab.tar)

have_bytes() {  # bytes actually written: pre-sized parallel downloads are sparse files
  local f=$1
  if [[ -f $f.ok ]]; then echo "${SIZE[$f]}"
  elif [[ -f $f ]]; then
    local b; b=$(du --block-size=1 "$f" | cut -f1)
    (( b > SIZE[$f] )) && b=${SIZE[$f]}; echo "$b"
  else echo 0; fi
}

prev_disk=-1
while true; do
  clear
  echo "Dataset downloads  ($(date +%H:%M:%S))"
  for f in "${ORDER[@]}"; do
    have=$(have_bytes "$f")
    state="waiting"
    [[ -f $f ]] && state="downloading"
    [[ -f $f.ok ]] && state="VERIFIED"
    awk -v f="$f" -v h="$have" -v s="${SIZE[$f]}" -v st="$state" \
      'BEGIN{printf "  %-40s %6.2f / %5.2f GB  %5.1f%%  %s\n", f, h/1e9, s/1e9, 100*h/s, st}'
  done
  # Speed from real disk usage (parallel downloads write inside pre-sized files).
  disk=$(du -s --block-size=1 asvspoof5 in_the_wild 2>/dev/null | awk '{s+=$1} END{print s}')
  if (( prev_disk >= 0 )); then
    awk -v a="$disk" -v b="$prev_disk" 'BEGIN{printf "\n  Total speed: %.2f MB/s\n", (a-b)/5/1e6}'
  fi
  prev_disk=$disk
  printf "  Disk free: %s\n" "$(df -h --output=avail . | tail -1 | tr -d ' ')"
  for log in pdownload_*.log; do
    [[ -f $log ]] && tail -1 "$log" | grep -v RuntimeWarning | sed 's/^/  last: /'
  done
  sleep 5
done
