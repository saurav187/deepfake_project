#!/usr/bin/env bash
# Read-only progress for the ASVspoof evaluation, including an already running job.
set -u
cd "$(dirname "$0")/../.." || exit 1
summary=deepfake_model/research/asvspoof_la_v1/summary.csv
unit=asvspoof-la-baseline-v1.service
if [[ -f "$summary" ]]; then
    echo "Evaluation complete"
    cat "$summary"
    exit 0
fi
pid=$(systemctl --user show "$unit" -p MainPID --value) || exit 1
if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
    systemctl --user status "$unit" --no-pager
    exit
fi
archive=$(realpath deepfake_model/datasets/asvspoof2021_la/ASVspoof2021_LA_eval.tar.gz)
for fd in /proc/"$pid"/fd/*; do
    [[ $(readlink "$fd") == "$archive" ]] || continue
    awk -v size="$(stat -c %s "$archive")" '$1 == "pos:" {
        printf "Archive pass: %.1f%% (%.2f / %.2f GB read)\n", 100*$2/size, $2/1e9, size/1e9
    }' "/proc/$pid/fdinfo/${fd##*/}"
    exit
done
echo "Service is running. No readable archive position at this instant."
