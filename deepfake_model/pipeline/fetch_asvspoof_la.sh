#!/usr/bin/env bash
# From a project checkout, resume the official archive and run the frozen comparison.
set -euo pipefail
cd "$(dirname "$0")/../.."
data=deepfake_model/datasets/asvspoof2021_la
archive="$data/ASVspoof2021_LA_eval.tar.gz"
partial="$archive.part"
results=deepfake_model/research/asvspoof_la_v1
mkdir -p "$data"
exec 9>"$data/fetch.lock"
flock -n 9 || { echo "Another downloader owns the lock." >&2; exit 1; }
valid_archive() {
    echo "2abee34d8b0b91159555fc4f016e4562  $1" | md5sum --check --status
}
if [[ ! -f "$data/LA-keys-full.tar.gz" ]]; then
    curl --fail --location --retry 4 -o "$data/keys.part" https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz
    gzip -t "$data/keys.part"
    mv "$data/keys.part" "$data/LA-keys-full.tar.gz"
fi
if [[ ! -f "$archive" ]]; then
    # Allow a transfer already started in the terminal to finish. If it stops
    # early, curl resumes from the bytes retained in the same .part file.
    while [[ -e "$partial" ]] && fuser -s "$partial"; do sleep 15; done
    if [[ ! -e "$partial" ]] || ! valid_archive "$partial"; then
        for attempt in {1..6}; do
            if curl --fail --silent --show-error --location --retry 6 --connect-timeout 30 --speed-time 120 --speed-limit 512 \
                --continue-at - -o "$partial" https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz; then
                break
            else
                status=$?
            fi
            # A fresh curl process resumes from the latest on-disk byte after a short response.
            if [[ $status != 18 || $attempt == 6 ]]; then exit "$status"; fi
            sleep 10
        done
    fi
    valid_archive "$partial" || { echo "Archive checksum failed" >&2; exit 1; }
    mv "$partial" "$archive"
fi
if [[ ! -d "$results" ]]; then
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    exec venv/bin/python deepfake_model/pipeline/evaluate_asvspoof_la.py --out "$results"
fi
