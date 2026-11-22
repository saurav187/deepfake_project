#!/usr/bin/env bash
# Resumable, checksum-verified downloads in priority order. A large file is
# skipped (not partially fetched) when the disk lacks room for it plus 2 GB.
# Re-run at any time: finished files are re-verified, partial ones resume.
set -euo pipefail
cd "$(dirname "$0")"
Z=https://zenodo.org/api/records/14498691/files

get() {  # url file algo digest bytes
    local url=$1 file=$2 algo=$3 digest=$4 bytes=$5
    if [[ -f $file.ok ]]; then echo "already verified: $file"; return; fi
    local have=0; [[ -f $file ]] && have=$(stat -c %s "$file")
    local free=$(( $(df --output=avail -B1 . | tail -1) ))
    if (( bytes - have + 2*1024**3 > free )); then
        echo "SKIPPED (needs $(( (bytes-have)/1024**3 )) GB, free $(( free/1024**3 )) GB): $file"; return
    fi
    # Abort a stalled transfer (<10 KB/s for 60 s) and resume, up to 50 times.
    local try
    for try in $(seq 1 50); do
        curl -L --fail --retry 5 --retry-delay 10 --speed-limit 10240 --speed-time 60 \
             -C - -o "$file" "$url" 2>>curl_progress.log && break
        echo "$(date +%H:%M:%S) transfer interrupted, resuming (attempt $try): $file"
        sleep 5
    done
    echo "$digest  $file" | "${algo}sum" -c - && touch "$file.ok"
}

mkdir -p asvspoof5 in_the_wild
get "$Z/ASVspoof5_protocols.tar.gz/content" asvspoof5/ASVspoof5_protocols.tar.gz md5 865d0e894ea9f686f0f37e5ae3ae3616 20669392
get "$Z/LICENSE.txt/content" asvspoof5/LICENSE.txt md5 9cbefc63251e8cbc8eb22e62cd6fe584 19940
get "$Z/README.txt/content" asvspoof5/README.txt md5 784002b8449ba1eeecc042bfad285de7 12131
get "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip" \
    in_the_wild/release_in_the_wild.zip sha256 46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a 8161489023
get "$Z/flac_T_aa.tar/content" asvspoof5/flac_T_aa.tar md5 b0cc86b14826a7701b52aad4f53daf9c 7522498560
get "$Z/flac_D_aa.tar/content" asvspoof5/flac_D_aa.tar md5 df0be44957623991028cce59792beb17 6647265280
get "$Z/flac_T_ab.tar/content" asvspoof5/flac_T_ab.tar md5 d05be3f4be7a343fdbdd0ed29fdff2e1 7489587200
echo FETCH_DONE
