#!/usr/bin/env bash
# Pack everything Git does not carry but the app needs, into one tar to copy to another computer:
#   the XLS-R encoder, release v1 (release.json + hash), the 3 trained heads, and the built web page.
# Paths inside the tar are relative to the repository root, so extract it there.
# Usage (repo root):  bash detector/make_bundle.sh [output.tar]     (default ~/deepfake_bundle_v1.tar)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$HOME/deepfake_bundle_v1.tar}
REL=detector/runs/release/v1
ENC=detector/pretrained/wav2vec2-xls-r-300m

[[ -f $REL/release.json ]] || { echo "missing $REL"; exit 1; }
[[ -f app/frontend/dist/index.html ]] || { echo "build the page first: npm --prefix app/frontend run build"; exit 1; }
mapfile -t RUNS < <(venv/bin/python -c "import json; [print(m['run']) for m in json.load(open('$REL/release.json'))['members']]")

FILES=("$ENC/config.json" "$ENC/preprocessor_config.json" "$ENC/pytorch_model.bin"
       "$REL/release.json" "$REL/release.sha256")
for run in "${RUNS[@]}"; do FILES+=("$run/best.pt" "$run/config.json" "$run/summary.json"); done
while IFS= read -r f; do FILES+=("$f"); done < <(find app/frontend/dist -type f | sort)

sha256sum "${FILES[@]}" > BUNDLE_SHA256SUMS
tar -cf "$OUT" BUNDLE_SHA256SUMS "${FILES[@]}"
rm BUNDLE_SHA256SUMS
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1), ${#FILES[@]} files). Extract it in the repository root on the other computer."
