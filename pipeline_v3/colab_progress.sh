#!/usr/bin/env bash
# Read-only progress: compute units, A100 wait, V1 exploratory prep/run, V2 downloads.
#   bash pipeline_v3/colab_progress.sh
#   watch -n 120 bash pipeline_v3/colab_progress.sh
# Env: SESSION (prep/download VM, default v3-t4p), GPU_SESSION (default v3-a100x),
#      START_CU (balance when this round started, default 183.44),
#      A100_LOG (local log of the A100 retry loop, optional)
set -u
COLAB="${COLAB:-$HOME/.local/bin/colab} --auth=oauth2"
SESSION="${SESSION:-v3-t4p}"
GPU_SESSION="${GPU_SESSION:-v3-a100x}"
START_CU="${START_CU:-183.44}"
A100_LOG="${A100_LOG:-$(ls -t /tmp/claude-1000/*/*/tasks/bdkpjyxy2.output 2>/dev/null | head -1)}"

echo "=== Compute  $(TZ=Asia/Kolkata date '+%F %T') IST ==="
usage=$($COLAB usage 2>&1)
echo "$usage" | grep -E "balance|rate|assignments" | sed 's/^/  /'
bal=$(echo "$usage" | grep -oE "balance: [0-9.]+" | grep -oE "[0-9.]+")
[ -n "$bal" ] && awk -v s="$START_CU" -v b="$bal" 'BEGIN{printf "  CU used since %.2f: %.2f\n", s, s-b}'
echo "  sessions:"; $COLAB sessions 2>&1 | grep "^\[" | sed 's/^/    /'

echo "=== A100 wait ==="
if [ -n "$A100_LOG" ] && [ -f "$A100_LOG" ]; then
  tries=$(grep -c "try" "$A100_LOG"); first=$(grep -m1 -oE "^[0-9:]+" "$A100_LOG")
  echo "  tries: $tries (first $first IST), last: $(tail -1 "$A100_LOG")"
else
  echo "  retry-loop log not found (set A100_LOG)"
fi

remote='
D=/content/drive/MyDrive/audio-v3; W=$D/stage0-en-ecapa-xtts-norm-exploratory
echo "=== V1 exploratory prep / run (VM $(hostname), $(TZ=Asia/Kolkata date +%T) IST) ==="
for f in /content/prep.log /content/run.log; do
  [ -f $f ] && { echo "  $(basename $f):"; grep -E ": start$|exit [0-9]+|PREP_DONE|CHAIN_|PILOT" $f | tail -8 | sed "s/^/    /"; }
done
echo "  crop cache: $(ls /content/crops 2>/dev/null | grep -c wav) / 10,433 crops"
b=$D/development/stage0-en-ecapa-xtts-norm-exploratory/bytes
[ -d $b ] && echo "  catalog bytes on Drive: $(ls $b | wc -l) / ~10,433 (catalog.csv: $([ -f $W/catalog.csv ] && echo yes || echo no))"
[ -d $W/packed ] && echo "  packed/: $(ls $W/packed | wc -l) files, index.json $([ -f $W/packed/index.json ] && echo present || echo missing)"
[ -f $W/audit.json ] && python3 -c "import json;r=json.load(open(\"$W/audit.json\"));print(\"  audit.json: pass=%s exception=%s index=%s\" % (r.get(\"pass\"), (r.get(\"exception\") or {}).get(\"exception\"), r.get(\"index_sha256\",\"\")[:12]))"
[ -f $W/pilot.json ] && python3 -c "import json;print(\"  pilot go:\", json.load(open(\"$W/pilot.json\")).get(\"go\"))"
ls $W/runs 2>/dev/null | sed "s/^/  runs: /"

echo "=== V2 downloads ==="
python3 - <<PY
import os, struct, glob
dl = "/content/dl"; drive = "/content/drive/MyDrive/audio-v3/development"
FILES = [("librisevoc","LibriSeVoc.zip",45046253997),("asvspoof5","ASVspoof5_protocols.tar.gz",20669392),
         ("asvspoof5","README.txt",12131),("asvspoof5","LICENSE.txt",19940)] + \
        [("asvspoof5",f"flac_T_a{c}.tar",s) for c,s in zip("abcde",(7522498560,7489587200,7497256960,7509688320,7482521600))]
def aria_done(ctrl):
    # aria2 control file: version, ext, infohash, piece length, total, upload, bitfield
    b = open(ctrl, "rb").read()
    o = 6; n = struct.unpack(">I", b[o:o+4])[0]; o += 4 + n
    piece = struct.unpack(">I", b[o:o+4])[0]; total = struct.unpack(">Q", b[o+4:o+12])[0]; o += 20
    m = struct.unpack(">I", b[o:o+4])[0]; bits = b[o+4:o+4+m]
    return min(total, sum(bin(x).count("1") for x in bits) * piece)
for dest, name, size in FILES:
    final = os.path.join(drive, dest, name); local = os.path.join(dl, name); part = local + ".part"
    if os.path.exists(final) and os.path.getsize(final) == size:
        state = "on Drive (size ok)"
    elif os.path.exists(final + ".part"):
        state = f"copying to Drive {os.path.getsize(final + ".part") / size:6.1%}"
    elif os.path.exists(local):
        state = "local verified, waiting for V1 pack before copy"
    elif os.path.exists(part + ".aria2"):
        d = aria_done(part + ".aria2"); state = f"downloading {d/1e9:6.2f} / {size/1e9:.2f} GB ({d/size:5.1%})"
    elif os.path.exists(part):
        d = os.path.getsize(part); state = f"downloading {d/1e9:6.2f} / {size/1e9:.2f} GB ({d/size:5.1%})"
    else:
        state = "queued"
    print(f"  {dest}/{name:28s} {state}")
PY
grep -E "VERIFIED|FAILED|NO_LOCAL|DONE" /content/v2dl.log 2>/dev/null | tail -6 | sed "s/^/  /"
rx() { awk "/eth0|ens|enp/ {s+=\$2} END {print s}" /proc/net/dev; }
a=$(rx); sleep 10; b=$(rx); echo "  VM network in: $(( (b-a)/10/1000000 )) MB/s (downloads + Drive reads)"
echo "  local disk free: $(df -h --output=avail /content | tail -1)"
'
for s in "$SESSION" "$GPU_SESSION"; do
  $COLAB sessions 2>&1 | grep -q "^\[$s\]" || continue
  for i in 1 2 3 4 5 6; do
    out=$(timeout 150 $COLAB ssh -s "$s" -- "$remote" </dev/null 2>&1)
    echo "$out" | grep -q "HTTP 429" || break
    sleep 15
  done
  echo; echo "$out" | grep -v "HTTP 429"
done
