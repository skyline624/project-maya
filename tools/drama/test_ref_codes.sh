#!/bin/sh
# Non-regression: a reference clip must come out of the codec's GPU encode with valid codes only (c0 0..4095, c1..c9 0..1023).
# clone1 (drama-3/comparaison/ab/clone1-reference.wav) got -1 for every codebook of its frame 1 from the CUDA encoder, which made the tower refuse the prompt
# ("forced codes need code0 >= 0"); maya-drama-codec now falls back to the CPU encoder for such a clip.
# Run ON the server: test_ref_codes.sh [ref.wav ...] (default: clone1 and the narratrice reference).  Needs the GPU free (maya-gpu-guard.sh).
set -eu
D=/mnt/donnees/serveur-ia-dev/drama-3; E=$D/maya-fork; W=$D/parite/ref-test
sh $D/maya-gpu-guard.sh | grep -q FREE || { echo "GPU BUSY"; exit 1; }
mkdir -p "$W"
[ $# -gt 0 ] || set -- $D/comparaison/ab/clone1-reference.wav $D/comparaison/reference-narratrice.wav
bad=0
for wav in "$@"; do
  cp "$wav" "$W/in.wav"
  GPU=1 DEV=1 TOOL=maya-drama-codec sh $E/tools/drama/ref/run.sh encode --mmproj $D/gguf/mmproj-drama3-bf16.gguf --wav "$W/in.wav" --out "$W/out.codes" --gpu 2>&1 | grep -i "negative\|frames"
  "$D/conv/bin/python" -I - "$W/out.codes" "$wav" <<'PYEOF' || bad=1
import struct, sys
b = open(sys.argv[1], "rb").read()
n, cb = struct.unpack("<II", b[4:12])
a = struct.unpack("<%di" % (n * cb), b[12:])
ok = all(0 <= a[i * cb] < 4096 and all(0 <= c < 1024 for c in a[i * cb + 1:(i + 1) * cb]) for i in range(n))
print(("PASS" if ok else "FAIL"), sys.argv[2].rsplit("/", 1)[-1], n, "frames")
sys.exit(0 if ok else 1)
PYEOF
done
sh $D/maya-gpu-guard.sh
exit $bad
