#!/bin/sh
# J6 final bench, ON the server: waits for the two cards to be FREE, starts serve/drama_server.py, runs drama-3/diag.py (the first part of each of the 8 sections,
# 3 attempts) with cloning (-> comparaison/maya-clone) and without (-> comparaison/maya-sans-clone), prints the tally and the RTF, stops the server.
# usage: bench_serve.sh [drama_server.py options]      env: ENGINE (the fork dir, default: the one this script sits in), SKIP=clone|plain
set -eu
ENGINE=${ENGINE:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)}
D=/mnt/donnees/serveur-ia-dev/drama-3; C=$D/comparaison; PY=$D/conv/bin/python; W=$D/parite/serve-work
mkdir -p "$W"
start() { # another job can take a card between the guard's FREE and the engine's allocation: the server then dies at once, and we wait again
  until sh "$D/maya-gpu-guard.sh" | grep -q FREE; do sleep 15; done
  "$PY" "$ENGINE/serve/drama_server.py" --engine-bin "$ENGINE/build-drama/maya-drama" "$@" > "$W/server.log" 2>&1 &
  SERVER=$!
  until curl -fs http://127.0.0.1:8000/health >/dev/null 2>&1; do
    kill -0 $SERVER 2>/dev/null || { tail -3 "$W/server.log"; return 1; }
    sleep 5
  done
}
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
until start "$@"; do echo "the server did not start, trying again"; sleep 20; done
echo "server up"
: > "$W/requests.jsonl"
rtf() { # sum of the generation seconds over the sum of the audio seconds of the diag log
  "$PY" -c "
import re, sys
g = a = 0
for l in open(sys.argv[1]):
    m = re.search(r', ([\d.]+) s, [\d.]+ chars/s, gen ([\d.]+) s', l)
    if m: a += float(m[1]); g += float(m[2])
print(f'RTF {g / a:.3f} ({g:.0f} s of generation for {a:.0f} s of audio)' if a else 'no take')" "$1"
}
if [ "${SKIP:-}" != clone ]; then
  DIAG_LABEL=../maya-clone DIAG_CLONE=1 "$PY" "$D/diag.py" | tee "$C/maya-clone.log"; rtf "$C/maya-clone.log"
  cp "$W/requests.jsonl" "$C/maya-clone/requests.jsonl"; : > "$W/requests.jsonl"
fi
if [ "${SKIP:-}" != plain ]; then
  DIAG_LABEL=../maya-sans-clone "$PY" "$D/diag.py" | tee "$C/maya-sans-clone.log"; rtf "$C/maya-sans-clone.log"
  cp "$W/requests.jsonl" "$C/maya-sans-clone/requests.jsonl"
fi
