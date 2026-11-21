#!/bin/sh
# Builds the reference tools against the read-only llama.cpp checkout, in docker. Run ON the server.
#   drama_ref        P1/P2/P3 dumps (rows, backbone, tower, a real generation)
#   maya-drama-codec P4 codec: decode codes -> wav, encode a reference wav -> codes
# libcuda.so.1 is absent from the image (the driver supplies it at run time): its symbols are left unresolved at link time.
# Output: <engineDir>/build-ref/. Run the tools with ref/run.sh.
set -eu
ENGINE=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd -P)
L=${LLAMA:-/mnt/donnees/serveur-ia-dev/drama-3/llama.cpp}
IMAGE=${IMAGE:-maya-drama-toolchain:cuda13}
mkdir -p "$ENGINE/build-ref"
docker run --rm --user "$(id -u):$(id -g)" -v "$ENGINE":"$ENGINE" -v "$L":"$L":ro -w "$ENGINE" --entrypoint /bin/sh "$IMAGE" -c "
  for t in ref/drama_ref:drama_ref codec/maya_drama_codec:maya-drama-codec; do
    g++ -std=c++17 -O2 -Wall -I$L/include -I$L/ggml/include -I$L/tools/mtmd tools/drama/\${t%%:*}.cpp -o build-ref/\${t##*:} \
        -L$L/build/bin -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -Wl,--allow-shlib-undefined
  done"
echo "built: $ENGINE/build-ref/drama_ref $ENGINE/build-ref/maya-drama-codec"
