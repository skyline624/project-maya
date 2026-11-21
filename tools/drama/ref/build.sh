#!/bin/sh
# Builds the P1 reference tool (drama_ref) against the read-only llama.cpp checkout, in docker. Run ON the server.
# libcuda.so.1 is absent from the image (the driver supplies it at run time): its symbols are left unresolved at link time.
# Output: <engineDir>/build-ref/drama_ref. Run it with ref/run.sh.
set -eu
ENGINE=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd -P)
L=${LLAMA:-/mnt/donnees/serveur-ia-dev/drama-3/llama.cpp}
IMAGE=${IMAGE:-maya-drama-toolchain:cuda13}
mkdir -p "$ENGINE/build-ref"
docker run --rm --user "$(id -u):$(id -g)" -v "$ENGINE":"$ENGINE" -v "$L":"$L":ro -w "$ENGINE" --entrypoint /bin/sh "$IMAGE" -c "
  g++ -std=c++17 -O2 -Wall -I$L/include -I$L/ggml/include tools/drama/ref/drama_ref.cpp -o build-ref/drama_ref \
      -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -Wl,--allow-shlib-undefined"
echo "built: $ENGINE/build-ref/drama_ref"
