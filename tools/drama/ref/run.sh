#!/bin/sh
# Runs build-ref/drama_ref in docker (the llama.cpp libs link the CUDA backend, whose libraries the host lacks).
# Usage: ref/run.sh <drama_ref args...>   (make-rows | run). Env: GPU=1 for --gpu runs (run maya-gpu-guard.sh first, FREE only).
# CPU runs see no GPU (CUDA_VISIBLE_DEVICES=-1). Paths in args must lie under drama-3/gguf (ro) or drama-3/parite (rw).
set -eu
ENGINE=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd -P)
D=/mnt/donnees/serveur-ia-dev/drama-3
L=${LLAMA:-$D/llama.cpp}
IMAGE=${IMAGE:-maya-drama-toolchain:cuda13}
if [ "${GPU:-0}" = 1 ]; then VIS=; else VIS="-e CUDA_VISIBLE_DEVICES=-1"; fi
mkdir -p "$D/parite"
docker run --rm --gpus all $VIS -e LD_LIBRARY_PATH=$L/build/bin --cpus "${CPUS:-12}" --memory "${MEMORY:-80g}" --user "$(id -u):$(id -g)" \
  -v "$ENGINE/build-ref":/ref:ro -v "$L":"$L":ro -v "$D/gguf":"$D/gguf":ro -v "$D/parite":"$D/parite" \
  --entrypoint /ref/drama_ref "$IMAGE" "$@"
