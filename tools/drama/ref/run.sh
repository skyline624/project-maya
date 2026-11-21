#!/bin/sh
# Runs build-ref/drama_ref in docker (the llama.cpp libs link the CUDA backend, whose libraries the host lacks).
# Usage: ref/run.sh <args...>   drama_ref (make-rows | run | tower | generate); TOOL=maya-drama-codec ref/run.sh decode|encode ... Env: GPU=1 for --gpu runs (DEV=n: only card n), DOCKER_I=1 to keep stdin open (maya-drama-codec serve) (run maya-gpu-guard.sh first, FREE only).
# CPU runs see no GPU (CUDA_VISIBLE_DEVICES=-1). Paths in args must lie under drama-3/gguf (ro) or drama-3/parite (rw).
set -eu
ENGINE=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd -P)
D=/mnt/donnees/serveur-ia-dev/drama-3
L=${LLAMA:-$D/llama.cpp}
IMAGE=${IMAGE:-maya-drama-toolchain:cuda13}
if [ "${GPU:-0}" = 1 ]; then VIS=${DEV:+"-e CUDA_VISIBLE_DEVICES=$DEV"}; else VIS="-e CUDA_VISIBLE_DEVICES=-1"; fi   # DEV=1: only card 1
mkdir -p "$D/parite"
docker run --rm ${DOCKER_I:+-i} --gpus all $VIS -e LD_LIBRARY_PATH=$L/build/bin --cpus "${CPUS:-12}" --memory "${MEMORY:-80g}" --user "$(id -u):$(id -g)" \
  -v "$ENGINE/build-ref":/ref:ro -v "$L":"$L":ro -v "$D/gguf":"$D/gguf":ro -v "$D/parite":"$D/parite" \
  --entrypoint /ref/${TOOL:-drama_ref} "$IMAGE" "$@"
