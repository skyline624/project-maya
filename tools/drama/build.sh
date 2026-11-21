#!/bin/sh
# Compile maya-drama (Drama 3 on Maya, docs/plan-drama3-maya.md) for the RTX 3090 (sm_86) in docker. Run ON the server
# from anywhere: it builds the fork it sits in. Output: <engineDir>/build-drama/maya-drama (+ the CUDA libraries it links).
# Options: env IMAGE, BASE (the image the toolchain is derived from), CPUS (12), MEMORY (48g), ARCH (86), WERROR (OFF).
# The nvidia/cuda devel image has no cmake: a small derived image (cmake, ninja, git, python3) is built once.
set -eu
ENGINE=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
BASE=${BASE:-nvidia/cuda:13.0.0-devel-ubuntu24.04}
IMAGE=${IMAGE:-maya-drama-toolchain:cuda13}
CPUS=${CPUS:-12}; MEMORY=${MEMORY:-48g}; ARCH=${ARCH:-86}; WERROR=${WERROR:-OFF}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" - <<DOCKERFILE
FROM $BASE
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake ninja-build git python3 \
    && rm -rf /var/lib/apt/lists/*
DOCKERFILE
fi

# STRATA_NATIVE_EXPERTS=OFF: maya-drama links no ggml (the default targets' CPU experts), so llama.cpp is not fetched
docker run --rm --cpus "$CPUS" --memory "$MEMORY" --pids-limit 4096 --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/build-home -e ARCH="$ARCH" -e WERROR="$WERROR" -e JOBS="$CPUS" \
  -v "$ENGINE":"$ENGINE" -w "$ENGINE" --entrypoint /bin/sh "$IMAGE" -c '
  set -eu; mkdir -p "$HOME"
  cmake -G Ninja -S . -B build-drama -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
    -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_WERROR=$WERROR -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda-13.0
  cmake --build build-drama --target maya-drama -j "$JOBS"
  # the CUDA runtime libraries it links, which the host does not have (the driver supplies libcuda only)
  for l in $(ldd build-drama/maya-drama | sed -n "s/^\s*\(libcu[a-z]*\.so\.13\).*/\1/p"); do cp -L /usr/local/cuda/lib64/$l build-drama/; done
'
echo "built: $ENGINE/build-drama/maya-drama"
