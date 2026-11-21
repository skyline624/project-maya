#!/bin/sh
# Builds maya-drama-codec (maya_drama_codec.cpp) against the read-only llama.cpp build and makes <engineDir>/build-drama/ (where tools/drama/build.sh puts
# maya-drama) self-contained for it: the libraries of llama.cpp's build and the CUDA ones they need are copied next to the binaries.
# The codec then runs natively, without docker, wherever the NVIDIA driver (libcuda) is, with LD_LIBRARY_PATH set to that folder (llama.cpp's libraries carry a RUNPATH
# of their own, /src/build/bin, so they find each other only through it; serve/drama_server.py sets the variable). Run ON the server, after tools/drama/build.sh.
# Env: LLAMA (llama.cpp checkout with its build/, default /mnt/donnees/serveur-ia-dev/drama-3/llama.cpp), IMAGE.
set -eu
ENGINE=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd -P)
L=${LLAMA:-/mnt/donnees/serveur-ia-dev/drama-3/llama.cpp}
IMAGE=${IMAGE:-maya-drama-toolchain:cuda13}
mkdir -p "$ENGINE/build-drama"
docker run --rm -i --user "$(id -u):$(id -g)" -e L="$L" -v "$ENGINE":"$ENGINE" -v "$L":"$L":ro -w "$ENGINE" --entrypoint /bin/sh "$IMAGE" -s <<'INNER'
set -eu
O=build-drama
g++ -std=c++17 -O2 -Wall -I$L/include -I$L/ggml/include -I$L/tools/mtmd tools/drama/codec/maya_drama_codec.cpp -o $O/maya-drama-codec \
    -L$L/build/bin -lmtmd -lllama -lggml -lggml-base -Wl,--allow-shlib-undefined
for l in libmtmd libllama libggml libggml-base libggml-cpu libggml-cuda; do cp -L $L/build/bin/$l.so.1 $O/; done
# the CUDA runtime, cuBLAS and NCCL that libggml-cuda links (the host has none of them; the driver supplies libcuda)
for p in $(LD_LIBRARY_PATH=$L/build/bin ldd $O/libggml-cuda.so.1 | sed -n 's/.*=> \(\/[^ ]*\/\(libcudart\|libcublas\|libcublasLt\|libnccl\)\.so\.[0-9]*\) .*/\1/p'); do cp -L $p $O/; done
# nothing may resolve outside the folder but the system libraries and the driver
if LD_LIBRARY_PATH=$PWD/$O ldd $O/maya-drama-codec | grep -v "libcuda.so.1\|=> $PWD/$O/" | grep -E 'not found|=> /(mnt|usr/local)'; then echo "maya-drama-codec: libraries resolved outside $O" >&2; exit 1; fi
INNER
echo "built: $ENGINE/build-drama/maya-drama-codec (libraries beside it)"
