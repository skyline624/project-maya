#!/bin/sh
# Compile Maya (targets maya and maya-ds4) for the RTX 3090 (sm_86) in the maya-toolchain:cuda13-nvtx image (CUDA 13.0,
# NVTX headers). Output: <engineDir>/build/{maya,maya-ds4,libcublas*.so.13,libcudart.so.13}.
# Options as /mnt/donnees/maya/compiler-v1.1.56.sh. Env: IMAGE, CPUS (12), MEMORY (48g), ARCH (86), VENV (host cmake).
# Needs docker and python3 on the host and network access (llama.cpp source, once).
set -eu
ENGINE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
IMAGE=${IMAGE:-maya-toolchain:cuda13-nvtx}
CPUS=${CPUS:-12}; MEMORY=${MEMORY:-48g}; ARCH=${ARCH:-86}
# cmake is not in the image: the wheel in Maya's venv does (mounted read-only).
VENV=${VENV:-/mnt/donnees/maya/source/.venv}
CMAKE=${CMAKE:-$VENV/lib/python3.12/site-packages/cmake/data/bin/cmake}
LLAMA=$ENGINE/third_party/llama.cpp

# llama.cpp at the commit setup.py pins; gitignored.
if [ ! -f "$LLAMA/ggml/CMakeLists.txt" ]; then
  python3 - "$ENGINE" <<'PY'
import re, sys, urllib.request, zipfile, shutil, pathlib, tempfile
e = pathlib.Path(sys.argv[1])
commit = re.search(r'^LLAMA_CPP_COMMIT = "(\w+)"', (e / "setup.py").read_text(), re.M).group(1)
tmp = pathlib.Path(tempfile.mkdtemp(dir=e / "third_party"))
z = tmp / "l.zip"
urllib.request.urlretrieve(f"https://github.com/ggml-org/llama.cpp/archive/{commit}.zip", z)
with zipfile.ZipFile(z) as f:
    f.extractall(tmp, [m for m in f.namelist() if "/tools/ui/" not in m])
shutil.move(str(next(p for p in tmp.iterdir() if p.is_dir())), e / "third_party" / "llama.cpp")
shutil.rmtree(tmp)
PY
fi

mkdir -p "$ENGINE/build"
docker run --rm --cpus "$CPUS" --memory "$MEMORY" --pids-limit 4096 --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/build-home -e ARCH="$ARCH" -e CMAKE="$CMAKE" -e ENGINE="$ENGINE" -e JOBS="$CPUS" \
  -v "$ENGINE":"$ENGINE" -v "$VENV":"$VENV":ro -w "$ENGINE" --entrypoint /bin/sh "$IMAGE" -c '
  set -eu; mkdir -p "$HOME"
  "$CMAKE" -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 -DSTRATA_GGML_DIR="$ENGINE/third_party/llama.cpp"
  "$CMAKE" --build build --target maya-ds4 maya -j "$JOBS"
  # CUDA runtime libraries the host does not have (the driver supplies libcuda only).
  for l in libcublas.so.13 libcublasLt.so.13 libcudart.so.13; do cp -L /usr/local/cuda/lib64/$l build/; done
'
echo "built: $ENGINE/build/maya $ENGINE/build/maya-ds4"
