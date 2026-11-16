# Maya in serveur-ia

Our notes; the upstream `README.md` is left untouched so subtree pulls stay clean. Engine manifest: `engine.yaml`.

## Build (once per version, about 10 min on 6 CPUs)

```sh
modules/llm/engines/maya/scripts/build.sh        # docker, image maya-toolchain:cuda13-nvtx, sm_86
```

Output in `build/` (ignored by git): `maya`, `maya-ds4`, plus the CUDA 13 runtime libraries (`libcublas*.so.13`,
`libcudart.so.13`) that the host does not have; the config templates list `build/` in `lib_dirs`. Env: `CPUS`, `MEMORY`,
`ARCH`. Options are those of `/mnt/donnees/maya/compiler-v1.1.56.sh`. The script downloads the pinned llama.cpp into
`third_party/llama.cpp` (ignored) when missing.

## Python environment

```sh
cd modules/llm/engines/maya
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -e ../../../../engines/python-sdk
```

## Weights layout (`--data-dir`; `--models-dir` is not used)

Read only, never copied (symlinks are fine), except `pack/slots/` which the engine writes:

```
{dataDir}/maya-s24/GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-0000{1,2,3}-of-00003.gguf
{dataDir}/maya-s24/pack/         (dense.bin, index.txt, native_experts.txt, sources.json, tokenizer/, slots/ writable)
{dataDir}/logs/                  (must exist, writable)
```

## Run by hand (what the brain does) and smoke test

```sh
cd modules/llm/engines/maya
CUDA_VISIBLE_DEVICES=0,1 STRATA_GLM_CPU_SHARED=1 .venv/bin/python -m serve.grpc_adapter --socket /tmp/sia-maya.sock \
  --engine-id maya --profile normal --config $PWD/config/maya-s24.json --models-dir <modelsDir> --data-dir <dataDir>
.venv/bin/python ../tests/gpu_smoke.py --socket /tmp/sia-maya.sock --shutdown
```

Game profile: `--config $PWD/config/maya-s24.game.json`. Maya has no vision here (`--vision` of the smoke test is for Strata).

## Measured (2x RTX 3090, driver 615, CUDA 13.0 build, 2026-11-16)

| | normal (ctx 262144, KV streaming 32768) | game (ctx 65536, 6 GiB reserve) |
|---|---|---|
| Start to READY | 126-128 s (files in page cache) | 131 s |
| VRAM per card | 22.8 GB | 18.3 GB |
| Host RAM | 61 GB lost, process maps 65 GB | same |
| Generation | 32-40 tok/s | 27-36 tok/s |

Production reference (`maya/bench`): 22.1 tok/s with `STRATA_GLM_CPU_SHARED=1`.

The templates carry the production flags (`--kv-streaming on --kv-resident 32768`, the expert profile
`data/expert-profile-glm.bin`, `--vram-reserve-mib 1536`). Not tested without them.
