# Maya in serveur-ia

Our notes; the upstream `README.md` is left untouched so subtree pulls stay clean. Engine manifest: `engine.yaml`.

## Build (once per version, about 10 min on 6 CPUs)

```sh
modules/llm/engines/maya/scripts/build.sh        # docker, image maya-toolchain:cuda13-nvtx, sm_86
```

Output in `build/` (ignored by git): `maya`, `maya-ds4`, `maya-vision` (the image encoder: llama.cpp mtmd plus GLM-5.3-Flash's
vision tower, added by `tools/vision/glm5next_patch.py`, which the script runs on `third_party/llama.cpp`), plus the CUDA 13 runtime libraries (`libcublas*.so.13`,
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
{modelsDir}/glm-5.3-flash/vision/mmproj-GLM-5.3-Flash-F16.gguf    (images: the vision tower, 1.13 GB)
{modelsDir}/glm-5.3-flash/vision/GLM-5.3-Flash-vocab.gguf         (images: the tokenizer the encoder reads its markers with)
```

## Images (vision)

The config's `"vision"` entry (`maya-s24.json`, `maya-s24.game.json`) makes Maya's service accept OpenAI `image_url` parts and
Anthropic image blocks, which the adapter gets from the brain as `ContentPart.image` (bytes + MIME, `llm_bridge._message`).
The service encodes each new picture with `build/maya-vision` (llama.cpp mtmd + the model's mmproj) and gives the embeddings
to the engine. `ListModels` then says `vision: true` (`svc.vision is not None`), which is what the brain's `/v1/models` shows
while the engine is READY. The encoder is on-demand: it starts when a request carries a new picture, in GPU 0 memory that the
engine lends from its expert cache (`STRATA_GLM_VISION_LEND_MB`, measured once per GPU and cached in
`{dataDir}/config/vision-memory.json`), and stops afterwards. `max_tokens` (1024) caps the image tokens of a picture.

The two vision files are ordinary manifest model files (same repo and pinned revision as the GGUFs, folder `vision/`), so the brain
downloads and checks them. The mmproj is derived from the official checkpoint, but Maya's author publishes it, and it is
byte-identical to what `tools/vision/glm5next_mmproj.py` makes from the last shard of `zai-org/GLM-5.3-Flash`
(`model-00062-of-00062.safetensors`, 1.26 GB, revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`; sha256 3627575d...fde882 both ways):

```sh
# only to reproduce it (needs numpy); the brain does not need this
python3 tools/vision/glm5next_mmproj.py --ckpt <dir with config.json, processor_config.json, model.safetensors.index.json and the last shard>   --out mmproj-GLM-5.3-Flash-F16.gguf --gguf-py third_party/llama.cpp/gguf-py
```

### Enable images on a Maya that is already installed (same version 1.1.56, so no new artifact install: it would answer 409)

1. `scripts/build.sh` (or take `build/maya-vision` and the libraries from the artifact built by `deploy/engines/build-artifact-maya.sh`) and copy
   `build/maya-vision` to `<engines>/maya/1.1.56/build/`.
2. Copy the new `config/maya-s24.json`, `config/maya-s24.game.json` and `engine.yaml` to `<engines>/maya/1.1.56/`.
3. Put the two vision files in a folder the brain can read (in prod: `/mnt/donnees/maya/models/Maya-S24/vision/`, seen as `/weights/maya/vision/`).
4. Restart the brain (it reads the manifests at start; the model then lacks two files) and import the model again with all five files:
   `POST /api/models/import {"engineId":"maya","modelId":"glm-5.3-flash","fileMap":{<the 3 GGUF paths as before>,"vision/mmproj-GLM-5.3-Flash-F16.gguf":"/weights/maya/vision/mmproj-GLM-5.3-Flash-F16.gguf","vision/GLM-5.3-Flash-vocab.gguf":"/weights/maya/vision/GLM-5.3-Flash-vocab.gguf"}}`
   (it hashes all the files again; downloading instead would fetch the GGUFs again).
5. Stop the idle Maya instance (`POST /api/scheduler/instances/{id}/stop`); the next request starts it with the encoder.

## Run by hand (what the brain does) and smoke test

```sh
cd modules/llm/engines/maya
CUDA_VISIBLE_DEVICES=0,1 STRATA_GLM_CPU_SHARED=1 .venv/bin/python -m serve.grpc_adapter --socket /tmp/sia-maya.sock \
  --engine-id maya --profile normal --config $PWD/config/maya-s24.json --models-dir <modelsDir> --data-dir <dataDir>
.venv/bin/python ../tests/gpu_smoke.py --socket /tmp/sia-maya.sock --shutdown
```

Game profile: `--config $PWD/config/maya-s24.game.json`. Add `--vision` to the smoke test to send a picture (the encoder starts on it).

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
