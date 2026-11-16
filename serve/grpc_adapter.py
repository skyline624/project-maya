"""serveur-ia: serve Maya's `Service` over gRPC (engine.proto + llm.proto) on a Unix socket.

    python -m serve.grpc_adapter --socket S --engine-id ID [--profile P] --config maya.json [service args]

The unknown arguments go to `serve.server.main` (for example `--engine mock --script "hi"` for tests).
The service's own HTTP server stays on 127.0.0.1, on a free port, for debugging only.
Needs the SDK in the same environment: `pip install -e engines/python-sdk`.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as `python serve/grpc_adapter.py` too

from serveur_ia_engine import (EngineNotReady, LlmBridge, StateTracker, free_port, openai_request,  # noqa: E402
                               poll_vram, render_config, run_hosted, with_prompt_tokens)
from serveur_ia_engine.llm_bridge import vram_from_telemetry  # noqa: E402
from serveur_ia_engine.proto.llm.v1 import llm_pb2  # noqa: E402

from serve import server  # noqa: E402
from serve.frontend import openai_to_messages  # noqa: E402


class MayaBridge(LlmBridge):
    svc: server.Service | None = None

    def _service(self) -> server.Service:
        if self.svc is None:
            raise EngineNotReady("the engine is still starting")
        return self.svc

    def _chat(self, svc, req):
        return openai_to_messages(req, deepseek=getattr(svc.template, "deepseek", False))

    def events(self, request, cancel):
        svc = self._service()
        req = svc.with_shared(openai_request(request), "openai")   # mirrors Handler._openai
        req["_api"] = "openai"
        svc.check_request(req)
        messages, tools, kw = self._chat(svc, req)
        if not messages:
            raise ValueError("messages is empty")
        try:
            ids, thinking, max_new = svc.prepare(messages, tools, kw, int(req.get("max_tokens") or 0),
                                                 cancel=cancel, guard=svc._loop_guard_on(req))
        except RuntimeError as e:   # the engine is starting, restarting or gone
            raise EngineNotReady(str(e)) from e
        if kw.get("_prefill"):   # a required tool call starts the answer
            req = {**req, "_prefill": kw["_prefill"]}
        req = {**req, "_blocked": kw.get("_blocked"), "_think_resume": kw.get("_think_resume")}
        return with_prompt_tokens(svc.run(ids, thinking, tools, max_new, req, cancel), len(ids))

    def count(self, request):
        svc = self._service()
        return int(svc.count_tokens(openai_request(request))["tokens"])

    def models(self):
        svc = self.svc
        if svc is None:
            return []
        return [llm_pb2.ModelInfo(id=svc.model, context_length=int(svc.engine.max_context or 0),
                                  vision=svc.vision is not None, tools=True, reasoning=True)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--engine-id", default=None)
    ap.add_argument("--profile", default=None)   # informational: the profile's args are already in --config
    ap.add_argument("--config", help="JSON config template; {engineDir} {modelsDir} {dataDir} are expanded")
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--data-dir", default=None)
    a, rest = ap.parse_known_args(argv)

    state, bridge = StateTracker(), MayaBridge()

    def ready(svc) -> None:
        bridge.svc = svc
        vram = vram_from_telemetry(svc)
        state.set_gpus(vram)
        state.set_models([(svc.model, sum(b for _, b in vram))])
        state.set_ready()
        poll_vram(svc, state)

    server.on_service_ready = ready
    args = ["--host", "127.0.0.1", "--port", str(free_port())]
    if "--engine" not in rest:
        args += ["--engine", "maya"]
    if a.config:
        data = a.data_dir or tempfile.mkdtemp(prefix="serveur-ia-")
        values = {"engineDir": str(Path(__file__).resolve().parents[1]), "modelsDir": a.models_dir, "dataDir": data}
        rendered = Path(data) / "config" / f"{a.engine_id or "maya"}.json"
        args += ["--config", render_config(a.config, str(rendered), values, os.environ.get("CUDA_VISIBLE_DEVICES"))]
    return run_hosted(lambda: server.main(args + rest), socket_path=a.socket, engine_id=a.engine_id,
                      version=server.MAYA_VERSION or "unknown", bridge=bridge, state=state)


if __name__ == "__main__":
    sys.exit(main())
