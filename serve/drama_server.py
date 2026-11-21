#!/usr/bin/env python3
"""HTTP service of Drama 3 on Maya (J6 of docs/plan-drama3-maya.md).

POST /v1/audio/speech  {"input": text, "references": [{"audio_path": wav, "text": transcript}], "seed": n, "max_frames": n, "guards": bool}
                       -> audio/wav, 44.1 kHz mono 16 bit.  Headers: X-Seed, X-Frames, X-Stop, X-Timing (JSON).  The bench of drama-3/comparaison sends exactly this.
GET  /health

Three processes: this one (HTTP, tokenizer, the Space's guards), `maya-drama --serve` (backbone + tower + sampling on the GPUs, see
include/strata/drama/drama_serve.hpp) and `maya-drama-codec serve` (the llama.cpp DAC codec in docker, card 1).  Prompts are tokenized here, with a
tokenizers BPE built from the backbone GGUF's own vocabulary (cached in the work dir); the engine turns the reference codes into embedding rows.
Requests are served one at a time.  Python standard library + numpy + tokenizers (+ gguf, once, to build the tokenizer)."""
import argparse
import hashlib
import io
import json
import os
import queue
import re
import secrets
import signal
import struct
import subprocess
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ENGINE_DIR = Path(__file__).resolve().parents[1]
DRAMA = Path("/mnt/donnees/serveur-ia-dev/drama-3")
WS = " \t\n\r\f\v"
INSTRUCTION = "convert the provided text to speech"
SPEAKER_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")
MAX_SPEAKER = 15
STRUCTURAL = ("<|im_start|>", "<|im_end|>", "<|voice|>")
# llama.cpp's pre-tokenizer for Qwen3.5 (llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_QWEN35)
PRE_REGEX = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

# the Space's guards (plan 9.5).  Calibrated on the narrator takes of parite/clone: silence sits at -76..-87 dBFS (5th-10th percentile of the 20 ms levels), speech
# around -29 dBFS (median), breaths and pauses between -50 and -80; the leading/trailing silence measured is the same at -60, -50, -45 and -40 dBFS.
GUARD_DB = -50.0        # a 20 ms frame under this level is silence
GUARD_HOP_S = 0.02
OPENING_S = 1.5         # speech starting later than this: regenerate once with another seed
EDGE_KEEP_FRAMES = 3    # silence kept before the first and after the last speech frame (60 ms)
MAX_PAUSE_S = 1.0       # an internal silence longer than this is cut to PAUSE_TO_S
PAUSE_TO_S = 0.5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- tokenizer
def build_tokenizer(gguf_path, out):
    """tokenizers BPE from the vocabulary stored in the backbone GGUF (the model's own tokens, merges and control tokens)."""
    import gguf
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

    reader = gguf.GGUFReader(str(gguf_path))

    def strings(key):
        f = reader.fields[key]
        return [str(bytes(f.parts[i]), "utf-8") for i in f.data]

    tokens, merges = strings("tokenizer.ggml.tokens"), strings("tokenizer.ggml.merges")
    f = reader.fields["tokenizer.ggml.token_type"]
    types = [int(f.parts[i][0]) for i in f.data]
    vocab = {t: i for i, t in enumerate(tokens)}
    assert len(vocab) == len(tokens), "duplicate tokens in the GGUF vocabulary"
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[tuple(m.split(" ", 1)) for m in merges]))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PRE_REGEX), behavior="isolated"), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    # llama.cpp splits the text on its control (3), user-defined (4) and unknown (2) tokens before the BPE
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False) for t, ty in zip(tokens, types) if ty in (2, 3, 4)])
    tk.save(str(out))


def load_tokenizer(gguf_path, path):
    from tokenizers import Tokenizer

    if not path.exists():
        t0 = time.monotonic()
        build_tokenizer(gguf_path, path)
        log(f"tokenizer built from {gguf_path} in {time.monotonic() - t0:.1f} s -> {path}")
    tk = Tokenizer.from_file(str(path))
    assert tk.token_to_id("<|im_end|>") == 248046 and tk.token_to_id("<|semantic:0|>") == 248099, "this is not Drama 3's vocabulary"
    return tk


def check_text(text, where):
    if not text.strip():
        raise ValueError(f"{where}: empty text")
    for s in STRUCTURAL:
        if s in text:
            raise ValueError(f"{where}: {s} is a structural token")
    for m in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG.match(text, m.start())
        if tag is None or int(tag.group(1)) > MAX_SPEAKER:
            raise ValueError(f"{where}: speaker tags are <|speaker:N|> with N in 0..{MAX_SPEAKER}")


def tagged(text):
    """The text, led by <|speaker:0|> unless a tag already leads it (SGLang's with_speaker_tag)."""
    return text if SPEAKER_TAG.match(text.lstrip(WS)) else "<|speaker:0|> " + text


class Prompter:
    def __init__(self, tk):
        self.tk = tk

    def enc(self, s):
        return self.tk.encode(s, add_special_tokens=False).ids

    def build(self, text, ref_text=None):
        """(token ids, index of the reference frames in them or -1, rows of the voice-only head): each segment is tokenized on its own, as llama.cpp's
        tokenize_segments and SGLang's build_prompt_rows do (reference-bracket-v1)."""
        check_text(text, "input")
        tail = self.enc("<|im_end|>\n") + self.enc("<|im_start|>user\n") + self.enc(tagged(text)) + self.enc("<|im_end|>\n") + self.enc("<|im_start|>assistant\n<|voice|>")
        if ref_text is None:
            head = self.enc("<|im_start|>system\n") + self.enc(INSTRUCTION)
            return head + tail, -1, len(head)
        check_text(ref_text, "references.0.text")
        ref_text = ref_text.lstrip(WS) if SPEAKER_TAG.match(ref_text.lstrip(WS)) else "<|speaker:0|> " + ref_text
        head = self.enc("<|im_start|>system\n") + self.enc(INSTRUCTION + " reference to the following:") + self.enc(f"\n\n[Reference 1]\nText: {ref_text}\nSpeech: ")
        return head + tail, len(head), len(head)


# ---------------------------------------------------------------- the two child processes
class Child:
    """A child process answering one line per request; a thread sorts its stdout into answers and log lines."""

    def __init__(self, cmd, env, logfile, name, is_answer):
        self.name = name
        self.q = queue.Queue()
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=logfile, text=True, bufsize=1, env={**os.environ, **env})
        threading.Thread(target=self._read, args=(is_answer, logfile), daemon=True).start()

    def _read(self, is_answer, logfile):
        for line in self.p.stdout:
            if is_answer(line):
                self.q.put(line.strip())
            else:
                logfile.write(f"[{self.name}] {line}")
                logfile.flush()
        self.q.put(None)

    def send(self, line, timeout):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        return self.get(timeout)

    def get(self, timeout):
        try:
            a = self.q.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"{self.name}: no answer in {timeout} s") from None
        if a is None:
            raise RuntimeError(f"{self.name} exited")
        return a

    def close(self, quit_line):
        try:
            self.p.stdin.write(quit_line + "\n")
            self.p.stdin.flush()
            self.p.stdin.close()
            self.p.wait(timeout=20)
        except Exception:  # noqa: BLE001 - going down anyway
            self.p.kill()


class Engine(Child):
    def __init__(self, cmd, logfile):
        super().__init__(cmd, {}, logfile, "engine", lambda l: l.startswith("{"))
        ready = json.loads(self.get(900))
        assert ready.get("event") == "ready", ready
        log("engine ready", ready)
        self.next_id = 0

    def gen(self, **req):
        self.next_id += 1
        a = json.loads(self.send(json.dumps({"op": "gen", "id": self.next_id, **req}), 900))
        if not a.get("ok"):
            raise RuntimeError("engine: " + a.get("error", "?"))
        return a

    def close(self, _=None):
        super().close('{"op":"quit"}')


class Codec(Child):
    def __init__(self, cmd, env, logfile):
        super().__init__(cmd, env, logfile, "codec", lambda l: l.startswith(("READY", "OK ", "ERR ")))
        assert self.get(900) == "READY"
        log("codec ready")

    def call(self, line):
        a = self.send(line, 300)
        if not a.startswith("OK "):
            raise RuntimeError("codec: " + a)
        return float(a.split()[1])

    def close(self, _=None):
        super().close("quit")


# ---------------------------------------------------------------- guards
def frame_db(x):
    n = int(44100 * GUARD_HOP_S)
    m = len(x) // n
    f = x[: m * n].astype(np.float64).reshape(m, n) / 32768.0
    return 20 * np.log10(np.sqrt((f * f).mean(axis=1)) + 1e-9)


def lead_silence_s(x):
    loud = np.flatnonzero(frame_db(x) > GUARD_DB)
    return loud[0] * GUARD_HOP_S if loud.size else len(x) / 44100


def tidy(x):
    """Trims the silence at both ends (keeping EDGE_KEEP_FRAMES frames) and cuts every internal silence longer than MAX_PAUSE_S to PAUSE_TO_S."""
    n, rate = int(44100 * GUARD_HOP_S), 44100
    loud = frame_db(x) > GUARD_DB
    idx = np.flatnonzero(loud)
    if idx.size == 0:
        return x
    first, last = idx[0], idx[-1]
    start, end = max(0, first - EDGE_KEEP_FRAMES) * n, min(len(x), (last + 1 + EDGE_KEEP_FRAMES) * n)
    keep, at = [], start
    run = None
    for i in range(first, last + 2):
        quiet = i <= last and not loud[i]
        if quiet and run is None:
            run = i
        elif not quiet and run is not None:
            if (i - run) * GUARD_HOP_S > MAX_PAUSE_S:
                keep.append(x[at : run * n + int(PAUSE_TO_S / 2 * rate)])
                at = i * n - int(PAUSE_TO_S / 2 * rate)
            run = None
    keep.append(x[at:end])
    return np.concatenate(keep)


def wav_bytes(x, rate=44100):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.astype("<i2").tobytes())
    return b.getvalue()


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 44100, "the codec gives mono 16 bit 44.1 kHz"
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()


# ---------------------------------------------------------------- service
class Service:
    def __init__(self, a):
        self.a = a
        self.work = Path(a.work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.engine_log = open(self.work / "engine.log", "a")
        self.codec_log = open(self.work / "codec.log", "a")
        self.requests = open(self.work / "requests.jsonl", "a")
        self.prompter = Prompter(load_tokenizer(a.model, Path(a.tokenizer) if a.tokenizer else self.work / "tokenizer.json"))
        self.lock = threading.Lock()
        self.refs = {}  # sha256 of the wav -> (codes path, frames)
        self.n = 0
        codec_cmd = a.codec_cmd.split() if a.codec_cmd else ["sh", str(ENGINE_DIR / "tools/drama/ref/run.sh"), "serve", "--mmproj", a.mmproj, "--gpu"]
        self.engine = Engine([a.engine_bin, "--serve", "--model", a.model, "--mmproj", a.mmproj, "--devs", a.devs, "--ctx", str(a.ctx), "--voices", str(a.voices),
                              "--sampling", a.sampling, *a.engine_arg], self.engine_log)
        self.codec = Codec(codec_cmd, {"TOOL": "maya-drama-codec", "GPU": "1", "DEV": "1", "DOCKER_I": "1"} if not a.codec_cmd else {}, self.codec_log)

    def close(self):
        self.engine.close()
        self.codec.close()

    def reference(self, ref):
        """(codes path, frames, voice id) of a reference: its wav goes through the codec's encoder once per content (cached by sha256)."""
        data = Path(ref["audio_path"]).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha not in self.refs:
            wav, codes = self.work / f"ref-{sha[:16]}.wav", self.work / f"ref-{sha[:16]}.codes"
            if not codes.exists():
                wav.write_bytes(data)  # the codec's container sees the work dir only
                self.codec.call(f"encode {wav} {codes}")
                wav.unlink()
            self.refs[sha] = (codes, struct.unpack("<I", codes.read_bytes()[4:8])[0])
        return (*self.refs[sha], sha)

    def synthesize(self, text, refs, seed, max_frames, guards):
        if len(refs) > 1:
            raise ValueError("one reference at most")
        t = {}
        t0 = time.monotonic()
        with self.lock:
            t["wait_ms"] = (time.monotonic() - t0) * 1e3
            t1 = time.monotonic()
            req = {}
            if refs:
                if not isinstance(refs[0], dict) or not isinstance(refs[0].get("audio_path"), str) or not isinstance(refs[0].get("text"), str):
                    raise ValueError("references.0 needs audio_path and text")
                codes, frames, sha = self.reference(refs[0])
                tokens, ref_at, head = self.prompter.build(text, refs[0]["text"])
                req.update(ref_at=ref_at, ref_codes=str(codes), prefix_rows=head + frames, voice=hashlib.sha256(f"{sha}|{tokens[:head]}".encode()).hexdigest()[:24])
            else:
                tokens, _, head = self.prompter.build(text)
                req.update(prefix_rows=head, voice="no-reference")
            t["prepare_ms"] = (time.monotonic() - t1) * 1e3
            self.n += 1
            tag = f"{os.getpid()}-{self.n}"
            out, wav = self.work / f"g-{tag}.codes", self.work / f"g-{tag}.wav"
            try:
                for attempt in range(2 if guards else 1):
                    g = self.engine.gen(tokens=tokens, seed=seed, max_frames=max_frames, out=str(out), **req)
                    t["codec_ms"] = self.codec.call(f"decode {out} {wav}") * 1e3
                    pcm = read_wav(wav)
                    t1 = time.monotonic()
                    lead = lead_silence_s(pcm) if guards else 0
                    t["guards_ms"] = t.get("guards_ms", 0) + (time.monotonic() - t1) * 1e3
                    t["attempts"] = attempt + 1
                    if guards and attempt == 0 and lead >= OPENING_S:
                        log(f"silent opening ({lead:.1f} s), seed {seed} -> {seed + 1}")
                        seed += 1
                        continue
                    break
            finally:
                for p in (out, wav):
                    p.unlink(missing_ok=True)
            t1 = time.monotonic()
            raw_s = len(pcm) / 44100
            if guards:
                pcm = tidy(pcm)
            t["guards_ms"] += (time.monotonic() - t1) * 1e3
        t["total_ms"] = (time.monotonic() - t0) * 1e3
        info = {"seed": seed, "frames": g["frames"], "stop": g["stop"], "prefix": g["prefix"], "rows": g["rows"], "chars": len(text), "clone": bool(refs), "audio_s": len(pcm) / 44100,
                "raw_audio_s": raw_s, "engine": {k: g[k] for k in ("prefix_ms", "prefill_ms", "gen_ms", "backbone_ms", "tower_ms")}, **t}
        self.requests.write(json.dumps(info) + "\n")
        self.requests.flush()
        log(f"{'clone' if refs else 'plain'} {len(text)} chars -> {g['frames']} frames {info['audio_s']:.1f} s ({g['stop']}), prefix {g['prefix']}, total {t['total_ms'] / 1e3:.1f} s")
        return pcm, info


def handler(svc):
    class H(BaseHTTPRequestHandler):
        def reply(self, code, body, ctype="application/json", headers=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def fail(self, code, msg):
            self.reply(code, json.dumps({"error": msg}).encode())

        def do_GET(self):
            self.reply(200, b'{"status":"ok"}') if self.path == "/health" else self.fail(404, "not found")

        def do_POST(self):
            if self.path != "/v1/audio/speech":
                return self.fail(404, "not found")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                text = body["input"]
                if not isinstance(text, str):
                    raise ValueError("input must be a string")
                refs = body.get("references") or []
                seed = int(body["seed"]) if body.get("seed") is not None else secrets.randbelow(2**31) + 1
                max_frames = int(body.get("max_frames") or svc.a.max_frames)
                guards = bool(body.get("guards", not svc.a.no_guards))
                if not 1 <= max_frames <= svc.a.ctx or seed < 1:
                    raise ValueError("max_frames must be in 1..ctx and seed >= 1")
            except (ValueError, KeyError, TypeError) as e:
                return self.fail(400, f"bad request: {e}")
            try:
                pcm, info = svc.synthesize(text, refs, seed, max_frames, guards)
            except (ValueError, OSError) as e:  # bad text or reference file
                return self.fail(400, str(e))
            except Exception as e:  # noqa: BLE001 - the engine or the codec failed
                log("ERROR", repr(e))
                return self.fail(500, str(e))
            self.reply(200, wav_bytes(pcm), "audio/wav", [("X-Seed", str(info["seed"])), ("X-Frames", str(info["frames"])), ("X-Stop", info["stop"]), ("X-Timing", json.dumps(info))])

        def log_message(self, *a):
            pass

    return H


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=str(DRAMA / "gguf/drama3-q8.gguf"))
    p.add_argument("--mmproj", default=str(DRAMA / "gguf/mmproj-drama3-bf16.gguf"))
    p.add_argument("--engine-bin", default=str(ENGINE_DIR / "build-drama/maya-drama"))
    p.add_argument("--codec-cmd", help="command of `maya-drama-codec serve` (default: tools/drama/ref/run.sh in docker, GPU 1)")
    p.add_argument("--work-dir", default=str(DRAMA / "parite/serve-work"), help="temporary files, reference codes, tokenizer, logs; must lie under drama-3/parite for the codec's container")
    p.add_argument("--tokenizer", help="tokenizer.json (default: built from the GGUF into the work dir)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--devs", default="0,1")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--voices", type=int, default=4, help="voices whose prefix the engine keeps (LRU)")
    p.add_argument("--max-frames", type=int, default=1024, help="frames of a take at most (1024 = 47.6 s, SGLang's cap)")
    p.add_argument("--sampling", default="official", choices=["official", "llama"])
    p.add_argument("--no-guards", action="store_true", help="the Space's guards off by default (a request can ask with \"guards\")")
    p.add_argument("--engine-arg", action="append", default=[], help="extra argument of maya-drama --serve")
    a = p.parse_args()
    svc = Service(a)
    httpd = ThreadingHTTPServer((a.host, a.port), handler(svc))
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=httpd.shutdown).start())
    log(f"listening on http://{a.host}:{a.port}/v1/audio/speech (guards {'off' if a.no_guards else 'on'}, sampling {a.sampling})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        svc.close()


if __name__ == "__main__":
    main()
