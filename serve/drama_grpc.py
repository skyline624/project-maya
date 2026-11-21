"""serveur-ia: Drama 3 (Fish Audio) on Maya behind the Audio gRPC service (audio.proto: Synthesize, ListVoices) on a Unix socket.

    python serve/drama_grpc.py --socket S --models-dir M --data-dir D [--engine-id ID] [--bin-dir B] [maya-drama --serve options]

Models: {M}/drama-3/{drama3-1014-q8.gguf, mmproj-drama3-1014-bf16.gguf}.  Work files, reference codes, tokenizer and logs: {D}/drama.
Binaries (maya-drama, maya-drama-codec and the libraries beside them): B, by default <engine>/build or <engine>/build-drama.
The cards are the ones CUDA_VISIBLE_DEVICES lists: maya-drama takes both, the codec the second one.  Unknown arguments (--expert-vram-mib N...) go to maya-drama --serve.
The service itself is serve/drama_server.py (the J6 service and its Space guards); the model loads in the background while Health says STARTING.
Needs the SDK in the same environment: `pip install -e engines/python-sdk`, plus requirements.txt, and ffmpeg.
"""
from __future__ import annotations

import argparse
import logging
import re
import secrets
import subprocess
import tempfile
import threading
import wave
from contextlib import contextmanager
from pathlib import Path

import grpc

from serveur_ia_engine.proto.audio.v1 import audio_pb2 as pb
from serveur_ia_engine.proto.audio.v1 import audio_pb2_grpc as rpc
from serveur_ia_engine.server import StateTracker, serve

import drama_server  # next to this file

LOG = logging.getLogger("drama")
VERSION = "0.1.0"
MODEL = "drama-3"
VOICE_PLAIN, VOICE_CLONE = "drama-3", "drama-3-clone"
RATE = 44100
CHUNK_BYTES = 32768       # 0.37 s of PCM
PART_CHARS = 500          # a take ends at 1024 frames (47.6 s, about 600-800 characters): longer texts go in several takes
PART_PAUSE_S = 0.2
MAX_REFERENCE_S = 30      # the codec's encoder truncates to 30 s anyway


def split_text(text, limit=PART_CHARS):
    """Whole sentences grouped in parts of `limit` characters at most (a longer sentence stays alone: the frame cap ends it)."""
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
        if current and len(current) + 1 + len(sentence) > limit:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    return [*parts, current] if current else [text]


def is_codec_wav(path):
    try:
        with wave.open(str(path)) as w:
            return (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
    except (wave.Error, EOFError):
        return False


def ffmpeg(args, data=None):
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], input=data, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise ValueError("cannot decode the reference audio: " + e.stderr.decode(errors="replace").strip()[-200:]) from e


def prepare_reference(source, directory):
    """A mono 44.1 kHz 16-bit wav the codec reads: the file itself when it already is one, else ffmpeg's conversion of the file or of the raw PCM."""
    out = Path(directory) / "reference.wav"
    kind = source.WhichOneof("source")
    if kind == "path":
        if not Path(source.path).is_file():
            raise ValueError(f"reference audio not found: {source.path}")
        if is_codec_wav(source.path) and Path(source.path).stat().st_size < 10_000_000:
            return Path(source.path)
        ffmpeg(["-i", source.path, "-t", str(MAX_REFERENCE_S), "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(out)])
    elif kind == "pcm_f32le":
        channels = source.channels or 1
        if not 1 <= channels <= 32 or not 1 <= source.sample_rate <= 384000 or len(source.pcm_f32le) % (4 * channels):
            raise ValueError("invalid reference PCM (sample rate, channels or length)")
        ffmpeg(["-f", "f32le", "-ar", str(source.sample_rate), "-ac", str(channels), "-i", "pipe:0", "-t", str(MAX_REFERENCE_S), "-ac", "1", "-ar", str(RATE),
                "-c:a", "pcm_s16le", str(out)], source.pcm_f32le)
    else:
        raise ValueError("reference_audio needs a path or PCM")
    return out


@contextmanager
def request_guard(context, engine):
    try:
        yield
    except ValueError as e:  # bad text, voice or reference
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
    except Exception as e:  # noqa: BLE001 - the child processes or the disk failed
        LOG.exception("Drama synthesis failed")
        engine.check_children(str(e))
        context.abort(grpc.StatusCode.INTERNAL, "Drama synthesis failed: " + str(e))


class DramaEngine(rpc.AudioServicer):
    def __init__(self, state):
        self.state = state
        self.svc = None  # drama_server.Service, once loaded

    def check_children(self, why):
        """Health FAILED when one of the two child processes is gone: the brain stops and restarts the engine."""
        if self.svc and (self.svc.engine.p.poll() is not None or self.svc.codec.p.poll() is not None):
            self.state.set_failed(why)

    def close(self):
        if self.svc:
            self.svc.close()

    def ListVoices(self, request, context):
        return pb.ListVoicesReply(voices=[pb.Voice(id=VOICE_PLAIN, language="auto", model=MODEL),
                                          pb.Voice(id=VOICE_CLONE, language="auto", model=MODEL, cloning=True)])

    def Synthesize(self, request, context):
        svc = self.svc
        if svc is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "the engine is still starting")
        with request_guard(context, self), tempfile.TemporaryDirectory(dir=svc.work, prefix="req-") as tmp:
            clone = request.HasField("reference_audio") and request.reference_audio.WhichOneof("source") is not None
            if request.voice not in ("", VOICE_PLAIN, VOICE_CLONE):
                raise ValueError(f"unknown voice '{request.voice}' ({VOICE_PLAIN} or {VOICE_CLONE})")
            if request.voice == VOICE_CLONE and not clone:
                raise ValueError(f"voice {VOICE_CLONE} needs reference_audio and reference_text")
            refs = []
            if clone:
                if not request.reference_text.strip():
                    raise ValueError("reference_text is required with reference_audio")
                refs = [{"audio_path": str(prepare_reference(request.reference_audio, tmp)), "text": request.reference_text}]
            parts, seed, held = split_text(request.text), secrets.randbelow(2**31) + 1, None
            for i, part in enumerate(parts):
                if not context.is_active():  # a take in progress cannot be interrupted: the cancel is seen between takes
                    return
                pcm, info = svc.synthesize(part, refs, seed, svc.a.max_frames, not svc.a.no_guards)
                if info["stop"] != "eos":
                    LOG.warning("take %d/%d stopped on %s: the audio is cut", i + 1, len(parts), info["stop"])
                data = pcm.astype("<i2").tobytes() + (bytes(int(RATE * PART_PAUSE_S) * 2) if i < len(parts) - 1 else b"")
                for at in range(0, len(data), CHUNK_BYTES):
                    if held is not None:
                        yield pb.AudioChunk(pcm_s16le=held, sample_rate=RATE)
                    held = data[at:at + CHUNK_BYTES]
            yield pb.AudioChunk(pcm_s16le=held or b"", sample_rate=RATE, final=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--engine-id", default=None)
    ap.add_argument("--profile", default=None)  # informational: the profile's arguments arrive as arguments
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bin-dir", default=None)
    ap.add_argument("--no-guards", action="store_true")
    a, rest = ap.parse_known_args(argv)

    engine_dir = Path(__file__).resolve().parents[1]
    bins = Path(a.bin_dir) if a.bin_dir else next((p for p in (engine_dir / "build", engine_dir / "build-drama") if (p / "maya-drama").is_file()), engine_dir / "build")
    models, data = Path(a.models_dir) / MODEL, Path(a.data_dir) / "drama"
    args = ["--model", str(models / "drama3-1014-q8.gguf"), "--mmproj", str(models / "mmproj-drama3-1014-bf16.gguf"), "--engine-bin", str(bins / "maya-drama"),
            "--codec-bin", str(bins / "maya-drama-codec"), "--work-dir", str(data)]
    args += ["--no-guards"] * a.no_guards + [f"--engine-arg={x}" for x in rest]
    options = drama_server.make_parser().parse_args(args)

    state = StateTracker()
    engine = DramaEngine(state)

    def load():
        try:
            engine.svc = drama_server.Service(options)
            state.set_models([(MODEL, 0)])
            state.set_ready()
        except Exception as e:  # noqa: BLE001 - reported by Health
            LOG.exception("Drama did not start")
            state.set_failed(str(e))

    def register(server):
        rpc.add_AudioServicer_to_server(engine, server)
        threading.Thread(target=load, name="drama-load", daemon=True).start()

    try:
        serve(a.socket, a.engine_id or "drama", "audio", VERSION, ["synthesize"], register=register, state=state)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
