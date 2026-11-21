#!/usr/bin/env python3
"""P4 / listening checks on 16-bit PCM wavs.
usage: compare_wav.py a.wav [b.wav]   one file: duration, RMS, leading/trailing silence;
                                      two files: the same for both, then the SNR of b against a (a = reference).
Silence = |sample| below SIL_DB dBFS. Exit 0 always (numbers only); P4 target: SNR >= 60 dB."""
import sys
import wave

import numpy as np

SIL_DB = -50.0


def load(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: not 16-bit")
        ch, rate = w.getnchannels(), w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    return x.reshape(-1, ch).mean(1), rate


def describe(name, x, rate):
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    loud = np.flatnonzero(np.abs(x) > 10 ** (SIL_DB / 20))
    lead = loud[0] / rate if loud.size else x.size / rate
    trail = (x.size - 1 - loud[-1]) / rate if loud.size else x.size / rate
    db = 20 * np.log10(rms) if rms > 0 else -np.inf
    print(f"{name}: {x.size / rate:.3f} s ({x.size} samples, {rate} Hz), RMS {rms:.4f} ({db:.1f} dBFS), "
          f"peak {np.abs(x).max() if x.size else 0:.3f}, leading silence {lead:.3f} s, trailing silence {trail:.3f} s")


def main(argv):
    if len(argv) not in (1, 2):
        sys.exit(__doc__)
    a, ra = load(argv[0])
    describe(argv[0], a, ra)
    if len(argv) == 1:
        return 0
    b, rb = load(argv[1])
    describe(argv[1], b, rb)
    if ra != rb:
        sys.exit(f"sample rates differ: {ra} vs {rb}")
    n = min(a.size, b.size)
    if a.size != b.size:
        print(f"length differs by {abs(a.size - b.size)} samples, comparing the first {n}")
    err = float(np.sum((a[:n] - b[:n]) ** 2))
    sig = float(np.sum(a[:n] ** 2))
    snr = np.inf if err == 0 else 10 * np.log10(sig / err) if sig > 0 else -np.inf
    print(f"SNR of {argv[1]} against {argv[0]}: {snr:.2f} dB (max abs diff {np.abs(a[:n] - b[:n]).max():.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
