#!/usr/bin/env python3
"""J4: one line per cloned part: frames, duration, chars/s, RMS, leading silence, and ms per frame / RTF from the maya-drama log.
usage: clone_stats.py <dir> <log> name1 name2 ...   (<dir>/<name>.wav, <dir>/<name>.txt the spoken text; the log's `INFO text i: N frames` lines in the same order)
Bench rule (docs/plan-drama3-maya.md 11): 12-25 chars/s and a take under 47.5 s."""
import re
import sys
import wave
import numpy as np

SILENCE = 0.01  # leading silence: time before the first sample above 1 % of full scale


def main():
    d, log, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    per = [(int(m[1]), float(m[2]), float(m[3])) for m in re.finditer(r"text \d+: (\d+) frames .*?total ([\d.]+) ms; RTF ([\d.]+)", open(log, encoding="utf-8").read())]
    if len(per) != len(names):
        sys.exit(f"{len(per)} generations in the log, {len(names)} names")
    print(f"{'part':<10} {'frames':>6} {'dur s':>6} {'chars':>5} {'ch/s':>5} {'rms':>6} {'lead ms':>7} {'ms/fr':>6} {'RTF':>5}  plausible")
    for name, (frames, ms, rtf) in zip(names, per):
        with wave.open(f"{d}/{name}.wav") as w:
            rate = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
        text = open(f"{d}/{name}.txt", encoding="utf-8").read()
        dur = len(x) / rate
        loud = np.flatnonzero(np.abs(x) > SILENCE)
        lead = 1000.0 * (loud[0] / rate if loud.size else dur)
        cps = len(text) / dur
        ok = 12 <= cps <= 25 and dur < 47.5
        print(f"{name:<10} {frames:>6} {dur:>6.2f} {len(text):>5} {cps:>5.1f} {np.sqrt((x * x).mean()):>6.3f} {lead:>7.0f} {ms:>6.1f} {rtf:>5.2f}  {'yes' if ok else 'NO'}")


main()
