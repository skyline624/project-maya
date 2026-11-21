#!/usr/bin/env python3
"""P2 tower parity: compare two .tower files (DRT1: tower_out of each layer for each backbone row). Numbers only, no pass/fail yet.
usage: compare_tower.py ref.tower test.tower [--rows A:B]
Per layer: cosine of tower_out over the rows (min / mean / row of the min), plus the relative L2 error (mean).
llama.cpp prunes the last layer's output on rows whose next row is a frame row (teacher forced): those are NaN in the .tower
and are skipped (the "rows" column gives the number of rows compared for the layer)."""
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        if f.read(4) != b"DRT1":
            sys.exit(f"{path}: bad magic")
        n, l, d = (int(x) for x in np.frombuffer(f.read(12), dtype="<u4"))
        a = np.frombuffer(f.read(), dtype="<f4")
    if a.size != n * l * d:
        sys.exit(f"{path}: size mismatch")
    return a.reshape(n, l, d).astype(np.float64)


def main(argv):
    rows = None
    if "--rows" in argv:
        i = argv.index("--rows")
        lo, hi = argv[i + 1].split(":")
        rows = slice(int(lo or 0), int(hi) if hi else None)
        del argv[i:i + 2]
    if len(argv) != 2:
        sys.exit(__doc__)
    r, t = load(argv[0]), load(argv[1])
    if r.shape != t.shape:
        sys.exit(f"shape mismatch: {r.shape} vs {t.shape}")
    base = 0
    if rows is not None:
        base = rows.start or 0
        r, t = r[rows], t[rows]
    if not np.isfinite(t[np.isfinite(r).all(2)]).all():
        print("WARNING: non-finite values in the test file where the reference has data")
    print(f"rows compared: {r.shape[0]}, layers: {r.shape[1]}, width: {r.shape[2]}")
    print("layer  rows  cos min    cos mean   min at row  rel L2 mean")
    for il in range(r.shape[1]):
        ok = np.isfinite(r[:, il]).all(1) & np.isfinite(t[:, il]).all(1)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            print(f"{il:5d}  no common row")
            continue
        a, b = r[idx, il], t[idx, il]
        c = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)
        rel = np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(a, axis=1), 1e-30)
        print(f"{il:5d}  {idx.size:4d}  {c.min():.6f}  {c.mean():.6f}  {base + int(idx[c.argmin()]):10d}  {rel.mean():.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
