#!/usr/bin/env python3
"""P1 parity: compare two .out files (DRO1: h + logits [eos, sem0..4095] per row). Exit 0 on PASS.
usage: compare_logits.py ref.out test.out [--rows A:B]"""
import sys
import numpy as np

# 3x llama.cpp's own CPU-vs-GPU noise on f2/f3 (docs/plan-drama3-maya.md §10); the prompt rows' top-1 (>= 95 %) is
# checked with --rows 0:42
TOP1_MIN, COS_MIN, KL_MAX, HCOS_MIN = 0.85, 0.9958, 8.1e-3, 0.981
IDS = np.concatenate(([248046], np.arange(248099, 248099 + 4096)))


def load(path):
    with open(path, "rb") as f:
        if f.read(4) != b"DRO1":
            sys.exit(f"{path}: bad magic")
        n, d, k = (int(x) for x in np.frombuffer(f.read(12), dtype="<u4"))
        a = np.frombuffer(f.read(), dtype="<f4")
    if a.size != n * (d + k):
        sys.exit(f"{path}: size mismatch")
    a = a.reshape(n, d + k)
    return a[:, :d].astype(np.float64), a[:, d:].astype(np.float64)


def cos(a, b):
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)


def log_softmax(x):
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def main(argv):
    rows = None
    if "--rows" in argv:
        i = argv.index("--rows")
        lo, hi = argv[i + 1].split(":")
        rows = slice(int(lo or 0), int(hi) if hi else None)
        del argv[i:i + 2]
    if len(argv) != 2:
        sys.exit(__doc__)
    h_r, l_r = load(argv[0])
    h_t, l_t = load(argv[1])
    if l_r.shape != l_t.shape or h_r.shape != h_t.shape:
        sys.exit(f"shape mismatch: {l_r.shape} vs {l_t.shape}")
    if rows is not None:
        h_r, l_r, h_t, l_t = h_r[rows], l_r[rows], h_t[rows], l_t[rows]
    if not (np.isfinite(l_t).all() and np.isfinite(h_t).all()):
        print("FAIL: non-finite values in test output")
        return 1
    a_r, a_t = l_r.argmax(1), l_t.argmax(1)
    top1 = (a_r == a_t).mean()
    c_l, c_h = cos(l_r, l_t), cos(h_r, h_t)
    lp_r, lp_t = log_softmax(l_r), log_softmax(l_t)
    kl = (np.exp(lp_r) * (lp_r - lp_t)).sum(1)
    print(f"rows compared: {len(kl)}")
    print(f"top-1 agreement: {100 * top1:.2f} %")
    print(f"logits cosine: min {c_l.min():.6f} mean {c_l.mean():.6f}")
    print(f"KL(ref||test): mean {kl.mean():.3e} max {kl.max():.3e}")
    print(f"h cosine: min {c_h.min():.6f} mean {c_h.mean():.6f}")
    base = (rows.start or 0) if rows is not None else 0
    print("5 worst rows by KL (row, KL, ref top-1 id, test top-1 id):")
    for r in np.argsort(-kl)[:5]:
        print(f"  {base + r}  {kl[r]:.3e}  {IDS[a_r[r]]}  {IDS[a_t[r]]}")
    ok = top1 >= TOP1_MIN and c_l.min() >= COS_MIN and kl.mean() <= KL_MAX and c_h.min() >= HCOS_MIN
    print("PASS" if ok else "FAIL", f"(top-1 >= {100 * TOP1_MIN:.0f} %, logits cosine min >= {COS_MIN}, mean KL <= {KL_MAX:g}, h cosine min >= {HCOS_MIN})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
