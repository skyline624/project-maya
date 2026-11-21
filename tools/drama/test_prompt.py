#!/usr/bin/env python3
"""J6: the prompts serve/drama_server.py builds (tokenizer from the GGUF vocabulary) against the rows llama.cpp's pipeline gave drama_ref make-rows.
usage: test_prompt.py <backbone.gguf> <dir> <tokenizer.json> [--plain rows_file text_file]...
<dir> holds ref.txt (the reference transcript) and, for each cloned part i, p<i>.rows (make-rows --ref-wav) and t<i>.txt (the spoken text).
--plain a.rows a.txt: a make-rows --prompt file of the no-reference prompt (one segment), its text in a.txt.
Token rows must be identical (an embedding row stands for a reference frame: one -1 each)."""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "serve"))
import drama_server as ds  # noqa: E402


def read_rows(path):
    b = Path(path).read_bytes()
    assert b[:4] == b"DRR1"
    n, e = struct.unpack("<II", b[4:12])
    at, rows = 12, []
    for _ in range(n):
        (t,) = struct.unpack("<i", b[at : at + 4])
        at += 4
        rows.append(t)
        if t == -1:
            at += 4 * e
    assert at == len(b)
    return rows


def main():
    gguf, d, tkpath, rest = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
    pr = ds.Prompter(ds.load_tokenizer(gguf, tkpath))
    ref = (d / "ref.txt").read_text(encoding="utf-8")
    bad = 0
    for i in range(8):
        text = (d / f"t{i}.txt").read_text(encoding="utf-8")
        want = read_rows(d / f"p{i}.rows")
        tokens, ref_at, head = pr.build(text, ref)
        n_frames = want.count(-1)
        got = tokens[:ref_at] + [-1] * n_frames + tokens[ref_at:]
        ok = got == want
        bad += not ok
        print(f"clone p{i}: {len(want)} rows ({n_frames} frame rows) {'IDENTICAL' if ok else 'DIFFERENT'}")
        if not ok:
            k = next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            print("  first difference at row", k, got[k - 2 : k + 3], want[k - 2 : k + 3])
    while rest and rest[0] == "--plain":
        want = read_rows(rest[1])
        tokens, _, _ = pr.build(Path(rest[2]).read_text(encoding="utf-8"))
        ok = tokens == want
        bad += not ok
        print(f"plain {rest[1]}: {len(want)} rows {'IDENTICAL' if ok else 'DIFFERENT'}")
        rest = rest[3:]
    sys.exit(1 if bad else 0)


main()
