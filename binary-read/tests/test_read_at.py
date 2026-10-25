"""binary-read's 64-bit and BF16 reads, through binary-read/tests/read_at.bend, on the C and the JavaScript lanes.

  (a) F32 and BF16 numbers at offsets just below and above 2^31, 2^32 and 2^33 of a sparse file of 8 GiB + 6 bytes
      (only a few pages are written): a read cut short by the end of the file, a read past it, a hole, a read of 0
      numbers; the file's size; read_f32 with its U32 offset still reads up to the 4 GiB boundary.
  (b) the _into reads: a tensor written from cell AT of a larger array, the cells before and after it left as they were,
      and a read clipped by the array's capacity.
  (c) a BF16 file written by PyTorch (tensor.to(torch.bfloat16), raw bytes) at offset 0 and at an offset that is not a
      multiple of 4: every value read equals bf16.to(torch.float32) bit for bit.

  reference/.venv/bin/python binary-read/tests/test_read_at.py BINARY        (C: BINARY built by bend ... -o)
  reference/.venv/bin/python binary-read/tests/test_read_at.py --js          (JavaScript: bend binary-read/tests/read_at.bend)
"""

import os
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

GIB = 1 << 30
NINE = 0x41100000  # 9.0, the value every probe's cells start with
HERE = Path(__file__).resolve().parent


def runner(cmd):
    def run(*args, ok=True):
        p = subprocess.run([*cmd, *map(str, args)], capture_output=True, text=True)
        if ok:
            assert p.returncode == 0, (args, p.returncode, p.stdout, p.stderr)
        else:
            assert p.returncode != 0, (args, "should have failed", p.stdout)
        return p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    return run


def parse(line):
    """'count: b0 b1 ..' -> (count, [b0, b1, ..])"""
    head, _, rest = line.partition(":")
    return int(head), [int(w) for w in rest.split()]


def js_bits(bits, js):
    # Bend's JavaScript F32 is a Number, and the runtime canonicalizes NaN payloads; C keeps them. Every other bit, -0
    # and subnormals included, must survive on both lanes.
    if not js:
        return bits
    return [0x7FC00000 if (b & 0x7F800000) == 0x7F800000 and b & 0x7FFFFF else b for b in bits]


def hl(off):
    return off >> 32, off & 0xFFFFFFFF


def expect_f32(data, off, n, cells=None, at=0):
    """read_f32_at's (or read_f32_into's) answer: the count of whole numbers and the cells 0 .. len(out)-1"""
    cells = n if cells is None else cells
    room = max(0, (1 << max(0, (cells - 1).bit_length())) - at)
    raw = data(off, 4 * min(n, room))
    whole = len(raw) // 4
    out = [NINE] * cells
    out[at:at + whole] = struct.unpack(f"<{whole}I", raw[:4 * whole])
    part = len(raw) % 4
    if part:  # a last partial number writes only its bytes over the cell
        out[at + whole] = struct.unpack("<I", raw[4 * whole:] + struct.pack("<I", NINE)[part:])[0]
    return whole, out


def expect_bf16(data, off, n, cells=None, at=0):
    cells = n if cells is None else cells
    room = max(0, (1 << max(0, (cells - 1).bit_length())) - at)
    raw = data(off, 2 * min(n, room))
    whole = len(raw) // 2
    out = [NINE] * cells
    out[at:at + whole] = [h << 16 for h in struct.unpack(f"<{whole}H", raw[:2 * whole])]
    return whole, out


def check_sparse(run, scratch, js):
    big = scratch / "big.bin"
    size = 8 * GIB + 6
    rng = random.Random(4)
    special = [0x3F800000, 0xC0200000, 0x80000000, 0x7F800000, 0xFF800000, 0x00000001, 0x007FFFFF, 0x7FC12345,
               0x7F800001, 0x3DCCCCCD]
    pieces = {2 * GIB - 8: 16, 4 * GIB - 8: 16, 4 * GIB + 4097: 12, 4 * GIB + 8192: 40, 8 * GIB - 10: 16}
    written = {}
    with open(big, "wb") as f:
        f.truncate(size)  # sparse: only the pieces below take disk space
        for at, length in pieces.items():
            words = special[: length // 4] if at == 4 * GIB + 8192 else [rng.getrandbits(32) for _ in range(length // 4)]
            raw = struct.pack(f"<{len(words)}I", *words)[:length]
            f.seek(at)
            f.write(raw)
            written[at] = raw

    def data(off, length):  # the file's bytes, from what was written; the rest is the sparse file's zeros
        end = min(off + length, size)
        out = bytearray(max(0, end - off))
        for at, raw in written.items():
            lo, hi = max(at, off), min(at + len(raw), end)
            if lo < hi:
                out[lo - off:hi - off] = raw[lo - at:hi - at]
        return bytes(out)

    assert run("size", big) == f"{size >> 32} {size & 0xFFFFFFFF}", run("size", big)
    probes = [
        ("f32", 2 * GIB - 8, 4), ("f32", 4 * GIB - 8, 4), ("f32", 4 * GIB - 4, 1), ("f32", 4 * GIB, 2),
        ("f32", 4 * GIB + 4097, 3), ("f32", 4 * GIB + 8192, 10), ("f32", 8 * GIB - 8, 4), ("f32", 8 * GIB - 8, 3),
        ("f32", 3 * GIB, 5), ("f32", 16 * GIB, 2), ("f32", 4 * GIB - 8, 0),
        ("bf16", 2 * GIB - 6, 6), ("bf16", 4 * GIB - 6, 6), ("bf16", 4 * GIB + 4097, 6), ("bf16", 4 * GIB + 8192, 20),
        ("bf16", 8 * GIB - 9, 8), ("bf16", 8 * GIB - 10, 8), ("bf16", 3 * GIB, 3), ("bf16", 16 * GIB, 2),
    ]
    for kind, off, n in probes:
        whole, bits = (expect_f32 if kind == "f32" else expect_bf16)(data, off, n)
        got, cells = parse(run(kind, big, *hl(off), n))
        assert (got, cells) == (whole, js_bits(bits, js)), (kind, off, n, got, cells, whole, bits)
    # the _into reads: from cell `at` of an array of 2^d >= at + n + 1 cells; the cells around the read keep 9.0
    into = [("f32into", 4 * GIB - 8, 4, 3), ("f32into", 4 * GIB + 8192, 10, 5), ("f32into", 8 * GIB - 8, 4, 1),
            ("f32into", 4 * GIB - 8, 4, 0), ("bf16into", 4 * GIB + 8192, 20, 7), ("bf16into", 8 * GIB - 9, 8, 2),
            ("bf16into", 2 * GIB - 6, 6, 9)]
    for kind, off, n, at in into:
        f = expect_f32 if kind == "f32into" else expect_bf16
        whole, bits = f(data, off, n, cells=at + n + 1, at=at)
        got, cells = parse(run(kind, big, *hl(off), n, at))
        assert (got, cells) == (whole, js_bits(bits, js)), (kind, off, n, at, got, cells, whole, bits)
    # read_f32 still reads with its U32 offset, up to the 4 GiB boundary, and answers bytes
    got, cells = parse(run("old", big, 4 * GIB - 8, 2))
    assert (got, cells) == (8, js_bits(expect_f32(data, 4 * GIB - 8, 2)[1], js)), (got, cells)
    print(f"binary-read: {len(probes) + len(into) + 2} reads around 2 GiB, 4 GiB and 8 GiB of a sparse {size}-byte file"
          " give the file's bits (short reads, a hole, past the end, into a cell offset, the U32 read_f32, size64)")


def check_capacity(run, scratch, js):
    # a read of more numbers than the array holds stops at its last cell and answers what it wrote
    path = scratch / "small.bin"
    path.write_bytes(struct.pack("<32I", *range(1, 33)))
    raw = path.read_bytes()
    for kind, width, n, d in [("f32cap", 4, 12, 3), ("f32cap", 4, 3, 2), ("bf16cap", 2, 40, 4), ("bf16cap", 2, 5, 3)]:
        room = 1 << d
        k = min(n, room)
        if width == 4:
            vals = list(struct.unpack(f"<{k}I", raw[:4 * k]))
        else:
            vals = [h << 16 for h in struct.unpack(f"<{k}H", raw[:2 * k])]
        got, cells = parse(run(kind, path, 0, 0, n, d))
        assert (got, cells) == (k, js_bits(vals + [NINE] * (room - k), js)), (kind, n, d, got, cells)
    print("binary-read: reads of more numbers than the array holds stop at its capacity (F32 and BF16)")


def check_bf16(run, scratch, js):
    g = torch.Generator().manual_seed(1021)
    x = torch.cat([torch.randn(10007, generator=g) * 10.0 ** torch.randint(-30, 30, (10007,), generator=g),
                   torch.tensor([0.0, -0.0, float("inf"), float("-inf"), 1e-40, -1e-40, 3.0e38, -3.0e38, 1.0, -1.0,
                                 1e-45, 9.2e-41])])
    b = x.to(torch.bfloat16)
    raw = b.view(torch.int16).numpy().tobytes()
    want = (b.to(torch.float32).view(torch.int32).numpy().astype(np.int64) & 0xFFFFFFFF).tolist()
    n = len(want)
    for prefix in [0, 6]:
        path = scratch / f"bf16_{prefix}.bin"
        path.write_bytes(b"\x55" * prefix + raw)
        got, cells = parse(run("bf16", path, 0, prefix, n))
        assert got == n and cells == js_bits(want, js), (prefix, got, [
            (i, c, w) for i, (c, w) in enumerate(zip(cells, want)) if c != w][:5])
    print(f"binary-read: {n} BF16 numbers written by PyTorch, read at offsets 0 and 6, equal bf16.to(torch.float32)"
          " bit for bit")


def main():
    js = "--js" in sys.argv[1:]
    cmd = ["bend", str(HERE / "read_at.bend")] if js else [sys.argv[1]]
    run = runner(cmd)
    with tempfile.TemporaryDirectory(prefix="bend_ml_read_at_") as d:
        scratch = Path(d)
        check_sparse(run, scratch, js)
        check_capacity(run, scratch, js)
        check_bf16(run, scratch, js)
    print("binary-read: PASS" + (" (JavaScript)" if js else " (C)"))


if __name__ == "__main__":
    main()
