"""tensor-gpu's random fills against a NumPy twin, word for word.

The fills use only + - * /, sqrt, integer operations and exact conversions, so NumPy's float32, doing the same
operations in the same order, gives the same 32 bits. Compiles tensor-gpu/tests/words.bend (the words of 4,096 cells
from the bangs, then 4,096 cells where the U32 indexes wrap, then two offset slices from fill_gauss_at and
fill_uniform_at, both from an odd index: 4,095 cells from 2^32 - 2047, which wrap, and 777 from 12345) and compares
each with the twin below at the cell's global index, so a slice's pairs are the full fill's.
"""
import os, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
np.seterr(over="ignore")
u32, f = np.uint32, np.float32
M24 = 16777215


def word(x):
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def of_word(w):
    return np.array([w], dtype=np.uint32).view(np.float32)[0]


def mix(z):  # mlbend-generative's mix32
    z = u32(z)
    z ^= z >> u32(16)
    z = u32(z * u32(2246822519))
    z ^= z >> u32(13)
    z = u32(z * u32(3266489917))
    return int(z ^ (z >> u32(16)))


def hash_(s, i):
    return mix((s * 2654435769 + i + 305419898) & 0xFFFFFFFF)


def uniform(s, i):
    return f(f((hash_(s, i) & M24) + 1) / f(16777216.0))


def ln(x):
    w = word(x)
    big = (w & 8388607) > 3474675
    m = of_word((w & 8388607) | (1056964608 if big else 1065353216))
    ex = f(f(w >> 23) - (f(126.0) if big else f(127.0)))
    s = f((m - f(1.0)) / (m + f(1.0)))
    z = f(s * s)
    c3 = f(f(0.14285715) + f(z * f(0.11111111)))
    c2 = f(f(0.2) + f(z * c3))
    p = f(z * f(f(0.33333334) + f(z * c2)))
    t = f(s + s)
    return f(f(ex * f(0.693145751953125)) + f(f(t + f(t * p)) + f(ex * f(0.0000014286067653))))


def sin_(x):
    z = f(x * x)
    c4 = f(f(0.0001984127) - f(z * f(0.0000027557319)))
    c3 = f(f(0.008333334) - f(z * c4))
    c2 = f(f(0.16666667) - f(z * c3))
    return f(x - f(f(x * z) * c2))


def cos_(x):
    z = f(x * x)
    c5 = f(f(0.0000248015873) - f(z * f(0.00000027557319)))
    c4 = f(f(0.0013888889) - f(z * c5))
    c3 = f(f(0.041666668) - f(z * c4))
    c2 = f(f(0.5) - f(z * c3))
    return f(f(1.0) - f(z * c2))


def sincos(v):  # of 2 pi v / 2^24
    q, w = v >> 22, v & 4194303
    hi = w > 2097152
    x = f(f(4194304 - w if hi else w) * f(0.00000037450702))
    a, b = sin_(x), cos_(x)
    sx, cx = (b, a) if hi else (a, b)
    return [sx, cx, -sx, -cx][q], [cx, -sx, -cx, sx][q]


def gauss(s, i):
    i2 = (i & ~1) & 0xFFFFFFFF
    r = np.sqrt(f(f(0.0) - f(f(2.0) * ln(uniform(s, i2)))), dtype=np.float32)
    sn, cs = sincos((hash_(s, (i2 + 1) & 0xFFFFFFFF) + 1) & M24)
    return f(r * cs) if i % 2 == 0 else f(r * sn)


def main():
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "words")
        r = subprocess.run([BEND, os.path.join(ROOT, "tensor-gpu/tests/words.bend"), "-o", exe], capture_output=True, text=True, env=ENV)
        if not os.path.exists(exe):
            print("ERROR could not compile tensor-gpu/tests/words.bend:", r.stdout[-400:], r.stderr[-400:])
            sys.exit(1)
        out = subprocess.run([exe, "+RTS", "--gpu", "off", "-RTS"], capture_output=True, text=True, env=ENV).stdout
    rows = [tuple(map(int, l.split())) for l in out.splitlines() if l.strip()]
    cells = list(range(4096)) + list(range(2**32 - 4096, 2**32))
    cells += [(2**32 - 2047 + k) % 2**32 for k in range(4095)] + [12345 + k for k in range(777)]
    if len(rows) != len(cells):
        print(f"ERROR expected {len(cells)} lines, got {len(rows)}")
        sys.exit(1)
    bad = sum((word(gauss(77, i)) != g) + (word(uniform(77, i)) != u) for i, (g, u) in zip(cells, rows))
    print(f"{len(cells)} cells, {bad} differing words")
    if bad:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
