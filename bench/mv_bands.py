"""Time per matrix · vector product: Mat.matmul_nt (bench/mv.bend) against Bands with the typed API
(bench/mv_bands.bend) and with the list API (bench/mv_bands_l.bend).

For each shape and setting, builds the benchmark with REPS = 1 and REPS = 1 + N and reports
(t(1 + N) - t(1)) / N, so building the matrix is not counted. Median of 3 runs per binary.

  reference/.venv/bin/python bench/mv_bands.py [scale of the number of products]
"""
import os, statistics, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
# (rows, columns, products): the four GPT-2 small products per layer and the logits
SHAPES = [(2304, 768, 200), (768, 768, 400), (3072, 768, 200), (768, 3072, 200), (50257, 768, 20)]
THREADS = [1, 4, 8, 16]


def build(src, subs, out):
    text = open(os.path.join(ROOT, "bench", src)).read()
    for k, v in subs.items():
        text = text.replace(k, v)
    path = os.path.join(ROOT, "bench", "_tmp_" + src)
    open(path, "w").write(text)
    try:
        r = subprocess.run([BEND, path, "-o", out], capture_output=True, text=True, env=ENV)
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
    finally:
        os.remove(path)


def run(exe, threads):
    ts, out = [], ""
    for _ in range(3):
        t0 = time.perf_counter()
        r = subprocess.run([exe, "--threads", str(threads)], capture_output=True, text=True)
        ts.append(time.perf_counter() - t0)
        out = r.stdout.strip()
    return statistics.median(ts), out


def per_product(src, subs, d, N):
    a, b = os.path.join(d, "one"), os.path.join(d, "many")
    build(src, dict(subs, REPS="0n"), a)
    build(src, dict(subs, REPS=f"{N}n"), b)
    rows = []
    for t in THREADS:
        t1, _ = run(a, t)
        t2, out = run(b, t)
        rows.append((t, (t2 - t1) / N * 1000, out))
    return rows


def main():
    with tempfile.TemporaryDirectory() as d:
        for (m, k, n) in SHAPES:
            N = max(1, int(n * SCALE))
            print(f"== W {m} x {k}, {N} products")
            settings = [("mv.bend", {"KN": f"{k}n", "MN": f"{m}n", "PAR": "0n"}, "matmul_nt, sequential")]
            settings += [("mv.bend", {"KN": f"{k}n", "MN": f"{m}n", "PAR": "3n"}, "matmul_nt, par = 3 (clones W)")]
            settings += [("mv_bands.bend", {"KN": f"{k}n", "MN": f"{m}n", "DEPTH": f"{dep}n"}, f"Bands, 2^{dep} bands") for dep in range(0, 7)]
            settings += [("mv_bands_l.bend", {"KN": f"{k}n", "MN": f"{m}n", "DEPTH": f"{dep}n"}, f"Bands list API, 2^{dep} bands") for dep in (0, 4, 5)]
            for src, subs, label in settings:
                rows = per_product(src, subs, d, N)
                cells = "  ".join(f"{t:2d}t {ms:7.3f} ms" for t, ms, _ in rows)
                print(f"{label:32s} {cells}   (output {rows[-1][2][:14]})", flush=True)


if __name__ == "__main__":
    main()
