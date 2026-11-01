"""Time softmax cross-entropy: tensor-fork's TF.softmax_ce (rows in 2^par blocks) against tensor-array's
TA.Mat.softmax_ce (par 0), on 1 thread (pinned to CPU 3) and 16 threads (bench/ce_fork.bend).

Builds each variant, checks that every variant gives the same loss and the same sum of gradient words, then runs
ROUNDS rounds; each round runs every (variant, threads) pair once in a shuffled order. Prints medians and ranges.
The fill (the logits made, MODE 0) is timed too, since every variant pays it. Use an idle machine.

  reference/.venv/bin/python bench/ce_fork.py [rounds] [rows,cols ...]
"""
import os, random, statistics, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 9
SHAPES = [tuple(int(x) for x in s.split(",")) for s in sys.argv[2:]] or [(256, 49152), (1024, 49152)]
PARS = [0, 4, 6]


def build(mode, n, c, par, out):
    lines = open(os.path.join(ROOT, "bench", "ce_fork.bend")).read().split("\n")
    for i, line in enumerate(lines):
        if line.startswith("  IO.print(go("):
            for k, v in (("MODE", mode), ("NR", n), ("NC", c), ("PAR", par)):
                line = line.replace(k, f"{v}n")
            lines[i] = line
    text = "\n".join(lines)
    path = os.path.join(ROOT, "bench", f"_tmp_ce_{mode}_{par}.bend")
    open(path, "w").write(text)
    try:
        r = subprocess.run([BEND, path, "-o", out], capture_output=True, text=True, env=ENV)
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
    finally:
        os.remove(path)


def cmd(exe, threads):
    run = [exe, "+RTS", "--threads", str(threads), "-RTS"]
    return ["taskset", "-c", "3", *run] if threads == 1 else run


def out(exe, threads):
    return subprocess.run(cmd(exe, threads), capture_output=True, text=True).stdout.strip()


with tempfile.TemporaryDirectory() as tmp:
    for n, c in SHAPES:
        names = {"fill": (0, 0)} | {f"par{p}": (1, p) for p in PARS}
        exes = {}
        for name, (mode, par) in names.items():
            exes[name] = os.path.join(tmp, f"{name}_{n}")
            build(mode, n, c, par, exes[name])
        sums = set()
        for p in PARS:
            build(2, n, c, p, os.path.join(tmp, "chk"))
            sums |= {out(os.path.join(tmp, "chk"), t) for t in (1, 16)}
        assert len(sums) == 1, sums
        print(f"{n} rows of {c}: every variant on 1 and 16 threads gives {sums.pop()}")
        pairs = [(name, t) for name in exes for t in (1, 16)]
        ts = {pair: [] for pair in pairs}
        for _ in range(ROUNDS):
            random.shuffle(pairs)
            for name, t in pairs:
                t0 = time.perf_counter()
                subprocess.run(cmd(exes[name], t), capture_output=True)
                ts[(name, t)].append(time.perf_counter() - t0)
        base = statistics.median(ts[("par0", 1)])
        for (name, t), v in sorted(ts.items()):
            m = statistics.median(v)
            print(f"  {name:5s} threads={t:2d}  median {m * 1000:7.1f} ms  [{min(v) * 1000:.1f}, {max(v) * 1000:.1f}]  par0@1 / this {base / m:.2f}x")
