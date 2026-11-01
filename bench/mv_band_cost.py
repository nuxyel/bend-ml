"""What one band of Bands.matvec_l costs, for two or more versions of tensor-array (NOTES.md, 2026-11-01).

Per product, with the matrix already built: each setting is built with REPS = 0 and REPS = N, and a round runs every
binary once in a shuffled order; the time per product is (t(N) - t(0)) / N within a round, and the instruction count
(perf, cpu_core/instructions/u) the same way. Settings: the bare gemm with x and y reused (bench/mv_band_cost.bend,
MODE 0), Bands.matvec_l with one band (MODE 1), both pinned to one thread with taskset -c 3, and Bands.matvec_l with
2^4 bands on 16 threads (bench/mv_bands_l.bend).

  python3 bench/mv_band_cost.py build <outdir> base=../_base/main.bend new=../tensor-array/main.bend
  python3 bench/mv_band_cost.py run <outdir> <rounds> <results.tsv> base=... new=...
  python3 bench/mv_band_cost.py bits <TAPATH>       sha256 of every output of bench/mv_bits.bend

Paths are relative to bench/. An old version can be laid out with `git show <rev>:tensor-array/main.bend`.
"""
import os, random, statistics, subprocess, sys, time, json
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
VERS = {}
SHAPES = [(50257, 768, 40), (768, 768, 400), (2304, 768, 200), (3072, 768, 200), (768, 3072, 200)]
# config: (label, source, threads, substitutions)
CONFIGS = [("kernel-1t", "mv_band_cost.bend", 1, {"MODE": "0n"}),
           ("band1-1t", "mv_band_cost.bend", 1, {"MODE": "1n"}),
           ("bands16-16t", "mv_bands_l.bend", 16, {"DEPTH": "4n"})]

def src(name, ta, subs):
    t = open(os.path.join(WT, "bench", name)).read()
    t = t.replace("import TAPATH as TA", f"import {ta} as TA")
    t = t.replace("import bend-ml-tensor-array@0.1.18.0/main.bend as TA", f"import {ta} as TA")
    for k, v in subs.items():
        t = t.replace(k, v)
    return t

def jobs():
    for v, ta in VERS.items():
        for (m, k, n) in SHAPES:
            for (lab, f, th, subs) in CONFIGS:
                for reps in (0, n):
                    yield (v, ta, m, k, n, lab, f, th, dict(subs, MN=f"{m}n", KN=f"{k}n", REPS=f"{reps}n"), reps)

def name(v, m, k, lab, reps):
    return f"{v}_{m}x{k}_{lab}_r{reps}"

def build(out):
    os.makedirs(out, exist_ok=True)
    for (v, ta, m, k, n, lab, f, th, subs, reps) in jobs():
        p = os.path.join(WT, "bench", "_tmp_h.bend")
        open(p, "w").write(src(f, ta, subs))
        r = subprocess.run([BEND, p, "-o", os.path.join(out, name(v, m, k, lab, reps))], capture_output=True, text=True, env=ENV, cwd=os.path.join(WT, "bench"))
        os.remove(p)
        if r.returncode:
            raise SystemExit(r.stdout + r.stderr)

def one(exe, th):
    cmd = ["perf", "stat", "-x,", "-e", "cpu_core/instructions/u", "--"]
    cmd += (["taskset", "-c", "3"] if th == 1 else []) + [exe, "+RTS", "--threads", str(th), "-RTS"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    ins = int(r.stderr.strip().splitlines()[-1].split(",")[0])
    return dt, ins, r.stdout.strip()

def run(out, rounds, res):
    js = list(jobs())
    with open(res, "a") as fo:
        for rd in range(rounds):
            random.shuffle(js)
            for (v, ta, m, k, n, lab, f, th, subs, reps) in js:
                dt, ins, o = one(os.path.join(out, name(v, m, k, lab, reps)), th)
                fo.write(f"{rd}\t{v}\t{m}x{k}\t{lab}\t{reps}\t{dt:.6f}\t{ins}\t{o}\n")
            fo.flush()
            print("round", rd, "done", time.strftime("%H:%M:%S"), "load", open("/proc/loadavg").read().split()[0], flush=True)

BITS = [(1, 1), (1, 3), (5, 4), (7, 5), (33, 771), (4097, 771), (2304, 768), (768, 3072), (50257, 768)]


def bits(ta):
    import hashlib, tempfile
    with tempfile.TemporaryDirectory() as d:
        for m, k in BITS:
            p = os.path.join(WT, "bench", "_tmp_bits.bend")
            t = open(os.path.join(WT, "bench", "mv_bits.bend")).read().replace("TAPATH", ta)
            t = t.replace("MN", f"{m}n").replace("KN", f"{k}n").replace("BANDS", "Nil{}" if m > 5000 else "bands(7)")
            open(p, "w").write(t)
            r = subprocess.run([BEND, p, "-o", os.path.join(d, "b")], capture_output=True, text=True, env=ENV, cwd=os.path.join(WT, "bench"))
            os.remove(p)
            if r.returncode:
                raise SystemExit(r.stdout + r.stderr)
            o = subprocess.run([os.path.join(d, "b"), "+RTS", "--threads", "16", "-RTS"], capture_output=True).stdout
            print(f"{m}x{k} {hashlib.sha256(o).hexdigest()[:16]} {len(o)} bytes", flush=True)


if __name__ == "__main__":
    for a in sys.argv[2:]:
        if "=" in a:
            k, v = a.split("=", 1)
            VERS[k] = v
    if sys.argv[1] == "build":
        build(sys.argv[2])
    elif sys.argv[1] == "bits":
        bits(sys.argv[2])
    else:
        run(sys.argv[2], int(sys.argv[3]), sys.argv[4])
