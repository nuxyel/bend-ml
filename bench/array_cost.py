"""The per-element cost of Array operations on one thread, Bend against C (NOTES.md, 2026-11-01).

bench/array_cost.bend runs one operation REPS times over an array of 2^DEPTH numbers; bench/array_cost.c is its C
twin (the same values, the same order of F32 additions). Each (op, depth) is built with REPS = 0 and REPS = N, and a
round runs every binary once in a shuffled order, pinned to CPU 3 with taskset, under perf stat
(cpu_core/instructions/u). The cost of one element is (x(N) - x(0)) / (N * 2^DEPTH) within a round, for the time and
the instruction count; the medians and the spread over the rounds are printed by `report`.

  python3 bench/array_cost.py build <outdir> [TAPATH]      TAPATH relative to bench/ (default ../tensor-array/main.bend)
  python3 bench/array_cost.py run <rounds> <results.tsv> <label>=<outdir> ...   C runs from the first outdir only
  python3 bench/array_cost.py report <results.tsv>

Two versions of tensor-array: lay the old one out with `git show <rev>:tensor-array/main.bend > ../_base/main.bend`
(a path without dots), build each into its own outdir, and pass both to one run so their rounds interleave.
"""
import os, random, statistics, struct, subprocess, sys, time
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
DEPTHS = [12, 16, 20, 24]
# op: (show, elements touched per run of REPS = N, at least 2 reps)
OPS = {"get": ("U32.show", 1 << 30), "fold": ("F32.show", 1 << 29), "set": ("U32.show", 1 << 30),
       "new": ("F32.show", 1 << 31), "newrt": ("F32.show", 1 << 30), "tolist": ("F32.show", 1 << 26), "round": ("F32.show", 1 << 26),
       "btolist": ("F32.show", 1 << 25)}


def reps(op, d):
    return max(2, OPS[op][1] >> d)


def jobs(labels=("bend",)):
    for op in OPS:
        for d in DEPTHS:
            for lang in tuple(labels) + ("c",):
                for r in (0, reps(op, d)):
                    yield (op, d, lang, r)


def exe(out, op, d, lang, r):
    return os.path.join(out, f"bend_{op}_d{d}_r{r}") if lang != "c" else os.path.join(out, "array_cost_c")


def build(out, ta="../tensor-array/main.bend"):
    os.makedirs(out, exist_ok=True)
    subprocess.run(["clang", "-std=c11", "-O3", os.path.join(WT, "bench", "array_cost.c"), "-o",
                    os.path.join(out, "array_cost_c")], check=True)
    t = open(os.path.join(WT, "bench", "array_cost.bend")).read()
    for (op, d, lang, r) in jobs():
        if lang == "c":
            continue
        s = (t.replace("TAPATH", ta).replace("OPNAME", "op_" + op).replace("SHOW", OPS[op][0])
             .replace("REPS", f"{r}n").replace("DEPTH", f"{d}n"))
        p = os.path.join(WT, "bench", "_tmp_ac.bend")
        open(p, "w").write(s)
        res = subprocess.run([BEND, p, "-o", exe(out, op, d, lang, r)], capture_output=True, text=True, env=ENV,
                             cwd=os.path.join(WT, "bench"))
        os.remove(p)
        if res.returncode:
            raise SystemExit(res.stdout + res.stderr)


def one(out, op, d, lang, r):
    cmd = ["perf", "stat", "-x,", "-e", "cpu_core/instructions/u", "--", "taskset", "-c", "3", exe(out, op, d, lang, r)]
    cmd += ["+RTS", "--threads", "1", "-RTS"] if lang != "c" else [op, str(d), str(r)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    ins = int(p.stderr.strip().splitlines()[-1].split(",")[0])
    return dt, ins, p.stdout.strip()


def run(rounds, res, *dirs):
    outs = dict(x.split("=", 1) for x in dirs)
    first = next(iter(outs.values()))
    js = list(jobs(outs))
    with open(res, "a") as fo:
        for rd in range(int(rounds)):
            random.shuffle(js)
            for j in js:
                dt, ins, o = one(outs.get(j[2], first), *j)
                fo.write("\t".join(map(str, (rd,) + j + (f"{dt:.6f}", ins, o))) + "\n")
            fo.flush()
            print("round", rd, "done", time.strftime("%H:%M:%S"), "load", open("/proc/loadavg").read().split()[0],
                  flush=True)


def report(res):
    rows = [l.rstrip("\n").split("\t") for l in open(res) if l.strip()]
    by = {}
    outs = {}
    for rd, op, d, lang, r, dt, ins, o in rows:
        by[(rd, op, int(d), lang, int(r))] = (float(dt), int(ins))
        if int(r):  # F32.show and %.9g print the same float32 differently: compare the bits
            v = o if op in ("get", "set") else struct.pack("<f", float(o)).hex()
            outs.setdefault((op, int(d)), {}).setdefault(lang, set()).add(v)
    rds = sorted({k[0] for k in by})
    print(f"rounds: {len(rds)}; ns and instructions per element, median [min-max] over the rounds")
    labels = sorted({r[3] for r in rows} - {"c"})
    print("label\top\tdepth\tbend ns\tc ns\tratio\tbend ins\tc ins\tratio\tsame output")
    for op, d, lab in [(op, d, lab) for op in OPS for d in DEPTHS for lab in labels]:
            n = reps(op, d)
            line = [lab, op, str(d)]
            cell = {}
            for lang in (lab, "c"):
                ts, is_ = [], []
                for rd in rds:
                    a, b = by.get((rd, op, d, lang, n)), by.get((rd, op, d, lang, 0))
                    if a and b:
                        ts.append((a[0] - b[0]) * 1e9 / (n << d))
                        is_.append((a[1] - b[1]) / (n << d))
                cell[lang] = (ts, is_)
            if not cell[lab][0]:
                continue
            f = lambda xs: f"{statistics.median(xs):.3f} [{min(xs):.3f}-{max(xs):.3f}]"
            tb, tc = statistics.median(cell[lab][0]), statistics.median(cell["c"][0])
            ib, ic = statistics.median(cell[lab][1]), statistics.median(cell["c"][1])
            o = outs.get((op, d), {})
            same = "yes" if len(o.get(lab, ())) == 1 and o.get(lab) == o.get("c") else f"NO {o}"
            print("\t".join(line + [f(cell[lab][0]), f(cell["c"][0]), f"{tb / tc:.2f}x", f"{ib:.2f}", f"{ic:.2f}",
                                    f"{ib / ic:.2f}x", same]))


if __name__ == "__main__":
    {"build": build, "run": run, "report": report}[sys.argv[1]](*sys.argv[2:])
