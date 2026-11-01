"""The GPT-2 demo's weight loading, at 1 and 16 threads, for loader variants (NOTES.md, 2026-11-01).

Variants, each a copy of demos/gpt2/fast.bend that prints the load time in milliseconds:
  base   fast.bend at REV (default 5a041a7: each band read by binary-read into a fresh Array.new, the band tree
         taken from a TA.Bands.zeros tree)
  split  base, with each band's Array split at its ANode halves 4 levels deep in parallel lets after the read, and
         joined again (the cost of handing out the halves of a band to 16 threads, with nothing to decode)
  pnew   base, with each band's Array built as ANode halves 4 levels deep in parallel lets instead of one Array.new
  tree   fast.bend in the working tree
A round runs every (variant, threads) pair once, in a shuffled order: 1 thread pinned with taskset -c 3, 16 threads
with +RTS --threads 16 -RTS, under perf stat (cpu_core/instructions/u, the whole process: loading and one token).

  python3 bench/gpt2_load.py build <outdir> [REV]
  python3 bench/gpt2_load.py bits <outdir>                 hash of every weight per variant (reference/test_gpt2_load.py)
  python3 bench/gpt2_load.py run <outdir> <rounds> <results.tsv>
"""
import os, random, re, statistics, subprocess, sys
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "reference"))
import test_gpt2_load as T  # noqa: E402

VARIANTS = ["base", "split", "pnew", "tree"]
THREADS = [1, 16]

ANODE = '''
# ---- added by bench/gpt2_load.py: an Array split at, or built from, its ANode halves in parallel lets ----
def pnew(+k: Nat, +d: Nat) -> Array<F32>:
  match k:
    case 0n:
      Array.new(F32, d, 0.0)
    case 1n+q:
      x y = pnew(q, Nat.sub(d, 1n)) pnew(q, Nat.sub(d, 1n))
      ANode{x, y}

def psplit(+k: Nat, a: Array<F32>) -> Array<F32>:
  match k a:
    case 0n _:
      a
    case 1n+q ALeaf{v}:
      ALeaf{v}
    case 1n+q ANode{xs, ys}:
      x y = psplit(q, xs) psplit(q, ys)
      ANode{x, y}
'''


def variant(t, v):
    t = T.sub(t, "IO.print(\"weights loaded in \" ++ tsec(t1, t0))",
              "IO.print(\"weights loaded in \" ++ Nat.show(Nat.sub(t1, t0)) ++ \" ms\")")
    if v in ("split", "pnew"):
        t = T.sub(t, "import ./tok.bend as K\n", "import ./tok.bend as K\n" + ANODE)
    if v == "split":
        t = T.sub(t, "        return TA.Mat{a}\n", "        return TA.Mat{psplit(4n, a)}\n")
    if v == "pnew":
        t = T.sub(t, "Array.new(F32, TA.cap_depth(Nat.mul(rr, c)), 0.0)",
                  "pnew(Nat.min(4n, TA.cap_depth(Nat.mul(rr, c))), TA.cap_depth(Nat.mul(rr, c)))")
    return t


def sources(rev):
    old = subprocess.run(["git", "show", f"{rev}:demos/gpt2/fast.bend"], capture_output=True, text=True, cwd=WT,
                         check=True).stdout
    cur = open(os.path.join(WT, "demos/gpt2/fast.bend")).read()
    return {"base": old, "split": old, "pnew": old, "tree": cur}


def build(out, rev):
    os.makedirs(out, exist_ok=True)
    for v, t in sources(rev).items():
        T.build(variant(t, v), os.path.join(out, v))
        T.build(T.hash_source(variant(t, v)), os.path.join(out, v + "_h"))
        print("built", v)


def bits(out):
    for th in THREADS:
        hs = {v: T.run_hash(os.path.join(out, v + "_h"), th) for v in VARIANTS}
        same = len(set(hs.values())) == 1
        print(f"{th} thread(s): " + " ".join(f"{v}={h}" for v, h in hs.items()) + ("  same" if same else "  DIFFERENT"))


def once(exe, th):
    cmd = ["perf", "stat", "-x", ",", "-e", "cpu_core/instructions/u"]
    cmd += ["taskset", "-c", "3", exe, "Hi", "1"] if th == 1 else [exe, "Hi", "1", "+RTS", "--threads", str(th), "-RTS"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=WT, env=T.ENV)
    ms = int(re.search(r"weights loaded in (\d+) ms", r.stdout).group(1))
    ins = int(re.search(r"^(\d+),,cpu_core/instructions/u", r.stderr, re.M).group(1))
    return ms, ins


def run(out, rounds, tsv):
    pairs = [(v, th) for v in VARIANTS for th in THREADS]
    rows = []
    with open(tsv, "w") as f:
        f.write("round\tvariant\tthreads\tload_ms\tinstructions_u\n")
        for rd in range(rounds):
            random.shuffle(pairs)
            for v, th in pairs:
                ms, ins = once(os.path.join(out, v), th)
                rows.append((v, th, ms, ins))
                f.write(f"{rd}\t{v}\t{th}\t{ms}\t{ins}\n")
                f.flush()
    print(f"{'variant':8} {'threads':>7} {'median ms':>9} {'min-max':>11} {'instr. (M)':>10} {'vs base':>8}")
    med = {}
    for v in VARIANTS:
        for th in THREADS:
            ms = [r[2] for r in rows if r[0] == v and r[1] == th]
            ins = [r[3] for r in rows if r[0] == v and r[1] == th]
            med[v, th] = statistics.median(ms)
            rel = med[v, th] / med["base", th] - 1 if ("base", th) in med else 0.0
            print(f"{v:8} {th:7} {med[v, th]:9.0f} {min(ms):5}-{max(ms):<5} {statistics.median(ins) / 1e6:10.0f} {rel:+8.0%}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "5a041a7")
    elif cmd == "bits":
        bits(sys.argv[2])
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        sys.exit(__doc__)
