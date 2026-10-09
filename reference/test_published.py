"""Imports every package from BendHub, at the version listed in the README, in a clean folder,
applies one proved law from each and runs a small computation. It catches a missing,
broken or mismatched publication.
"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")

README = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
VERS = dict(re.findall(r"\[`(bend-ml-[a-z-]+)`\]\([a-z-]+\) \| (\d+\.\d+\.\d+\.\d+) \|", README))
NEED = ["bend-ml-nat-lemmas", "bend-ml-bpe-tokenizer", "bend-ml-tensor", "bend-ml-tensor-array", "bend-ml-autograd"]

PROGRAM = """import Base
import {nl}/main.bend as NL
import {bpe}/main.bend as BPE
import {t}/main.bend as T
import {ta}/main.bend as TA
import {ag}/main.bend as AG

# a law from each package, applied to symbolic arguments (type-checked against the statement)
def law_nat(a: Nat, +b: Nat) -> {{Nat.add(a, b) == Nat.add(b, a) : Nat}}:
  NL.add_comm(a, b)

def law_bpe(+n: Nat, +corpus: List<&2, Nat>, +bs: List<&2, Nat>) -> {{BPE.decode(BPE.train(n, 0n, BPE.lift(corpus)), BPE.encode(BPE.train(n, 0n, BPE.lift(corpus)), BPE.lift(bs))) == bs : List<&2, Nat>}}:
  BPE.roundtrip_trained(n, corpus, bs)

def law_tensor(+r: Nat, +c: Nat) -> {{Nat.mul(r, c) == Nat.mul(c, r) : Nat}}:
  T.reshape_swap(r, c)

def law_bands(+n: Nat) -> {{n == Nat.add(TA.half(n), Nat.sub(n, TA.half(n))) : Nat}}:
  TA.half_cover(n)

def law_band(-r: Nat, +c: Nat, +rows: Nat, x: Array<F32>, w: Array<F32>) -> {{TA.bl_len(r, c, TA.bv_gemm(r, c, rows, rows, x, w)) == rows : Nat}}:
  TA.band_len(r, c, rows, x, w)

def law_ad(+e: AG.NE, +x: Nat) -> {{AG.nbwd(e, x, 1n) == AG.nfwd(e, x) : Nat}}:
  AG.reverse_eq_forward(e, x)

# a small computation per package; the shape errors are caught by the types
def prod() -> TA.MMul<2n, 3n, 4n>:
  TA.Mat.matmul(2n, 3n, 4n, 0n, TA.Mat.fill(2n, 3n, 2.0), TA.Mat.fill(3n, 4n, 3.0))

def len_of(-r: Nat, -c: Nat, x: TA.ML<r, c>) -> Nat:
  match x:
    case TA.ML{{m, l}}:
      List.length(&2, F32, l)

def after(x: TA.MMul<2n, 3n, 4n>) -> Nat:
  match x:
    case TA.MMul{{a, b, c}}:
      len_of(2n, 4n, TA.Mat.to_list(2n, 4n, c))

# 6 x 3 in 2^2 bands, times a vector of 3 numbers: 6 numbers
def bands_len(x: TA.BV<6n, 3n>) -> Nat:
  match x:
    case TA.BV{{b, y}}:
      len_of(1n, 6n, TA.Mat.to_list(1n, 6n, y))

def main() -> Nat:
  Nat.add(after(prod()), bands_len(TA.Bands.matvec(6n, 3n, TA.Bands.zeros(2n, 6n, 3n), TA.Mat.fill(1n, 3n, 1.0))))
"""


def main():
    missing = [p for p in NEED if p not in VERS]
    if missing:
        sys.exit(f"ERROR: version missing from the README table: {missing}")
    src = PROGRAM.format(nl=f"bend-ml-nat-lemmas@{VERS['bend-ml-nat-lemmas']}",
                         bpe=f"bend-ml-bpe-tokenizer@{VERS['bend-ml-bpe-tokenizer']}",
                         t=f"bend-ml-tensor@{VERS['bend-ml-tensor']}",
                         ta=f"bend-ml-tensor-array@{VERS['bend-ml-tensor-array']}",
                         ag=f"bend-ml-autograd@{VERS['bend-ml-autograd']}")
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "published.bend")
        open(f, "w").write(src)
        r = subprocess.run([BEND, f], capture_output=True, text=True, env=ENV, cwd=d)
    out = (r.stdout + r.stderr).strip()
    ok = r.returncode == 0 and out == "14n"
    print(("ok   " if ok else "ERROR ") + "published packages " + ", ".join(f"{k.replace('bend-ml-', '')}@{v}" for k, v in VERS.items() if k in NEED))
    if not ok:
        print(out[:800])
    sys.exit(0 if ok else 1)


main()
