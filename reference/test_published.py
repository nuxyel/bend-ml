"""Imports every package from BendHub, at the version listed in the README, in a clean folder,
applies one proved law from each and runs a small computation. It catches a missing,
broken or mismatched publication.
"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
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


# bend-ml-binary-read is foreign code (no laws): read 1.0 and -2.5 back from a file, on the JS and the C lane
READ = """import Base
import {br}/main.bend as R

def show3(r: Array<F32> & F32) -> String:
  match r:
    case Tuple{{a, x}}:
      F32.show(x)

def show2(r: Array<F32> & F32) -> String:
  match r:
    case Tuple{{a, x}}:
      F32.show(x) ++ " " ++ show3(Array.get(F32, a, 1))

def show(p: File & (Array<F32> & Result<&1, &1, U32 & String, U32>)) -> IO(Unit):
  match p:
    case Tuple{{f, Tuple{{a, r}}}}:
      do IO<Unit>:
        +n : U32 <- IO.pass(U32, r)
        File.close(f)
        IO.print(U32.show(n) ++ " " ++ show2(Array.get(F32, a, 0)))

def main() -> IO(Unit):
  do IO<Unit>:
    f : File <- IO.try(File, File.open("w.bin", "r"))
    p : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- R.read_f32(f, 0, 2, Array.new(F32, 1n, 0.0))
    show(p)
"""


def check_read():
    br = f"bend-ml-binary-read@{VERS['bend-ml-binary-read']}"
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "w.bin"), "wb").write(bytes([0, 0, 128, 63, 0, 0, 32, 192]))
        f = os.path.join(d, "read.bend")
        open(f, "w").write(READ.format(br=br))
        js = subprocess.run([BEND, f], capture_output=True, text=True, env=ENV, cwd=d)
        cc = subprocess.run([BEND, f, "-o", os.path.join(d, "read")], capture_output=True, text=True, env=ENV, cwd=d)
        c = subprocess.run([os.path.join(d, "read")], capture_output=True, text=True, cwd=d) if cc.returncode == 0 else cc
    outs = [js.stdout.strip(), c.stdout.strip()]
    ok = outs == ["8 1 -2.5", "8 1 -2.5"]
    print(("ok   " if ok else "ERROR ") + f"published {br.replace('bend-ml-', '')} on the JS and C lanes")
    if not ok:
        print(outs, (js.stderr + c.stderr)[:800])
    return ok


def main():
    missing = [p for p in NEED + ["bend-ml-binary-read"] if p not in VERS]
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
    ok = check_read() and ok
    sys.exit(0 if ok else 1)


main()
