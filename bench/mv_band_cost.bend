# What one band of Bands.matvec_l costs: y(MN) = W(MN x KN) · x(KN) repeated REPS times, either through
# Bands.matvec_l with one band (MODE = 1n) or through the bare gemm it runs, with x and y kept as arrays and
# reused (MODE = 0n). W is filled in place (no list of MN * KN numbers). Prints a sum over y so every
# product is used.
import Base
import TAPATH as TA

# w[i] = (i mod 13) / 10 for i < n, the same data as bench/mv_bands.bend
def fillw(n: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      a
    case 1n+p:
      fillw(p, (i + 1 : U32), Array.set(F32, a, i, (U32.to_f32((i - (U32.div(i, 13) * 13 : U32) : U32)) / 10.0 : F32)))

def gen(n: Nat, +i: U32, acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      acc
    case 1n+p:
      gen(p, (i + 1 : U32), (U32.to_f32((i - (U32.div(i, 13) * 13 : U32) : U32)) / 10.0 : F32) <> acc)

def lsum(xs: List<&2, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      lsum(t, (acc + h : F32))

def asum(n: Nat, +i: U32, acc: F32, r: Array<F32> & F32) -> F32:
  match n r:
    case 0n Tuple{a, +v}:
      (acc + v : F32)
    case 1n+p Tuple{a, +v}:
      asum(p, (i + 1 : U32), (acc + v : F32), Array.get(F32, a, (i + 1 : U32)))

# the bare kernel: x (array a), W (array b) and y (array c) reused across products
def lg(k: Nat, acc: F32, g: TA.G) -> F32:
  match k g:
    case 0n TA.G{a, b, c}:
      asum(Nat.sub(MN, 1n), 0, acc, Array.get(F32, c, 0))
    case 1n+p TA.G{a, b, c}:
      lg(p, acc, TA.gemm(1n, MN, KN, U32.from_nat(KN), 1, 1, U32.from_nat(KN), a, b, c))

# Bands.matvec_l with one band; the sum of y is taken at the end only, as in lg
def ll(k: Nat, +x: List<&2, F32>, r: TA.BL<MN, KN>) -> F32:
  match k r:
    case 0n TA.BL{w, y}:
      lsum(y, 0.0)
    case 1n+p TA.BL{w, y}:
      ll(p, x, TA.Bands.matvec_l(MN, KN, w, x))

def run(m: Nat) -> F32:
  match m:
    case 0n:
      lg(1n+REPS, 0.0, TA.G{TA.fill_list(gen(KN, 0, Nil{}), 0, Array.new(F32, TA.cap_depth(KN), 0.0)), fillw(Nat.mul(MN, KN), 0, Array.new(F32, TA.cap_depth(Nat.mul(MN, KN)), 0.0)), Array.new(F32, TA.cap_depth(MN), 0.0)})
    case 1n+_:
      ll(REPS, gen(KN, 0, Nil{}), TA.Bands.matvec_l(MN, KN, TA.BLeaf{TA.Mat{fillw(Nat.mul(MN, KN), 0, Array.new(F32, TA.cap_depth(Nat.mul(MN, KN)), 0.0))}}, gen(KN, 0, Nil{})))

def main() -> IO(Unit):
  IO.print(F32.show(run(MODE)))
