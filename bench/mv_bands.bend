# matrix · vector as in bench/mv.bend, but with W in 2^DEPTH row bands (Bands): y(MN) = W(MN x KN) · x(KN),
# repeated REPS times. The bands run in parallel and no weight is copied.
import Base
import bend-ml-tensor@0.1.2.0/main.bend as T
import bend-ml-tensor-array@0.1.5.0/main.bend as TA

# list of n values ((i mod 13) / 10), the same data as bench/mv.bend

def gen(n: Nat, +i: U32, acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      acc
    case 1n+p:
      gen(p, (i + 1 : U32), (U32.to_f32((i - (U32.div(i, 13) * 13 : U32) : U32)) / 10.0 : F32) <> acc)

def ml_max(-r: Nat, -c: Nat, x: TA.ML<r, c>) -> F32:
  match x:
    case TA.ML{m, l}:
      T.max.l(l)

# r is the product of the previous iteration
def loop(k: Nat, +x: List<&2, F32>, acc: F32, r: TA.BV<MN, KN>) -> F32:
  match k r:
    case 0n TA.BV{w, y}:
      (acc + ml_max(1n, MN, TA.Mat.to_list(1n, MN, y)) : F32)
    case 1n+p TA.BV{w, y}:
      loop(p, x, (acc + ml_max(1n, MN, TA.Mat.to_list(1n, MN, y)) : F32), TA.Bands.matvec(MN, KN, w, TA.Mat.from_list(1n, KN, x)))

def main() -> IO(Unit):
  IO.print(F32.show(loop(REPS, gen(KN, 0, Nil{}), 0.0, TA.Bands.matvec(MN, KN, TA.Bands.fill_at(MN, KN, 0n, Nat.mul(MN, KN), gen(Nat.mul(MN, KN), 0, Nil{}), TA.Bands.zeros(DEPTH, MN, KN)), TA.Mat.from_list(1n, KN, gen(KN, 0, Nil{}))))))
