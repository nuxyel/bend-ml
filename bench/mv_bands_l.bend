# bench/mv_bands.bend with the list API, Bands.matvec_l (x and y are lists, no Mat conversions):
# y(MN) = W(MN x KN) · x(KN) with W in 2^DEPTH row bands, repeated REPS times.
import Base
import bend-ml-tensor@0.1.2.0/main.bend as T
import bend-ml-tensor-array@0.1.6.0/main.bend as TA

# list of n values ((i mod 13) / 10), the same data as bench/mv.bend

def gen(n: Nat, +i: U32, acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      acc
    case 1n+p:
      gen(p, (i + 1 : U32), (U32.to_f32((i - (U32.div(i, 13) * 13 : U32) : U32)) / 10.0 : F32) <> acc)

# r is the product of the previous iteration
def loop(k: Nat, +x: List<&2, F32>, acc: F32, r: TA.BL<MN, KN>) -> F32:
  match k r:
    case 0n TA.BL{w, y}:
      (acc + T.max.l(y) : F32)
    case 1n+p TA.BL{w, y}:
      loop(p, x, (acc + T.max.l(y) : F32), TA.Bands.matvec_l(MN, KN, w, x))

def main() -> IO(Unit):
  IO.print(F32.show(loop(REPS, gen(KN, 0, Nil{}), 0.0, TA.Bands.matvec_l(MN, KN, TA.Bands.fill_at(MN, KN, 0n, Nat.mul(MN, KN), gen(Nat.mul(MN, KN), 0, Nil{}), TA.Bands.zeros(DEPTH, MN, KN)), gen(KN, 0, Nil{})))))
