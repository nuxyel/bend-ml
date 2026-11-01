# Prints every output of several matrix · vector products (Bands.matvec_l with 1 and 2^4 bands, Mat.matmul_nt
# with one row) so two versions of tensor-array can be compared by a hash of the printed text (F32.show prints the
# shortest exact form). TAPATH is the tensor-array file, MN x KN the shape, BANDS `bands(7)` or `Nil{}`; the
# driver is bench/mv_band_cost.py bits.
import Base
import TAPATH as TA

def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  Bool.pick(F32, U32.is_eq(h, 7), (0.0 - 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 23), 0.00000000000000000000000000000000000001, (((U32.to_f32(h) / 250.0 : F32) - 2.0 : F32))))

def fill(k: Nat, +i: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, (i + 1 : U32), seed, Array.set(F32, a, i, val(i, seed)))

def gen(n: Nat, +i: U32, +seed: U32, acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      gen(p, (i + 1 : U32), seed, val(i, seed) <> acc)

def ys(r: TA.BL<MN, KN>) -> List<&2, F32>:
  match r:
    case TA.BL{w, y}: y

def yv(r: TA.ML<1n, MN>) -> List<&2, F32>:
  match r:
    case TA.ML{m, l}: l

def nt(r: TA.MMulNT<1n, KN, MN>) -> List<&2, F32>:
  match r:
    case TA.MMulNT{a, b, c}: yv(TA.Mat.to_list(1n, MN, c))

def one(+seed: U32) -> List<&2, F32>:
  ys(TA.Bands.matvec_l(MN, KN, TA.BLeaf{TA.Mat{fill(Nat.mul(MN, KN), 0, seed, Array.new(F32, TA.cap_depth(Nat.mul(MN, KN)), 0.0))}}, gen(KN, 0, (seed + 1 : U32), Nil{})))

def bands(+seed: U32) -> List<&2, F32>:
  ys(TA.Bands.matvec_l(MN, KN, TA.Bands.fill_at(MN, KN, 0n, Nat.mul(MN, KN), gen(Nat.mul(MN, KN), 0, seed, Nil{}), TA.Bands.zeros(4n, MN, KN)), gen(KN, 0, (seed + 1 : U32), Nil{})))

def mnt(+seed: U32) -> List<&2, F32>:
  nt(TA.Mat.matmul_nt(1n, KN, MN, 0n, TA.Mat{fill(KN, 0, (seed + 1 : U32), Array.new(F32, TA.cap_depth(KN), 0.0))}, TA.Mat{fill(Nat.mul(MN, KN), 0, seed, Array.new(F32, TA.cap_depth(Nat.mul(MN, KN)), 0.0))}))

def main() -> List<&2, F32>:
  List.append(&2, F32, one(5), List.append(&2, F32, BANDS, mnt(9)))
