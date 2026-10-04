# matrix · vector as in GPT-2: y(1 x M) = x(1 x K) · W(M x K)ᵀ, repeated REPS times; W with distinct data.
import Base
import bend-ml-tensor@0.1.1.0/main.bend as T
import bend-ml-tensor-array@0.1.1.0/main.bend as TA

# list of n values ((i mod 13) / 10)

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
def loop(k: Nat, +par: Nat, acc: F32, r: TA.MMulNT<1n, KN, MN>) -> F32:
  match k r:
    case 0n TA.MMulNT{x, w, c}:
      (acc + ml_max(1n, MN, TA.Mat.to_list(1n, MN, c)) : F32)
    case 1n+p TA.MMulNT{x, w, c}:
      loop(p, par, (acc + ml_max(1n, MN, TA.Mat.to_list(1n, MN, c)) : F32), TA.Mat.matmul_nt(1n, KN, MN, par, x, w))

def main() -> IO(Unit):
  IO.print(F32.show(loop(REPS, PAR, 0.0, TA.Mat.matmul_nt(1n, KN, MN, PAR, TA.Mat.from_list(1n, KN, gen(KN, 0, Nil{})), TA.Mat.from_list(MN, KN, gen(Nat.mul(MN, KN), 0, Nil{}))))))
