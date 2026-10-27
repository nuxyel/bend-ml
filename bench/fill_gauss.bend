# 100 fills of a 256 x 3072 Mat of Gaussians (tensor-gpu's gauss_mat, the size of bend-ml #23), counters s0, s0 + 1, ..,
# one cell of each read back; s0 is the number of arguments. Time it on the GPU and with +RTS --gpu off --threads N -RTS.
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ../tensor-gpu/main.bend as TG

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def cell(r: Array<F32> & F32) -> U32:
  match r:
    case Tuple{a, x}: bits(x)

def chk(m: TA.Mat<256n, 3072n>) -> U32:
  match m:
    case TA.Mat{d}: cell(Array.get(F32, d, 12345))

# k fills of 256 x 3072, counters s0, s0 + 1, ..; one cell of each read back
@unsafe
def go(k: Nat, +s: U32, acc: U32) -> U32:
  match k:
    case 0n: acc
    case 1n+p: go(p, (s + 1 : U32), U32.xor(acc, chk(TG.gauss_mat(s, 256n, 3072n))))

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    IO.print(U32.show(go(100n, U32.from_nat(List.length(&1, String, args)), 0)))
