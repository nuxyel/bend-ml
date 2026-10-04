# dot product of two Array<F32> with 1024 slots (784 used), repeated
# 128000 times = the same number of multiplications as mm_list (10 * 100 * 128 products of 784).
import Base

type R is Type:
  R{a: Array<F32>, b: Array<F32>, x: F32}

# reads a[i] and b[i], accumulates; returns both arrays for the next step
def dot2(n: Nat, +i: U32, acc: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> R:
  match n ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      R{a, b, acc}
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      dot2(p, (i + 1 : U32), (acc + (x * y : F32) : F32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def dot(a: Array<F32>, b: Array<F32>) -> R:
  dot2(784n, 0, 0.0, Array.get(F32, a, 0), Array.get(F32, b, 0))

def many(k: Nat, acc: F32, r: R) -> F32:
  match k r:
    case 0n R{a, b, x}:
      (acc + x : F32)
    case 1n+p R{a, b, +x}:
      many(p, (acc + x : F32), dot(a, b))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(F32.show(many(128000n, 0.0, dot([0.5 : F32*1024n], [0.25 : F32*1024n]))))
