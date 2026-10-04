import Base

def fill(n: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      a
    case 1n+p:
      fill(p, (i + 1 : U32), Array.set(F32, a, i, U32.to_f32(i)))

# k clones encadeados; cada um LÊ o último elemento da cópia (obriga a materializar)
def loop(k: Nat, acc: F32, r: Array<F32> & Array<F32>, g: Array<F32> & F32) -> F32:
  match k r g:
    case 0n Tuple{x, y} Tuple{z, +v}:
      (acc + v : F32)
    case 1n+p Tuple{x, y} Tuple{z, +v}:
      loop(p, (acc + v : F32), Array.clone(F32, x), Array.get(F32, y, 131071))

def run(k: Nat, a: Array<F32>) -> F32:
  loop(k, 0.0, Array.clone(F32, a), Array.get(F32, [0.0 : F32^1n], 0))

def main() -> IO(Unit):
  IO.print(F32.show(run(K, fill(131072n, 0, [0.0 : F32^17n]))))
