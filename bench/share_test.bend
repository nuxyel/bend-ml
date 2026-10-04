# Cada tarefa faz K produtos escalares de 784 elementos. MODO privado: a tarefa constrói seus
# arrays. MODO clone: recebe clones de um par construído uma vez na raiz.
import Base

type R is Type:
  R{a: Array<F32>, b: Array<F32>, x: F32}

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

def fill(n: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      a
    case 1n+p:
      fill(p, (i + 1 : U32), Array.set(F32, a, i, (U32.to_f32(i) / 100.0 : F32)))

def go(k: Nat, a: Array<F32>, b: Array<F32>) -> F32:
  many(k, 0.0, dot(a, b))

# privado
def tp(+d: Nat, +k: Nat) -> F32:
  match d:
    case 0n:
      go(k, fill(784n, 0, [0.0 : F32^10n]), fill(784n, 0, [0.0 : F32^10n]))
    case 1n+q:
      x y = tp(q, k) tp(q, k)
      (x + y : F32)

# clone: cada folha recebe um par (a, b) clonado do pai
def tc(d: Nat, +k: Nat, ra: Array<F32> & Array<F32>, rb: Array<F32> & Array<F32>) -> F32:
  match d ra rb:
    case 0n Tuple{a, a2} Tuple{b, b2}:
      go(k, a, b)
    case 1n++q Tuple{a, a2} Tuple{b, b2}:
      x y = tc(q, k, Array.clone(F32, a), Array.clone(F32, b)) tc(q, k, Array.clone(F32, a2), Array.clone(F32, b2))
      (x + y : F32)

def main() -> IO(Unit):
  IO.print(F32.show(MODE))
