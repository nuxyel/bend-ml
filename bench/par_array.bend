# confere o ganho do Array com dados DISTINTOS: a[i] = i/100, b[i] = (783-i)/100
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

# preenche a[i] = i/100 e b[i] = (783-i)/100 para i = 0..783
def fill(n: Nat, +i: U32, a: Array<F32>, b: Array<F32>) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (a, b)
    case 1n+p:
      fill(p, (i + 1 : U32), Array.set(F32, a, i, (U32.to_f32(i) / 100.0 : F32)), Array.set(F32, b, i, (U32.to_f32((783 - i : U32)) / 100.0 : F32)))

def go(r: Array<F32> & Array<F32>, k: Nat) -> F32:
  match r:
    case Tuple{a, b}:
      many(k, 0.0, dot(a, b))

# uma tarefa: cria seus arrays e faz k produtos
def task(k: Nat) -> F32:
  go(fill(784n, 0, [0.0 : F32*1024n], [0.0 : F32*1024n]), k)

# 2^d tarefas em paralelo (fork-join binário)
def tree(d: Nat, +k: Nat) -> F32:
  match d:
    case 0n:
      task(k)
    case 1n++q:
      a b = tree(q, k) tree(q, k)
      (a + b : F32)

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(F32.show(tree(5n, 80000n)))
