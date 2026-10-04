import Base

# Vetor de F32 com o tamanho no TIPO: Vec(3n) é um tipo diferente de Vec(4n).
# O tipo é calculado por recursão em n, como Word(n) na Base.
type Vec.Nil is Data:
  VNil{}

type Vec.Con<-p: Nat> is Data:
  VCon{head: F32, tail: Vec(p)}

def Vec(n: Nat) -> Data:
  match n:
    case 0n:
      Vec.Nil
    case 1n+p:
      Vec.Con<p>

def Vec.zero(n: Nat) -> Vec(n):
  match n:
    case 0n:
      VNil{}
    case 1n+p:
      VCon{0.0, Vec.zero(p)}

def Vec.add(n: Nat, a: Vec(n), b: Vec(n)) -> Vec(n):
  match n:
    case 0n:
      VNil{}
    case 1n+p:
      VCon{x, xs} = a
      VCon{y, ys} = b
      VCon{(x + y : F32), Vec.add(p, xs, ys)}

# produto escalar acumulando em acc
def Vec.dot(n: Nat, a: Vec(n), b: Vec(n), acc: F32) -> F32:
  match n:
    case 0n:
      acc
    case 1n+p:
      VCon{x, xs} = a
      VCon{y, ys} = b
      Vec.dot(p, xs, ys, (acc + (x * y : F32) : F32))

def Vec.fill(n: Nat, +v: F32) -> Vec(n):
  match n:
    case 0n:
      VNil{}
    case 1n+p:
      VCon{v, Vec.fill(p, v)}

def main() -> IO(Unit):
  do IO<Unit>:
    a : Vec(784n) = Vec.fill(784n, 0.5)
    b : Vec(784n) = Vec.fill(784n, 2.0)
    IO.print(F32.show(Vec.dot(784n, a, b, 0.0)))
