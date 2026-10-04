import Base

# F32 vector with its size in the TYPE: Vec(3n) is a different type from Vec(4n).
# The type is computed by recursion on n, like Word(n) in Base.
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

# dot product accumulating in acc
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
    a : Vec(3n) = Vec.fill(3n, 2.0)
    b : Vec(3n) = Vec.fill(3n, 3.0)
    IO.print(F32.show(Vec.dot(3n, a, b, 0.0)))
