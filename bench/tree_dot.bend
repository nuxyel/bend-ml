# dot product walking BOTH trees together (no indices); returns the rebuilt trees.
import Base

type R is Type:
  R{a: Array<F32>, b: Array<F32>, x: F32}

def dotT2(l: R, r: R) -> R:
  match l r:
    case R{a1, b1, x1} R{a2, b2, x2}:
      R{ANode{a1, a2}, ANode{b1, b2}, (x1 + x2 : F32)}

# sum of the products; the trees are returned rebuilt
def dotT(a: Array<F32>, b: Array<F32>) -> R:
  match a b:
    case ALeaf{+x} ALeaf{+y}:
      R{ALeaf{x}, ALeaf{y}, (x * y : F32)}
    case ANode{xa, ya} ANode{xb, yb}:
      dotT2(dotT(xa, xb), dotT(ya, yb))
    case ALeaf{y1} ANode{y2, y3}:
      R{ALeaf{y1}, ANode{y2, y3}, 0.0}
    case ANode{y1, y2} ALeaf{y3}:
      R{ANode{y1, y2}, ALeaf{y3}, 0.0}

def many(k: Nat, acc: F32, r: R) -> F32:
  match k r:
    case 0n R{a, b, x}:
      (acc + x : F32)
    case 1n+p R{a, b, +x}:
      many(p, (acc + x : F32), dotT(a, b))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(F32.show(many(100000n, 0.0, dotT([0.5 : F32*1024n], [0.25 : F32*1024n]))))
