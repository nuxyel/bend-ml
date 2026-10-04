# matmul 100x784 · 784x128 com listas (como o bend-ml-tensor), repetido N vezes.
import Base

def dot(xs: List<&2, F32>, ys: List<&2, F32>, acc: F32) -> F32:
  match xs ys:
    case Con{+h, t} Con{+g, u}:
      dot(t, u, (acc + (h * g : F32) : F32))
    case _ _:
      acc

def row(+ra: List<&2, F32>, bt: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match bt:
    case Nil{}:
      Nil{}
    case Con{col, t}:
      dot(ra, col, 0.0) <> row(ra, t)

def rows(a: List<&2, List<&2, F32>>, +bt: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match a:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      row(r, bt) <> rows(t, bt)

def rep(n: Nat, +v: F32, +c: Nat) -> List<&2, List<&2, F32>>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      List.replicate(F32, c, v) <> rep(p, v, c)

def tot(rs: List<&2, List<&2, F32>>, acc: F32) -> F32:
  match rs:
    case Nil{}:
      acc
    case Con{+r, t}:
      tot(t, (dot(r, r, 0.0) + acc : F32))

def many(n: Nat, +a: List<&2, List<&2, F32>>, +bt: List<&2, List<&2, F32>>, acc: F32) -> F32:
  match n:
    case 0n:
      acc
    case 1n+p:
      many(p, a, bt, (acc + tot(rows(a, bt), 0.0) : F32))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(F32.show(many(10n, rep(100n, 0.5, 784n), rep(128n, 0.25, 784n), 0.0)))
