import Base
import ../main.bend as T

def fshow(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      F32.show(h) ++ " " ++ fshow(t)

def rshow(rs: List<&2, List<&2, F32>>) -> String:
  match rs:
    case Nil{}:
      ""
    case Con{r, t}:
      fshow(r) ++ "| " ++ rshow(t)

def showm(-r: Nat, -c: Nat, m: T.Mat<r, c>) -> String:
  match m:
    case T.Mat{x}:
      rshow(x)

# 2x3 · 3x4 = 2x4
def prod() -> T.Mat<2n, 4n>:
  T.Mat.matmul(2n, 3n, 4n, T.Mat.fill(2n, 3n, 2.0), T.Mat.fill(3n, 4n, 3.0))

# 2x6 -> 3x4: 12 == 12, the proof is just computing
def resh() -> T.Mat<3n, 4n>:
  T.Mat.reshape(2n, 6n, 3n, 4n, {==}, T.Mat.fill(2n, 6n, 1.0))

def tr() -> T.Mat<3n, 2n>:
  T.Mat.transpose(2n, 3n, T.Mat.fill(2n, 3n, 5.0))

def flat() -> T.Mat<1n, 12n>:
  T.Mat.flatten(3n, 4n, T.Mat.fill(3n, 4n, 7.0))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(showm(2n, 4n, prod()))
    IO.print(showm(3n, 4n, resh()))
    IO.print(showm(3n, 2n, tr()))
    IO.print(showm(1n, 12n, flat()))
