import Base
import ../main.bend as TA

def len_of(-r: Nat, -c: Nat, x: TA.ML<r, c>) -> Nat:
  match x:
    case TA.ML{m, l}:
      List.length(&2, F32, l)

def after(x: TA.MMul<2n, 3n, 4n>) -> Nat:
  match x:
    case TA.MMul{a, b, c}:
      len_of(2n, 4n, TA.Mat.to_list(2n, 4n, c))

def main() -> Nat:
  after(TA.Mat.matmul(2n, 3n, 4n, 0n, TA.Mat.fill(2n, 3n, 2.0), TA.Mat.fill(3n, 4n, 3.0)))
