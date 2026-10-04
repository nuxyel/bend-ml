import Base
import ../main.bend as T
# MUST FAIL: (2x3) · (4x5), the inner dimensions 3 and 4 do not match
def bad() -> T.Mat<2n, 5n>:
  T.Mat.matmul(2n, 3n, 5n, T.Mat.fill(2n, 3n, 1.0), T.Mat.fill(4n, 5n, 1.0))
def main() -> Nat:
  0n
