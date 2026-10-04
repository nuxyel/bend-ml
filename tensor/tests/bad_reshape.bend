import Base
import ../main.bend as T
# MUST FAIL: 2x6 = 12 elements, 5x3 = 15; there is no proof of 12 == 15
def bad() -> T.Mat<5n, 3n>:
  T.Mat.reshape(2n, 6n, 5n, 3n, {==}, T.Mat.fill(2n, 6n, 1.0))
def main() -> Nat:
  0n
