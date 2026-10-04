import Base
import ../main.bend as T
# DEVE FALHAR: (2x3) · (4x5), as dimensões internas 3 e 4 não batem
def bad() -> T.Mat<2n, 5n>:
  T.Mat.matmul(2n, 3n, 5n, T.Mat.fill(2n, 3n, 1.0), T.Mat.fill(4n, 5n, 1.0))
def main() -> Nat:
  0n
