import Base
import ../main.bend as TA
# DEVE FALHAR: (2x3) · (4x5): as dimensões internas 3 e 4 não batem
def bad() -> TA.MMul<2n, 3n, 5n>:
  TA.Mat.matmul(2n, 3n, 5n, 0n, TA.Mat.zeros(2n, 3n), TA.Mat.zeros(4n, 5n))
def main() -> Nat:
  0n
