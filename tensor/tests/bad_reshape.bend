import Base
import ../main.bend as T
# DEVE FALHAR: 2x6 = 12 elementos, 5x3 = 15; não existe prova de 12 == 15
def bad() -> T.Mat<5n, 3n>:
  T.Mat.reshape(2n, 6n, 5n, 3n, {==}, T.Mat.fill(2n, 6n, 1.0))
def main() -> Nat:
  0n
