import Base
import ../main.bend as TA
# DEVE FALHAR: o gradiente da camada (dW) tem de ser (i x o) = Xᵀ·dY; aqui o produto sai (o x i)
def bad(x: TA.Mat<100n, 784n>, dy: TA.Mat<100n, 128n>) -> TA.MMulTN<128n, 100n, 784n>:
  TA.Mat.matmul_tn(784n, 100n, 128n, 0n, x, dy)
def main() -> Nat:
  0n
