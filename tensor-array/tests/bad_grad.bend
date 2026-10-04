import Base
import ../main.bend as TA
# MUST FAIL: the layer gradient (dW) must be (i x o) = Xᵀ·dY; here the product comes out (o x i)
def bad(x: TA.Mat<100n, 784n>, dy: TA.Mat<100n, 128n>) -> TA.MMulTN<128n, 100n, 784n>:
  TA.Mat.matmul_tn(784n, 100n, 128n, 0n, x, dy)
def main() -> Nat:
  0n
