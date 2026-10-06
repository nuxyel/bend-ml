import Base
import ../main.bend as TA
# MUST FAIL: the weights are 2304 x 768 (one row per output), so B · x needs x with 768
# numbers; a vector of 2304 numbers would be the transposed product and does not fit
def bad(b: TA.Bands<2304n, 768n>, x: TA.Mat<1n, 2304n>) -> TA.BV<2304n, 768n>:
  TA.Bands.matvec(2304n, 768n, b, x)
def main() -> Nat:
  0n
