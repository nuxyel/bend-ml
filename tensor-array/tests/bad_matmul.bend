import Base
import ../main.bend as TA
# MUST FAIL: (2x3) · (4x5): the inner dimensions 3 and 4 do not match
def bad() -> TA.MMul<2n, 3n, 5n>:
  TA.Mat.matmul(2n, 3n, 5n, 0n, TA.Mat.zeros(2n, 3n), TA.Mat.zeros(4n, 5n))
def main() -> Nat:
  0n
