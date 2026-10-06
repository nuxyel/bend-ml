import Base
import ../main.bend as TA
# Compiled to C by reference/check_all.py, which requires the join task of TA.bmv: without it the
# bands of a matrix · vector product run one after another (see NOTES.md, exp. 11).
def ys(r: TA.BV<64n, 8n>) -> Nat:
  match r:
    case TA.BV{b, y}:
      7n
def main() -> Nat:
  ys(TA.Bands.matvec(64n, 8n, TA.Bands.zeros(3n, 64n, 8n), TA.Mat.fill(1n, 8n, 1.0)))
