# The layer from runtime_batch.bend with W of the wrong shape (5 x 3 instead of 4 x 3).
# X has 4 columns, so X·W needs W to have 4 rows. This must NOT compile, and it does not,
# even though the batch size n is unknown until the program reads its input file:
# the checker proves the mismatch for every n at once.
#
#   bend examples/runtime_batch_bad.bend
#   Error:
#   - expected : TA.Mat<4n, 3n>
#   - observed : TA.Mat<5n, 3n>

import Base
import bend-ml-tensor-array@0.1.3.0/main.bend as TA

def wrong_weights() -> TA.Mat<5n, 3n>:
  TA.Mat.zeros(5n, 3n)

def layer(+n: Nat, x: TA.Mat<n, 4n>) -> TA.MMul<n, 4n, 3n>:
  TA.Mat.matmul(n, 4n, 3n, 0n, x, wrong_weights())
