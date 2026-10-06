# examples/symbolic_reshape.bend asking for the wrong shape: n x 6 -> n x 5. This must NOT
# compile, and it does not: the reshape needs a proof that n*6 == n*5, which is false for
# every n > 0, so no proof can be written. Reflexivity ({==}) is refused because the checker
# cannot make n*6 and n*5 equal for an unknown n.
#
#   bend examples/symbolic_reshape_bad.bend
#   Error:
#   - expected : Nat.mul(n, 6n)
#   - observed : Nat.mul(n, 5n)

import Base
import bend-ml-tensor@0.1.2.0/main.bend as T

def regroup(+n: Nat, x: T.Mat<n, 6n>) -> T.Mat<n, 5n>:
  T.Mat.reshape(n, 6n, n, 5n, {==}, x)
