# A square weight matrix used the wrong way round, the kind of shape bug that runs and
# gives wrong numbers instead of crashing.
#
# The weights are stored d_out x d_in (one row per output), so the right product is
# X · Wᵀ (Mat.matmul_nt). The bug below computes X · W (Mat.matmul). With concrete sizes
# like 768 x 768 both products fit, and no shape checker can tell them apart: the same
# function written with 768n everywhere compiles. Written once for any d_in and d_out,
# the two sizes have different names, so the checker sees that W's rows are d_out, not
# d_in, and the program does not compile, even if it is only ever used with
# d_in = d_out = 768.
#
#   bend examples/square_transpose_bad.bend
#   Error:
#   - expected : TA.Mat<din, dout>
#   - observed : TA.Mat<dout, din>

import Base
import bend-ml-tensor-array@0.1.3.0/main.bend as TA

# the bug: X · W instead of X · Wᵀ
def proj(+n: Nat, +din: Nat, +dout: Nat, x: TA.Mat<n, din>, w: TA.Mat<dout, din>) -> TA.MMul<n, din, dout>:
  TA.Mat.matmul(n, din, dout, 0n, x, w)

def attn_out(x: TA.Mat<1n, 768n>, w: TA.Mat<768n, 768n>) -> TA.MMul<1n, 768n, 768n>:
  proj(1n, 768n, 768n, x, w)
