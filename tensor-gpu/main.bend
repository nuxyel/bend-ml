# bend-ml-tensor-gpu: tensor-array's products on the GPU, with tensor-array's bits.
#
#   import ../tensor-array/main.bend as TA
#   import bend-ml-tensor-gpu@0.1.4.0/main.bend as TG
#   TG.matmul(n, k, m, par, a, b)      # the same arguments, result type and bits as TA.Mat.matmul
#
# The six products (matmul, matmul_nt, matmul_tn and their accumulating forms) take tensor-array's Mat values and return
# its result types; par is ignored (the GPU decides the split). Each product is one `!` call (a bang): a balanced fork
# tree whose leaves each compute a 4 x 4 tile of C, every node sharing A, B and C through Base's Array.fork (two
# handles to one block, no copy), as upstream's one-block pattern does (bendlang/bend tests/run/stencil3d.bend). On a
# machine without a GPU, or with `--gpu off`, the same bang runs on the CPU threads.
#
# The bits: a cell of C is computed in tensor-array's order. Product p of a dot goes to partial sum p mod 4, in
# increasing p, each sum starting at +0.0, and the four sums are combined by TA.t8_comb with rot = k mod 4, as
# tensor-array's dotr and t8_dot do; an accumulating product adds the cell to C once with TA.cwr. The loop takes four
# steps of k a turn; the steps past k in the last turn add +0.0, which leaves a sum unchanged (a sum that starts at +0.0
# is never -0.0). Bend builds the device code with --fmad=false, so no multiply-add is fused. tests/bits.bend checks the
# bits against tensor-array on many shapes, on the GPU and the CPU.
#
# Not proved, and marked so: Array.fork and Array.join are @unsafe in Base, so the defs below that call them, or take
# a shared array, are @unsafe, and `bend main.bend` names every def that relies on them. Nothing checks that a leaf
# writes only its own tile of C, or never writes A or B; the tiles are disjoint by construction (leaf t writes tile t).
# All of it is Bend: no foreign code. Written by bend-ml's scripts/gen_tensor_gpu.py (4 x 4 tiles).
import Base
import bend-ml-tensor-array@0.1.16.0/main.bend as TA

# the sizes and strides of one product: C[i][j] = sum_p A[oa + i*arow + p*sa] * B[ob + p*sb + j*bcol], n x m, k steps
# (oa, ob: 0 except for matmul_precise's blocks)
type E is Data:
  E{n: U32, m: U32, k: U32, tcols: U32, tiles: U32, arow: U32, sa: U32, sb: U32, bcol: U32, kg: Nat, rot: Nat, rem: U32, ac: Bool, oa: U32, ob: U32}

# a product that counts only when its step is inside k (the last turn's steps past k add +0.0)
def pk(ok: Bool, +x: F32) -> F32:
  match ok:
    case True{}: x
    case False{}: 0.0

# the values of one turn: A's 4 rows and B's 4 columns at four consecutive steps, read through the arrays in order
type GR is Type:
  GR{a: Array<F32>, b: Array<F32>, x0_0: F32, x0_1: F32, x0_2: F32, x0_3: F32, x1_0: F32, x1_1: F32, x1_2: F32, x1_3: F32, x2_0: F32, x2_1: F32, x2_2: F32, x2_3: F32, x3_0: F32, x3_1: F32, x3_2: F32, x3_3: F32, y0_0: F32, y0_1: F32, y0_2: F32, y0_3: F32, y1_0: F32, y1_1: F32, y1_2: F32, y1_3: F32, y2_0: F32, y2_1: F32, y2_2: F32, y2_3: F32, y3_0: F32, y3_1: F32, y3_2: F32, y3_3: F32}

def gr_31(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, +y2_2: F32, +y2_3: F32, +y3_0: F32, +y3_1: F32, +y3_2: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y3_3}:
      GR{a, b, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, y2_3, y3_0, y3_1, y3_2, y3_3}
def gr_30(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, +y2_2: F32, +y2_3: F32, +y3_0: F32, +y3_1: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y3_2}:
      gr_31(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, y2_3, y3_0, y3_1, y3_2, a, Array.get(F32, b, (ib + ((sb * 3 : U32) + (bcol * 3 : U32) : U32) : U32)))
def gr_29(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, +y2_2: F32, +y2_3: F32, +y3_0: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y3_1}:
      gr_30(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, y2_3, y3_0, y3_1, a, Array.get(F32, b, (ib + ((sb * 3 : U32) + (bcol * 2 : U32) : U32) : U32)))
def gr_28(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, +y2_2: F32, +y2_3: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y3_0}:
      gr_29(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, y2_3, y3_0, a, Array.get(F32, b, (ib + ((sb * 3 : U32) + (bcol * 1 : U32) : U32) : U32)))
def gr_27(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, +y2_2: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y2_3}:
      gr_28(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, y2_3, a, Array.get(F32, b, (ib + ((sb * 3 : U32) + (bcol * 0 : U32) : U32) : U32)))
def gr_26(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y2_2}:
      gr_27(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, a, Array.get(F32, b, (ib + ((sb * 2 : U32) + (bcol * 3 : U32) : U32) : U32)))
def gr_25(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y2_1}:
      gr_26(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, a, Array.get(F32, b, (ib + ((sb * 2 : U32) + (bcol * 2 : U32) : U32) : U32)))
def gr_24(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y2_0}:
      gr_25(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, a, Array.get(F32, b, (ib + ((sb * 2 : U32) + (bcol * 1 : U32) : U32) : U32)))
def gr_23(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y1_3}:
      gr_24(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, a, Array.get(F32, b, (ib + ((sb * 2 : U32) + (bcol * 0 : U32) : U32) : U32)))
def gr_22(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y1_2}:
      gr_23(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, a, Array.get(F32, b, (ib + ((sb * 1 : U32) + (bcol * 3 : U32) : U32) : U32)))
def gr_21(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y1_1}:
      gr_22(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, a, Array.get(F32, b, (ib + ((sb * 1 : U32) + (bcol * 2 : U32) : U32) : U32)))
def gr_20(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y1_0}:
      gr_21(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, a, Array.get(F32, b, (ib + ((sb * 1 : U32) + (bcol * 1 : U32) : U32) : U32)))
def gr_19(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y0_3}:
      gr_20(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, a, Array.get(F32, b, (ib + ((sb * 1 : U32) + (bcol * 0 : U32) : U32) : U32)))
def gr_18(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y0_2}:
      gr_19(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, a, Array.get(F32, b, (ib + ((sb * 0 : U32) + (bcol * 3 : U32) : U32) : U32)))
def gr_17(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y0_1}:
      gr_18(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, a, Array.get(F32, b, (ib + ((sb * 0 : U32) + (bcol * 2 : U32) : U32) : U32)))
def gr_16(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y0_0}:
      gr_17(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, a, Array.get(F32, b, (ib + ((sb * 0 : U32) + (bcol * 1 : U32) : U32) : U32)))
def gr_15(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x3_3}:
      gr_16(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, a, Array.get(F32, b, (ib + ((sb * 0 : U32) + (bcol * 0 : U32) : U32) : U32)))
def gr_14(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x3_2}:
      gr_15(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, Array.get(F32, a, (ia + ((sa * 3 : U32) + (arow * 3 : U32) : U32) : U32)), b)
def gr_13(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x3_1}:
      gr_14(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, Array.get(F32, a, (ia + ((sa * 3 : U32) + (arow * 2 : U32) : U32) : U32)), b)
def gr_12(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x3_0}:
      gr_13(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, Array.get(F32, a, (ia + ((sa * 3 : U32) + (arow * 1 : U32) : U32) : U32)), b)
def gr_11(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x2_3}:
      gr_12(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, Array.get(F32, a, (ia + ((sa * 3 : U32) + (arow * 0 : U32) : U32) : U32)), b)
def gr_10(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x2_2}:
      gr_11(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, Array.get(F32, a, (ia + ((sa * 2 : U32) + (arow * 3 : U32) : U32) : U32)), b)
def gr_9(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x2_1}:
      gr_10(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, Array.get(F32, a, (ia + ((sa * 2 : U32) + (arow * 2 : U32) : U32) : U32)), b)
def gr_8(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x2_0}:
      gr_9(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, Array.get(F32, a, (ia + ((sa * 2 : U32) + (arow * 1 : U32) : U32) : U32)), b)
def gr_7(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x1_3}:
      gr_8(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, Array.get(F32, a, (ia + ((sa * 2 : U32) + (arow * 0 : U32) : U32) : U32)), b)
def gr_6(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x1_2}:
      gr_7(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, Array.get(F32, a, (ia + ((sa * 1 : U32) + (arow * 3 : U32) : U32) : U32)), b)
def gr_5(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x1_1}:
      gr_6(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, Array.get(F32, a, (ia + ((sa * 1 : U32) + (arow * 2 : U32) : U32) : U32)), b)
def gr_4(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x1_0}:
      gr_5(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, Array.get(F32, a, (ia + ((sa * 1 : U32) + (arow * 1 : U32) : U32) : U32)), b)
def gr_3(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x0_3}:
      gr_4(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, Array.get(F32, a, (ia + ((sa * 1 : U32) + (arow * 0 : U32) : U32) : U32)), b)
def gr_2(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x0_2}:
      gr_3(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, Array.get(F32, a, (ia + ((sa * 0 : U32) + (arow * 3 : U32) : U32) : U32)), b)
def gr_1(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x0_1}:
      gr_2(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, Array.get(F32, a, (ia + ((sa * 0 : U32) + (arow * 2 : U32) : U32) : U32)), b)
def gr_0(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x0_0}:
      gr_1(ia, ib, arow, sa, sb, bcol, x0_0, Array.get(F32, a, (ia + ((sa * 0 : U32) + (arow * 1 : U32) : U32) : U32)), b)
# the turn whose A values start at ia and B values at ib
def turn(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>) -> GR:
  gr_0(ia, ib, arow, sa, sb, bcol, Array.get(F32, a, ia), b)

# the tile's cells after the four sums are combined, row by row
type T is Type:
  T{a: Array<F32>, b: Array<F32>, c0_0: F32, c0_1: F32, c0_2: F32, c0_3: F32, c1_0: F32, c1_1: F32, c1_2: F32, c1_3: F32, c2_0: F32, c2_1: F32, c2_2: F32, c2_3: F32, c3_0: F32, c3_1: F32, c3_2: F32, c3_3: F32}

# kg more full turns; g holds the current turn's values. The last call (kg = 0) adds the turn that holds the steps
# past the last full turn (rem = k mod 4 of them), counting only those, and combines the sums.
@unsafe
def mm(kg: Nat, +ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +rem: U32, +rot: Nat, +s0_0_0: F32, +s1_0_0: F32, +s2_0_0: F32, +s3_0_0: F32, +s0_0_1: F32, +s1_0_1: F32, +s2_0_1: F32, +s3_0_1: F32, +s0_0_2: F32, +s1_0_2: F32, +s2_0_2: F32, +s3_0_2: F32, +s0_0_3: F32, +s1_0_3: F32, +s2_0_3: F32, +s3_0_3: F32, +s0_1_0: F32, +s1_1_0: F32, +s2_1_0: F32, +s3_1_0: F32, +s0_1_1: F32, +s1_1_1: F32, +s2_1_1: F32, +s3_1_1: F32, +s0_1_2: F32, +s1_1_2: F32, +s2_1_2: F32, +s3_1_2: F32, +s0_1_3: F32, +s1_1_3: F32, +s2_1_3: F32, +s3_1_3: F32, +s0_2_0: F32, +s1_2_0: F32, +s2_2_0: F32, +s3_2_0: F32, +s0_2_1: F32, +s1_2_1: F32, +s2_2_1: F32, +s3_2_1: F32, +s0_2_2: F32, +s1_2_2: F32, +s2_2_2: F32, +s3_2_2: F32, +s0_2_3: F32, +s1_2_3: F32, +s2_2_3: F32, +s3_2_3: F32, +s0_3_0: F32, +s1_3_0: F32, +s2_3_0: F32, +s3_3_0: F32, +s0_3_1: F32, +s1_3_1: F32, +s2_3_1: F32, +s3_3_1: F32, +s0_3_2: F32, +s1_3_2: F32, +s2_3_2: F32, +s3_3_2: F32, +s0_3_3: F32, +s1_3_3: F32, +s2_3_3: F32, +s3_3_3: F32, g: GR) -> T:
  match kg g:
    case 0n GR{a, b, +x0_0, +x0_1, +x0_2, +x0_3, +x1_0, +x1_1, +x1_2, +x1_3, +x2_0, +x2_1, +x2_2, +x2_3, +x3_0, +x3_1, +x3_2, +x3_3, +y0_0, +y0_1, +y0_2, +y0_3, +y1_0, +y1_1, +y1_2, +y1_3, +y2_0, +y2_1, +y2_2, +y2_3, +y3_0, +y3_1, +y3_2, +y3_3}:
      T{a, b, TA.t8_comb(rot, (s0_0_0 + pk(U32.is_lt(0, rem), (x0_0 * y0_0 : F32)) : F32), (s1_0_0 + pk(U32.is_lt(1, rem), (x1_0 * y1_0 : F32)) : F32), (s2_0_0 + pk(U32.is_lt(2, rem), (x2_0 * y2_0 : F32)) : F32), (s3_0_0 + pk(U32.is_lt(3, rem), (x3_0 * y3_0 : F32)) : F32)), TA.t8_comb(rot, (s0_0_1 + pk(U32.is_lt(0, rem), (x0_0 * y0_1 : F32)) : F32), (s1_0_1 + pk(U32.is_lt(1, rem), (x1_0 * y1_1 : F32)) : F32), (s2_0_1 + pk(U32.is_lt(2, rem), (x2_0 * y2_1 : F32)) : F32), (s3_0_1 + pk(U32.is_lt(3, rem), (x3_0 * y3_1 : F32)) : F32)), TA.t8_comb(rot, (s0_0_2 + pk(U32.is_lt(0, rem), (x0_0 * y0_2 : F32)) : F32), (s1_0_2 + pk(U32.is_lt(1, rem), (x1_0 * y1_2 : F32)) : F32), (s2_0_2 + pk(U32.is_lt(2, rem), (x2_0 * y2_2 : F32)) : F32), (s3_0_2 + pk(U32.is_lt(3, rem), (x3_0 * y3_2 : F32)) : F32)), TA.t8_comb(rot, (s0_0_3 + pk(U32.is_lt(0, rem), (x0_0 * y0_3 : F32)) : F32), (s1_0_3 + pk(U32.is_lt(1, rem), (x1_0 * y1_3 : F32)) : F32), (s2_0_3 + pk(U32.is_lt(2, rem), (x2_0 * y2_3 : F32)) : F32), (s3_0_3 + pk(U32.is_lt(3, rem), (x3_0 * y3_3 : F32)) : F32)), TA.t8_comb(rot, (s0_1_0 + pk(U32.is_lt(0, rem), (x0_1 * y0_0 : F32)) : F32), (s1_1_0 + pk(U32.is_lt(1, rem), (x1_1 * y1_0 : F32)) : F32), (s2_1_0 + pk(U32.is_lt(2, rem), (x2_1 * y2_0 : F32)) : F32), (s3_1_0 + pk(U32.is_lt(3, rem), (x3_1 * y3_0 : F32)) : F32)), TA.t8_comb(rot, (s0_1_1 + pk(U32.is_lt(0, rem), (x0_1 * y0_1 : F32)) : F32), (s1_1_1 + pk(U32.is_lt(1, rem), (x1_1 * y1_1 : F32)) : F32), (s2_1_1 + pk(U32.is_lt(2, rem), (x2_1 * y2_1 : F32)) : F32), (s3_1_1 + pk(U32.is_lt(3, rem), (x3_1 * y3_1 : F32)) : F32)), TA.t8_comb(rot, (s0_1_2 + pk(U32.is_lt(0, rem), (x0_1 * y0_2 : F32)) : F32), (s1_1_2 + pk(U32.is_lt(1, rem), (x1_1 * y1_2 : F32)) : F32), (s2_1_2 + pk(U32.is_lt(2, rem), (x2_1 * y2_2 : F32)) : F32), (s3_1_2 + pk(U32.is_lt(3, rem), (x3_1 * y3_2 : F32)) : F32)), TA.t8_comb(rot, (s0_1_3 + pk(U32.is_lt(0, rem), (x0_1 * y0_3 : F32)) : F32), (s1_1_3 + pk(U32.is_lt(1, rem), (x1_1 * y1_3 : F32)) : F32), (s2_1_3 + pk(U32.is_lt(2, rem), (x2_1 * y2_3 : F32)) : F32), (s3_1_3 + pk(U32.is_lt(3, rem), (x3_1 * y3_3 : F32)) : F32)), TA.t8_comb(rot, (s0_2_0 + pk(U32.is_lt(0, rem), (x0_2 * y0_0 : F32)) : F32), (s1_2_0 + pk(U32.is_lt(1, rem), (x1_2 * y1_0 : F32)) : F32), (s2_2_0 + pk(U32.is_lt(2, rem), (x2_2 * y2_0 : F32)) : F32), (s3_2_0 + pk(U32.is_lt(3, rem), (x3_2 * y3_0 : F32)) : F32)), TA.t8_comb(rot, (s0_2_1 + pk(U32.is_lt(0, rem), (x0_2 * y0_1 : F32)) : F32), (s1_2_1 + pk(U32.is_lt(1, rem), (x1_2 * y1_1 : F32)) : F32), (s2_2_1 + pk(U32.is_lt(2, rem), (x2_2 * y2_1 : F32)) : F32), (s3_2_1 + pk(U32.is_lt(3, rem), (x3_2 * y3_1 : F32)) : F32)), TA.t8_comb(rot, (s0_2_2 + pk(U32.is_lt(0, rem), (x0_2 * y0_2 : F32)) : F32), (s1_2_2 + pk(U32.is_lt(1, rem), (x1_2 * y1_2 : F32)) : F32), (s2_2_2 + pk(U32.is_lt(2, rem), (x2_2 * y2_2 : F32)) : F32), (s3_2_2 + pk(U32.is_lt(3, rem), (x3_2 * y3_2 : F32)) : F32)), TA.t8_comb(rot, (s0_2_3 + pk(U32.is_lt(0, rem), (x0_2 * y0_3 : F32)) : F32), (s1_2_3 + pk(U32.is_lt(1, rem), (x1_2 * y1_3 : F32)) : F32), (s2_2_3 + pk(U32.is_lt(2, rem), (x2_2 * y2_3 : F32)) : F32), (s3_2_3 + pk(U32.is_lt(3, rem), (x3_2 * y3_3 : F32)) : F32)), TA.t8_comb(rot, (s0_3_0 + pk(U32.is_lt(0, rem), (x0_3 * y0_0 : F32)) : F32), (s1_3_0 + pk(U32.is_lt(1, rem), (x1_3 * y1_0 : F32)) : F32), (s2_3_0 + pk(U32.is_lt(2, rem), (x2_3 * y2_0 : F32)) : F32), (s3_3_0 + pk(U32.is_lt(3, rem), (x3_3 * y3_0 : F32)) : F32)), TA.t8_comb(rot, (s0_3_1 + pk(U32.is_lt(0, rem), (x0_3 * y0_1 : F32)) : F32), (s1_3_1 + pk(U32.is_lt(1, rem), (x1_3 * y1_1 : F32)) : F32), (s2_3_1 + pk(U32.is_lt(2, rem), (x2_3 * y2_1 : F32)) : F32), (s3_3_1 + pk(U32.is_lt(3, rem), (x3_3 * y3_1 : F32)) : F32)), TA.t8_comb(rot, (s0_3_2 + pk(U32.is_lt(0, rem), (x0_3 * y0_2 : F32)) : F32), (s1_3_2 + pk(U32.is_lt(1, rem), (x1_3 * y1_2 : F32)) : F32), (s2_3_2 + pk(U32.is_lt(2, rem), (x2_3 * y2_2 : F32)) : F32), (s3_3_2 + pk(U32.is_lt(3, rem), (x3_3 * y3_2 : F32)) : F32)), TA.t8_comb(rot, (s0_3_3 + pk(U32.is_lt(0, rem), (x0_3 * y0_3 : F32)) : F32), (s1_3_3 + pk(U32.is_lt(1, rem), (x1_3 * y1_3 : F32)) : F32), (s2_3_3 + pk(U32.is_lt(2, rem), (x2_3 * y2_3 : F32)) : F32), (s3_3_3 + pk(U32.is_lt(3, rem), (x3_3 * y3_3 : F32)) : F32))}
    case 1n+q GR{a, b, +x0_0, +x0_1, +x0_2, +x0_3, +x1_0, +x1_1, +x1_2, +x1_3, +x2_0, +x2_1, +x2_2, +x2_3, +x3_0, +x3_1, +x3_2, +x3_3, +y0_0, +y0_1, +y0_2, +y0_3, +y1_0, +y1_1, +y1_2, +y1_3, +y2_0, +y2_1, +y2_2, +y2_3, +y3_0, +y3_1, +y3_2, +y3_3}:
      mm(q, (ia + (sa * 4 : U32) : U32), (ib + (sb * 4 : U32) : U32), arow, sa, sb, bcol, rem, rot, (s0_0_0 + (x0_0 * y0_0 : F32) : F32), (s1_0_0 + (x1_0 * y1_0 : F32) : F32), (s2_0_0 + (x2_0 * y2_0 : F32) : F32), (s3_0_0 + (x3_0 * y3_0 : F32) : F32), (s0_0_1 + (x0_0 * y0_1 : F32) : F32), (s1_0_1 + (x1_0 * y1_1 : F32) : F32), (s2_0_1 + (x2_0 * y2_1 : F32) : F32), (s3_0_1 + (x3_0 * y3_1 : F32) : F32), (s0_0_2 + (x0_0 * y0_2 : F32) : F32), (s1_0_2 + (x1_0 * y1_2 : F32) : F32), (s2_0_2 + (x2_0 * y2_2 : F32) : F32), (s3_0_2 + (x3_0 * y3_2 : F32) : F32), (s0_0_3 + (x0_0 * y0_3 : F32) : F32), (s1_0_3 + (x1_0 * y1_3 : F32) : F32), (s2_0_3 + (x2_0 * y2_3 : F32) : F32), (s3_0_3 + (x3_0 * y3_3 : F32) : F32), (s0_1_0 + (x0_1 * y0_0 : F32) : F32), (s1_1_0 + (x1_1 * y1_0 : F32) : F32), (s2_1_0 + (x2_1 * y2_0 : F32) : F32), (s3_1_0 + (x3_1 * y3_0 : F32) : F32), (s0_1_1 + (x0_1 * y0_1 : F32) : F32), (s1_1_1 + (x1_1 * y1_1 : F32) : F32), (s2_1_1 + (x2_1 * y2_1 : F32) : F32), (s3_1_1 + (x3_1 * y3_1 : F32) : F32), (s0_1_2 + (x0_1 * y0_2 : F32) : F32), (s1_1_2 + (x1_1 * y1_2 : F32) : F32), (s2_1_2 + (x2_1 * y2_2 : F32) : F32), (s3_1_2 + (x3_1 * y3_2 : F32) : F32), (s0_1_3 + (x0_1 * y0_3 : F32) : F32), (s1_1_3 + (x1_1 * y1_3 : F32) : F32), (s2_1_3 + (x2_1 * y2_3 : F32) : F32), (s3_1_3 + (x3_1 * y3_3 : F32) : F32), (s0_2_0 + (x0_2 * y0_0 : F32) : F32), (s1_2_0 + (x1_2 * y1_0 : F32) : F32), (s2_2_0 + (x2_2 * y2_0 : F32) : F32), (s3_2_0 + (x3_2 * y3_0 : F32) : F32), (s0_2_1 + (x0_2 * y0_1 : F32) : F32), (s1_2_1 + (x1_2 * y1_1 : F32) : F32), (s2_2_1 + (x2_2 * y2_1 : F32) : F32), (s3_2_1 + (x3_2 * y3_1 : F32) : F32), (s0_2_2 + (x0_2 * y0_2 : F32) : F32), (s1_2_2 + (x1_2 * y1_2 : F32) : F32), (s2_2_2 + (x2_2 * y2_2 : F32) : F32), (s3_2_2 + (x3_2 * y3_2 : F32) : F32), (s0_2_3 + (x0_2 * y0_3 : F32) : F32), (s1_2_3 + (x1_2 * y1_3 : F32) : F32), (s2_2_3 + (x2_2 * y2_3 : F32) : F32), (s3_2_3 + (x3_2 * y3_3 : F32) : F32), (s0_3_0 + (x0_3 * y0_0 : F32) : F32), (s1_3_0 + (x1_3 * y1_0 : F32) : F32), (s2_3_0 + (x2_3 * y2_0 : F32) : F32), (s3_3_0 + (x3_3 * y3_0 : F32) : F32), (s0_3_1 + (x0_3 * y0_1 : F32) : F32), (s1_3_1 + (x1_3 * y1_1 : F32) : F32), (s2_3_1 + (x2_3 * y2_1 : F32) : F32), (s3_3_1 + (x3_3 * y3_1 : F32) : F32), (s0_3_2 + (x0_3 * y0_2 : F32) : F32), (s1_3_2 + (x1_3 * y1_2 : F32) : F32), (s2_3_2 + (x2_3 * y2_2 : F32) : F32), (s3_3_2 + (x3_3 * y3_2 : F32) : F32), (s0_3_3 + (x0_3 * y0_3 : F32) : F32), (s1_3_3 + (x1_3 * y1_3 : F32) : F32), (s2_3_3 + (x2_3 * y2_3 : F32) : F32), (s3_3_3 + (x3_3 * y3_3 : F32) : F32), turn((ia + (sa * 4 : U32) : U32), (ib + (sb * 4 : U32) : U32), arow, sa, sb, bcol, a, b))

# writes cell (i, j) of C when it is inside n x m
def wr(ok: Bool, +ac: Bool, +idx: U32, +x: F32, c: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: TA.cwr(ac, idx, x, c)
    case False{}: c

# the tile at rows i0.., columns j0.. of C, from the loop's result; A and B are dropped (the caller holds them)
@unsafe
def put(+ac: Bool, +n: U32, +m: U32, +i0: U32, +j0: U32, c: Array<F32>, t: T) -> Array<F32>:
  match t:
    case T{a, b, +c0_0, +c0_1, +c0_2, +c0_3, +c1_0, +c1_1, +c1_2, +c1_3, +c2_0, +c2_1, +c2_2, +c2_3, +c3_0, +c3_1, +c3_2, +c3_3}:
      wr(Bool.and(U32.is_lt((i0 + 3 : U32), n), U32.is_lt((j0 + 3 : U32), m)), ac, (((i0 + 3 : U32) * m : U32) + (j0 + 3 : U32) : U32), c3_3, wr(Bool.and(U32.is_lt((i0 + 3 : U32), n), U32.is_lt((j0 + 2 : U32), m)), ac, (((i0 + 3 : U32) * m : U32) + (j0 + 2 : U32) : U32), c3_2, wr(Bool.and(U32.is_lt((i0 + 3 : U32), n), U32.is_lt((j0 + 1 : U32), m)), ac, (((i0 + 3 : U32) * m : U32) + (j0 + 1 : U32) : U32), c3_1, wr(Bool.and(U32.is_lt((i0 + 3 : U32), n), U32.is_lt((j0 + 0 : U32), m)), ac, (((i0 + 3 : U32) * m : U32) + (j0 + 0 : U32) : U32), c3_0, wr(Bool.and(U32.is_lt((i0 + 2 : U32), n), U32.is_lt((j0 + 3 : U32), m)), ac, (((i0 + 2 : U32) * m : U32) + (j0 + 3 : U32) : U32), c2_3, wr(Bool.and(U32.is_lt((i0 + 2 : U32), n), U32.is_lt((j0 + 2 : U32), m)), ac, (((i0 + 2 : U32) * m : U32) + (j0 + 2 : U32) : U32), c2_2, wr(Bool.and(U32.is_lt((i0 + 2 : U32), n), U32.is_lt((j0 + 1 : U32), m)), ac, (((i0 + 2 : U32) * m : U32) + (j0 + 1 : U32) : U32), c2_1, wr(Bool.and(U32.is_lt((i0 + 2 : U32), n), U32.is_lt((j0 + 0 : U32), m)), ac, (((i0 + 2 : U32) * m : U32) + (j0 + 0 : U32) : U32), c2_0, wr(Bool.and(U32.is_lt((i0 + 1 : U32), n), U32.is_lt((j0 + 3 : U32), m)), ac, (((i0 + 1 : U32) * m : U32) + (j0 + 3 : U32) : U32), c1_3, wr(Bool.and(U32.is_lt((i0 + 1 : U32), n), U32.is_lt((j0 + 2 : U32), m)), ac, (((i0 + 1 : U32) * m : U32) + (j0 + 2 : U32) : U32), c1_2, wr(Bool.and(U32.is_lt((i0 + 1 : U32), n), U32.is_lt((j0 + 1 : U32), m)), ac, (((i0 + 1 : U32) * m : U32) + (j0 + 1 : U32) : U32), c1_1, wr(Bool.and(U32.is_lt((i0 + 1 : U32), n), U32.is_lt((j0 + 0 : U32), m)), ac, (((i0 + 1 : U32) * m : U32) + (j0 + 0 : U32) : U32), c1_0, wr(Bool.and(U32.is_lt((i0 + 0 : U32), n), U32.is_lt((j0 + 3 : U32), m)), ac, (((i0 + 0 : U32) * m : U32) + (j0 + 3 : U32) : U32), c0_3, wr(Bool.and(U32.is_lt((i0 + 0 : U32), n), U32.is_lt((j0 + 2 : U32), m)), ac, (((i0 + 0 : U32) * m : U32) + (j0 + 2 : U32) : U32), c0_2, wr(Bool.and(U32.is_lt((i0 + 0 : U32), n), U32.is_lt((j0 + 1 : U32), m)), ac, (((i0 + 0 : U32) * m : U32) + (j0 + 1 : U32) : U32), c0_1, wr(Bool.and(U32.is_lt((i0 + 0 : U32), n), U32.is_lt((j0 + 0 : U32), m)), ac, (((i0 + 0 : U32) * m : U32) + (j0 + 0 : U32) : U32), c0_0, c))))))))))))))))

@unsafe
def tile(+e: E, +i0: U32, +j0: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match e:
    case E{+n, +m, +k, +tcols, +tiles, +arow, +sa, +sb, +bcol, +kg, +rot, +rem, +ac, +oa, +ob}:
      put(ac, n, m, i0, j0, c, mm(kg, (oa + (i0 * arow : U32) : U32), (ob + (j0 * bcol : U32) : U32), arow, sa, sb, bcol, rem, rot, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, turn((oa + (i0 * arow : U32) : U32), (ob + (j0 * bcol : U32) : U32), arow, sa, sb, bcol, a, b)))

# leaf t computes tile t (row-major over the tiles); leaves past the last tile write nothing
@unsafe
def leaf_if(ok: Bool, +e: E, +t: U32, +tcols: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: tile(e, (U32.div(t, tcols) * 4 : U32), (U32.mod(t, tcols) * 4 : U32), a, b, c)
    case False{}: c

@unsafe
def leaf(+e: E, +t: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match e:
    case E{+n, +m, +k, +tcols, +tiles, +arow, +sa, +sb, +bcol, +kg, +rot, +rem, +ac, +oa, +ob}:
      leaf_if(U32.is_lt(t, tiles), e, t, tcols, a, b, c)

@unsafe
def tree(+e: E, d: Nat, +t: U32, +a: Array<F32>, +b: Array<F32>, +c: Array<F32>) -> Array<F32>:
  match d:
    case 0n: leaf(e, t, a, b, c)
    case 1n++q:
      l r = tree(e, q, (t * 2 : U32), a, b, c) tree(e, q, ((t * 2 : U32) + 1 : U32), a, b, c)
      Array.join(F32, l, r)

# C made inside the bang: the root of the ! call allocates it on the device. Made on the host instead, a new C often
# reuses a block whose pages sit on the device, so zeroing it on the host pulls them back over PCIe and the bang moves
# them again (a 256 x 1536 product: 9.2 ms against 6.0, measured 2026-10-18). The leaves write every cell of C.
@unsafe
def tree_new(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, +cd: Nat) -> Array<F32>:
  tree(e, d, 0, a, b, Array.new(F32, cd, 0.0))

@unsafe
def bang_new(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, +cd: Nat) -> Array<F32>:
  tree_new!(e, d, a, b, cd)

# one bang over the tiles; A and B are shared with every leaf and come back unchanged
@unsafe
def bang(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  tree!(e, d, 0, a, b, c)

type ABC is Type:
  ABC{a: Array<F32>, b: Array<F32>, c: Array<F32>}

@unsafe
def run_e(+e: E, +tiles: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> ABC:
  ABC{a, b, bang(e, TA.cap_depth(U32.to_nat(tiles)), a, b, c)}

# the product's sizes and strides
def e_ofo(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +oa: U32, +ob: U32) -> E:
  E{U32.from_nat(n), U32.from_nat(m), U32.from_nat(k), U32.from_nat(Nat.div(Nat.add(m, 3n), 4n)), U32.from_nat(Nat.mul(Nat.div(Nat.add(n, 3n), 4n), Nat.div(Nat.add(m, 3n), 4n))), arow, sa, sb, bcol, Nat.div(k, 4n), Nat.mod(k, 4n), U32.from_nat(Nat.mod(k, 4n)), ac, oa, ob}

def e_of(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32) -> E:
  e_ofo(ac, n, m, k, arow, sa, sb, bcol, 0, 0)

def tiles(+n: Nat, +m: Nat) -> Nat:
  Nat.mul(Nat.div(Nat.add(n, 3n), 4n), Nat.div(Nat.add(m, 3n), 4n))

# C (n x m) = A · B with k steps into a new C made on the device
@unsafe
def gemm_new(+n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +a: Array<F32>, +b: Array<F32>) -> ABC:
  ABC{a, b, bang_new(e_of(False{}, n, m, k, arow, sa, sb, bcol), TA.cap_depth(tiles(n, m)), a, b, TA.cap_depth(Nat.mul(n, m)))}

# C (n x m) = A · B with k steps, or C + A · B when ac, C[i][j] = sum_p A[i*arow + p*sa] * B[p*sb + j*bcol]
@unsafe
def gemm(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> ABC:
  run_e(e_of(ac, n, m, k, arow, sa, sb, bcol), U32.from_nat(tiles(n, m)), a, b, c)

def mm_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMul<n, k, m>:
  match g:
    case ABC{a, b, c}: TA.MMul{TA.Mat{a}, TA.Mat{b}, TA.Mat{c}}

def mm_nt_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMulNT<n, k, m>:
  match g:
    case ABC{a, b, c}: TA.MMulNT{TA.Mat{a}, TA.Mat{b}, TA.Mat{c}}

def mm_tn_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMulTN<n, k, m>:
  match g:
    case ABC{a, b, c}: TA.MMulTN{TA.Mat{a}, TA.Mat{b}, TA.Mat{c}}

# C = A · B   (n x k) · (k x m), tensor-array's TA.Mat.matmul bits
@unsafe
def matmul(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      mm_fin(n, k, m, gemm_new(n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y))

# C + A · B, into C (n x m)
@unsafe
def matmul_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMul<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      mm_fin(n, k, m, gemm(True{}, n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y, z))

# C = A · Bᵀ   (n x k) · (m x k)ᵀ: for weights stored with one row per output
@unsafe
def matmul_nt(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>) -> TA.MMulNT<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      mm_nt_fin(n, k, m, gemm_new(n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y))

# C + A · Bᵀ, into C (n x m)
@unsafe
def matmul_nt_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>, c: TA.Mat<n, m>) -> TA.MMulNT<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      mm_nt_fin(n, k, m, gemm(True{}, n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y, z))

# C = Aᵀ · B   (k x n)ᵀ · (k x m): weight gradient, Xᵀ · dY
@unsafe
def matmul_tn(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>) -> TA.MMulTN<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      mm_tn_fin(n, k, m, gemm_new(n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y))

# C + Aᵀ · B, into C (n x m): dW += Xᵀ · dY
@unsafe
def matmul_tn_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMulTN<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      mm_tn_fin(n, k, m, gemm(True{}, n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y, z))

# =====================================================================================================================
# Bangs over any leaf code (0.1.1.0). runK and runK_new (K = 1..4 outputs) run one bang over 2^d leaves: leaf t of
# l = 2^d gets t, l, a record e of scalars (the same for every leaf), four input arrays and the K outputs, and returns
# the outputs. The leaf code is a template (~f), written by the caller in plain Bend: it takes every array as an ordinary
# argument (it reads them with Array.get and may write the outputs), so it needs no @unsafe; the sharing is here, where
# every node of the tree holds a handle to each array (Array.fork in Base, upstream's one-block pattern) and joins the
# outputs (Array.join). runK takes the outputs from the caller and writes them in place (an optimizer's weights and
# moments); runK_new makes them inside the bang with 2^c1.. cells, since an array made on the host pulls device pages
# back (see bang_new above). Both return the inputs unchanged with the outputs (RK). An unused input is any array, such
# as none(Unit{}).
#
# Not checked, as for the products: that a leaf writes only its own cells of the outputs (two leaves writing one cell
# race) and never an input. Leaves past the work must return their outputs unchanged; share(t, n, l) is how many of the
# cells t, t + l, t + 2l, .. lie below n (leaf t's cells when n cells are dealt round-robin, so neighbouring lanes read
# neighbouring cells), and depth(n) is a tree for n cells: 2^14 leaves once there are that many, else the fewest that
# cover them. The leaves run in a run-time order inside each group of 128 (bendlang/bend's CUDA lowering), so a leaf's
# result must not depend on another's. Every number a leaf computes is the caller's own code, so the bits are the
# caller's: the same on the GPU as on the CPU threads, except where CUDA's math functions differ from the host's (an exp
# on the device takes the float overload; the known limit of bendlang/bend#1305).
# =====================================================================================================================

def depth2(big: Bool, +n: Nat) -> Nat:
  match big:
    case True{}: 14n
    case False{}: TA.cap_depth(n)

# leaves for n cells: 2^14 when there are that many, else the fewest powers of two that cover them
def depth(+n: Nat) -> Nat:
  depth2(Nat.is_le(16384n, n), n)

# how many of the cells t, t + l, t + 2l, .. lie below n
def share(+t: U32, +n: U32, +l: U32) -> Nat:
  U32.to_nat(Bool.pick(U32, U32.is_lt(t, n), U32.div(((n - t : U32) + (l - 1 : U32) : U32), l), 0))

# an array of one cell, for an unused input
def none(u: Unit) -> Array<F32>:
  Array.new(F32, 0n, 0.0)

type O2 is Type:
  O2{a: Array<F32>, b: Array<F32>}

type O3 is Type:
  O3{a: Array<F32>, b: Array<F32>, c: Array<F32>}

type O4 is Type:
  O4{a: Array<F32>, b: Array<F32>, c: Array<F32>, d: Array<F32>}

type R1 is Type:
  R1{x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, a: Array<F32>}

type R2 is Type:
  R2{x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: O2}

type R3 is Type:
  R3{x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: O3}

type R4 is Type:
  R4{x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: O4}

@unsafe
def join2(x: O2, y: O2) -> O2:
  match x y:
    case O2{a, b} O2{a2, b2}: O2{Array.join(F32, a, a2), Array.join(F32, b, b2)}

@unsafe
def join3(x: O3, y: O3) -> O3:
  match x y:
    case O3{a, b, c} O3{a2, b2, c2}: O3{Array.join(F32, a, a2), Array.join(F32, b, b2), Array.join(F32, c, c2)}

@unsafe
def join4(x: O4, y: O4) -> O4:
  match x y:
    case O4{a, b, c, d} O4{a2, b2, c2, d2}: O4{Array.join(F32, a, a2), Array.join(F32, b, b2), Array.join(F32, c, c2), Array.join(F32, d, d2)}

@unsafe
def tree1(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32>, +d: Nat, +t: U32, +l: U32, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +oa: Array<F32>) -> Array<F32>:
  match d:
    case 0n: f(t, l, e, x1, x2, x3, x4, oa)
    case 1n+q:
      x y = tree1(~E, ~f, q, (t * 2 : U32), l, e, x1, x2, x3, x4, oa) tree1(~E, ~f, q, ((t * 2 : U32) + 1 : U32), l, e, x1, x2, x3, x4, oa)
      Array.join(F32, x, y)

@unsafe
def root1(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32>, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, oa: Array<F32>) -> Array<F32>:
  tree1(~E, ~f, d, 0, U32.pow(2, d), e, x1, x2, x3, x4, oa)

@unsafe
def root1_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32>, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +ca: Nat) -> Array<F32>:
  root1(~E, ~f, d, e, x1, x2, x3, x4, Array.new(F32, ca, 0.0))

@unsafe
def go1(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32>, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, oa: Array<F32>) -> R1:
  R1{x1, x2, x3, x4, root1!(~E, ~f, d, e, x1, x2, x3, x4, oa)}

@unsafe
def go1_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32>, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +ca: Nat) -> R1:
  R1{x1, x2, x3, x4, root1_new!(~E, ~f, d, e, x1, x2, x3, x4, ca)}

# one bang over 2^d leaves with 1 output written in place
@unsafe
def run1(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32>, +d: Nat, +e: E, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, oa: Array<F32>) -> R1:
  go1(~E, ~f, d, e, x1, x2, x3, x4, oa)

# one bang over 2^d leaves with 1 new output of 2^c.. cells, made on the device
@unsafe
def run1_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32>, +d: Nat, +e: E, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, +ca: Nat) -> R1:
  go1_new(~E, ~f, d, e, x1, x2, x3, x4, ca)

@unsafe
def tree2(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O2 -> O2, +d: Nat, +t: U32, +l: U32, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +oa: Array<F32>, +ob: Array<F32>) -> O2:
  match d:
    case 0n: f(t, l, e, x1, x2, x3, x4, O2{oa, ob})
    case 1n+q:
      x y = tree2(~E, ~f, q, (t * 2 : U32), l, e, x1, x2, x3, x4, oa, ob) tree2(~E, ~f, q, ((t * 2 : U32) + 1 : U32), l, e, x1, x2, x3, x4, oa, ob)
      join2(x, y)

@unsafe
def root2(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O2 -> O2, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, oa: Array<F32>, ob: Array<F32>) -> O2:
  tree2(~E, ~f, d, 0, U32.pow(2, d), e, x1, x2, x3, x4, oa, ob)

@unsafe
def root2_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O2 -> O2, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +ca: Nat, +cb: Nat) -> O2:
  root2(~E, ~f, d, e, x1, x2, x3, x4, Array.new(F32, ca, 0.0), Array.new(F32, cb, 0.0))

@unsafe
def go2(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O2 -> O2, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, oa: Array<F32>, ob: Array<F32>) -> R2:
  R2{x1, x2, x3, x4, root2!(~E, ~f, d, e, x1, x2, x3, x4, oa, ob)}

@unsafe
def go2_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O2 -> O2, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +ca: Nat, +cb: Nat) -> R2:
  R2{x1, x2, x3, x4, root2_new!(~E, ~f, d, e, x1, x2, x3, x4, ca, cb)}

# one bang over 2^d leaves with 2 outputs written in place
@unsafe
def run2(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O2 -> O2, +d: Nat, +e: E, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, oa: Array<F32>, ob: Array<F32>) -> R2:
  go2(~E, ~f, d, e, x1, x2, x3, x4, oa, ob)

# one bang over 2^d leaves with 2 new outputs of 2^c.. cells, made on the device
@unsafe
def run2_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O2 -> O2, +d: Nat, +e: E, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, +ca: Nat, +cb: Nat) -> R2:
  go2_new(~E, ~f, d, e, x1, x2, x3, x4, ca, cb)

@unsafe
def tree3(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O3 -> O3, +d: Nat, +t: U32, +l: U32, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +oa: Array<F32>, +ob: Array<F32>, +oc: Array<F32>) -> O3:
  match d:
    case 0n: f(t, l, e, x1, x2, x3, x4, O3{oa, ob, oc})
    case 1n+q:
      x y = tree3(~E, ~f, q, (t * 2 : U32), l, e, x1, x2, x3, x4, oa, ob, oc) tree3(~E, ~f, q, ((t * 2 : U32) + 1 : U32), l, e, x1, x2, x3, x4, oa, ob, oc)
      join3(x, y)

@unsafe
def root3(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O3 -> O3, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, oa: Array<F32>, ob: Array<F32>, oc: Array<F32>) -> O3:
  tree3(~E, ~f, d, 0, U32.pow(2, d), e, x1, x2, x3, x4, oa, ob, oc)

@unsafe
def root3_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O3 -> O3, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +ca: Nat, +cb: Nat, +cc: Nat) -> O3:
  root3(~E, ~f, d, e, x1, x2, x3, x4, Array.new(F32, ca, 0.0), Array.new(F32, cb, 0.0), Array.new(F32, cc, 0.0))

@unsafe
def go3(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O3 -> O3, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, oa: Array<F32>, ob: Array<F32>, oc: Array<F32>) -> R3:
  R3{x1, x2, x3, x4, root3!(~E, ~f, d, e, x1, x2, x3, x4, oa, ob, oc)}

@unsafe
def go3_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O3 -> O3, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +ca: Nat, +cb: Nat, +cc: Nat) -> R3:
  R3{x1, x2, x3, x4, root3_new!(~E, ~f, d, e, x1, x2, x3, x4, ca, cb, cc)}

# one bang over 2^d leaves with 3 outputs written in place
@unsafe
def run3(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O3 -> O3, +d: Nat, +e: E, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, oa: Array<F32>, ob: Array<F32>, oc: Array<F32>) -> R3:
  go3(~E, ~f, d, e, x1, x2, x3, x4, oa, ob, oc)

# one bang over 2^d leaves with 3 new outputs of 2^c.. cells, made on the device
@unsafe
def run3_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O3 -> O3, +d: Nat, +e: E, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, +ca: Nat, +cb: Nat, +cc: Nat) -> R3:
  go3_new(~E, ~f, d, e, x1, x2, x3, x4, ca, cb, cc)

@unsafe
def tree4(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O4 -> O4, +d: Nat, +t: U32, +l: U32, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +oa: Array<F32>, +ob: Array<F32>, +oc: Array<F32>, +od: Array<F32>) -> O4:
  match d:
    case 0n: f(t, l, e, x1, x2, x3, x4, O4{oa, ob, oc, od})
    case 1n+q:
      x y = tree4(~E, ~f, q, (t * 2 : U32), l, e, x1, x2, x3, x4, oa, ob, oc, od) tree4(~E, ~f, q, ((t * 2 : U32) + 1 : U32), l, e, x1, x2, x3, x4, oa, ob, oc, od)
      join4(x, y)

@unsafe
def root4(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O4 -> O4, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, oa: Array<F32>, ob: Array<F32>, oc: Array<F32>, od: Array<F32>) -> O4:
  tree4(~E, ~f, d, 0, U32.pow(2, d), e, x1, x2, x3, x4, oa, ob, oc, od)

@unsafe
def root4_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O4 -> O4, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +ca: Nat, +cb: Nat, +cc: Nat, +cd: Nat) -> O4:
  root4(~E, ~f, d, e, x1, x2, x3, x4, Array.new(F32, ca, 0.0), Array.new(F32, cb, 0.0), Array.new(F32, cc, 0.0), Array.new(F32, cd, 0.0))

@unsafe
def go4(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O4 -> O4, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, oa: Array<F32>, ob: Array<F32>, oc: Array<F32>, od: Array<F32>) -> R4:
  R4{x1, x2, x3, x4, root4!(~E, ~f, d, e, x1, x2, x3, x4, oa, ob, oc, od)}

@unsafe
def go4_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O4 -> O4, +d: Nat, +e: E, +x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>, +ca: Nat, +cb: Nat, +cc: Nat, +cd: Nat) -> R4:
  R4{x1, x2, x3, x4, root4_new!(~E, ~f, d, e, x1, x2, x3, x4, ca, cb, cc, cd)}

# one bang over 2^d leaves with 4 outputs written in place
@unsafe
def run4(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O4 -> O4, +d: Nat, +e: E, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, oa: Array<F32>, ob: Array<F32>, oc: Array<F32>, od: Array<F32>) -> R4:
  go4(~E, ~f, d, e, x1, x2, x3, x4, oa, ob, oc, od)

# one bang over 2^d leaves with 4 new outputs of 2^c.. cells, made on the device
@unsafe
def run4_new(~E: Data, ~f: @+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> O4 -> O4, +d: Nat, +e: E, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, +ca: Nat, +cb: Nat, +cc: Nat, +cd: Nat) -> R4:
  go4_new(~E, ~f, d, e, x1, x2, x3, x4, ca, cb, cc, cd)

# ---- tensor-array 0.1.14.0's precise product on the GPU (opt-in) ----
# TA.Mat.matmul_precise's sums, a bang a block of 1024 steps that starts at the block's place in A and B (oa, ob; no
# copy): the first block's bang makes C on the device, each next one adds its cells into C as the accumulating kernel
# does (once, as tensor-array's cwr), so every cell is ((x0 + x1) + x2) .., the same bits as tensor-array's
# (tests/precise.bend). Only plain (not accumulating) products, as tensor-array's matmul_precise.

def blk_len(+bi: U32, +k: U32) -> Nat:
  U32.to_nat(Bool.pick(U32, U32.is_lt((k - (bi * 1024 : U32) : U32), 1024), (k - (bi * 1024 : U32) : U32), 1024))

# the blocks after the first: nb more, from block bi, each added into C
@unsafe
def blocks_acc(nb: Nat, +bi: U32, +n: Nat, +m: Nat, +k: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match nb:
    case 0n: c
    case 1n++q: blocks_acc(q, (bi + 1 : U32), n, m, k, arow, sa, sb, bcol, a, b, bang(e_ofo(True{}, n, m, blk_len(bi, k), arow, sa, sb, bcol, ((bi * 1024 : U32) * sa : U32), ((bi * 1024 : U32) * sb : U32)), TA.cap_depth(tiles(n, m)), a, b, c))

@unsafe
def precise_c(+n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +a: Array<F32>, +b: Array<F32>) -> Array<F32>:
  blocks_acc(Nat.sub(U32.to_nat(U32.div((U32.from_nat(k) + 1023 : U32), 1024)), 1n), 1, n, m, U32.from_nat(k), arow, sa, sb, bcol, a, b, bang_new(e_ofo(False{}, n, m, 1024n, arow, sa, sb, bcol, 0, 0), TA.cap_depth(tiles(n, m)), a, b, TA.cap_depth(Nat.mul(n, m))))

@unsafe
def mm_precise3(+n: Nat, +k: Nat, +m: Nat, +a: Array<F32>, +b: Array<F32>) -> TA.MMul<n, k, m>:
  mm_fin(n, k, m, ABC{a, b, precise_c(n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, a, b)})

@unsafe
def mm_precise2(long: Bool, +n: Nat, +k: Nat, +m: Nat, +par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  match long a b:
    case True{} TA.Mat{x} TA.Mat{y}: mm_precise3(n, k, m, x, y)
    case False{} a b: matmul(n, k, m, par, a, b)

# C = A · B with k summed in blocks of 1024: TA.Mat.matmul_precise's bits (opt-in)
@unsafe
def matmul_precise(+n: Nat, +k: Nat, +m: Nat, +par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  mm_precise2(Nat.is_lt(1024n, k), n, k, m, par, a, b)
