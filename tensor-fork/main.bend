# bend-ml-tensor-fork: bend-ml-tensor-array's products, with the parallel blocks sharing their operands.
#
#   import bend-ml-tensor-array@0.1.12.0/main.bend as TA
#   import bend-ml-tensor-fork@0.1.0.0/main.bend as TF
#   TF.matmul(n, k, m, par, a, b)      # the same arguments, result type and bits as TA.Mat.matmul
#
# The six products (matmul, matmul_nt, matmul_tn and their accumulating forms) take tensor-array's Mat values and
# return its result types. A product that splits (tensor-array's split_depth, two or more rows) packs A and B once into
# cells of 8, as the sequential path does, and every node of the fork tree shares the two packed operands and C through
# Base's Array.fork: two handles to one block, no copy. Each leaf runs tensor-array's cells (t8_rows, t8_cols) straight
# into its own rows (or columns) of C. tensor-array's parallel path instead clones an operand at every node and hands
# each block's cells back as a list that is then written into C. A product that does not split, and a product with one
# row, runs tensor-array's sequential code. The bits are those of tensor-array at depth 0, at every depth and thread
# count (tests/bits.bend).
#
# Not proved, and marked so: Array.fork and Array.join are @unsafe in Base (two handles to one array break the single
# owner that the proofs rely on), so the defs below that call them are @unsafe, and `bend main.bend` names every def
# that relies on them. Nothing checks that a leaf never writes A or B, or that two leaves' blocks of C are disjoint;
# the blocks are the halves of the fork tree, as in tensor-array's own split. tests/bits.bend checks both by sampling.
# All of it is Bend: no foreign code.
import Base
import bend-ml-tensor-array@0.1.12.0/main.bend as TA

# ---- blocks: any kernel over ranges of shared arrays ----
# parN runs a kernel f over 2^d blocks of the range 0 .. cnt-1 in a tree of parallel lets; every block gets a handle to
# each of the N arrays (no copy) and the start and length of its range, plus e, a reusable record of whatever else the
# kernel needs (sizes, rates). The caller's kernel must write only inside its block's range, and a block's result may not
# depend on another's writes; then the arrays hold the same bits at every depth and thread count as f over the whole
# range at once. Nothing checks this (Array.fork and Array.join are @unsafe).

type B2<-A: Type, -B: Type> is Type:
  B2{a: Array<A>, b: Array<B>}

type B3<-A: Type, -B: Type, -C: Type> is Type:
  B3{a: Array<A>, b: Array<B>, c: Array<C>}

type B4<-A: Type, -B: Type, -C: Type, -D: Type> is Type:
  B4{a: Array<A>, b: Array<B>, c: Array<C>, d: Array<D>}

@unsafe
def blocks1(~E: Data, ~A: Type, ~f: @+e: E -> @+i0: U32 -> @+cnt: Nat -> Array<A> -> Array<A>, +e: E, d: Nat, +i0: U32, +cnt: Nat, pa: Array<A> & Array<A>) -> Array<A>:
  match d pa:
    case 0n Tuple{a, a2}:
      f(e, i0, cnt, a)
    case 1n++q Tuple{a, a2}:
      l r = blocks1(~E, ~A, ~f, e, q, i0, Nat.div(cnt, 2n), Array.fork(A, a)) blocks1(~E, ~A, ~f, e, q, (i0 + U32.from_nat(Nat.div(cnt, 2n)) : U32), Nat.sub(cnt, Nat.div(cnt, 2n)), Array.fork(A, a2))
      Array.join(A, l, r)

@unsafe
def join2(-A: Type, -B: Type, l: B2<A, B>, r: B2<A, B>) -> B2<A, B>:
  match l r:
    case B2{a, b} B2{a2, b2}:
      B2{Array.join(A, a, a2), Array.join(B, b, b2)}

@unsafe
def blocks2(~E: Data, ~A: Type, ~B: Type, ~f: @+e: E -> @+i0: U32 -> @+cnt: Nat -> B2<A, B> -> B2<A, B>, +e: E, d: Nat, +i0: U32, +cnt: Nat, pa: Array<A> & Array<A>, pb: Array<B> & Array<B>) -> B2<A, B>:
  match d pa pb:
    case 0n Tuple{a, a2} Tuple{b, b2}:
      f(e, i0, cnt, B2{a, b})
    case 1n++q Tuple{a, a2} Tuple{b, b2}:
      l r = blocks2(~E, ~A, ~B, ~f, e, q, i0, Nat.div(cnt, 2n), Array.fork(A, a), Array.fork(B, b)) blocks2(~E, ~A, ~B, ~f, e, q, (i0 + U32.from_nat(Nat.div(cnt, 2n)) : U32), Nat.sub(cnt, Nat.div(cnt, 2n)), Array.fork(A, a2), Array.fork(B, b2))
      join2(A, B, l, r)

@unsafe
def join3(-A: Type, -B: Type, -C: Type, l: B3<A, B, C>, r: B3<A, B, C>) -> B3<A, B, C>:
  match l r:
    case B3{a, b, c} B3{a2, b2, c2}:
      B3{Array.join(A, a, a2), Array.join(B, b, b2), Array.join(C, c, c2)}

@unsafe
def blocks3(~E: Data, ~A: Type, ~B: Type, ~C: Type, ~f: @+e: E -> @+i0: U32 -> @+cnt: Nat -> B3<A, B, C> -> B3<A, B, C>, +e: E, d: Nat, +i0: U32, +cnt: Nat, pa: Array<A> & Array<A>, pb: Array<B> & Array<B>, pc: Array<C> & Array<C>) -> B3<A, B, C>:
  match d pa pb pc:
    case 0n Tuple{a, a2} Tuple{b, b2} Tuple{c, c2}:
      f(e, i0, cnt, B3{a, b, c})
    case 1n++q Tuple{a, a2} Tuple{b, b2} Tuple{c, c2}:
      l r = blocks3(~E, ~A, ~B, ~C, ~f, e, q, i0, Nat.div(cnt, 2n), Array.fork(A, a), Array.fork(B, b), Array.fork(C, c)) blocks3(~E, ~A, ~B, ~C, ~f, e, q, (i0 + U32.from_nat(Nat.div(cnt, 2n)) : U32), Nat.sub(cnt, Nat.div(cnt, 2n)), Array.fork(A, a2), Array.fork(B, b2), Array.fork(C, c2))
      join3(A, B, C, l, r)

@unsafe
def join4(-A: Type, -B: Type, -C: Type, -D: Type, l: B4<A, B, C, D>, r: B4<A, B, C, D>) -> B4<A, B, C, D>:
  match l r:
    case B4{a, b, c, d} B4{a2, b2, c2, d2}:
      B4{Array.join(A, a, a2), Array.join(B, b, b2), Array.join(C, c, c2), Array.join(D, d, d2)}

@unsafe
def blocks4(~E: Data, ~A: Type, ~B: Type, ~C: Type, ~D: Type, ~f: @+e: E -> @+i0: U32 -> @+cnt: Nat -> B4<A, B, C, D> -> B4<A, B, C, D>, +e: E, d: Nat, +i0: U32, +cnt: Nat, pa: Array<A> & Array<A>, pb: Array<B> & Array<B>, pc: Array<C> & Array<C>, pd: Array<D> & Array<D>) -> B4<A, B, C, D>:
  match d pa pb pc pd:
    case 0n Tuple{a, a2} Tuple{b, b2} Tuple{c, c2} Tuple{x, x2}:
      f(e, i0, cnt, B4{a, b, c, x})
    case 1n++q Tuple{a, a2} Tuple{b, b2} Tuple{c, c2} Tuple{x, x2}:
      l r = blocks4(~E, ~A, ~B, ~C, ~D, ~f, e, q, i0, Nat.div(cnt, 2n), Array.fork(A, a), Array.fork(B, b), Array.fork(C, c), Array.fork(D, x)) blocks4(~E, ~A, ~B, ~C, ~D, ~f, e, q, (i0 + U32.from_nat(Nat.div(cnt, 2n)) : U32), Nat.sub(cnt, Nat.div(cnt, 2n)), Array.fork(A, a2), Array.fork(B, b2), Array.fork(C, c2), Array.fork(D, x2))
      join4(A, B, C, D, l, r)

# f over 2^d blocks of 0 .. cnt-1, sharing the arrays
@unsafe
def par1(~E: Data, ~A: Type, ~f: @+e: E -> @+i0: U32 -> @+cnt: Nat -> Array<A> -> Array<A>, +e: E, d: Nat, +cnt: Nat, a: Array<A>) -> Array<A>:
  blocks1(~E, ~A, ~f, e, d, 0, cnt, Array.fork(A, a))

@unsafe
def par2(~E: Data, ~A: Type, ~B: Type, ~f: @+e: E -> @+i0: U32 -> @+cnt: Nat -> B2<A, B> -> B2<A, B>, +e: E, d: Nat, +cnt: Nat, a: Array<A>, b: Array<B>) -> B2<A, B>:
  blocks2(~E, ~A, ~B, ~f, e, d, 0, cnt, Array.fork(A, a), Array.fork(B, b))

@unsafe
def par3(~E: Data, ~A: Type, ~B: Type, ~C: Type, ~f: @+e: E -> @+i0: U32 -> @+cnt: Nat -> B3<A, B, C> -> B3<A, B, C>, +e: E, d: Nat, +cnt: Nat, a: Array<A>, b: Array<B>, c: Array<C>) -> B3<A, B, C>:
  blocks3(~E, ~A, ~B, ~C, ~f, e, d, 0, cnt, Array.fork(A, a), Array.fork(B, b), Array.fork(C, c))

@unsafe
def par4(~E: Data, ~A: Type, ~B: Type, ~C: Type, ~D: Type, ~f: @+e: E -> @+i0: U32 -> @+cnt: Nat -> B4<A, B, C, D> -> B4<A, B, C, D>, +e: E, d: Nat, +cnt: Nat, a: Array<A>, b: Array<B>, c: Array<C>, x: Array<D>) -> B4<A, B, C, D>:
  blocks4(~E, ~A, ~B, ~C, ~D, ~f, e, d, 0, cnt, Array.fork(A, a), Array.fork(B, b), Array.fork(C, c), Array.fork(D, x))

def tg_c(g: TA.TG) -> Array<F32>:
  match g:
    case TA.TG{a, b, c}:
      c

# rows i0 .. i0+cnt-1 of C, every column
def rblock(+ac: Bool, cnt: Nat, +m: Nat, +tk: Nat, +tku: U32, +rot: Nat, +i0: U32, +mu: U32, a: Array<TA.T8>, b: Array<TA.T8>, c: Array<F32>) -> Array<F32>:
  tg_c(TA.t8_rows(ac, cnt, m, tk, tku, rot, i0, mu, TA.TG{a, b, c}))

# columns j0 .. j0+cnt-1 of C, rows i.. (n more)
def cbrows(+ac: Bool, n: Nat, +cnt: Nat, +tk: Nat, +tku: U32, +rot: Nat, +i: U32, +j0: U32, +mu: U32, g: TA.TG) -> TA.TG:
  match n:
    case 0n:
      g
    case 1n+p:
      cbrows(ac, p, cnt, tk, tku, rot, (i + 1 : U32), j0, mu, TA.t8_cols(ac, cnt, tk, tku, rot, i, j0, ((i * mu : U32) + j0 : U32), g))

# 2^d blocks of rows of C, halved at each node; each pair holds two handles to one block
@unsafe
def rpar(+ac: Bool, d: Nat, +cnt: Nat, +m: Nat, +tk: Nat, +tku: U32, +rot: Nat, +i0: U32, +mu: U32, pa: Array<TA.T8> & Array<TA.T8>, pb: Array<TA.T8> & Array<TA.T8>, pc: Array<F32> & Array<F32>) -> Array<F32>:
  match d pa pb pc:
    case 0n Tuple{a, a2} Tuple{b, b2} Tuple{c, c2}:
      rblock(ac, cnt, m, tk, tku, rot, i0, mu, a, b, c)
    case 1n++q Tuple{a, a2} Tuple{b, b2} Tuple{c, c2}:
      l r = rpar(ac, q, Nat.div(cnt, 2n), m, tk, tku, rot, i0, mu, Array.fork(TA.T8, a), Array.fork(TA.T8, b), Array.fork(F32, c)) rpar(ac, q, Nat.sub(cnt, Nat.div(cnt, 2n)), m, tk, tku, rot, (i0 + U32.from_nat(Nat.div(cnt, 2n)) : U32), mu, Array.fork(TA.T8, a2), Array.fork(TA.T8, b2), Array.fork(F32, c2))
      Array.join(F32, l, r)

# 2^d blocks of columns of C
@unsafe
def cpar(+ac: Bool, d: Nat, +cnt: Nat, +n: Nat, +tk: Nat, +tku: U32, +rot: Nat, +j0: U32, +mu: U32, pa: Array<TA.T8> & Array<TA.T8>, pb: Array<TA.T8> & Array<TA.T8>, pc: Array<F32> & Array<F32>) -> Array<F32>:
  match d pa pb pc:
    case 0n Tuple{a, a2} Tuple{b, b2} Tuple{c, c2}:
      tg_c(cbrows(ac, n, cnt, tk, tku, rot, 0, j0, mu, TA.TG{a, b, c}))
    case 1n++q Tuple{a, a2} Tuple{b, b2} Tuple{c, c2}:
      l r = cpar(ac, q, Nat.div(cnt, 2n), n, tk, tku, rot, j0, mu, Array.fork(TA.T8, a), Array.fork(TA.T8, b), Array.fork(F32, c)) cpar(ac, q, Nat.sub(cnt, Nat.div(cnt, 2n)), n, tk, tku, rot, (j0 + U32.from_nat(Nat.div(cnt, 2n)) : U32), mu, Array.fork(TA.T8, a2), Array.fork(TA.T8, b2), Array.fork(F32, c2))
      Array.join(F32, l, r)

# by rows when C has at least as many rows as columns, as tensor-array splits
@unsafe
def t8_tree(byrows: Bool, +ac: Bool, d: Nat, +n: Nat, +m: Nat, +tk: Nat, +tku: U32, +rot: Nat, pa: Array<TA.T8> & Array<TA.T8>, pb: Array<TA.T8> & Array<TA.T8>, pc: Array<F32> & Array<F32>) -> Array<F32>:
  match byrows:
    case True{}:
      rpar(ac, d, n, m, tk, tku, rot, 0, U32.from_nat(m), pa, pb, pc)
    case False{}:
      cpar(ac, d, m, n, tk, tku, rot, 0, U32.from_nat(m), pa, pb, pc)

# the packed copies are dropped after the tree; A and B (unpacked) go back to the caller
@unsafe
def t8_go(+ac: Bool, d: Nat, +n: Nat, +m: Nat, +tk: Nat, +tku: U32, +rot: Nat, c: Array<F32>, p: TA.PKK) -> TA.G:
  match p:
    case TA.PKK{a, b, ta, tb}:
      TA.G{a, b, t8_tree(Nat.is_le(m, n), ac, d, n, m, tk, tku, rot, Array.fork(TA.T8, ta), Array.fork(TA.T8, tb), Array.fork(F32, c))}

# C (n x m) = A · B (or C + A · B when ac), n >= 2, with 2^d blocks sharing A, B and C; in any layout of TA.gemm_any
# (arow, sa, sb, bcol); A and B come back intact
@unsafe
def gemm_fork(+ac: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> TA.G:
  t8_go(ac, d, n, m, Nat.div(Nat.add(k, 7n), 8n), U32.from_nat(Nat.div(Nat.add(k, 7n), 8n)), Nat.mod(k, 4n), c, TA.t8_pk2(TA.pk_pack(n, Nat.div(Nat.add(k, 7n), 8n), k, arow, sa, a), TA.pk_pack(m, Nat.div(Nat.add(k, 7n), 8n), k, bcol, sb, b)))

# two or more rows: the shared tree when d > 0, tensor-array's sequential cells at d = 0
def gemm_rows(+ac: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> TA.G:
  match d:
    case 0n:
      TA.gemm_t8_ac(ac, 0n, n, m, k, arow, sa, sb, bcol, a, b, c)
    case 1n+q:
      gemm_fork(ac, 1n+q, n, m, k, arow, sa, sb, bcol, a, b, c)

def gemm_any2(+ac: Bool, one: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> TA.G:
  match one:
    case True{}:
      TA.gemm_ac(ac, n, m, k, arow, sa, sb, bcol, a, b, c)
    case False{}:
      gemm_rows(ac, d, n, m, k, arow, sa, sb, bcol, a, b, c)

# TA.gemm_any's arguments and result; one row runs the scalar loop at every depth, as in tensor-array 0.1.12.0
def gemm_any(+ac: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> TA.G:
  gemm_any2(ac, Nat.is_le(n, 1n), d, n, m, k, arow, sa, sb, bcol, a, b, c)

# ---- the six products: tensor-array's signatures; par goes through TA.split_depth, as there ----

# C = A · B   (n x k) · (k x m)
def matmul(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      TA.mmul_fin(n, k, m, gemm_any(False{}, TA.split_depth(par, n, m, k), n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y, Array.new(F32, TA.cap_depth(Nat.mul(n, m)), 0.0)))

# C + A · B, into C
def matmul_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMul<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      TA.mmul_fin(n, k, m, gemm_any(True{}, TA.split_depth(par, n, m, k), n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y, z))

# C = A · Bᵀ   (n x k) · (m x k)ᵀ: weights with one row per output
def matmul_nt(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>) -> TA.MMulNT<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      TA.mmul_nt_fin(n, k, m, gemm_any(False{}, TA.split_depth(par, n, m, k), n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y, Array.new(F32, TA.cap_depth(Nat.mul(n, m)), 0.0)))

# C + A · Bᵀ, into C
def matmul_nt_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>, c: TA.Mat<n, m>) -> TA.MMulNT<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      TA.mmul_nt_fin(n, k, m, gemm_any(True{}, TA.split_depth(par, n, m, k), n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y, z))

# C = Aᵀ · B   (k x n)ᵀ · (k x m): a weight gradient, Xᵀ · dY
def matmul_tn(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>) -> TA.MMulTN<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      TA.mmul_tn_fin(n, k, m, gemm_any(False{}, TA.split_depth(par, n, m, k), n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y, Array.new(F32, TA.cap_depth(Nat.mul(n, m)), 0.0)))

# C + Aᵀ · B, into C: dW += Xᵀ · dY
def matmul_tn_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMulTN<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      TA.mmul_tn_fin(n, k, m, gemm_any(True{}, TA.split_depth(par, n, m, k), n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y, z))
