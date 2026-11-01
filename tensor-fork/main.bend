# bend-ml-tensor-fork: bend-ml-tensor-array's products, with the parallel blocks sharing their operands; and par1..par4,
# any kernel over blocks of a range sharing up to four arrays.
#
#   import bend-ml-tensor-array@0.1.19.0/main.bend as TA
#   import bend-ml-tensor-fork@0.1.6.0/main.bend as TF
#   TF.matmul(n, k, m, par, a, b)      # the same arguments, result type and bits as TA.Mat.matmul
#
# The six products (matmul, matmul_nt, matmul_tn and their accumulating forms) take tensor-array's Mat values and
# return its result types. A product that splits (tensor-array's split_depth, two or more rows) packs A and B once into
# cells of 8, as the sequential path does (in blocks of lines when an operand holds 16384 cells or more), and every
# node of the fork tree shares the two packed operands and C through Base's Array.fork: two handles to one block, no
# copy. Each leaf runs tensor-array's cells (t8_rows, t8_cols) straight
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
import bend-ml-tensor-array@0.1.19.0/main.bend as TA

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

# the smallest d with 2^d >= n (TA.cap_depth's answer) by doubling a U32: log2(n) steps, where cap_depth compares Nats
# (a share of every product's time at 512 x 1536)
def depth_go(f: Nat, ok: Bool, +n: U32, +p: U32, d: Nat) -> Nat:
  match f ok:
    case _ True{}:
      d
    case 0n False{}:
      d
    case 1n+g False{}:
      depth_go(g, U32.is_le(n, (p + p : U32)), n, (p + p : U32), 1n+d)

def depth(+n: Nat) -> Nat:
  depth_go(32n, U32.is_le(U32.from_nat(n), 1), U32.from_nat(n), 1, 0n)

# ---- packing in parallel: tensor-array's pk_pack, its lines in 2^d blocks ----

type PE is Data:
  PE{tk: Nat, tku: U32, ku: U32, lstep: U32, st: U32}

def pk_b2(r: TA.PK) -> B2<F32, TA.T8>:
  match r:
    case TA.PK{a, t}:
      B2{a, t}

# lines i0 .. i0+cnt-1 into their cells
def pk_leaf(+e: PE, +i0: U32, +cnt: Nat, r: B2<F32, TA.T8>) -> B2<F32, TA.T8>:
  match e r:
    case PE{+tk, +tku, +ku, +lstep, +st} B2{a, t}:
      pk_b2(TA.pk_lines(cnt, tk, tku, ku, i0, i0, lstep, st, TA.PK{a, t}))

def pk_of(r: B2<F32, TA.T8>) -> TA.PK:
  match r:
    case B2{a, t}:
      TA.PK{a, t}

# one block below 16384 cells (MNIST's 100 x 784 and 128 x 784 pack faster in one), else 2^d
def pk_d(big: Bool, d: Nat) -> Nat:
  match big:
    case True{}:
      d
    case False{}:
      0n

@unsafe
def pk_par(d: Nat, +cnt: Nat, +tk: Nat, +k: Nat, +lstep: U32, +st: U32, a: Array<F32>) -> TA.PK:
  pk_of(par2(~PE, ~F32, ~TA.T8, ~pk_leaf, PE{tk, U32.from_nat(tk), U32.from_nat(k), lstep, st}, pk_d(Nat.is_le(16384n, Nat.mul(cnt, tk)), d), cnt, a, Array.new(TA.T8, depth(Nat.mul(cnt, tk)), TA.t8_zero())))

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
def gemm_fork(+ac: Bool, +d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> TA.G:
  t8_go(ac, d, n, m, Nat.div(Nat.add(k, 7n), 8n), U32.from_nat(Nat.div(Nat.add(k, 7n), 8n)), Nat.mod(k, 4n), c, TA.t8_pk2(pk_par(d, n, Nat.div(Nat.add(k, 7n), 8n), k, arow, sa, a), pk_par(d, m, Nat.div(Nat.add(k, 7n), 8n), k, bcol, sb, b)))

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
      TA.mmul_fin(n, k, m, gemm_any(False{}, TA.split_depth(par, n, m, k), n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y, Array.new(F32, depth(Nat.mul(n, m)), 0.0)))

# C + A · B, into C
def matmul_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMul<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      TA.mmul_fin(n, k, m, gemm_any(True{}, TA.split_depth(par, n, m, k), n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y, z))

# C = A · Bᵀ   (n x k) · (m x k)ᵀ: weights with one row per output
def matmul_nt(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>) -> TA.MMulNT<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      TA.mmul_nt_fin(n, k, m, gemm_any(False{}, TA.split_depth(par, n, m, k), n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y, Array.new(F32, depth(Nat.mul(n, m)), 0.0)))

# C + A · Bᵀ, into C
def matmul_nt_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>, c: TA.Mat<n, m>) -> TA.MMulNT<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      TA.mmul_nt_fin(n, k, m, gemm_any(True{}, TA.split_depth(par, n, m, k), n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y, z))

# C = Aᵀ · B   (k x n)ᵀ · (k x m): a weight gradient, Xᵀ · dY
def matmul_tn(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>) -> TA.MMulTN<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      TA.mmul_tn_fin(n, k, m, gemm_any(False{}, TA.split_depth(par, n, m, k), n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y, Array.new(F32, depth(Nat.mul(n, m)), 0.0)))

# C + Aᵀ · B, into C: dW += Xᵀ · dY
def matmul_tn_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMulTN<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      TA.mmul_tn_fin(n, k, m, gemm_any(True{}, TA.split_depth(par, n, m, k), n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y, z))

# ---- tensor-array 0.1.14.0's precise product with these blocks (opt-in) ----
# TA.Mat.matmul_precise's composition, each block's product computed by gemm_any above: k in blocks of 1024, the first
# block's product written into C, each next one's added to it, so the same bits as tensor-array's (tests/bits.bend).

@unsafe
def mp_s3(+ac: Bool, +par: Nat, +n: Nat, +m: Nat, +w: Nat, +w32: U32, +m32: U32, a: Array<F32>, as: Array<F32>, c: Array<F32>, sb: TA.SL) -> TA.G:
  match sb:
    case TA.SL{b, bs}: TA.mp_s4(a, b, gemm_any(ac, TA.split_depth(par, n, m, w), n, m, w, w32, 1, m32, 1, as, bs, c))

@unsafe
def mp_s2(+ac: Bool, +par: Nat, +n: Nat, +m: Nat, +w: Nat, +w32: U32, +m32: U32, +o: U32, c: Array<F32>, b: Array<F32>, sa: TA.SL) -> TA.G:
  match sa:
    case TA.SL{a, as}: mp_s3(ac, par, n, m, w, w32, m32, a, as, c, TA.sl_row(Nat.mul(w, m), (o * m32 : U32), 0, Array.new(F32, depth(Nat.mul(w, m)), 0.0), Array.get(F32, b, (o * m32 : U32))))

@unsafe
def mp_step(+ac: Bool, +par: Nat, +n: Nat, +m: Nat, +k32: U32, +bi: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> TA.G:
  mp_s2(ac, par, n, m, TA.blk_len(bi, k32), U32.from_nat(TA.blk_len(bi, k32)), U32.from_nat(m), (bi * 1024 : U32), c, b, TA.sl_rows(n, 0, TA.blk_len(bi, k32), U32.from_nat(TA.blk_len(bi, k32)), k32, (bi * 1024 : U32), TA.SL{a, Array.new(F32, depth(Nat.mul(n, TA.blk_len(bi, k32))), 0.0)}))

@unsafe
def mp_go(nb: Nat, +bi: U32, +par: Nat, +n: Nat, +m: Nat, +k32: U32, g: TA.G) -> TA.G:
  match nb g:
    case 0n g: g
    case 1n++q TA.G{a, b, c}: mp_go(q, (bi + 1 : U32), par, n, m, k32, mp_step(U32.is_lt(0, bi), par, n, m, k32, bi, a, b, c))

@unsafe
def mm_precise2(long: Bool, +n: Nat, +k: Nat, +m: Nat, +par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  match long a b:
    case True{} TA.Mat{x} TA.Mat{y}: TA.mmul_fin(n, k, m, mp_go(U32.to_nat(U32.div((U32.from_nat(k) + 1023 : U32), 1024)), 0, par, n, m, U32.from_nat(k), TA.G{x, y, Array.new(F32, depth(Nat.mul(n, m)), 0.0)}))
    case False{} a b: matmul(n, k, m, par, a, b)

# C = A · B with k summed in blocks of 1024: TA.Mat.matmul_precise's bits (opt-in)
@unsafe
def matmul_precise(+n: Nat, +k: Nat, +m: Nat, +par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  mm_precise2(Nat.is_lt(1024n, k), n, k, m, par, a, b)

# ---- softmax_ce with the rows in blocks (TA.Mat.softmax_ce's bits) ----
# tensor-array walks the rows one after another (cea_rows), adding each row's loss term to the running loss. Here the
# rows run in 2^d blocks through par3, sharing the logits (each block writes only its own rows), the labels (read) and
# an array t of one loss term per row (each block writes only its own cells). The loss is then summed from t in row
# order, from +0.0, with the same subtraction as cea_rows (loss - log(max(p, 1e-12))), so the bits are the same at
# every depth and thread count. The rows are tensor-array's own loops (cea_max, cea_exp, cea_p, cea_grad).
# Labels past the n rows (a list longer than the matrix) run after the blocks through TA.cea_rows, in order, as in
# tensor-array: those rows reach past the matrix and may wrap around onto its first rows, so they cannot share a
# block. A matrix with no columns, or par = 0, runs TA.Mat.softmax_ce itself.

type SE is Data:
  SE{mm1: Nat, mu: U32, nf: F32}

# a row's loss term log(max(p, 1e-12)) and its gradient: cea_row3..cea_row0 without the running loss
def sr3(+mm1: Nat, +o: U32, +label: Nat, +s: F32, +nf: F32, q: TA.AF) -> TA.L2:
  match q:
    case TA.AF{c, +p}:
      TA.L2{F32.log(F32.max(p, 0.000000000001)), TA.cea_grad(mm1, o, 0n, label, s, nf, Array.get(F32, c, o))}

def sr2(+mm1: Nat, +o: U32, +label: Nat, +nf: F32, q: TA.AF) -> TA.L2:
  match q:
    case TA.AF{c, +s}:
      sr3(mm1, o, label, s, nf, TA.cea_p(Nat.is_le(label, mm1), o, label, s, c))

def sr1(+mm1: Nat, +o: U32, +label: Nat, +nf: F32, q: TA.AF) -> TA.L2:
  match q:
    case TA.AF{c, +m}:
      sr2(mm1, o, label, nf, TA.cea_exp(mm1, o, m, 0.0, Array.get(F32, c, o)))

def sr0(+mm1: Nat, +o: U32, +label: Nat, +nf: F32, r: Array<F32> & F32) -> TA.L2:
  match r:
    case Tuple{c, +x0}:
      sr1(mm1, o, label, nf, TA.cea_max(mm1, o, x0, Array.get(F32, c, o)))

def ce_put(+i: U32, lab: Array<U32>, t: Array<F32>, q: TA.L2) -> B3<F32, U32, F32>:
  match q:
    case TA.L2{+x, z}:
      B3{z, lab, Array.set(F32, t, i, x)}

def ce_row(+mm1: Nat, +mu: U32, +nf: F32, +i: U32, z: Array<F32>, t: Array<F32>, r: Array<U32> & U32) -> B3<F32, U32, F32>:
  match r:
    case Tuple{lab, +y}:
      ce_put(i, lab, t, sr0(mm1, (i * mu : U32), U32.to_nat(y), nf, Array.get(F32, z, (i * mu : U32))))

def ce_step(+mm1: Nat, +mu: U32, +nf: F32, +i: U32, g: B3<F32, U32, F32>) -> B3<F32, U32, F32>:
  match g:
    case B3{z, lab, t}:
      ce_row(mm1, mu, nf, i, z, t, Array.get(U32, lab, i))

def ce_loop(k: Nat, +mm1: Nat, +mu: U32, +nf: F32, +i: U32, g: B3<F32, U32, F32>) -> B3<F32, U32, F32>:
  match k:
    case 0n:
      g
    case 1n+p:
      ce_loop(p, mm1, mu, nf, (i + 1 : U32), ce_step(mm1, mu, nf, i, g))

# rows i0 .. i0+cnt-1: their gradients in the logits, their loss terms in t
def ce_leaf(+e: SE, +i0: U32, +cnt: Nat, g: B3<F32, U32, F32>) -> B3<F32, U32, F32>:
  match e:
    case SE{+mm1, +mu, +nf}:
      ce_loop(cnt, mm1, mu, nf, i0, g)

# the loss over k terms, in row order from +0.0, as cea_rows adds them
def ce_sum(k: Nat, +i: U32, +loss: F32, r: Array<F32> & F32) -> F32:
  match k r:
    case 0n Tuple{t, x}:
      loss
    case 1n+p Tuple{t, +x}:
      ce_sum(p, (i + 1 : U32), (loss - x : F32), Array.get(F32, t, (i + 1 : U32)))

type CL is Type:
  CL{a: Array<U32>, k: Nat, rest: List<&2, Nat>}

# a label past the row gives p = 0 and no one-hot cell, as c does: kept as c so that it fits a U32
def lab_fit(inside: Bool, +y: Nat, +c: Nat) -> Nat:
  match inside:
    case True{}:
      y
    case False{}:
      c

# up to n labels into a, counted in k; the rest stays a list
def lab_take(n: Nat, labels: List<&2, Nat>, +mm1: Nat, +c: Nat, +i: U32, +k: Nat, a: Array<U32>) -> CL:
  match n labels:
    case 0n ls:
      CL{a, k, ls}
    case 1n+p Nil{}:
      CL{a, k, Nil{}}
    case 1n+p Con{+y, t}:
      lab_take(p, t, mm1, c, (i + 1 : U32), 1n+k, Array.set(U32, a, i, U32.from_nat(lab_fit(Nat.is_le(y, mm1), y, c))))

# after the blocks: the loss from t, then any labels past the n rows, in order, then tensor-array's mean
def ce_fin2(-n: Nat, -c: Nat, +nf: F32, +mm1: Nat, +mu: U32, +k: Nat, rest: List<&2, Nat>, g: B3<F32, U32, F32>) -> TA.MCE<n, c>:
  match g:
    case B3{z, lab, t}:
      TA.ce_fin(n, c, nf, TA.cea_rows(rest, mm1, mu, nf, U32.from_nat(k), TA.L2{ce_sum(k, 0, 0.0, Array.get(F32, t, 0)), z}))

@unsafe
def ce_par3(-n: Nat, -c: Nat, +nf: F32, +mm1: Nat, +mu: U32, +d: Nat, z: Array<F32>, lt: CL) -> TA.MCE<n, c>:
  match lt:
    case CL{lab, +k, rest}:
      ce_fin2(n, c, nf, mm1, mu, k, rest, par3(~SE, ~F32, ~U32, ~F32, ~ce_leaf, SE{mm1, mu, nf}, Nat.min(d, depth(k)), k, z, lab, Array.new(F32, depth(k), 0.0)))

def softmax_ce2(seq: Bool, +n: Nat, +c: Nat, +d: Nat, z: TA.Mat<n, c>, labels: List<&2, Nat>) -> TA.MCE<n, c>:
  match seq z:
    case True{} z:
      TA.Mat.softmax_ce(n, c, z, labels)
    case False{} TA.Mat{x}:
      ce_par3(n, c, F32.from_nat(n), Nat.sub(c, 1n), U32.from_nat(c), d, x, lab_take(n, labels, Nat.sub(c, 1n), c, 0, 0n, Array.new(U32, depth(n), 0)))

# mean cross-entropy and its gradient, TA.Mat.softmax_ce's arguments and bits, with the rows in up to 2^par blocks
# (par = 0: tensor-array's own loop)
def softmax_ce(+n: Nat, +c: Nat, +par: Nat, z: TA.Mat<n, c>, labels: List<&2, Nat>) -> TA.MCE<n, c>:
  softmax_ce2(Bool.or(Nat.is_eq(par, 0n), Nat.is_eq(c, 0n)), n, c, par, z, labels)
