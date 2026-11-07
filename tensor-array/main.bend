# bend-ml-tensor-array: tensors over a flat Array<F32>, with the shape in the type.
#
#   import bend-ml-tensor-array@0.1.20.0/main.bend as TA
#
# Mat<r, c> stores r*c numbers in an Array<F32>, row by row (index i*c + j). The
# dimensions are erased type parameters: the same guarantee as bend-ml-tensor
# (a product with wrong dimensions does not compile), but ~50x faster than lists:
# Array.get/set by index, and products in parallel blocks of rows.
#
# An Array is affine (a single owner): every operation that reads an operand returns it
# together with the result (as Array.get does), in records MM, AR, ... Use Mat.clone
# when you need two copies.
#
# What is NOT proved: the invariant "the Array has capacity >= r*c" holds because
# the constructors (Mat.zeros, Mat.of_list) allocate the right size; and F32 is
# validated by tests against PyTorch (reference/test_tensor_array.py), not by proof.

import Base

# ---------------------------------------------------------------------
# helper types
# ---------------------------------------------------------------------

type R is Type:
  R{a: Array<F32>, b: Array<F32>, x: F32}

type G is Type:
  G{a: Array<F32>, b: Array<F32>, c: Array<F32>}

# A dot product with four partial sums. Each step adds its product to a0 and rotates the sums
# (a1, a2, a3, a0), so each sum gets every fourth product and the processor can work on the next
# addition before the last one has finished; at the end the sums are added in pairs. The order of
# the additions is fixed, so the result is the same on every run and every thread count, but it is
# not the order of a single running sum (NOTES.md, exp. 17).
def dotr(k: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, a0: F32, a1: F32, a2: F32, a3: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> R:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      R{a, b, ((a0 + a1 : F32) + (a2 + a3 : F32) : F32)}
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      dotr(p, (ia + sa : U32), (ib + sb : U32), sa, sb, a1, a2, a3, (a0 + (x * y : F32) : F32), Array.get(F32, a, (ia + sa : U32)), Array.get(F32, b, (ib + sb : U32)))

# dotr's state between two steps: the four partial sums and the two pending reads
type D4 is Type:
  D4{a0: F32, a1: F32, a2: F32, a3: F32, ra: Array<F32> & F32, rb: Array<F32> & F32}

# one step of dotr: adds the pending product to a0, rotates the sums, reads the next pair at ia, ib
def dstep(+ia: U32, +ib: U32, s: D4) -> D4:
  match s:
    case D4{a0, a1, a2, a3, ra, rb}:
      match ra rb:
        case Tuple{a, +x} Tuple{b, +y}:
          D4{a1, a2, a3, (a0 + (x * y : F32) : F32), Array.get(F32, a, ia), Array.get(F32, b, ib)}

def dfin(k: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, s: D4) -> R:
  match s:
    case D4{a0, a1, a2, a3, ra, rb}:
      dotr(k, ia, ib, sa, sb, a0, a1, a2, a3, ra, rb)

# dotr with its loop unrolled by four: q rounds of four dstep, then the last rem (< 4) steps by dotr itself.
# Every product goes to the same sum in the same order as in dotr (four rotations bring the names back), so the
# result has dotr's bits (tests/dot4.bend). The compiled loop runs ~16% fewer instructions; one thread on a
# 50,257 x 768 matrix · vector goes from 26.7 to 20.0 ms, smaller ones and 16 threads stay within noise (NOTES.md,
# 2026-11-01). Eight steps a round were no faster.
def dot4(q: Nat, +rem: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, s: D4) -> R:
  match q:
    case 0n:
      dfin(rem, ia, ib, sa, sb, s)
    case 1n+p:
      dot4(p, rem, (ia + (4 * sa : U32) : U32), (ib + (4 * sb : U32) : U32), sa, sb, dstep((ia + (4 * sa : U32) : U32), (ib + (4 * sb : U32) : U32), dstep((ia + (3 * sa : U32) : U32), (ib + (3 * sb : U32) : U32), dstep((ia + (2 * sa : U32) : U32), (ib + (2 * sb : U32) : U32), dstep((ia + sa : U32), (ib + sb : U32), s)))))

def dotg(+k: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, acc: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> R:
  dot4(Nat.div(k, 4n), Nat.sub(k, Nat.mul(4n, Nat.div(k, 4n))), ia, ib, sa, sb, D4{acc, 0.0, 0.0, 0.0, ra, rb})

# writes one cell of C: c[idx] = x, or c[idx] = c[idx] + x when ac (an accumulating product, 0.1.10.0). x is computed
# exactly as without ac and added once, so C + A·B has the bits of adding a separately computed A·B to C.

def cwr_add(+idx: U32, +x: F32, r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{c, +v}:
      Array.set(F32, c, idx, (v + x : F32))

def cwr(ac: Bool, +idx: U32, +x: F32, c: Array<F32>) -> Array<F32>:
  match ac:
    case True{}:
      cwr_add(idx, x, Array.get(F32, c, idx))
    case False{}:
      Array.set(F32, c, idx, x)

# row i of C, columns j.. ; r is the product of the current column (already computed)

def gcols(+ac: Bool, mleft: Nat, +kn: Nat, +ia0: U32, +ib0: U32, +sa: U32, +sb: U32, +bcol: U32, +cidx: U32, c: Array<F32>, r: R) -> G:
  match mleft r:
    case 0n R{a, b, +x}:
      G{a, b, cwr(ac, cidx, x, c)}
    case 1n+p R{a, b, +x}:
      gcols(ac, p, kn, ia0, (ib0 + bcol : U32), sa, sb, bcol, (cidx + 1 : U32), cwr(ac, cidx, x, c), dotg(kn, ia0, (ib0 + bcol : U32), sa, sb, 0.0, Array.get(F32, a, ia0), Array.get(F32, b, (ib0 + bcol : U32))))

def grows(+ac: Bool, nleft: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +mu: U32, g: G) -> G:
  match nleft g:
    case 0n G{a, b, c}:
      G{a, b, c}
    case 1n+p G{a, b, c}:
      grows(ac, p, mm1, kn, (i + 1 : U32), arow, sa, sb, bcol, mu, gcols(ac, mm1, kn, (i * arow : U32), 0, sa, sb, bcol, (i * mu : U32), c, dotg(kn, (i * arow : U32), 0, sa, sb, 0.0, Array.get(F32, a, (i * arow : U32)), Array.get(F32, b, 0))))

# C (n x m) = A · B with k steps, or C + A · B when ac; c is the output array (with capacity n*m)

def gemm_ac(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  grows(ac, n, Nat.sub(m, 1n), k, 0, arow, sa, sb, bcol, U32.from_nat(m), G{a, b, c})

def gemm(+n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  gemm_ac(False{}, n, m, k, arow, sa, sb, bcol, a, b, c)


# ---------------------------------------------------------------------
# parallel gemm: 2^d blocks of rows of C, each with its own copy of A and B,
# returning lists that are written into C at the end
# ---------------------------------------------------------------------

type S3 is Type:
  S3{a: Array<F32>, b: Array<F32>, out: List<&2, F32>}

def lcols(mleft: Nat, +kn: Nat, +ia0: U32, +ib0: U32, +sa: U32, +sb: U32, +bcol: U32, out: List<&2, F32>, r: R) -> S3:
  match mleft r:
    case 0n R{a, b, +x}:
      S3{a, b, x <> out}
    case 1n+p R{a, b, +x}:
      lcols(p, kn, ia0, (ib0 + bcol : U32), sa, sb, bcol, x <> out, dotg(kn, ia0, (ib0 + bcol : U32), sa, sb, 0.0, Array.get(F32, a, ia0), Array.get(F32, b, (ib0 + bcol : U32))))

def lrows(nleft: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, st: S3) -> S3:
  match nleft st:
    case 0n S3{a, b, out}:
      S3{a, b, out}
    case 1n+p S3{a, b, out}:
      lrows(p, mm1, kn, (i + 1 : U32), arow, sa, sb, bcol, lcols(mm1, kn, (i * arow : U32), 0, sa, sb, bcol, out, dotg(kn, (i * arow : U32), 0, sa, sb, 0.0, Array.get(F32, a, (i * arow : U32)), Array.get(F32, b, 0))))

def lblock_fin(st: S3) -> List<&2, F32>:
  match st:
    case S3{a, b, out}:
      List.reverse(&2, F32, out)

def lblock(cnt: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>) -> List<&2, F32>:
  lblock_fin(lrows(cnt, mm1, kn, i, arow, sa, sb, bcol, S3{a, b, Nil{}}))

def lpar(d: Nat, +cnt: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, ra: Array<F32> & Array<F32>, rb: Array<F32> & Array<F32>) -> List<&2, F32>:
  match d ra rb:
    case 0n Tuple{a, a2} Tuple{b, b2}:
      lblock(cnt, mm1, kn, i, arow, sa, sb, bcol, a, b)
    case 1n++q Tuple{a, a2} Tuple{b, b2}:
      x y = lpar(q, Nat.div(cnt, 2n), mm1, kn, i, arow, sa, sb, bcol, Array.clone(F32, a), Array.clone(F32, b)) lpar(q, Nat.sub(cnt, Nat.div(cnt, 2n)), mm1, kn, (i + U32.from_nat(Nat.div(cnt, 2n)) : U32), arow, sa, sb, bcol, Array.clone(F32, a2), Array.clone(F32, b2))
      List.append(&2, F32, x, y)

def write_l(xs: List<&2, F32>, +idx: U32, c: Array<F32>) -> Array<F32>:
  match xs:
    case Nil{}:
      c
    case Con{h, t}:
      write_l(t, (idx + 1 : U32), Array.set(F32, c, idx, h))

# c[idx..] = list, or c[idx..] += list when ac

def write_la(+ac: Bool, xs: List<&2, F32>, +idx: U32, c: Array<F32>) -> Array<F32>:
  match xs:
    case Nil{}:
      c
    case Con{+h, t}:
      write_la(ac, t, (idx + 1 : U32), cwr(ac, idx, h, c))

def gp3(+ac: Bool, a: Array<F32>, b: Array<F32>, c: Array<F32>, xs: List<&2, F32>) -> G:
  G{a, b, write_la(ac, xs, 0, c)}

def gp2(d: Nat, +n: Nat, +mm1: Nat, +kn: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, c: Array<F32>, pa: Array<F32> & Array<F32>, pb: Array<F32> & Array<F32>) -> G:
  match pa pb:
    case Tuple{a1, a2} Tuple{b1, b2}:
      gp3(False{}, a1, b1, c, lpar(d, n, mm1, kn, 0, arow, sa, sb, bcol, Array.clone(F32, a2), Array.clone(F32, b2)))

# C (n x m) = A · B, with 2^d blocks of rows in parallel; returns A, B (intact) and C

def gemm_par(d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  gp2(d, n, Nat.sub(m, 1n), k, arow, sa, sb, bcol, c, Array.clone(F32, a), Array.clone(F32, b))


# ---------------------------------------------------------------------
# parallel gemm by COLUMNS, for n = 1 (matrix · vector): each block computes a
# range of columns of C, with its own copy of the operands
# ---------------------------------------------------------------------

def cblock_fin(st: S3) -> List<&2, F32>:
  match st:
    case S3{a, b, out}:
      List.reverse(&2, F32, out)

# cnt columns starting at j0 (cnt >= 1); row 0 of A is at ia0

def cblock_go(cnt1: Nat, +kn: Nat, +ia0: U32, +j0: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>) -> List<&2, F32>:
  cblock_fin(lcols(cnt1, kn, ia0, (j0 * bcol : U32), sa, sb, bcol, Nil{}, dotg(kn, ia0, (j0 * bcol : U32), sa, sb, 0.0, Array.get(F32, a, ia0), Array.get(F32, b, (j0 * bcol : U32)))))

def cblock(cnt: Nat, +kn: Nat, +ia0: U32, +j0: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>) -> List<&2, F32>:
  match cnt:
    case 0n:
      Nil{}
    case 1n+p:
      cblock_go(p, kn, ia0, j0, sa, sb, bcol, a, b)

def cpar(d: Nat, +cnt: Nat, +kn: Nat, +ia0: U32, +j0: U32, +sa: U32, +sb: U32, +bcol: U32, ra: Array<F32> & Array<F32>, rb: Array<F32> & Array<F32>) -> List<&2, F32>:
  match d ra rb:
    case 0n Tuple{a, a2} Tuple{b, b2}:
      cblock(cnt, kn, ia0, j0, sa, sb, bcol, a, b)
    case 1n++q Tuple{a, a2} Tuple{b, b2}:
      x y = cpar(q, Nat.div(cnt, 2n), kn, ia0, j0, sa, sb, bcol, Array.clone(F32, a), Array.clone(F32, b)) cpar(q, Nat.sub(cnt, Nat.div(cnt, 2n)), kn, ia0, (j0 + U32.from_nat(Nat.div(cnt, 2n)) : U32), sa, sb, bcol, Array.clone(F32, a2), Array.clone(F32, b2))
      List.append(&2, F32, x, y)

def cp2(+ac: Bool, d: Nat, +m: Nat, +kn: Nat, +sa: U32, +sb: U32, +bcol: U32, c: Array<F32>, pa: Array<F32> & Array<F32>, pb: Array<F32> & Array<F32>) -> G:
  match pa pb:
    case Tuple{a1, a2} Tuple{b1, b2}:
      gp3(ac, a1, b1, c, cpar(d, m, kn, 0, 0, sa, sb, bcol, Array.clone(F32, a2), Array.clone(F32, b2)))

# C (1 x m) = A (1 x k) · B, with 2^d blocks of columns in parallel. Since 0.1.12.0 no product calls it (one row runs
# the scalar loop, see gemm_any3): it clones both operands at every node. Kept for callers of earlier versions.

def gemm_par_cols(+ac: Bool, d: Nat, +m: Nat, +k: Nat, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  cp2(ac, d, m, k, sa, sb, bcol, c, Array.clone(F32, a), Array.clone(F32, b))

# ---------------------------------------------------------------------
# element-wise operations (in place, by index)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# gemm: C[i][j] = sum_p A[i*arow + p*sa] * B[p*sb + j*bcol]
# (nn: arow=K, sa=1, sb=M, bcol=1 | nt: B stored m x k: sb=1, bcol=K | tn: A stored k x n: arow=1, sa=N)
# ---------------------------------------------------------------------

# acc += A[ia + t*sa] * B[ib + t*sb] for k steps; ra, rb already carry the first pair

type P2 is Type:
  P2{a: Array<F32>, b: Array<F32>}

type AL is Type:
  AL{c: Array<F32>, l: List<&2, F32>}

# list of mleft + 1 elements ending at idx (reads with Array.get, returning the array). It reads from
# idx down and conses each number onto acc, so the list comes out in index order with no reverse
# (NOTES.md, 2026-11-01): callers pass the last index and its read.

def read_l(mleft: Nat, +idx: U32, acc: List<&2, F32>, rc: Array<F32> & F32) -> AL:
  match mleft rc:
    case 0n Tuple{c, +v}:
      AL{c, v <> acc}
    case 1n+p Tuple{c, +v}:
      read_l(p, (idx - 1 : U32), v <> acc, Array.get(F32, c, (idx - 1 : U32)))

# writes the list starting at idx

def addl(xs: List<&2, F32>, +idx: U32, rc: Array<F32> & F32) -> Array<F32>:
  match xs rc:
    case Nil{} Tuple{c, v}:
      c
    case Con{+h, t} Tuple{c, +v}:
      addl(t, (idx + 1 : U32), Array.get(F32, Array.set(F32, c, idx, (v + h : F32)), (idx + 1 : U32)))

# adds a bias (a list of m) to each of the n rows of c (n x m)

def bias_rows(nleft: Nat, +i: U32, +mu: U32, +bl: List<&2, F32>, c: Array<F32>) -> Array<F32>:
  match nleft:
    case 0n:
      c
    case 1n+p:
      bias_rows(p, (i + 1 : U32), mu, bl, addl(bl, (i * mu : U32), Array.get(F32, c, (i * mu : U32))))

# in-place relu on the first n elements

def relu_l(nleft: Nat, +t: U32, rc: Array<F32> & F32) -> Array<F32>:
  match nleft rc:
    case 0n Tuple{c, v}:
      c
    case 1n+p Tuple{c, +v}:
      relu_l(p, (t + 1 : U32), Array.get(F32, Array.set(F32, c, t, F32.max(v, 0.0)), (t + 1 : U32)))

def pos2(b: Bool) -> F32:
  match b:
    case True{}:
      1.0
    case False{}:
      0.0

def pos(+y: F32) -> F32:
  pos2(F32.is_gt(y, 0.0))

# d[t] *= (h[t] > 0): the relu gradient; returns both arrays

def mask_l(nleft: Nat, +t: U32, rd: Array<F32> & F32, rh: Array<F32> & F32) -> P2:
  match nleft rd rh:
    case 0n Tuple{d, x} Tuple{h, y}:
      P2{d, h}
    case 1n+p Tuple{d, +x} Tuple{h, +y}:
      mask_l(p, (t + 1 : U32), Array.get(F32, Array.set(F32, d, t, (x * pos(y) : F32)), (t + 1 : U32)), Array.get(F32, h, (t + 1 : U32)))

# w[t] -= lr * dw[t]

def maxl2(xs: List<&2, F32>, cur: F32) -> F32:
  match xs:
    case Nil{}:
      cur
    case Con{+h, t}:
      maxl2(t, F32.max(cur, h))

def maxl(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{+h, t}:
      maxl2(t, h)

def expl(xs: List<&2, F32>, +m: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      F32.exp((h - m : F32)) <> expl(t, m)

def suml(xs: List<&2, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      suml(t, (acc + h : F32))

def divl(xs: List<&2, F32>, +s: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      (h / s : F32) <> divl(t, s)

def softmax2(+es: List<&2, F32>) -> List<&2, F32>:
  divl(es, suml(es, 0.0))

def softmax(+xs: List<&2, F32>) -> List<&2, F32>:
  softmax2(expl(xs, maxl(xs)))

# p[label]

def sgd_l(nleft: Nat, +t: U32, +lr: F32, rw: Array<F32> & F32, rd: Array<F32> & F32) -> P2:
  match nleft rw rd:
    case 0n Tuple{w, x} Tuple{d, y}:
      P2{w, d}
    case 1n+p Tuple{w, +x} Tuple{d, +y}:
      sgd_l(p, (t + 1 : U32), lr, Array.get(F32, Array.set(F32, w, t, (x - (lr * y : F32) : F32)), (t + 1 : U32)), Array.get(F32, d, (t + 1 : U32)))

# ---------------------------------------------------------------------
# softmax + cross-entropy per row (lists of 10 elements: negligible cost)
# ---------------------------------------------------------------------

def pick(ps: List<&2, F32>, n: Nat) -> F32:
  match ps n:
    case Con{+h, t} 0n:
      h
    case Con{h, t} 1n+q:
      pick(t, q)
    case Nil{} _:
      0.0

# (p - onehot(label)) / n

def hit(b: Bool) -> F32:
  match b:
    case True{}:
      1.0
    case False{}:
      0.0

def grad_l(ps: List<&2, F32>, +label: Nat, +j: Nat, +nf: F32) -> List<&2, F32>:
  match ps:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      (((h - hit(Nat.is_eq(label, j)) : F32) / nf) : F32) <> grad_l(t, label, 1n+j, nf)

type L2 is Type:
  L2{loss: F32, c: Array<F32>}

def ce_row2(+ps: List<&2, F32>, +label: Nat, +nf: F32, +idx: U32, loss: F32, c: Array<F32>) -> L2:
  L2{(loss - F32.log(F32.max(pick(ps, label), 0.000000000001)) : F32), write_l(grad_l(ps, label, 0n, nf), idx, c)}

def ce_row(al: AL, +label: Nat, +nf: F32, +idx: U32, loss: F32) -> L2:
  match al:
    case AL{c, l}:
      ce_row2(softmax(l), label, nf, idx, loss, c)

# walks the rows of z2 (n x m): r carries the accumulated loss and the array; at the end the array has become the gradient

type Mat<-r: Nat, -c: Nat> is Type:
  Mat{d: Array<F32>}

def ce_rows(labels: List<&2, Nat>, +mm1: Nat, +mu: U32, +nf: F32, +i: U32, r: L2) -> L2:
  match labels r:
    case Nil{} L2{loss, c}:
      L2{loss, c}
    case Con{+y, t} L2{loss, c}:
      ce_rows(t, mm1, mu, nf, (i + 1 : U32), ce_row(read_l(mm1, ((i * mu : U32) + U32.from_nat(mm1) : U32), Nil{}, Array.get(F32, c, ((i * mu : U32) + U32.from_nat(mm1) : U32))), y, nf, (i * mu : U32), loss))

# ---- the same rows over the array itself: no lists ----
# The list version above reads each row into a list and walks it five times. This one walks the row in place with tail
# loops, with the same operations in the same order (the max from the first cell, exp(x - max) summed from +0.0 left to
# right, p = e / s, the loss term read before the row is overwritten, (p - onehot) / n written back), so the same bits;
# MetalBend measured 3.4x fewer cycles on 64 rows of 49,152 (2026-10-25). The list version stays as the reference
# tests/ce_array.bend compares it with.

type AF is Type:
  AF{c: Array<F32>, x: F32}

# the max of k + 1 cells from idx, from m (the first cell's value, already read in r)
def cea_max(k: Nat, +idx: U32, +m: F32, r: Array<F32> & F32) -> AF:
  match k r:
    case 0n Tuple{c, +x}:
      AF{c, F32.max(m, x)}
    case 1n+p Tuple{c, +x}:
      cea_max(p, (idx + 1 : U32), F32.max(m, x), Array.get(F32, c, (idx + 1 : U32)))

# cells idx..idx + k become exp(cell - m); returns their sum from s, added left to right
def cea_exp(k: Nat, +idx: U32, +m: F32, +s: F32, r: Array<F32> & F32) -> AF:
  match k r:
    case 0n Tuple{c, +x}:
      +e = F32.exp((x - m : F32))
      AF{Array.set(F32, c, idx, e), (s + e : F32)}
    case 1n+p Tuple{c, +x}:
      +e = F32.exp((x - m : F32))
      cea_exp(p, (idx + 1 : U32), m, (s + e : F32), Array.get(F32, Array.set(F32, c, idx, e), (idx + 1 : U32)))

# cell idx (column j) becomes ((e / s) - onehot) / nf, for k + 1 cells
def cea_grad(k: Nat, +idx: U32, +j: Nat, +label: Nat, +s: F32, +nf: F32, r: Array<F32> & F32) -> Array<F32>:
  match k r:
    case 0n Tuple{c, +x}:
      Array.set(F32, c, idx, ((((x / s : F32) - hit(Nat.is_eq(label, j)) : F32) / nf) : F32))
    case 1n+p Tuple{c, +x}:
      cea_grad(p, (idx + 1 : U32), 1n+j, label, s, nf, Array.get(F32, Array.set(F32, c, idx, ((((x / s : F32) - hit(Nat.is_eq(label, j)) : F32) / nf) : F32)), (idx + 1 : U32)))

def cea_p2(+s: F32, r: Array<F32> & F32) -> AF:
  match r:
    case Tuple{c, +e}:
      AF{c, (e / s : F32)}

# p[label] = e[label] / s, or 0.0 for a label past the row (pick's Nil case)
def cea_p(inside: Bool, +o: U32, +label: Nat, +s: F32, c: Array<F32>) -> AF:
  match inside:
    case True{}:
      cea_p2(s, Array.get(F32, c, (o + U32.from_nat(label) : U32)))
    case False{}:
      AF{c, 0.0}

def cea_row3(+mm1: Nat, +o: U32, +label: Nat, +s: F32, +nf: F32, +loss: F32, q: AF) -> L2:
  match q:
    case AF{c, +p}:
      L2{(loss - F32.log(F32.max(p, 0.000000000001)) : F32), cea_grad(mm1, o, 0n, label, s, nf, Array.get(F32, c, o))}

def cea_row2(+mm1: Nat, +o: U32, +label: Nat, +nf: F32, +loss: F32, q: AF) -> L2:
  match q:
    case AF{c, +s}:
      cea_row3(mm1, o, label, s, nf, loss, cea_p(Nat.is_le(label, mm1), o, label, s, c))

def cea_row1(+mm1: Nat, +o: U32, +label: Nat, +nf: F32, +loss: F32, q: AF) -> L2:
  match q:
    case AF{c, +m}:
      cea_row2(mm1, o, label, nf, loss, cea_exp(mm1, o, m, 0.0, Array.get(F32, c, o)))

# a row's max starts from its first cell (maxl), then folds over the k + 1 cells from it
def cea_row0(+mm1: Nat, +o: U32, +label: Nat, +nf: F32, +loss: F32, r: Array<F32> & F32) -> L2:
  match r:
    case Tuple{c, +x0}:
      cea_row1(mm1, o, label, nf, loss, cea_max(mm1, o, x0, Array.get(F32, c, o)))

# one row per label, in order, as ce_rows
def cea_rows(labels: List<&2, Nat>, +mm1: Nat, +mu: U32, +nf: F32, +i: U32, r: L2) -> L2:
  match labels r:
    case Nil{} L2{loss, c}:
      L2{loss, c}
    case Con{+y, t} L2{+loss, c}:
      cea_rows(t, mm1, mu, nf, (i + 1 : U32), cea_row0(mm1, (i * mu : U32), y, nf, loss, Array.get(F32, c, (i * mu : U32))))

# ---- helper: hits per row ----

type CR is Type:
  CR{c: Array<F32>, n: Nat}

def gt_head(xs: List<&2, F32>, +v: F32) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Con{+h, t}:
      F32.is_gt(h, v)

def argmax2(xs: List<&2, F32>, g: Bool, +i: Nat, +bi: Nat, +bv: F32) -> Nat:
  match xs g:
    case Nil{} _:
      bi
    case Con{+h, +t} True{}:
      argmax2(t, gt_head(t, h), 1n+i, i, h)
    case Con{h, +t} False{}:
      argmax2(t, gt_head(t, bv), 1n+i, bi, bv)

def argmax(xs: List<&2, F32>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{+h, +t}:
      argmax2(t, gt_head(t, h), 1n, 0n, h)

def one(b: Bool) -> Nat:
  match b:
    case True{}:
      1n
    case False{}:
      0n

def count_rows3(+y: Nat, n: Nat, al: AL) -> CR:
  match al:
    case AL{c, l}:
      CR{c, Nat.add(n, one(Nat.is_eq(argmax(l), y)))}

def count_rows(labels: List<&2, Nat>, +mm1: Nat, +mu: U32, +i: U32, r: CR) -> CR:
  match labels r:
    case Nil{} CR{c, n}:
      CR{c, n}
    case Con{+y, t} CR{c, n}:
      count_rows(t, mm1, mu, (i + 1 : U32), count_rows3(y, n, read_l(mm1, ((i * mu : U32) + U32.from_nat(mm1) : U32), Nil{}, Array.get(F32, c, ((i * mu : U32) + U32.from_nat(mm1) : U32)))))


# =====================================================================
# Typed API: Mat<r, c> (the dimensions enter the type)
# =====================================================================

# =====================================================================
# Array capacity: a PROVED depth for the number of slots
# =====================================================================
#
# cap_depth(n) is the d such that an Array with 2^d slots holds n elements. It is computed by
# structural recursion on n (not with U32.log2) so that the law cap_ok below can prove
# `n <= 2^cap_depth(n)`. What stays trusted is only that Array.new(T, d, v) allocates 2^d slots.

# a type chosen by a Bool: rewriting through it refutes True == False
def BD(b: Bool, t: Type, f: Type) -> Type:
  match b:
    case True{}:
      t
    case False{}:
      f

# a <= b, by direct recursion
def leb(a: Nat, b: Nat) -> Bool:
  match a b:
    case 0n _:
      True{}
    case 1n+p 0n:
      False{}
    case 1n+p 1n+q:
      leb(p, q)

def pow2(d: Nat) -> Nat:
  match d:
    case 0n:
      1n
    case 1n++q:
      # bend-lint: disable-next bend-lint-rules/performance/repeated_call -- pow2 appears only in proofs' types not-expected-because: the package never runs it (NOTES 2026-10-17)
      Nat.add(pow2(q), pow2(q))

# smallest power of two >= n, by structural recursion on n: returns (depth, 2^depth)
def cap_pick(ok: Bool, d: Nat, +p: Nat) -> Nat & Nat:
  match ok:
    case True{}:
      (d, p)
    case False{}:
      (1n+d, Nat.add(p, p))

def cap_step(r: Nat & Nat, +n: Nat) -> Nat & Nat:
  match r:
    case Tuple{+d, +p}:
      cap_pick(leb(n, p), d, p)

def cap_pair(n: Nat) -> Nat & Nat:
  match n:
    case 0n:
      (0n, 1n)
    case 1n++m:
      cap_step(cap_pair(m), 1n+m)

def fst_nat(r: Nat & Nat) -> Nat:
  match r:
    case Tuple{d, p}:
      d

def snd_nat(r: Nat & Nat) -> Nat:
  match r:
    case Tuple{d, p}:
      p

# leb is Base's Nat.is_le, case for case. leb walks its arguments one step at a time in the compiled program; Base's
# comparison is one native comparison there (0.1.13.0).
def le_leb(a: Nat, b: Nat) -> {Nat.is_le(a, b) == leb(a, b) : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      le_leb(p, q)

def leb_of(+a: Nat, +b: Nat, e: {True{} == Nat.is_le(a, b) : Bool}) -> {True{} == leb(a, b) : Bool}:
  %le_leb(a, b) : {True{} == _ : Bool}
  e

# the same pair by doubling (0.1.10.0): p doubles until n <= p, log2(n) steps in a tail loop, where cap_pair takes n
# steps of a recursion that is not a tail call, each with a comparison of up to n steps. Its fuel is n, which always
# suffices (2^n >= n); if it ran out, the answer would be cap_pair's, so the law below needs no arithmetic about how
# many doublings there are. ok is Nat.is_le(n, p), computed by the caller (match only inspects a parameter); with leb
# it was n steps in all, 152 ms for n = 786,432 on one GPU lane (0.1.13.0).
def cap_dbl(f: Nat, ok: Bool, +n: Nat, +d: Nat, +p: Nat) -> Nat & Nat:
  match f ok:
    case _ True{}:
      (d, p)
    case 0n False{}:
      cap_pair(n)
    case 1n+g False{}:
      cap_dbl(g, Nat.is_le(n, Nat.add(p, p)), n, 1n+d, Nat.add(p, p))

# the depth d such that an Array with 2^d slots holds n elements (the smallest such d)
def cap_depth(+n: Nat) -> Nat:
  fst_nat(cap_dbl(n, Nat.is_le(n, 1n), n, 0n, 1n))

# ---- lemmas ----

# a <= b  ->  a <= 1 + b
def leb_succ(a: Nat, b: Nat, e: {True{} == leb(a, b) : Bool}) -> {True{} == leb(a, 1n+b) : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+p 0n:
      %e : BD(_, Unit, {True{} == leb(1n+p, 1n) : Bool})
      Unit{}
    case 1n+p 1n+q:
      leb_succ(p, q, e)

# a <= b  ->  a <= c + b
def leb_up_l(+a: Nat, +b: Nat, c: Nat, +e: {True{} == leb(a, b) : Bool}) -> {True{} == leb(a, Nat.add(c, b)) : Bool}:
  match c:
    case 0n:
      e
    case 1n++c2:
      leb_succ(a, Nat.add(c2, b), leb_up_l(a, b, c2, e))

# a <= b  ->  a <= b + c
def leb_up(a: Nat, b: Nat, c: Nat, e: {True{} == leb(a, b) : Bool}) -> {True{} == leb(a, Nat.add(b, c)) : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+p 0n:
      %e : BD(_, Unit, {True{} == leb(1n+p, Nat.add(0n, c)) : Bool})
      Unit{}
    case 1n+p 1n+q:
      leb_up(p, q, c, e)

# a <= b  and  c <= d  ->  a + c <= b + d
def leb_add(a: Nat, b: Nat, c: Nat, d: Nat, e1: {True{} == leb(a, b) : Bool}, e2: {True{} == leb(c, d) : Bool}) -> {True{} == leb(Nat.add(a, c), Nat.add(b, d)) : Bool}:
  match a b:
    case 0n _:
      leb_up_l(c, d, b, e2)
    case 1n+p 0n:
      %e1 : BD(_, Unit, {True{} == leb(Nat.add(1n+p, c), Nat.add(0n, d)) : Bool})
      Unit{}
    case 1n+p 1n+q:
      leb_add(p, q, c, d, e1, e2)

# 1 <= 2^d
def pow2_pos(d: Nat) -> {True{} == leb(1n, pow2(d)) : Bool}:
  match d:
    case 0n:
      {==}
    case 1n++q:
      leb_up(1n, pow2(q), pow2(q), pow2_pos(q))

# 1 <= p when p = 2^d
def pos_p(d: Nat, p: Nat, i1: {p == pow2(d) : Nat}) -> {True{} == leb(1n, p) : Bool}:
  %Equal.sym(Nat, p, pow2(d), i1) : {True{} == leb(1n, _) : Bool}
  pow2_pos(d)

# p + p == 2^(1+d) when p == 2^d
def step_eq(d: Nat, p: Nat, i1: {p == pow2(d) : Nat}) -> {Nat.add(p, p) == pow2(1n+d) : Nat}:
  %i1 : {Nat.add(p, p) == Nat.add(_, _) : Nat}
  {==}

# one step: given the invariant for m, the invariant for 1+m holds whichever way the comparison goes
def step_ok(ok: Bool, +d: Nat, +p: Nat, +m: Nat, +e: {ok == leb(1n+m, p) : Bool}, +i1: {p == pow2(d) : Nat}, +i2: {True{} == leb(m, p) : Bool}) -> {snd_nat(cap_pick(ok, d, p)) == pow2(fst_nat(cap_pick(ok, d, p))) : Nat} & {True{} == leb(1n+m, snd_nat(cap_pick(ok, d, p))) : Bool}:
  match ok:
    case True{}:
      (i1, e)
    case False{}:
      (step_eq(d, p, i1), leb_add(1n, p, m, p, pos_p(d, p, i1), i2))


# the invariant for cap_step on a pair, opened
def step_pair_ok(+m: Nat, r: Nat & Nat, i1: {snd_nat(r) == pow2(fst_nat(r)) : Nat}, i2: {True{} == leb(m, snd_nat(r)) : Bool}) -> {snd_nat(cap_step(r, 1n+m)) == pow2(fst_nat(cap_step(r, 1n+m))) : Nat} & {True{} == leb(1n+m, snd_nat(cap_step(r, 1n+m))) : Bool}:
  match r:
    case Tuple{+d, +p}:
      step_ok(leb(1n+m, p), d, p, m, {==}, i1, i2)

def cap_succ(+m: Nat, ih: {snd_nat(cap_pair(m)) == pow2(fst_nat(cap_pair(m))) : Nat} & {True{} == leb(m, snd_nat(cap_pair(m))) : Bool}) -> {snd_nat(cap_pair(1n+m)) == pow2(fst_nat(cap_pair(1n+m))) : Nat} & {True{} == leb(1n+m, snd_nat(cap_pair(1n+m))) : Bool}:
  match ih:
    case Tuple{i1, i2}:
      step_pair_ok(m, cap_pair(m), i1, i2)

# the invariant of cap_pair: the second component is 2^(first) and is >= n
def cap_inv(n: Nat) -> {snd_nat(cap_pair(n)) == pow2(fst_nat(cap_pair(n))) : Nat} & {True{} == leb(n, snd_nat(cap_pair(n))) : Bool}:
  match n:
    case 0n:
      ({==}, {==})
    case 1n++m:
      cap_succ(m, cap_inv(m))

# the invariant of cap_dbl: whichever way it ends (n <= p found, or the fuel out and cap_pair's answer), the second
# component is 2^(first) and is >= n
def dbl_inv(f: Nat, ok: Bool, +n: Nat, +d: Nat, +p: Nat, e: {ok == Nat.is_le(n, p) : Bool}, i1: {p == pow2(d) : Nat}) -> {snd_nat(cap_dbl(f, ok, n, d, p)) == pow2(fst_nat(cap_dbl(f, ok, n, d, p))) : Nat} & {True{} == leb(n, snd_nat(cap_dbl(f, ok, n, d, p))) : Bool}:
  match f ok:
    case _ True{}:
      (i1, leb_of(n, p, e))
    case 0n False{}:
      cap_inv(n)
    case 1n+g False{}:
      dbl_inv(g, Nat.is_le(n, Nat.add(p, p)), n, 1n+d, Nat.add(p, p), {==}, step_eq(d, p, i1))

def dbl_final(n: Nat, pr: {snd_nat(cap_dbl(n, Nat.is_le(n, 1n), n, 0n, 1n)) == pow2(fst_nat(cap_dbl(n, Nat.is_le(n, 1n), n, 0n, 1n))) : Nat} & {True{} == leb(n, snd_nat(cap_dbl(n, Nat.is_le(n, 1n), n, 0n, 1n))) : Bool}) -> {True{} == leb(n, pow2(cap_depth(n))) : Bool}:
  match pr:
    case Tuple{i1, i2}:
      %i1 : {True{} == leb(n, _) : Bool}
      i2

# LAW (cap_ok): an Array with 2^cap_depth(n) slots always has room for n elements.
law cap_ok:
  for +n: Nat
  {True{} == leb(n, pow2(cap_depth(n))) : Bool}

def cap_ok(n):
  dbl_final(n, dbl_inv(n, Nat.is_le(n, 1n), n, 0n, 1n, {==}, {==}))

# r x c of zeros, with the exact capacity (smallest power of 2)

def Mat.zeros(+r: Nat, +c: Nat) -> Mat<r, c>:
  Mat{Array.new(F32, cap_depth(Nat.mul(r, c)), 0.0)}

def Mat.fill(+r: Nat, +c: Nat, +v: F32) -> Mat<r, c>:
  Mat{Array.new(F32, cap_depth(Nat.mul(r, c)), v)}

def fill_list(xs: List<&2, F32>, +i: U32, a: Array<F32>) -> Array<F32>:
  match xs:
    case Nil{}:
      a
    case Con{h, t}:
      fill_list(t, (i + 1 : U32), Array.set(F32, a, i, h))

def len_is(xs: List<&2, F32>, n: Nat) -> Bool:
  match xs n:
    case Nil{} 0n:
      True{}
    case Con{h, t} 1n+p:
      len_is(t, p)
    case _ _:
      False{}

def Mat.of_list2(-r: Nat, -c: Nat, +rr: Nat, +cc: Nat, ok: Bool, xs: List<&2, F32>) -> Maybe<&1, Mat<r, c>>:
  match ok:
    case True{}:
      Some{Mat{fill_list(xs, 0, Array.new(F32, cap_depth(Nat.mul(rr, cc)), 0.0))}}
    case False{}:
      None{}

# Mat<r, c> from r*c numbers in row order; None if the size does not match

def Mat.of_list(+r: Nat, +c: Nat, +xs: List<&2, F32>) -> Maybe<&1, Mat<r, c>>:
  Mat.of_list2(r, c, r, c, len_is(xs, Nat.mul(r, c)), xs)

# reads the first n numbers, returning the matrix

# Mat<r, c> without checking the list size (a short list leaves the rest at zero, a long one is cut
# by the capacity): only for when the size is known by construction. Prefer Mat.of_list.

def Mat.from_list(+r: Nat, +c: Nat, xs: List<&2, F32>) -> Mat<r, c>:
  Mat{fill_list(xs, 0, Array.new(F32, cap_depth(Nat.mul(r, c)), 0.0))}

# writes the numbers of xs into m starting at flat index i (row-major); numbers past the
# capacity are dropped. For loading a large matrix block by block without building one big list.
def Mat.fill_at(-r: Nat, -c: Nat, +i: U32, xs: List<&2, F32>, m: Mat<r, c>) -> Mat<r, c>:
  match m:
    case Mat{d}:
      Mat{fill_list(xs, i, d)}

type ML<-r: Nat, -c: Nat> is Type:
  ML{m: Mat<r, c>, l: List<&2, F32>}

def Mat.to_list2(-r: Nat, -c: Nat, al: AL) -> ML<r, c>:
  match al:
    case AL{d, l}:
      ML{Mat{d}, l}

def ML.len(-r: Nat, -c: Nat, ml: ML<r, c>) -> Nat:
  match ml:
    case ML{m, l}:
      List.length(&2, F32, l)

# n numbers from index 0, in row order; n = 0 gives the empty list (a Mat with no numbers)
def Mat.to_list_count(-r: Nat, -c: Nat, d: Array<F32>, +n: Nat) -> ML<r, c>:
  match n:
    case 0n:
      ML{Mat{d}, Nil{}}
    case 1n+p:
      Mat.to_list2(r, c, read_l(p, (U32.from_nat(n) - 1 : U32), Nil{}, Array.get(F32, d, (U32.from_nat(n) - 1 : U32))))

# two independent copies

def Mat.to_list(+r: Nat, +c: Nat, m: Mat<r, c>) -> ML<r, c>:
  match m:
    case Mat{d}:
      Mat.to_list_count(r, c, d, Nat.mul(r, c))

type MM2<-r: Nat, -c: Nat> is Type:
  MM2{a: Mat<r, c>, b: Mat<r, c>}

def Mat.clone2(-r: Nat, -c: Nat, p: Array<F32> & Array<F32>) -> MM2<r, c>:
  match p:
    case Tuple{x, y}:
      MM2{Mat{x}, Mat{y}}

# ---- products: C (n x m) = A · B, with 2^par blocks of rows in parallel ----

# C = A(n x k) · B(k x m)

def Mat.clone(-r: Nat, -c: Nat, m: Mat<r, c>) -> MM2<r, c>:
  match m:
    case Mat{d}:
      Mat.clone2(r, c, Array.clone(F32, d))

type MMul<-n: Nat, -k: Nat, -m: Nat> is Type:
  MMul{a: Mat<n, k>, b: Mat<k, m>, c: Mat<n, m>}

# C = A(n x k) · Bᵀ, with B stored m x k

type MMulNT<-n: Nat, -k: Nat, -m: Nat> is Type:
  MMulNT{a: Mat<n, k>, b: Mat<m, k>, c: Mat<n, m>}

# C = Aᵀ · B, with A stored k x n

type MMulTN<-n: Nat, -k: Nat, -m: Nat> is Type:
  MMulTN{a: Mat<k, n>, b: Mat<k, m>, c: Mat<n, m>}

# ---------------------------------------------------------------------
# gemm in tiles of 8 (0.1.8.0): for products with 2 or more rows, A's rows and B's columns are packed once into cells of
# 8 F32 (a row of k numbers is ceil(k/8) cells, zeros after the last), so the inner loop reads 8 numbers per array
# access instead of one. The order of the additions is that of `dotr`: product p of a dot goes to partial sum p mod 4,
# in order, and the four sums are combined as dotr combines them after k steps (its rotation depends on k mod 4). The
# padding adds +0 to a sum that starts at +0, which changes nothing, so the results are the same bits as the scalar
# loop's. Products with one row (matrix · vector) keep the scalar loop: packing B would cost as much as the product.
# ---------------------------------------------------------------------

type T8 is Data:
  T8{v0: F32, v1: F32, v2: F32, v3: F32, v4: F32, v5: F32, v6: F32, v7: F32}

def t8_zero() -> T8:
  T8{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

def pick0(b: Bool, +v: F32) -> F32:
  match b:
    case True{}:
      v
    case False{}:
      0.0

type PK is Type:
  PK{a: Array<F32>, t: Array<T8>}

# one cell: the 8 numbers of a at idx, idx + st, ..., those at positions q0 + q >= lim (past the end of the line) as 0
def pk_g7(+v0: F32, +v1: F32, +v2: F32, +v3: F32, +v4: F32, +v5: F32, +v6: F32, +q0: U32, +lim: U32, +idx: U32, +st: U32, +o: U32, t: Array<T8>, r: Array<F32> & F32) -> PK:
  match r:
    case Tuple{a, +v}:
      PK{a, Array.set(T8, t, o, T8{v0, v1, v2, v3, v4, v5, v6, pick0(U32.is_lt((q0 + 7 : U32), lim), v)})}

def pk_g6(+v0: F32, +v1: F32, +v2: F32, +v3: F32, +v4: F32, +v5: F32, +q0: U32, +lim: U32, +idx: U32, +st: U32, +o: U32, t: Array<T8>, r: Array<F32> & F32) -> PK:
  match r:
    case Tuple{a, +v}:
      pk_g7(v0, v1, v2, v3, v4, v5, pick0(U32.is_lt((q0 + 6 : U32), lim), v), q0, lim, idx, st, o, t, Array.get(F32, a, (idx + (st * 7 : U32) : U32)))

def pk_g5(+v0: F32, +v1: F32, +v2: F32, +v3: F32, +v4: F32, +q0: U32, +lim: U32, +idx: U32, +st: U32, +o: U32, t: Array<T8>, r: Array<F32> & F32) -> PK:
  match r:
    case Tuple{a, +v}:
      pk_g6(v0, v1, v2, v3, v4, pick0(U32.is_lt((q0 + 5 : U32), lim), v), q0, lim, idx, st, o, t, Array.get(F32, a, (idx + (st * 6 : U32) : U32)))

def pk_g4(+v0: F32, +v1: F32, +v2: F32, +v3: F32, +q0: U32, +lim: U32, +idx: U32, +st: U32, +o: U32, t: Array<T8>, r: Array<F32> & F32) -> PK:
  match r:
    case Tuple{a, +v}:
      pk_g5(v0, v1, v2, v3, pick0(U32.is_lt((q0 + 4 : U32), lim), v), q0, lim, idx, st, o, t, Array.get(F32, a, (idx + (st * 5 : U32) : U32)))

def pk_g3(+v0: F32, +v1: F32, +v2: F32, +q0: U32, +lim: U32, +idx: U32, +st: U32, +o: U32, t: Array<T8>, r: Array<F32> & F32) -> PK:
  match r:
    case Tuple{a, +v}:
      pk_g4(v0, v1, v2, pick0(U32.is_lt((q0 + 3 : U32), lim), v), q0, lim, idx, st, o, t, Array.get(F32, a, (idx + (st * 4 : U32) : U32)))

def pk_g2(+v0: F32, +v1: F32, +q0: U32, +lim: U32, +idx: U32, +st: U32, +o: U32, t: Array<T8>, r: Array<F32> & F32) -> PK:
  match r:
    case Tuple{a, +v}:
      pk_g3(v0, v1, pick0(U32.is_lt((q0 + 2 : U32), lim), v), q0, lim, idx, st, o, t, Array.get(F32, a, (idx + (st * 3 : U32) : U32)))

def pk_g1(+v0: F32, +q0: U32, +lim: U32, +idx: U32, +st: U32, +o: U32, t: Array<T8>, r: Array<F32> & F32) -> PK:
  match r:
    case Tuple{a, +v}:
      pk_g2(v0, pick0(U32.is_lt((q0 + 1 : U32), lim), v), q0, lim, idx, st, o, t, Array.get(F32, a, (idx + (st * 2 : U32) : U32)))

def pk_g0(+q0: U32, +lim: U32, +idx: U32, +st: U32, +o: U32, t: Array<T8>, r: Array<F32> & F32) -> PK:
  match r:
    case Tuple{a, +v}:
      pk_g1(pick0(U32.is_lt((q0 + 0 : U32), lim), v), q0, lim, idx, st, o, t, Array.get(F32, a, (idx + (st * 1 : U32) : U32)))

# the cells of one line, cell o.. of t; q0 counts the line's numbers before the cell
def pk_line(tl: Nat, +q0: U32, +lim: U32, +base: U32, +st: U32, +o: U32, r: PK) -> PK:
  match tl r:
    case 0n r:
      r
    case 1n+q PK{a, t}:
      pk_line(q, (q0 + 8 : U32), lim, (base + (st * 8 : U32) : U32), st, (o + 1 : U32), pk_g0(q0, lim, base, st, o, t, Array.get(F32, a, base)))

# lines count lines; source line i starts at i * lstep (in a), and goes to cell row o (cells o * tku .. in t)
def pk_lines(cnt: Nat, +tk: Nat, +tku: U32, +ku: U32, +i: U32, +o: U32, +lstep: U32, +st: U32, r: PK) -> PK:
  match cnt:
    case 0n:
      r
    case 1n+p:
      pk_lines(p, tk, tku, ku, (i + 1 : U32), (o + 1 : U32), lstep, st, pk_line(tk, 0, ku, (i * lstep : U32), st, (o * tku : U32), r))

def pk_pack(+cnt: Nat, +tk: Nat, +k: Nat, +lstep: U32, +st: U32, a: Array<F32>) -> PK:
  pk_lines(cnt, tk, U32.from_nat(tk), U32.from_nat(k), 0, 0, lstep, st, PK{a, Array.new(T8, cap_depth(Nat.mul(cnt, tk)), t8_zero())})

# the dot of two lines of cells: tk cells from ia (in a) and ib (in b); ra, rb hold the arrays and the cells at ia, ib,
# already read. Product q of a cell goes to sum q mod 4, the first four before the last four, as dotr adds them.
def t8_dot(tk: Nat, +ia: U32, +ib: U32, s0: F32, s1: F32, s2: F32, s3: F32, ra: Array<T8> & T8, rb: Array<T8> & T8) -> TD:
  match tk ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      TD{a, b, s0, s1, s2, s3}
    case 1n+p Tuple{a, T8{x0, x1, x2, x3, x4, x5, x6, x7}} Tuple{b, T8{y0, y1, y2, y3, y4, y5, y6, y7}}:
      t8_dot(p, (ia + 1 : U32), (ib + 1 : U32), ((s0 + (x0 * y0 : F32) : F32) + (x4 * y4 : F32) : F32), ((s1 + (x1 * y1 : F32) : F32) + (x5 * y5 : F32) : F32), ((s2 + (x2 * y2 : F32) : F32) + (x6 * y6 : F32) : F32), ((s3 + (x3 * y3 : F32) : F32) + (x7 * y7 : F32) : F32), Array.get(T8, a, (ia + 1 : U32)), Array.get(T8, b, (ib + 1 : U32)))

type TD is Type:
  TD{a: Array<T8>, b: Array<T8>, s0: F32, s1: F32, s2: F32, s3: F32}

# dotr's final sum after k products: the sums sit rotated by k mod 4 when its loop ends
def t8_comb(+rot: Nat, +s0: F32, +s1: F32, +s2: F32, +s3: F32) -> F32:
  match rot:
    case 0n:
      ((s0 + s1 : F32) + (s2 + s3 : F32) : F32)
    case 1n:
      ((s1 + s2 : F32) + (s3 + s0 : F32) : F32)
    case 2n:
      ((s2 + s3 : F32) + (s0 + s1 : F32) : F32)
    case _:
      ((s3 + s0 : F32) + (s1 + s2 : F32) : F32)

type TR is Type:
  TR{a: Array<T8>, b: Array<T8>, x: F32}

def t8_fin(+rot: Nat, d: TD) -> TR:
  match d:
    case TD{a, b, +s0, +s1, +s2, +s3}:
      TR{a, b, t8_comb(rot, s0, s1, s2, s3)}

def t8_cell(+tk: Nat, +tku: U32, +rot: Nat, +i: U32, +j: U32, a: Array<T8>, b: Array<T8>) -> TR:
  t8_fin(rot, t8_dot(tk, (i * tku : U32), (j * tku : U32), 0.0, 0.0, 0.0, 0.0, Array.get(T8, a, (i * tku : U32)), Array.get(T8, b, (j * tku : U32))))

type TG is Type:
  TG{a: Array<T8>, b: Array<T8>, c: Array<F32>}

def t8_set(+ac: Bool, +idx: U32, c: Array<F32>, r: TR) -> TG:
  match r:
    case TR{a, b, +x}:
      TG{a, b, cwr(ac, idx, x, c)}

def t8_cols(+ac: Bool, mleft: Nat, +tk: Nat, +tku: U32, +rot: Nat, +i: U32, +j: U32, +cidx: U32, g: TG) -> TG:
  match mleft g:
    case 0n g:
      g
    case 1n+p TG{a, b, c}:
      t8_cols(ac, p, tk, tku, rot, i, (j + 1 : U32), (cidx + 1 : U32), t8_set(ac, cidx, c, t8_cell(tk, tku, rot, i, j, a, b)))

def t8_rows(+ac: Bool, nleft: Nat, +m: Nat, +tk: Nat, +tku: U32, +rot: Nat, +i: U32, +mu: U32, g: TG) -> TG:
  match nleft:
    case 0n:
      g
    case 1n+p:
      t8_rows(ac, p, m, tk, tku, rot, (i + 1 : U32), mu, t8_cols(ac, m, tk, tku, rot, i, 0, (i * mu : U32), g))

# rows i0 .. i0+cnt-1 of C as a list, in order (the parallel version's leaves)
type TL is Type:
  TL{a: Array<T8>, b: Array<T8>, out: List<&2, F32>}

# cells j.. of row i, m - j of them; r is cell j, already computed
def t8_lcols(mleft: Nat, +tk: Nat, +tku: U32, +rot: Nat, +i: U32, +j: U32, out: List<&2, F32>, r: TR) -> TL:
  match mleft r:
    case 0n TR{a, b, +x}:
      TL{a, b, x <> out}
    case 1n+p TR{a, b, +x}:
      t8_lcols(p, tk, tku, rot, i, (j + 1 : U32), x <> out, t8_cell(tk, tku, rot, i, (j + 1 : U32), a, b))

def t8_lrows(nleft: Nat, +m: Nat, +tk: Nat, +tku: U32, +rot: Nat, +i: U32, l: TL) -> TL:
  match nleft l:
    case 0n l:
      l
    case 1n+p TL{a, b, out}:
      t8_lrows(p, m, tk, tku, rot, (i + 1 : U32), t8_lcols(Nat.sub(m, 1n), tk, tku, rot, i, 0, out, t8_cell(tk, tku, rot, i, 0, a, b)))

def t8_lfin(l: TL) -> List<&2, F32>:
  match l:
    case TL{a, b, out}:
      List.reverse(&2, F32, out)

type PKK is Type:
  PKK{a: Array<F32>, b: Array<F32>, ta: Array<T8>, tb: Array<T8>}

def t8_pk2(pa: PK, pb: PK) -> PKK:
  match pa pb:
    case PK{a, ta} PK{b, tb}:
      PKK{a, b, ta, tb}

def t8_g(a: Array<F32>, b: Array<F32>, g: TG) -> G:
  match g:
    case TG{ta, tb, c}:
      G{a, b, c}

def t8_out_seq(+ac: Bool, +n: Nat, +m: Nat, +tk: Nat, +tku: U32, +rot: Nat, c: Array<F32>, p: PKK) -> G:
  match p:
    case PKK{a, b, ta, tb}:
      t8_g(a, b, t8_rows(ac, n, m, tk, tku, rot, 0, U32.from_nat(m), TG{ta, tb, c}))

# ---- the parallel path (0.1.9.0): the larger operand is packed into one piece per block, so a block owns its piece;
# the other operand is copied once per node of the fork tree (2^d - 1 copies), instead of both at every node ----

type PT is Type:
  PLeaf{t: Array<T8>}
  PNode{l: PT, r: PT}

type PKT is Type:
  PKT{a: Array<F32>, t: PT}

def pkt_leaf(r: PK) -> PKT:
  match r:
    case PK{a, t}:
      PKT{a, PLeaf{t}}

def pkt_then(r: PKT, k: Array<F32> -> PT -> PKT) -> PKT:
  match r:
    case PKT{a, t}:
      k(a, t)

# lines i .. i+cnt-1 of a, packed into 2^d pieces (halves of the line count, as the fork tree splits them)
def pk_tree(+d: Nat, +cnt: Nat, +i: U32, +tk: Nat, +tku: U32, +ku: U32, +lstep: U32, +st: U32, a: Array<F32>) -> PKT:
  match d:
    case 0n:
      pkt_leaf(pk_lines(cnt, tk, tku, ku, i, 0, lstep, st, PK{a, Array.new(T8, cap_depth(Nat.mul(cnt, tk)), t8_zero())}))
    case 1n+q:
      pkt_then(pk_tree(q, Nat.div(cnt, 2n), i, tk, tku, ku, lstep, st, a), a2 => l => pkt_then(pk_tree(q, Nat.sub(cnt, Nat.div(cnt, 2n)), (i + U32.from_nat(Nat.div(cnt, 2n)) : U32), tk, tku, ku, lstep, st, a2), a3 => r => PKT{a3, PNode{l, r}}))

def t8_with2(-R: Type, p: Array<T8> & Array<T8>, k: Array<T8> -> Array<T8> -> R) -> R:
  match p:
    case Tuple{x, y}:
      k(x, y)

# split by rows of C: the pieces are A's rows, B is shared; the result is C's rows in order
def t8_rpar(t: PT, +cnt: Nat, +m: Nat, +tk: Nat, +tku: U32, +rot: Nat, b: Array<T8>) -> List<&2, F32>:
  match t:
    case PLeaf{a}:
      t8_lfin(t8_lrows(cnt, m, tk, tku, rot, 0, TL{a, b, Nil{}}))
    case PNode{l, r}:
      t8_with2(List<&2, F32>, Array.clone(T8, b), b1 => b2 => (
        x y = t8_rpar(l, Nat.div(cnt, 2n), m, tk, tku, rot, b1) t8_rpar(r, Nat.sub(cnt, Nat.div(cnt, 2n)), m, tk, tku, rot, b2)
        List.append(&2, F32, x, y)))

# a block of C's columns j0 .. j0+w-1 for every row, row by row
type CB is Data:
  CB{j0: U32, w: Nat, xs: List<&2, F32>}

# split by columns of C: the pieces are B's columns, A is shared
def t8_cpar(t: PT, +cnt: Nat, +j0: U32, +n: Nat, +tk: Nat, +tku: U32, +rot: Nat, a: Array<T8>) -> List<&2, CB>:
  match t:
    case PLeaf{b}:
      [CB{j0, cnt, t8_lfin(t8_lrows(n, cnt, tk, tku, rot, 0, TL{a, b, Nil{}}))}]
    case PNode{l, r}:
      t8_with2(List<&2, CB>, Array.clone(T8, a), a1 => a2 => (
        x y = t8_cpar(l, Nat.div(cnt, 2n), j0, n, tk, tku, rot, a1) t8_cpar(r, Nat.sub(cnt, Nat.div(cnt, 2n)), (j0 + U32.from_nat(Nat.div(cnt, 2n)) : U32), n, tk, tku, rot, a2)
        List.append(&2, CB, x, y)))

type RW is Type:
  RW{c: Array<F32>, xs: List<&2, F32>}

# w numbers of xs into c from idx; what is left of xs comes back
def cb_row(+ac: Bool, w: Nat, +idx: U32, c: Array<F32>, xs: List<&2, F32>) -> RW:
  match w xs:
    case 1n+p Con{+h, t}:
      cb_row(ac, p, (idx + 1 : U32), cwr(ac, idx, h, c), t)
    case _ _:
      RW{c, xs}

def cb_rows(+ac: Bool, n: Nat, +i: U32, +j0: U32, +w: Nat, +mu: U32, r: RW) -> Array<F32>:
  match n r:
    case 0n RW{c, xs}:
      c
    case 1n+p RW{c, xs}:
      cb_rows(ac, p, (i + 1 : U32), j0, w, mu, cb_row(ac, w, ((i * mu : U32) + j0 : U32), c, xs))

def cb_write(+ac: Bool, bs: List<&2, CB>, +n: Nat, +mu: U32, c: Array<F32>) -> Array<F32>:
  match bs:
    case Nil{}:
      c
    case Con{CB{+j0, +w, xs}, t}:
      cb_write(ac, t, n, mu, cb_rows(ac, n, 0, j0, w, mu, RW{c, xs}))

def t8_pr2(+ac: Bool, +n: Nat, +m: Nat, +tk: Nat, +tku: U32, +rot: Nat, c: Array<F32>, ra: PKT, rb: PK) -> G:
  match ra rb:
    case PKT{a, t} PK{b, tb}:
      G{a, b, write_la(ac, t8_rpar(t, n, m, tk, tku, rot, tb), 0, c)}

def t8_pc2(+ac: Bool, +n: Nat, +m: Nat, +tk: Nat, +tku: U32, +rot: Nat, c: Array<F32>, ra: PK, rb: PKT) -> G:
  match ra rb:
    case PK{a, ta} PKT{b, t}:
      G{a, b, cb_write(ac, t8_cpar(t, m, 0, n, tk, tku, rot, ta), n, U32.from_nat(m), c)}

# 2^d blocks along the larger of n (rows of C, pieces of A) and m (columns of C, pieces of B)
def t8_par2(+ac: Bool, byrows: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +tk: Nat, +tku: U32, +rot: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  match byrows:
    case True{}:
      t8_pr2(ac, n, m, tk, tku, rot, c, pk_tree(d, n, 0, tk, tku, U32.from_nat(k), arow, sa, a), pk_pack(m, tk, k, bcol, sb, b))
    case False{}:
      t8_pc2(ac, n, m, tk, tku, rot, c, pk_pack(n, tk, k, arow, sa, a), pk_tree(d, m, 0, tk, tku, U32.from_nat(k), bcol, sb, b))

def gemm_t8b(+ac: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +tk: Nat, +tku: U32, +rot: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  match d:
    case 0n:
      t8_out_seq(ac, n, m, tk, tku, rot, c, t8_pk2(pk_pack(n, tk, k, arow, sa, a), pk_pack(m, tk, k, bcol, sb, b)))
    case 1n+q:
      t8_par2(ac, Nat.is_le(m, n), 1n+q, n, m, k, tk, tku, rot, arow, sa, sb, bcol, a, b, c)

# C (n x m) = A · B (or C + A · B when ac) in cells of 8, sequential (d = 0) or with 2^d blocks of rows in parallel; A
# and B come back intact
def gemm_t8_ac(+ac: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  gemm_t8b(ac, d, n, m, k, Nat.div(Nat.add(k, 7n), 8n), U32.from_nat(Nat.div(Nat.add(k, 7n), 8n)), Nat.mod(k, 4n), arow, sa, sb, bcol, a, b, c)

def gemm_t8(d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  gemm_t8_ac(False{}, d, n, m, k, arow, sa, sb, bcol, a, b, c)

# one row (matrix · vector): the scalar loop at every depth; two or more rows: cells of 8. Up to 0.1.11.0 one row in
# parallel ran gemm_par_cols, which clones both operands at every node of the fork tree: a 50,257 x 768 weight matrix at
# depth 4 ran out of the default heap (5.1 GB) and 8192 x 2048 took 124 ms against 10 ms sequentially. Packing the
# matrix into cells of 8 instead costs more than the product, which reads each weight once (64 ms against 24 at
# 50,257 x 768). Matrix · vector in parallel is Bands.matvec. Measured 2026-10-16 (NOTES.md).
def gemm_any3(+ac: Bool, one: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  match one:
    case True{}:
      gemm_ac(ac, n, m, k, arow, sa, sb, bcol, a, b, c)
    case False{}:
      gemm_t8_ac(ac, d, n, m, k, arow, sa, sb, bcol, a, b, c)

def gemm_any2(+ac: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  gemm_any3(ac, Nat.is_le(n, 1n), d, n, m, k, arow, sa, sb, bcol, a, b, c)

def gemm_any(+ac: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  gemm_any2(ac, d, n, m, k, arow, sa, sb, bcol, a, b, c)

def mmul_fin(-n: Nat, -k: Nat, -m: Nat, g: G) -> MMul<n, k, m>:
  match g:
    case G{a, b, c}:
      MMul{Mat{a}, Mat{b}, Mat{c}}

def mmul_nt_fin(-n: Nat, -k: Nat, -m: Nat, g: G) -> MMulNT<n, k, m>:
  match g:
    case G{a, b, c}:
      MMulNT{Mat{a}, Mat{b}, Mat{c}}

def mmul_tn_fin(-n: Nat, -k: Nat, -m: Nat, g: G) -> MMulTN<n, k, m>:
  match g:
    case G{a, b, c}:
      MMulTN{Mat{a}, Mat{b}, Mat{c}}

# ---- the split depth of a product (0.1.11.0): par is an upper bound, and a product too small to gain splits 0 times ----
#
# A product of n x k by k x m splits into 2^par blocks only when it does at least 2^23 multiply-adds and each cell of C
# sums at least 64 of them; otherwise it runs sequentially. Measured on 16 threads (2026-10-16, 18 shapes): below 2^23 the
# split costs more than it saves, and with a short inner dimension (a weight gradient over a few tokens) every split
# was 2-3.6x slower, because each block hands its rows of C back as a list that is then written into C, a cost per cell
# of C that k multiply-adds per cell do not pay for. The result does not depend on the depth: the same bits at every
# depth (tests/t8.bend, tests/acc.bend, tests/split.bend).
def split_pick(big: Bool, d: Nat) -> Nat:
  match big:
    case True{}:
      d
    case False{}:
      0n

def split_depth(d: Nat, +n: Nat, +m: Nat, +k: Nat) -> Nat:
  split_pick(Bool.and(Nat.is_le(64n, k), Nat.is_le(8388608n, Nat.mul(Nat.mul(n, m), k))), d)

def mm_a(+n: Nat, +k: Nat, +m: Nat, d: Nat, a: Array<F32>, b: Array<F32>) -> MMul<n, k, m>:
  mmul_fin(n, k, m, gemm_any(False{}, d, n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, a, b, Array.new(F32, cap_depth(Nat.mul(n, m)), 0.0)))

# C = A · B   (n x k) · (k x m); par = log2 of the most parallel blocks (0 = sequential); split_depth picks 0 or par

def Mat.matmul(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<n, k>, b: Mat<k, m>) -> MMul<n, k, m>:
  match a b:
    case Mat{x} Mat{y}:
      mm_a(n, k, m, split_depth(par, n, m, k), x, y)

# C + A · B, into C (n x m); A and B come back with the result, as Mat.matmul returns them. Each cell of A · B is
# computed as Mat.matmul computes it and added once to C's cell: the same bits as Mat.add(c, Mat.matmul(a, b)'s c),
# without allocating the product or walking it a second time.

def Mat.matmul_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<n, k>, b: Mat<k, m>, c: Mat<n, m>) -> MMul<n, k, m>:
  match a b c:
    case Mat{x} Mat{y} Mat{z}:
      mmul_fin(n, k, m, gemm_any(True{}, split_depth(par, n, m, k), n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y, z))

def mm_nt(+n: Nat, +k: Nat, +m: Nat, d: Nat, a: Array<F32>, b: Array<F32>) -> MMulNT<n, k, m>:
  mmul_nt_fin(n, k, m, gemm_any(False{}, d, n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), a, b, Array.new(F32, cap_depth(Nat.mul(n, m)), 0.0)))

# C = A · Bᵀ   (n x k) · (m x k)ᵀ: for weights stored with one row per output

def Mat.matmul_nt(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<n, k>, b: Mat<m, k>) -> MMulNT<n, k, m>:
  match a b:
    case Mat{x} Mat{y}:
      mm_nt(n, k, m, split_depth(par, n, m, k), x, y)

# C + A · Bᵀ, into C (n x m)

def Mat.matmul_nt_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<n, k>, b: Mat<m, k>, c: Mat<n, m>) -> MMulNT<n, k, m>:
  match a b c:
    case Mat{x} Mat{y} Mat{z}:
      mmul_nt_fin(n, k, m, gemm_any(True{}, split_depth(par, n, m, k), n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y, z))

def mm_tn(+n: Nat, +k: Nat, +m: Nat, d: Nat, a: Array<F32>, b: Array<F32>) -> MMulTN<n, k, m>:
  mmul_tn_fin(n, k, m, gemm_any(False{}, d, n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, a, b, Array.new(F32, cap_depth(Nat.mul(n, m)), 0.0)))

# C = Aᵀ · B   (k x n)ᵀ · (k x m): weight gradient, X^T · dY

def Mat.matmul_tn(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<k, n>, b: Mat<k, m>) -> MMulTN<n, k, m>:
  match a b:
    case Mat{x} Mat{y}:
      mm_tn(n, k, m, split_depth(par, n, m, k), x, y)

# C + Aᵀ · B, into C (n x m): a weight gradient added into the gradient it accumulates, dW += Xᵀ · dY

def Mat.matmul_tn_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<k, n>, b: Mat<k, m>, c: Mat<n, m>) -> MMulTN<n, k, m>:
  match a b c:
    case Mat{x} Mat{y} Mat{z}:
      mmul_tn_fin(n, k, m, gemm_any(True{}, split_depth(par, n, m, k), n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y, z))

# ---- a precise product for long dots (0.1.14.0) ----
#
# Mat.matmul_precise is Mat.matmul with k summed in blocks of 1024 (the last one 1..1024): block b's columns of A and
# rows of B are copied out, the first block's product written into C, each next one's added to it as matmul_acc adds,
# so every cell is ((x0 + x1) + x2) .., x_b the block's dot with the usual four partial sums. A dot of 49,152 products
# as four running sums of 12,288 lost the small terms (SmolLM2's head: its gradient 1.8x further from float64 than
# PyTorch's float32, 0.03x in blocks; BendTorch NOTES, 2026-10-18). Opt-in: it copies the slices and runs one product
# a block, and the default products keep their speed and bits. Up to 1024 steps it is Mat.matmul itself.

type SL is Type:
  SL{a: Array<F32>, s: Array<F32>}

# d[di..] = src[si..], k numbers; r holds src[si], already read
def sl_row(k: Nat, +si: U32, +di: U32, d: Array<F32>, r: Array<F32> & F32) -> SL:
  match k r:
    case 0n Tuple{a, x}: SL{a, d}
    case 1n+p Tuple{a, +x}: sl_row(p, (si + 1 : U32), (di + 1 : U32), Array.set(F32, d, di, x), Array.get(F32, a, (si + 1 : U32)))

# rows i.. of a slice: w numbers a row from src[i*st + o], into d[i*w]
def sl_rows(r: Nat, +i: U32, +w: Nat, +w32: U32, +st: U32, +o: U32, sl: SL) -> SL:
  match r sl:
    case 0n sl: sl
    case 1n+p SL{a, d}: sl_rows(p, (i + 1 : U32), w, w32, st, o, sl_row(w, ((i * st : U32) + o : U32), (i * w32 : U32), d, Array.get(F32, a, ((i * st : U32) + o : U32))))

def blk_len(+bi: U32, +k: U32) -> Nat:
  U32.to_nat(Bool.pick(U32, U32.is_lt((k - (bi * 1024 : U32) : U32), 1024), (k - (bi * 1024 : U32) : U32), 1024))

def mp_s4(a: Array<F32>, b: Array<F32>, g: G) -> G:
  match g:
    case G{x, y, c}: G{a, b, c}

def mp_s3(+ac: Bool, +par: Nat, +n: Nat, +m: Nat, +w: Nat, +w32: U32, +m32: U32, a: Array<F32>, as: Array<F32>, c: Array<F32>, sb: SL) -> G:
  match sb:
    case SL{b, bs}: mp_s4(a, b, gemm_any(ac, split_depth(par, n, m, w), n, m, w, w32, 1, m32, 1, as, bs, c))

def mp_s2(+ac: Bool, +par: Nat, +n: Nat, +m: Nat, +w: Nat, +w32: U32, +m32: U32, +o: U32, c: Array<F32>, b: Array<F32>, sa: SL) -> G:
  match sa:
    case SL{a, as}: mp_s3(ac, par, n, m, w, w32, m32, a, as, c, sl_row(Nat.mul(w, m), (o * m32 : U32), 0, Array.new(F32, cap_depth(Nat.mul(w, m)), 0.0), Array.get(F32, b, (o * m32 : U32))))

# block bi: A's columns o..o+w (A is n x k) and B's rows o..o+w (B is k x m), multiplied into C (added when ac)
def mp_step(+ac: Bool, +par: Nat, +n: Nat, +m: Nat, +k32: U32, +bi: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  mp_s2(ac, par, n, m, blk_len(bi, k32), U32.from_nat(blk_len(bi, k32)), U32.from_nat(m), (bi * 1024 : U32), c, b, sl_rows(n, 0, blk_len(bi, k32), U32.from_nat(blk_len(bi, k32)), k32, (bi * 1024 : U32), SL{a, Array.new(F32, cap_depth(Nat.mul(n, blk_len(bi, k32))), 0.0)}))

def mp_go(nb: Nat, +bi: U32, +par: Nat, +n: Nat, +m: Nat, +k32: U32, g: G) -> G:
  match nb g:
    case 0n g: g
    case 1n++q G{a, b, c}: mp_go(q, (bi + 1 : U32), par, n, m, k32, mp_step(U32.is_lt(0, bi), par, n, m, k32, bi, a, b, c))

def mm_precise2(long: Bool, +n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<n, k>, b: Mat<k, m>) -> MMul<n, k, m>:
  match long a b:
    case True{} Mat{x} Mat{y}: mmul_fin(n, k, m, mp_go(U32.to_nat(U32.div((U32.from_nat(k) + 1023 : U32), 1024)), 0, par, n, m, U32.from_nat(k), G{x, y, Array.new(F32, cap_depth(Nat.mul(n, m)), 0.0)}))
    case False{} a b: Mat.matmul(n, k, m, par, a, b)

# C = A · B (n x k) · (k x m), k summed in blocks of 1024 (above); opt-in
def Mat.matmul_precise(+n: Nat, +k: Nat, +m: Nat, +par: Nat, a: Mat<n, k>, b: Mat<k, m>) -> MMul<n, k, m>:
  mm_precise2(Nat.is_lt(1024n, k), n, k, m, par, a, b)

# ---- element-wise and training operations ----

# adds the bias (one row, m numbers) to each row of C (n x m)

type MBias<-n: Nat, -m: Nat> is Type:
  MBias{c: Mat<n, m>, b: Mat<1n, m>}

def add_row2(-n: Nat, -m: Nat, +nn: Nat, +mm: Nat, cd: Array<F32>, al: AL) -> MBias<n, m>:
  match al:
    case AL{bd, bl}:
      MBias{Mat{bias_rows(nn, 0, U32.from_nat(mm), bl, cd)}, Mat{bd}}

def Mat.add_row(+n: Nat, +m: Nat, c: Mat<n, m>, b: Mat<1n, m>) -> MBias<n, m>:
  match c b:
    case Mat{cd} Mat{bd}:
      add_row2(n, m, n, m, cd, read_l(Nat.sub(m, 1n), U32.from_nat(Nat.sub(m, 1n)), Nil{}, Array.get(F32, bd, U32.from_nat(Nat.sub(m, 1n)))))

# pair of matrices of the same shape (the result of an operation that reads two operands)

type MPair<-r: Nat, -c: Nat> is Type:
  MPair{a: Mat<r, c>, b: Mat<r, c>}

def mpair_fin(-r: Nat, -c: Nat, p: P2) -> MPair<r, c>:
  match p:
    case P2{a, b}:
      MPair{Mat{a}, Mat{b}}

def relu_fin(-r: Nat, -c: Nat, d: Array<F32>) -> Mat<r, c>:
  Mat{d}

def Mat.relu(+r: Nat, +c: Nat, a: Mat<r, c>) -> Mat<r, c>:
  match a:
    case Mat{d}:
      relu_fin(r, c, relu_l(Nat.mul(r, c), 0, Array.get(F32, d, 0)))

# (dy * (h > 0), h): the gradient that passes through the relu

def Mat.relu_bwd(+r: Nat, +c: Nat, dy: Mat<r, c>, h: Mat<r, c>) -> MPair<r, c>:
  match dy h:
    case Mat{d} Mat{x}:
      mpair_fin(r, c, mask_l(Nat.mul(r, c), 0, Array.get(F32, d, 0), Array.get(F32, x, 0)))

# (w - lr * dw, dw): one SGD step

def Mat.sgd(+r: Nat, +c: Nat, +lr: F32, w: Mat<r, c>, dw: Mat<r, c>) -> MPair<r, c>:
  match w dw:
    case Mat{x} Mat{y}:
      mpair_fin(r, c, sgd_l(Nat.mul(r, c), 0, lr, Array.get(F32, x, 0), Array.get(F32, y, 0)))

# (a + b, b): element-wise sum (a - (-1)*b is exactly a + b)

def Mat.add(+r: Nat, +c: Nat, a: Mat<r, c>, b: Mat<r, c>) -> MPair<r, c>:
  match a b:
    case Mat{x} Mat{y}:
      mpair_fin(r, c, sgd_l(Nat.mul(r, c), 0, (0.0 - 1.0 : F32), Array.get(F32, x, 0), Array.get(F32, y, 0)))

# sum of each column (n x m -> 1 x m), returning the matrix

type MColSum<-n: Nat, -m: Nat> is Type:
  MColSum{a: Mat<n, m>, s: Mat<1n, m>}

def colsum_fin(-n: Nat, -m: Nat, g: G) -> MColSum<n, m>:
  match g:
    case G{ones, a, s}:
      MColSum{Mat{a}, Mat{s}}

def Mat.col_sums(+n: Nat, +m: Nat, a: Mat<n, m>) -> MColSum<n, m>:
  match a:
    case Mat{d}:
      colsum_fin(n, m, gemm(1n, m, n, 0, 1, U32.from_nat(m), 1, Array.new(F32, cap_depth(n), 1.0), d, Array.new(F32, cap_depth(m), 0.0)))

# s + the sum of each column, into s (1 x m): a bias gradient added into its accumulator, with the bits of
# Mat.add(s, Mat.col_sums(a)'s s)

def Mat.col_sums_acc(+n: Nat, +m: Nat, a: Mat<n, m>, s: Mat<1n, m>) -> MColSum<n, m>:
  match a s:
    case Mat{d} Mat{z}:
      colsum_fin(n, m, gemm_ac(True{}, 1n, m, n, 0, 1, U32.from_nat(m), 1, Array.new(F32, cap_depth(n), 1.0), d, z))

# mean cross-entropy and its gradient on the logits: (softmax - one-hot) / n.
# labels has one entry per row (n); the gradient replaces the logits.

type MCE<-n: Nat, -c: Nat> is Type:
  MCE{loss: F32, grad: Mat<n, c>}

def ce_fin(-n: Nat, -c: Nat, +nf: F32, r: L2) -> MCE<n, c>:
  match r:
    case L2{loss, d}:
      MCE{(loss / nf : F32), Mat{d}}

def Mat.softmax_ce(+n: Nat, +c: Nat, z: Mat<n, c>, labels: List<&2, Nat>) -> MCE<n, c>:
  match z:
    case Mat{d}:
      ce_fin(n, c, F32.from_nat(n), cea_rows(labels, Nat.sub(c, 1n), U32.from_nat(c), F32.from_nat(n), 0, L2{0.0, d}))

# argmax hits per row against labels

type MHits<-n: Nat, -c: Nat> is Type:
  MHits{z: Mat<n, c>, hits: Nat}

def hits_fin(-n: Nat, -c: Nat, r: CR) -> MHits<n, c>:
  match r:
    case CR{d, k}:
      MHits{Mat{d}, k}

def Mat.count_correct(+n: Nat, +c: Nat, z: Mat<n, c>, labels: List<&2, Nat>) -> MHits<n, c>:
  match z:
    case Mat{d}:
      hits_fin(n, c, count_rows(labels, Nat.sub(c, 1n), U32.from_nat(c), 0, CR{d, 0n}))

# a row as a list (returning the matrix) and the other way around

type MRow<-r: Nat, -c: Nat> is Type:
  MRow{m: Mat<r, c>, row: List<&2, F32>}

def row_fin(-r: Nat, -c: Nat, al: AL) -> MRow<r, c>:
  match al:
    case AL{d, l}:
      MRow{Mat{d}, l}

def Mat.read_row(+r: Nat, +c: Nat, m: Mat<r, c>, +i: U32) -> MRow<r, c>:
  match m:
    case Mat{d}:
      row_fin(r, c, read_l(Nat.sub(c, 1n), ((i * U32.from_nat(c) : U32) + U32.from_nat(Nat.sub(c, 1n)) : U32), Nil{}, Array.get(F32, d, ((i * U32.from_nat(c) : U32) + U32.from_nat(Nat.sub(c, 1n)) : U32))))

def Mat.write_row(-r: Nat, +c: Nat, m: Mat<r, c>, +i: U32, xs: List<&2, F32>) -> Mat<r, c>:
  match m:
    case Mat{d}:
      Mat{write_l(xs, (i * U32.from_nat(c) : U32), d)}

# ---- variants that check the number of labels ----

# does the list of labels have exactly n entries (one per row)?
def labels_len_is(xs: List<&2, Nat>, n: Nat) -> Bool:
  match xs n:
    case Nil{} 0n:
      True{}
    case Con{h, t} 1n+p:
      labels_len_is(t, p)
    case _ _:
      False{}

def ce_checked(ok: Bool, +n: Nat, +c: Nat, z: Mat<n, c>, labels: List<&2, Nat>) -> Maybe<&1, MCE<n, c>>:
  match ok:
    case True{}:
      Some{Mat.softmax_ce(n, c, z, labels)}
    case False{}:
      None{}

# like Mat.softmax_ce, but returns None unless labels has exactly n entries
def Mat.softmax_ce_checked(+n: Nat, +c: Nat, z: Mat<n, c>, +labels: List<&2, Nat>) -> Maybe<&1, MCE<n, c>>:
  ce_checked(labels_len_is(labels, n), n, c, z, labels)

def hits_checked(ok: Bool, +n: Nat, +c: Nat, z: Mat<n, c>, labels: List<&2, Nat>) -> Maybe<&1, MHits<n, c>>:
  match ok:
    case True{}:
      Some{Mat.count_correct(n, c, z, labels)}
    case False{}:
      None{}

# like Mat.count_correct, but returns None unless labels has exactly n entries
def Mat.count_correct_checked(+n: Nat, +c: Nat, z: Mat<n, c>, +labels: List<&2, Nat>) -> Maybe<&1, MHits<n, c>>:
  hits_checked(labels_len_is(labels, n), n, c, z, labels)

# ---------------------------------------------------------------------
# Bands<r, c>: a matrix cut into bands of rows, for matrix · vector in parallel
# without copying the weights.
#
# Mat.matmul_nt with n = 1 can only run in parallel by cloning the whole weight
# matrix for every task, and in matrix · vector each weight is read once, so the
# copies cost more than the products (NOTES.md, exp. 9). Bands keeps each band in
# its own Array: a task takes its band and nothing is copied except the input vector.
#
# A node with r rows has children with half(r) and r - half(r) rows. The split is
# part of the type, so the row count of every band follows from r, and the law
# half_cover says that the two halves always cover the r rows exactly.
# ---------------------------------------------------------------------

# n / 2 rounded down. A tail loop with an accumulator: a non-tail recursion here would put a
# continuation on every node of the band tree and keep the bands from running in parallel.
def half.go(n: Nat, acc: Nat) -> Nat:
  match n:
    case 0n:
      acc
    case 1n+p:
      match p:
        case 0n:
          acc
        case 1n+q:
          half.go(q, 1n+acc)

def half(n: Nat) -> Nat:
  half.go(n, 0n)

# a <= a
def leb_refl(+a: Nat) -> {True{} == leb(a, a) : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      leb_refl(p)

# 1 + (a + b) == a + (1 + b)
def add_succ_r(+a: Nat, -b: Nat) -> {1n+Nat.add(a, b) == Nat.add(a, 1n+b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_succ_r(p, b) : {2n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

# a + 0 == a
def add_zero_r(+a: Nat) -> {Nat.add(a, 0n) == a : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_zero_r(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

# half.go(n, acc) <= n + acc
def half_go_le(+n: Nat, +acc: Nat) -> {True{} == leb(half.go(n, acc), Nat.add(n, acc)) : Bool}:
  match n:
    case 0n:
      leb_refl(acc)
    case 1n+p:
      match p:
        case 0n:
          leb_succ(acc, acc, leb_refl(acc))
        case 1n+q:
          %Equal.sym(Nat, 1n+Nat.add(q, acc), Nat.add(q, 1n+acc), add_succ_r(q, acc)) : {True{} == leb(half.go(q, 1n+acc), 1n+_) : Bool}
          leb_succ(half.go(q, 1n+acc), Nat.add(q, 1n+acc), half_go_le(q, 1n+acc))

# half(n) <= n
def half_le(+n: Nat) -> {True{} == leb(half(n), n) : Bool}:
  %add_zero_r(n) : {True{} == leb(half(n), _) : Bool}
  half_go_le(n, 0n)

# a <= r  ->  r == a + (r - a)
def add_sub(+a: Nat, +r: Nat, e: {True{} == leb(a, r) : Bool}) -> {r == Nat.add(a, Nat.sub(r, a)) : Nat}:
  match a r:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      %e : BD(_, Unit, {0n == Nat.add(1n+p, Nat.sub(0n, 1n+p)) : Nat})
      Unit{}
    case 1n+p 1n+q:
      %add_sub(p, q, e) : {1n+q == 1n+_ : Nat}
      {==}

# LAW (half_cover): the two bands of a node, half(r) rows and r - half(r) rows,
# add up to exactly r rows: no row is lost and none is counted twice.
law half_cover:
  for +n: Nat
  {n == Nat.add(half(n), Nat.sub(n, half(n))) : Nat}

def half_cover(n):
  add_sub(half(n), n, half_le(n))

type Bands<-r: Nat, -c: Nat> is Type:
  BLeaf{m: Mat<r, c>}
  BNode{x: Bands<half(r), c>, y: Bands<Nat.sub(r, half(r)), c>}

# r x c of zeros in 2^d bands (d = 0: a single band, the same as a Mat)
def Bands.zeros(+d: Nat, +r: Nat, +c: Nat) -> Bands<r, c>:
  match d:
    case 0n:
      BLeaf{Mat.zeros(r, c)}
    case 1n+q:
      # bend-lint: disable-next bend-lint-rules/performance/missed_fork -- bands are built once at load not-expected-because: forking them measured as no change (NOTES 2026-10-16)
      BNode{Bands.zeros(q, half(r), c), Bands.zeros(q, Nat.sub(r, half(r)), c)}

# the row counts of the two bands follow the row count of the node: if rr == r at run
# time, the same holds for both halves (the bands carry r only in their type)
def eq_half(-a: Nat, -b: Nat, e: {a == b : Nat}) -> {half(a) == half(b) : Nat}:
  %e : {half(a) == half(_) : Nat}
  {==}

def eq_rest(-a: Nat, -b: Nat, e: {a == b : Nat}) -> {Nat.sub(a, half(a)) == Nat.sub(b, half(b)) : Nat}:
  %e : {Nat.sub(a, half(a)) == Nat.sub(_, half(_)) : Nat}
  {==}

# does a block of n numbers starting at flat index i end before the second band of a
# node with r rows? does it start in the second band?
def bfill_l(+r: Nat, +c: Nat, +i: Nat, +n: Nat) -> Bool:
  Nat.is_le(Nat.add(i, n), Nat.mul(half(r), c))

def bfill_r(+r: Nat, +c: Nat, +i: Nat) -> Bool:
  Nat.is_le(Nat.mul(half(r), c), i)

# writes the first n numbers of xs at index i (and nothing past them)
def fill_n(n: Nat, xs: List<&2, F32>, +i: U32, a: Array<F32>) -> Array<F32>:
  match n xs:
    case 1n+p Con{h, t}:
      fill_n(p, t, (i + 1 : U32), Array.set(F32, a, i, h))
    case _ _:
      a

def fill_band(-r: Nat, -c: Nat, n: Nat, xs: List<&2, F32>, +i: U32, m: Mat<r, c>) -> Mat<r, c>:
  match m:
    case Mat{d}:
      Mat{fill_n(n, xs, i, d)}

# writes the first n numbers of xs at flat index i; l and rt say which bands they touch
# (bfill_l, bfill_r); rr is the row count r at run time. A block that crosses into the second
# band is not copied: the first band writes only its part, the second gets the rest by List.drop.
def bfill(+c: Nat, -r: Nat, b: Bands<r, c>, l: Bool, rt: Bool, +rr: Nat, -e: {rr == r : Nat}, +i: Nat, +n: Nat, +xs: List<&2, F32>) -> Bands<r, c>:
  match b l rt:
    case BLeaf{m} _ _:
      BLeaf{fill_band(r, c, n, xs, U32.from_nat(i), m)}
    case BNode{x, y} True{} _:
      BNode{bfill(c, half(r), x, bfill_l(half(rr), c, i, n), bfill_r(half(rr), c, i), half(rr), eq_half(rr, r, e), i, n, xs), y}
    case BNode{x, y} False{} True{}:
      BNode{x, bfill(c, Nat.sub(r, half(r)), y, bfill_l(Nat.sub(rr, half(rr)), c, Nat.sub(i, Nat.mul(half(rr), c)), n), bfill_r(Nat.sub(rr, half(rr)), c, Nat.sub(i, Nat.mul(half(rr), c))), Nat.sub(rr, half(rr)), eq_rest(rr, r, e), Nat.sub(i, Nat.mul(half(rr), c)), n, xs)}
    case BNode{x, y} False{} False{}:
      # bend-lint: disable-next bend-lint-rules/performance/missed_fork -- bands are built once at load not-expected-because: forking them measured as no change (NOTES 2026-10-16)
      BNode{bfill(c, half(r), x, bfill_l(half(rr), c, i, Nat.sub(Nat.mul(half(rr), c), i)), bfill_r(half(rr), c, i), half(rr), eq_half(rr, r, e), i, Nat.sub(Nat.mul(half(rr), c), i), xs), bfill(c, Nat.sub(r, half(r)), y, bfill_l(Nat.sub(rr, half(rr)), c, 0n, Nat.sub(n, Nat.sub(Nat.mul(half(rr), c), i))), bfill_r(Nat.sub(rr, half(rr)), c, 0n), Nat.sub(rr, half(rr)), eq_rest(rr, r, e), 0n, Nat.sub(n, Nat.sub(Nat.mul(half(rr), c), i)), List.drop(&2, F32, xs, Nat.sub(Nat.mul(half(rr), c), i)))}

def bfill_top(ok: Bool, +r: Nat, +c: Nat, b: Bands<r, c>, +i: Nat, +n: Nat, +xs: List<&2, F32>) -> Bands<r, c>:
  match ok:
    case True{}:
      bfill(c, r, b, bfill_l(r, c, i, n), bfill_r(r, c, i), r, {==}, i, n, xs)
    case False{}:
      bfill(c, r, b, bfill_l(r, c, i, Nat.sub(Nat.mul(r, c), i)), bfill_r(r, c, i), r, {==}, i, Nat.sub(Nat.mul(r, c), i), xs)

# writes the n numbers of xs into b starting at flat index i (row-major, as Mat.fill_at);
# numbers past r*c are dropped. For loading a large matrix block by block.
def Bands.fill_at(+r: Nat, +c: Nat, +i: Nat, +n: Nat, +xs: List<&2, F32>, b: Bands<r, c>) -> Bands<r, c>:
  bfill_top(Nat.is_le(Nat.add(i, n), Nat.mul(r, c)), r, c, b, i, n, xs)

# ---- y = B · x for B with r rows of c numbers (a weight matrix stored out x in) ----

# B (intact) and y = B · x, typed like Mat.matmul_nt: x is 1 x c, y is 1 x r
type BV<-r: Nat, -c: Nat> is Type:
  BV{b: Bands<r, c>, y: Mat<1n, r>}

# the bands and y as a list in row order (the result of Bands.matvec_l)
type BL<-r: Nat, -c: Nat> is Type:
  BL{b: Bands<r, c>, y: List<&2, F32>}

def bv_g2(-r: Nat, -c: Nat, w: Array<F32>, al: AL) -> BL<r, c>:
  match al:
    case AL{d, l}:
      BL{BLeaf{Mat{w}}, l}

# the band's 1 + m rows came out in an Array (one number per row); read them back as a list
def bv_g(-r: Nat, -c: Nat, +m: Nat, g: G) -> BL<r, c>:
  match g:
    case G{a, b, y}:
      bv_g2(r, c, b, read_l(m, U32.from_nat(m), Nil{}, Array.get(F32, y, U32.from_nat(m))))

# the rows of one band, one dot product each, with the gemm of Mat.matmul_nt (n = 1), so the
# results are identical. Writing the results into an Array and reading them back is faster than
# building the list row by row (NOTES.md, exp. 14).
def bv_gemm(-r: Nat, +c: Nat, rows: Nat, +rr: Nat, x: Array<F32>, w: Array<F32>) -> BL<r, c>:
  match rows:
    case 0n:
      BL{BLeaf{Mat{w}}, Nil{}}
    case 1n+p:
      bv_g(r, c, p, gemm(1n, rr, c, U32.from_nat(c), 1, 1, U32.from_nat(c), x, w, Array.new(F32, cap_depth(rr), 0.0)))

def bv_mat(-r: Nat, +c: Nat, +rr: Nat, x: Array<F32>, m: Mat<r, c>) -> BL<r, c>:
  match m:
    case Mat{w}:
      bv_gemm(r, c, rr, rr, x, w)

def bv_join(-r: Nat, -c: Nat, p: BL<half(r), c>, q: BL<Nat.sub(r, half(r)), c>) -> BL<r, c>:
  match p q:
    case BL{bx, yx} BL{by, yy}:
      BL{BNode{bx, by}, List.append(&2, F32, yx, yy)}

# the two bands of a node run in parallel, each with its own copy of x (c numbers);
# xp is a pair of copies of x (a leaf uses one). xp comes after the erased e: up to Bend 2.0.36 an erased
# parameter at the end of the list turned the parallel let into two sequential calls (NOTES.md, exp. 11; fixed in
# 2.0.37 by bendlang/bend#1377). The order is kept, and harmless.
def bmv(+c: Nat, -r: Nat, b: Bands<r, c>, +rr: Nat, -e: {rr == r : Nat}, xp: Array<F32> & Array<F32>) -> BL<r, c>:
  match b xp:
    case BLeaf{m} Tuple{x, x2}:
      bv_mat(r, c, rr, x, m)
    case BNode{bx, by} Tuple{x1, x2}:
      p q = bmv(c, half(r), bx, half(rr), eq_half(rr, r, e), Array.clone(F32, x1)) bmv(c, Nat.sub(r, half(r)), by, Nat.sub(rr, half(rr)), eq_rest(rr, r, e), Array.clone(F32, x2))
      bv_join(r, c, p, q)

def bv_fin(+r: Nat, -c: Nat, bl: BL<r, c>) -> BV<r, c>:
  match bl:
    case BL{b, y}:
      BV{b, Mat.from_list(1n, r, y)}

# y = B · x for B (r x c, one row per output) and x (1 x c); returns B intact and y (1 x r).
# The same product as Mat.matmul_nt(1n, c, r, ..), and the same numbers, bit for bit.
def Bands.matvec(+r: Nat, +c: Nat, b: Bands<r, c>, x: Mat<1n, c>) -> BV<r, c>:
  match x:
    case Mat{xa}:
      bv_fin(r, c, bmv(c, r, b, r, {==}, Array.clone(F32, xa)))

# the same product with x and y as lists (x: c numbers, y: r numbers in row order), without the
# Mat conversions of Bands.matvec: for callers that already hold the vector as a list
def Bands.matvec_l(+r: Nat, +c: Nat, b: Bands<r, c>, +x: List<&2, F32>) -> BL<r, c>:
  bmv(c, r, b, r, {==}, Array.clone(F32, fill_list(x, 0, Array.new(F32, cap_depth(c), 0.0))))

# ---- reading rows and the whole matrix ----

type BRow<-r: Nat, -c: Nat> is Type:
  BRow{b: Bands<r, c>, row: List<&2, F32>}

def brow_fin(-r: Nat, -c: Nat, al: AL) -> BRow<r, c>:
  match al:
    case AL{d, l}:
      BRow{BLeaf{Mat{d}}, l}

# n numbers starting at flat index i of one band (a band may have no rows)
def brow_leaf2(-r: Nat, -c: Nat, +n: Nat, +i: U32, d: Array<F32>) -> BRow<r, c>:
  match n:
    case 0n:
      BRow{BLeaf{Mat{d}}, Nil{}}
    case 1n+p:
      brow_fin(r, c, read_l(p, ((i + U32.from_nat(n) : U32) - 1 : U32), Nil{}, Array.get(F32, d, ((i + U32.from_nat(n) : U32) - 1 : U32))))

def brow_leaf(-r: Nat, -c: Nat, m: Mat<r, c>, +i: U32, n: Nat) -> BRow<r, c>:
  match m:
    case Mat{d}:
      brow_leaf2(r, c, n, i, d)

def brow_l(-r: Nat, -c: Nat, p: BRow<half(r), c>, y: Bands<Nat.sub(r, half(r)), c>) -> BRow<r, c>:
  match p:
    case BRow{bx, row}:
      BRow{BNode{bx, y}, row}

def brow_r(-r: Nat, -c: Nat, x: Bands<half(r), c>, q: BRow<Nat.sub(r, half(r)), c>) -> BRow<r, c>:
  match q:
    case BRow{by, row}:
      BRow{BNode{x, by}, row}

# row i; left says whether i is in the first band of the node (i < half(rr))
def brow(+c: Nat, -r: Nat, b: Bands<r, c>, left: Bool, +rr: Nat, -e: {rr == r : Nat}, +i: Nat) -> BRow<r, c>:
  match b left:
    case BLeaf{m} _:
      brow_leaf(r, c, m, U32.from_nat(Nat.mul(i, c)), c)
    case BNode{x, y} True{}:
      brow_l(r, c, brow(c, half(r), x, Nat.is_lt(i, half(half(rr))), half(rr), eq_half(rr, r, e), i), y)
    case BNode{x, y} False{}:
      brow_r(r, c, x, brow(c, Nat.sub(r, half(r)), y, Nat.is_lt(Nat.sub(i, half(rr)), half(Nat.sub(rr, half(rr)))), Nat.sub(rr, half(rr)), eq_rest(rr, r, e), Nat.sub(i, half(rr))))

# row i (c numbers), for example an embedding lookup
def Bands.read_row(+r: Nat, +c: Nat, b: Bands<r, c>, +i: Nat) -> BRow<r, c>:
  brow(c, r, b, Nat.is_lt(i, half(r)), r, {==}, i)

def bl_join(-r: Nat, -c: Nat, p: BRow<half(r), c>, q: BRow<Nat.sub(r, half(r)), c>) -> BRow<r, c>:
  match p q:
    case BRow{bx, xs} BRow{by, ys}:
      BRow{BNode{bx, by}, List.append(&2, F32, xs, ys)}

def blist(+c: Nat, -r: Nat, b: Bands<r, c>, +rr: Nat, -e: {rr == r : Nat}) -> BRow<r, c>:
  match b:
    case BLeaf{m}:
      brow_leaf(r, c, m, 0, Nat.mul(rr, c))
    case BNode{x, y}:
      # bend-lint: disable-next bend-lint-rules/performance/missed_fork -- bands are built once at load not-expected-because: forking them measured as no change (NOTES 2026-10-16)
      bl_join(r, c, blist(c, half(r), x, half(rr), eq_half(rr, r, e)), blist(c, Nat.sub(r, half(r)), y, Nat.sub(rr, half(rr)), eq_rest(rr, r, e)))

# all r*c numbers in row order (in BRow.row)
def Bands.to_list(+r: Nat, +c: Nat, b: Bands<r, c>) -> BRow<r, c>:
  blist(c, r, b, r, {==})

# ---- the list path returns exactly r numbers ----
#
# The numbers do not matter here, only how many there are: each lemma below counts the list a
# function builds, following the function's own recursion.

def bl_len(-r: Nat, -c: Nat, bl: BL<r, c>) -> Nat:
  match bl:
    case BL{b, y}:
      List.length(&2, F32, y)

# s == a + (1 + b)  ->  s == 1 + (a + b)
def succ_out(-s: Nat, +a: Nat, +b: Nat, e: {s == Nat.add(a, 1n+b) : Nat}) -> {s == 1n+Nat.add(a, b) : Nat}:
  %Equal.sym(Nat, 1n+Nat.add(a, b), Nat.add(a, 1n+b), add_succ_r(a, b)) : {s == _ : Nat}
  e

# s == 1 + (p + (1 + l))  ->  s == 2 + (p + l)
def len_step(-s: Nat, +p: Nat, +l: Nat, e: {s == 1n+Nat.add(p, 1n+l) : Nat}) -> {s == 2n+Nat.add(p, l) : Nat}:
  %Equal.sym(Nat, 1n+Nat.add(p, l), Nat.add(p, 1n+l), add_succ_r(p, l)) : {s == 1n+_ : Nat}
  e

# s == n + 0  ->  s == n
def zero_out(-s: Nat, +n: Nat, e: {s == Nat.add(n, 0n) : Nat}) -> {s == n : Nat}:
  %add_zero_r(n) : {s == _ : Nat}
  e

# appending adds the lengths
def app_len(+xs: List<&2, F32>, +ys: List<&2, F32>) -> {List.length(&2, F32, List.append(&2, F32, xs, ys)) == Nat.add(List.length(&2, F32, xs), List.length(&2, F32, ys)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case Con{h, t}:
      %app_len(t, ys) : {1n+List.length(&2, F32, List.append(&2, F32, t, ys)) == 1n+_ : Nat}
      {==}

# the same count for the list of a whole Mat: to_list2 wraps read_l's answer, and ML.len counts it
def read_ml_len(-r: Nat, -c: Nat, +mleft: Nat, +idx: U32, +acc: List<&2, F32>, rc: Array<F32> & F32) -> {ML.len(r, c, Mat.to_list2(r, c, read_l(mleft, idx, acc, rc))) == Nat.add(1n+mleft, List.length(&2, F32, acc)) : Nat}:
  match mleft rc:
    case 0n Tuple{d, +v}:
      {==}
    case 1n+p Tuple{d, +v}:
      len_step(ML.len(r, c, Mat.to_list2(r, c, read_l(p, (idx - 1 : U32), v <> acc, Array.get(F32, d, (idx - 1 : U32))))), p, List.length(&2, F32, acc), read_ml_len(r, c, p, (idx - 1 : U32), v <> acc, Array.get(F32, d, (idx - 1 : U32))))

# a leaf's list: read_l reads mleft + 1 numbers onto acc. By induction on mleft: with 0 left it
# reads one number onto acc (1 + the length of acc, by the definitions); with 1 + p left it reads
# one and recurses, and len_step moves the 1 out of the sum.
law leaf_len:
  for -r: Nat
  for -c: Nat
  for w: Array<F32>
  for +mleft: Nat
  for +idx: U32
  for +acc: List<&2, F32>
  for rc: Array<F32> & F32
  {bl_len(r, c, bv_g2(r, c, w, read_l(mleft, idx, acc, rc))) == Nat.add(1n+mleft, List.length(&2, F32, acc)) : Nat}

def leaf_len(r, c, w, mleft, idx, acc, rc):
  match mleft rc:
    case 0n Tuple{d, +v}:
      {==}
    case 1n+p Tuple{d, +v}:
      len_step(bl_len(r, c, bv_g2(r, c, w, read_l(p, (idx - 1 : U32), v <> acc, Array.get(F32, d, (idx - 1 : U32))))), p, List.length(&2, F32, acc), leaf_len(r, c, w, p, (idx - 1 : U32), v <> acc, Array.get(F32, d, (idx - 1 : U32))))

# whatever the gemm wrote, reading 1 + p rows back gives 1 + p numbers (G has one constructor, so
# matching g opens it, and leaf_len counts the list)
def g_len(-r: Nat, -c: Nat, +p: Nat, g: G) -> {bl_len(r, c, bv_g(r, c, p, g)) == 1n+p : Nat}:
  match g:
    case G{a, b, y}:
      zero_out(bl_len(r, c, bv_g2(r, c, b, read_l(p, U32.from_nat(p), Nil{}, Array.get(F32, y, U32.from_nat(p))))), 1n+p, leaf_len(r, c, b, p, U32.from_nat(p), Nil{}, Array.get(F32, y, U32.from_nat(p))))

# LAW (band_len): the product of one band with `rows` rows gives exactly `rows` numbers.
# 0 rows give the empty list; 1 + p rows go through the gemm, and g_len counts what is read back.
law band_len:
  for -r: Nat
  for +c: Nat
  for +rows: Nat
  for x: Array<F32>
  for w: Array<F32>
  {bl_len(r, c, bv_gemm(r, c, rows, rows, x, w)) == rows : Nat}

def band_len(r, c, rows, x, w):
  match rows:
    case 0n:
      {==}
    case 1n+p:
      g_len(r, c, p, gemm(1n, 1n+p, c, U32.from_nat(c), 1, 1, U32.from_nat(c), x, w, Array.new(F32, cap_depth(1n+p), 0.0)))

# joining two halves of h1 and h2 numbers gives h1 + h2 numbers. (The law for the whole tree, matvec_l_len below, cannot
# use it: it would need each band's Array twice. It goes through witnesses instead.)
def join_len(-r: Nat, -c: Nat, -h1: Nat, -h2: Nat, p: BL<half(r), c>, q: BL<Nat.sub(r, half(r)), c>, hp: {bl_len(half(r), c, p) == h1 : Nat}, hq: {bl_len(Nat.sub(r, half(r)), c, q) == h2 : Nat}) -> {bl_len(r, c, bv_join(r, c, p, q)) == Nat.add(h1, h2) : Nat}:
  match p q:
    case BL{bx, +yx} BL{by, +yy}:
      %hp : {List.length(&2, F32, List.append(&2, F32, yx, yy)) == Nat.add(_, h2) : Nat}
      %hq : {List.length(&2, F32, List.append(&2, F32, yx, yy)) == Nat.add(List.length(&2, F32, yx), _) : Nat}
      app_len(yx, yy)

# ---- the whole tree: Bands.matvec_l gives exactly r numbers ----
#
# The obvious induction applies join_len to the two recursive results and to the induction hypotheses about the same
# calls, which uses each band's Array twice, and Bend's affine rules refuse that. The way round: every lemma below
# returns a witness instead of a bare count, a record holding the list itself (Data, so it may be used freely), an
# equation saying the result is that list in its constructor (the arrays appear only in the types, which never run),
# and the list's length. So each array is used once, by the one call that builds the witness; the node case rewrites
# both halves into constructors with the two equations, after which bv_join unfolds and app_len counts the appended
# list. The same structure as the code: read_w follows read_l, g_w a band's gemm, bmv_w the tree.

def al_c(al: AL) -> Array<F32>:
  match al:
    case AL{c, l}:
      c

def bl_b(-r: Nat, -c: Nat, bl: BL<r, c>) -> Bands<r, c>:
  match bl:
    case BL{b, y}:
      b

# a read's list y, with s == AL{s's array, y} and y of length n
type WAL<-s: AL, -n: Nat> is Type:
  WAL{y: List<&2, F32>, e: {AL{al_c(s), y} == s : AL}, l: {List.length(&2, F32, y) == n : Nat}}

# a product's list y, with s == BL{s's bands, y} and y of length n
type WBL<-r: Nat, -c: Nat, -s: BL<r, c>, -n: Nat> is Type:
  WBL{y: List<&2, F32>, e: {BL{bl_b(r, c, s), y} == s : BL<r, c>}, l: {List.length(&2, F32, y) == n : Nat}}

# moves the 1 out of the length's sum (len_step on the witness)
def wal_step(-s: AL, +p: Nat, +l: Nat, w: WAL<s, 1n+Nat.add(p, 1n+l)>) -> WAL<s, 2n+Nat.add(p, l)>:
  match w:
    case WAL{+y, e, h}:
      WAL{y, e, len_step(List.length(&2, F32, y), p, l, h)}

# read_l reads mleft + 1 numbers onto acc: by induction on mleft, as leaf_len
def read_w(+mleft: Nat, +idx: U32, +acc: List<&2, F32>, rc: Array<F32> & F32) -> WAL<read_l(mleft, idx, acc, rc), Nat.add(1n+mleft, List.length(&2, F32, acc))>:
  match mleft rc:
    case 0n Tuple{d, +v}:
      WAL{v <> acc, {==}, {==}}
    case 1n+p Tuple{d, +v}:
      wal_step(read_l(p, (idx - 1 : U32), v <> acc, Array.get(F32, d, (idx - 1 : U32))), p, List.length(&2, F32, acc), read_w(p, (idx - 1 : U32), v <> acc, Array.get(F32, d, (idx - 1 : U32))))

# n + 0 -> n in the length
def wal_zero(-s: AL, +n: Nat, w: WAL<s, Nat.add(n, 0n)>) -> WAL<s, n>:
  match w:
    case WAL{+y, e, h}:
      WAL{y, e, zero_out(List.length(&2, F32, y), n, h)}

# a band's list is the read's list: rewrite the read into its constructor, then bv_g2 unfolds
def g2_w(-r: Nat, -c: Nat, -w: Array<F32>, -al: AL, -n: Nat, a: WAL<al, n>) -> WBL<r, c, bv_g2(r, c, w, al), n>:
  match a:
    case WAL{+y, e, h}:
      %e : WBL<r, c, bv_g2(r, c, w, _), n>
      WBL{y, {==}, h}

# whatever the gemm wrote, reading 1 + p rows back gives 1 + p numbers
def g_w(-r: Nat, -c: Nat, +p: Nat, g: G) -> WBL<r, c, bv_g(r, c, p, g), 1n+p>:
  match g:
    case G{a, b, y}:
      g2_w(r, c, b, read_l(p, U32.from_nat(p), Nil{}, Array.get(F32, y, U32.from_nat(p))), 1n+p, wal_zero(read_l(p, U32.from_nat(p), Nil{}, Array.get(F32, y, U32.from_nat(p))), 1n+p, read_w(p, U32.from_nat(p), Nil{}, Array.get(F32, y, U32.from_nat(p)))))

# one band of `rows` rows: no numbers for 0 rows, else g_w
def gemm_w(-r: Nat, +c: Nat, rows: Nat, +rr: Nat, x: Array<F32>, w: Array<F32>) -> WBL<r, c, bv_gemm(r, c, rows, rr, x, w), rows>:
  match rows:
    case 0n:
      WBL{Nil{}, {==}, {==}}
    case 1n+p:
      g_w(r, c, p, gemm(1n, rr, c, U32.from_nat(c), 1, 1, U32.from_nat(c), x, w, Array.new(F32, cap_depth(rr), 0.0)))

# a leaf of the tree is one band of rr rows
def mat_w(-r: Nat, +c: Nat, +rr: Nat, x: Array<F32>, m: Mat<r, c>) -> WBL<r, c, bv_mat(r, c, rr, x, m), rr>:
  match m:
    case Mat{w}:
      gemm_w(r, c, rr, rr, x, w)

# two halves of half(rr) and rr - half(rr) numbers give rr numbers
def join_n(+rr: Nat, +y1: List<&2, F32>, +y2: List<&2, F32>, l1: {List.length(&2, F32, y1) == half(rr) : Nat}, l2: {List.length(&2, F32, y2) == Nat.sub(rr, half(rr)) : Nat}) -> {List.length(&2, F32, List.append(&2, F32, y1, y2)) == rr : Nat}:
  %Equal.sym(Nat, rr, Nat.add(half(rr), Nat.sub(rr, half(rr))), half_cover(rr)) : {List.length(&2, F32, List.append(&2, F32, y1, y2)) == _ : Nat}
  %l1 : {List.length(&2, F32, List.append(&2, F32, y1, y2)) == Nat.add(_, Nat.sub(rr, half(rr))) : Nat}
  %l2 : {List.length(&2, F32, List.append(&2, F32, y1, y2)) == Nat.add(List.length(&2, F32, y1), _) : Nat}
  app_len(y1, y2)

# a node: both halves rewritten into constructors (e1, e2), so bv_join unfolds to BL{BNode{..}, y1 ++ y2}
def node_w(-r: Nat, -c: Nat, +rr: Nat, -p: BL<half(r), c>, -q: BL<Nat.sub(r, half(r)), c>, w1: WBL<half(r), c, p, half(rr)>, w2: WBL<Nat.sub(r, half(r)), c, q, Nat.sub(rr, half(rr))>) -> WBL<r, c, bv_join(r, c, p, q), rr>:
  match w1 w2:
    case WBL{+y1, e1, l1} WBL{+y2, e2, l2}:
      %e1 : WBL<r, c, bv_join(r, c, _, q), rr>
      %e2 : WBL<r, c, bv_join(r, c, BL{bl_b(half(r), c, p), y1}, _), rr>
      WBL{List.append(&2, F32, y1, y2), {==}, join_n(rr, y1, y2, l1, l2)}

# the tree, by induction on the bands; each band and each copy of x is used once, by its own call
def bmv_w(+c: Nat, -r: Nat, b: Bands<r, c>, +rr: Nat, -e: {rr == r : Nat}, xp: Array<F32> & Array<F32>) -> WBL<r, c, bmv(c, r, b, rr, e, xp), rr>:
  match b xp:
    case BLeaf{m} Tuple{x, x2}:
      mat_w(r, c, rr, x, m)
    case BNode{bx, by} Tuple{x1, x2}:
      # bend-lint: disable-next bend-lint-rules/performance/missed_fork -- a proof of matvec_l_len not-expected-because: the package never runs it (NOTES 2026-11-01)
      node_w(r, c, rr, bmv(c, half(r), bx, half(rr), eq_half(rr, r, e), Array.clone(F32, x1)), bmv(c, Nat.sub(r, half(r)), by, Nat.sub(rr, half(rr)), eq_rest(rr, r, e), Array.clone(F32, x2)), bmv_w(c, half(r), bx, half(rr), eq_half(rr, r, e), Array.clone(F32, x1)), bmv_w(c, Nat.sub(r, half(r)), by, Nat.sub(rr, half(rr)), eq_rest(rr, r, e), Array.clone(F32, x2)))

# a witness gives the count
def w_len(-r: Nat, -c: Nat, -s: BL<r, c>, -n: Nat, w: WBL<r, c, s, n>) -> {bl_len(r, c, s) == n : Nat}:
  match w:
    case WBL{+y, e, h}:
      %e : {bl_len(r, c, _) == n : Nat}
      h

# LAW (matvec_l_len): Bands.matvec_l gives exactly r numbers, one per row of the matrix, whatever the bands' depth.
law matvec_l_len:
  for +r: Nat
  for +c: Nat
  for b: Bands<r, c>
  for +x: List<&2, F32>
  {bl_len(r, c, Bands.matvec_l(r, c, b, x)) == r : Nat}

def matvec_l_len(r, c, b, x):
  w_len(r, c, Bands.matvec_l(r, c, b, x), r, bmv_w(c, r, b, r, {==}, Array.clone(F32, fill_list(x, 0, Array.new(F32, cap_depth(c), 0.0)))))

# Mat.to_list_count reads exactly n numbers, and n = 0 reads none.
def to_list_count_len(-r: Nat, -c: Nat, d: Array<F32>, +n: Nat) -> {ML.len(r, c, Mat.to_list_count(r, c, d, n)) == n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      zero_out(ML.len(r, c, Mat.to_list_count(r, c, d, 1n+p)), 1n+p, read_ml_len(r, c, p, (U32.from_nat(1n+p) - 1 : U32), Nil{}, Array.get(F32, d, (U32.from_nat(1n+p) - 1 : U32))))

# LAW (to_list_len): Mat.to_list reads exactly r*c numbers, no more and no fewer.
law to_list_len:
  for +r: Nat
  for +c: Nat
  for m: Mat<r, c>
  {ML.len(r, c, Mat.to_list(r, c, m)) == Nat.mul(r, c) : Nat}

def to_list_len(r, c, m):
  match m:
    case Mat{d}:
      to_list_count_len(r, c, d, Nat.mul(r, c))

# ---------------------------------------------------------------------
# Base's Array: three laws about Array.set and Array.get
# ---------------------------------------------------------------------
#
# - size_set: Array.set keeps Array.size;
# - get_after_set: what Array.set writes at i, Array.get reads back at i;
# - get_other_set: a write at i leaves alone the cell that a read at k reaches, when the two descents part.
# They are about Base's Array, for any element type, and live here until Base states them
# (bendlang/bend#1934; nuxyel/bend-ml#26). The helpers are named ASet.*.
#
# The first two hold for every i (the two descents wrap i the same way) and for every tree Base can build,
# balanced or not, since a write replaces one leaf and leaves the tree's shape alone. "The size" is the
# second component of the pair Array.size returns, read with Base's Pair.snd (ASet.sz); the value is
# Pair.snd of the pair Array.get returns.
#
# The third law's hypothesis says that the descent Array.set makes for i and the descent Array.get makes
# for k take different branches at some node before either reaches a leaf (ASet.Apart). It is not "i and
# k differ after the mask", because that is false on an unbalanced tree: Array.size follows the leftmost
# path, so in ANode{ANode{ALeaf, ALeaf}, ALeaf} (size 4) the indices 2 and 3 differ but both reach the
# last leaf, and a write at 2 changes what a read at 3 returns (tests/array_laws.bend prints it: 9.0). On a
# perfect tree of 2^d leaves, which Array.new and literals build, the descents part exactly when the masked
# indices differ; that equivalence is not proved here, since it needs lemmas about U32 bits. For a concrete
# tree ASet.Apart computes to Unit, so Unit{} proves it (tests/array_laws.bend).
#
# Why the proofs are shaped this way: an Array is affine, and so is it inside a proof, so a proof may use a
# given array once. The obvious lemmas each need the same subtree again (one to say Array.size of it is
# (it, its size), one for the induction step, one to open the pair the descent returns), so they cannot
# be combined. Instead, one pass over the tree (ASet.facts) proves all six facts the step needs at once;
# each fact is an equation, and equations can be reused (+). A second pass (ASet.copy) hands back the same
# array and its size as a plain value, which the top-level proof needs to pick the first branch. The third
# law uses one pass too (ASet.both): it proves the six facts and four more about the read along k, and on
# the half the write does not enter it runs a smaller pass (ASet.side).
#
# The trailing Unit in ASet.Facts, ASet.Copy, ASet.Side and ASet.Reads works around bendlang/bend#1077: a +
# binder on the last field of a nested pair fails to check, so no fact sits in the last field.


# ASet.sz(T, a): the size of a as a plain U32, the second component of the pair Array.size returns
def ASet.sz(-T: Data, a: Array<T>) -> U32:
  Pair.snd(Array<T>, U32, Array.size(T, a))

# the three results the descents build: the swapped pair, the written array, and the read pair
def ASet.G(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Array<T> & T:
  Array.swap.go(T, a, n, j, v, z)

def ASet.NA(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Array<T>:
  Array.set.fin(T, ASet.G(T, a, n, j, v, z))

def ASet.K(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Array<T> & T:
  Array.get.go(T, ASet.NA(T, a, n, j, v, z), n, j, z)

# ASet.Facts: what one pass over the tree a proves, for any n, j, v, z (n need not be a's real size):
# 1. Array.size(a) is the pair (a, ASet.sz(a));
# 2. the pair the write's descent returns is (the written array, the old value);
# 3. Array.size of the written array is the pair (written array, its sz);
# 4. the written array has a's sz;
# 5. reading the written array along the same path gives v;
# 6. the pair the read returns is (its first component, its second).
def ASet.Facts(-T: Data, -a: Array<T>, -n: U32, -j: U32, -v: T, -z: Bool) -> Type:
  {(a, ASet.sz(T, a)) == Array.size(T, a) : Array<T> & U32} & {(ASet.NA(T, a, n, j, v, z), Pair.snd(Array<T>, T, ASet.G(T, a, n, j, v, z))) == ASet.G(T, a, n, j, v, z) : Array<T> & T} & {(ASet.NA(T, a, n, j, v, z), ASet.sz(T, ASet.NA(T, a, n, j, v, z))) == Array.size(T, ASet.NA(T, a, n, j, v, z)) : Array<T> & U32} & {ASet.sz(T, ASet.NA(T, a, n, j, v, z)) == ASet.sz(T, a) : U32} & {Pair.snd(Array<T>, T, ASet.K(T, a, n, j, v, z)) == v : T} & {(Pair.fst(Array<T>, T, ASet.K(T, a, n, j, v, z)), Pair.snd(Array<T>, T, ASet.K(T, a, n, j, v, z))) == ASet.K(T, a, n, j, v, z) : Array<T> & T} & Unit

# Array.size hands back the array it measured, next to ASet.sz(T, a)
def ASet.size_pair(-T: Data, a: Array<T>) -> {(a, ASet.sz(T, a)) == Array.size(T, a) : Array<T> & U32}:
  match a:
    case ALeaf{x}:
      {==}
    case ANode{xs, ys}:
      %ASet.size_pair(T, xs) : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}
      {==}

def ASet.node_t(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, f: ASet.Facts(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))) -> ASet.Facts(T, ANode{xs, ys}, n, j, v, True{}):
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  match r4:
                    case Tuple{+c5, r5}:
                      match r5:
                        case Tuple{+c6, u}:
                          ({(%c1 : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{xs, ys}, ASet.sz(T, ANode{xs, ys})) == Array.size(T, ANode{xs, ys}) : Array<T> & U32}}, {(%c2 : {(Array.set.fin(T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(ASet.NA(T, ANode{xs, ys}, n, j, v, True{}), Pair.snd(Array<T>, T, ASet.G(T, ANode{xs, ys}, n, j, v, True{}))) == ASet.G(T, ANode{xs, ys}, n, j, v, True{}) : Array<T> & T}}, {(%c2 : {(Array.set.fin(T, Array.swap.lo(T, ys, _)), ASet.sz(T, Array.set.fin(T, Array.swap.lo(T, ys, _)))) == Array.size(T, Array.set.fin(T, Array.swap.lo(T, ys, _))) : Array<T> & U32}; %c3 : {(ANode{Array.set.fin(T, ASet.G(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))), ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ASet.NA(T, ANode{xs, ys}, n, j, v, True{}), ASet.sz(T, ASet.NA(T, ANode{xs, ys}, n, j, v, True{}))) == Array.size(T, ASet.NA(T, ANode{xs, ys}, n, j, v, True{})) : Array<T> & U32}}, {(%c2 : {ASet.sz(T, Array.set.fin(T, Array.swap.lo(T, ys, _))) == ASet.sz(T, ANode{xs, ys}) : U32}; %c3 : {Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) == ASet.sz(T, ANode{xs, ys}) : U32}; %c1 : {U32.shl(ASet.sz(T, Array.set.fin(T, ASet.G(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))))) == Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) : U32}; %c4 : {U32.shl(ASet.sz(T, Array.set.fin(T, ASet.G(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))))) == U32.shl(_) : U32}; {==}) : {ASet.sz(T, ASet.NA(T, ANode{xs, ys}, n, j, v, True{})) == ASet.sz(T, ANode{xs, ys}) : U32}}, {(%c2 : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{})) == v : T}; %c6 : {Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _)) == v : T}; c5) : {Pair.snd(Array<T>, T, ASet.K(T, ANode{xs, ys}, n, j, v, True{})) == v : T}}, {(%c2 : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{}) : Array<T> & T}; %c6 : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, ASet.K(T, ANode{xs, ys}, n, j, v, True{})), Pair.snd(Array<T>, T, ASet.K(T, ANode{xs, ys}, n, j, v, True{}))) == ASet.K(T, ANode{xs, ys}, n, j, v, True{}) : Array<T> & T}}, Unit{})

def ASet.node_f(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +s: {(xs, ASet.sz(T, xs)) == Array.size(T, xs) : Array<T> & U32}, f: ASet.Facts(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))) -> ASet.Facts(T, ANode{xs, ys}, n, j, v, False{}):
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  match r4:
                    case Tuple{+c5, r5}:
                      match r5:
                        case Tuple{+c6, u}:
                          ({(%s : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{xs, ys}, ASet.sz(T, ANode{xs, ys})) == Array.size(T, ANode{xs, ys}) : Array<T> & U32}}, {(%c2 : {(Array.set.fin(T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(ASet.NA(T, ANode{xs, ys}, n, j, v, False{}), Pair.snd(Array<T>, T, ASet.G(T, ANode{xs, ys}, n, j, v, False{}))) == ASet.G(T, ANode{xs, ys}, n, j, v, False{}) : Array<T> & T}}, {(%c2 : {(Array.set.fin(T, Array.swap.hi(T, xs, _)), ASet.sz(T, Array.set.fin(T, Array.swap.hi(T, xs, _)))) == Array.size(T, Array.set.fin(T, Array.swap.hi(T, xs, _))) : Array<T> & U32}; %s : {(ANode{xs, Array.set.fin(T, ASet.G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n)))))}, Pair.snd(Array<T>, U32, Array.size.node(T, Array.set.fin(T, ASet.G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))), _))) == Array.size.node(T, Array.set.fin(T, ASet.G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))), _) : Array<T> & U32}; {==}) : {(ASet.NA(T, ANode{xs, ys}, n, j, v, False{}), ASet.sz(T, ASet.NA(T, ANode{xs, ys}, n, j, v, False{}))) == Array.size(T, ASet.NA(T, ANode{xs, ys}, n, j, v, False{})) : Array<T> & U32}}, {(%c2 : {ASet.sz(T, Array.set.fin(T, Array.swap.hi(T, xs, _))) == ASet.sz(T, ANode{xs, ys}) : U32}; %s : {Pair.snd(Array<T>, U32, Array.size.node(T, Array.set.fin(T, ASet.G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))), _)) == Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) : U32}; {==}) : {ASet.sz(T, ASet.NA(T, ANode{xs, ys}, n, j, v, False{})) == ASet.sz(T, ANode{xs, ys}) : U32}}, {(%c2 : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{})) == v : T}; %c6 : {Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _)) == v : T}; c5) : {Pair.snd(Array<T>, T, ASet.K(T, ANode{xs, ys}, n, j, v, False{})) == v : T}}, {(%c2 : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{}) : Array<T> & T}; %c6 : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, ASet.K(T, ANode{xs, ys}, n, j, v, False{})), Pair.snd(Array<T>, T, ASet.K(T, ANode{xs, ys}, n, j, v, False{}))) == ASet.K(T, ANode{xs, ys}, n, j, v, False{}) : Array<T> & T}}, Unit{})

def ASet.facts(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> ASet.Facts(T, a, n, j, v, z):
  match a z:
    case ALeaf{x} _:
      ({==}, {==}, {==}, {==}, {==}, {==}, Unit{})
    case ANode{xs, ys} True{}:
      ASet.node_t(T, xs, ys, n, j, v, ASet.facts(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n)))))
    case ANode{xs, ys} False{}:
      ASet.node_f(T, xs, ys, n, j, v, ASet.size_pair(T, xs), ASet.facts(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n)))))

# ASet.copy(T, a): one pass over a hands back the same array b, its size n as a value, and the facts
# that tie them to a: b is a, n is ASet.sz(T, b), and Array.size(b) is the pair (b, ASet.sz(T, b))
def ASet.Copy(-T: Data, -a: Array<T>) -> Type:
  &b: Array<T> -> &n: U32 -> {b == a : Array<T>} & {n == ASet.sz(T, b) : U32} & {(b, ASet.sz(T, b)) == Array.size(T, b) : Array<T> & U32} & Unit

def ASet.copy_node(-T: Data, -xs: Array<T>, ys: Array<T>, r: ASet.Copy(T, xs)) -> ASet.Copy(T, ANode{xs, ys}):
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+m, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  match r4:
                    case Tuple{+s, u}:
                      (ANode{b, ys}, U32.shl(m),
                        {(%p : {ANode{b, ys} == ANode{_, ys} : Array<T>}; {==}) : {ANode{b, ys} == ANode{xs, ys} : Array<T>}},
                        {(%s : {U32.shl(m) == Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) : U32}; %q : {U32.shl(m) == U32.shl(_) : U32}; {==}) : {U32.shl(m) == ASet.sz(T, ANode{b, ys}) : U32}},
                        {(%s : {(ANode{b, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{b, ys}, ASet.sz(T, ANode{b, ys})) == Array.size(T, ANode{b, ys}) : Array<T> & U32}},
                        Unit{})

def ASet.copy(-T: Data, a: Array<T>) -> ASet.Copy(T, a):
  match a:
    case ALeaf{x}:
      (ALeaf{x}, 1, {==}, {==}, {==}, Unit{})
    case ANode{xs, ys}:
      ASet.copy_node(T, xs, ys, ASet.copy(T, xs))

# the masked index and the first branch, as Array.swap.at and Array.get.at compute them from a size n
def ASet.J(+i: U32, +n: U32) -> U32:
  U32.and(i, U32.sub(n, 1))

def ASet.Z(+i: U32, +n: U32) -> Bool:
  U32.is_lt(ASet.J(i, n), U32.shr(n))

def ASet.close_size(-T: Data, -b: Array<T>, +n: U32, +i: U32, +v: T, +q: {n == ASet.sz(T, b) : U32}, f: ASet.Facts(T, b, n, ASet.J(i, n), v, ASet.Z(i, n))) -> {ASet.sz(T, Array.set(T, b, i, v)) == ASet.sz(T, b) : U32}:
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  %c1 : {ASet.sz(T, Array.set.fin(T, Array.swap.at(T, i, v, _))) == ASet.sz(T, b) : U32}
                  %q : {ASet.sz(T, Array.set.fin(T, Array.swap.go(T, b, _, ASet.J(i, _), v, ASet.Z(i, _)))) == _ : U32}
                  Equal.trans(U32, ASet.sz(T, ASet.NA(T, b, n, ASet.J(i, n), v, ASet.Z(i, n))), ASet.sz(T, b), n, c4, Equal.sym(U32, n, ASet.sz(T, b), q))

def ASet.close_get(-T: Data, -b: Array<T>, +n: U32, +i: U32, +v: T, +q: {n == ASet.sz(T, b) : U32}, f: ASet.Facts(T, b, n, ASet.J(i, n), v, ASet.Z(i, n))) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, b, i, v), i)) == v : T}:
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  match r4:
                    case Tuple{+c5, r5}:
                      %c1 : {Pair.snd(Array<T>, T, Array.get.at(T, i, Array.size(T, Array.set.fin(T, Array.swap.at(T, i, v, _))))) == v : T}
                      %q : {Pair.snd(Array<T>, T, Array.get.at(T, i, Array.size(T, Array.set.fin(T, Array.swap.go(T, b, _, ASet.J(i, _), v, ASet.Z(i, _)))))) == v : T}
                      %c3 : {Pair.snd(Array<T>, T, Array.get.at(T, i, _)) == v : T}
                      %Equal.trans(U32, n, ASet.sz(T, b), ASet.sz(T, ASet.NA(T, b, n, ASet.J(i, n), v, ASet.Z(i, n))), q, Equal.sym(U32, ASet.sz(T, ASet.NA(T, b, n, ASet.J(i, n), v, ASet.Z(i, n))), ASet.sz(T, b), c4)) : {Pair.snd(Array<T>, T, Array.get.go(T, ASet.NA(T, b, n, ASet.J(i, n), v, ASet.Z(i, n)), _, ASet.J(i, _), ASet.Z(i, _))) == v : T}
                      c5

def ASet.top_size(-T: Data, -a: Array<T>, +i: U32, +v: T, r: ASet.Copy(T, a)) -> {ASet.sz(T, Array.set(T, a, i, v)) == ASet.sz(T, a) : U32}:
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+n, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  %p : {ASet.sz(T, Array.set(T, _, i, v)) == ASet.sz(T, _) : U32}
                  ASet.close_size(T, b, n, i, v, q, ASet.facts(T, b, n, ASet.J(i, n), v, ASet.Z(i, n)))

def ASet.top_get(-T: Data, -a: Array<T>, +i: U32, +v: T, r: ASet.Copy(T, a)) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, a, i, v), i)) == v : T}:
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+n, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  %p : {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, _, i, v), i)) == v : T}
                  ASet.close_get(T, b, n, i, v, q, ASet.facts(T, b, n, ASet.J(i, n), v, ASet.Z(i, n)))

# LAW: writing v at i keeps the array's size (any i: the index wraps around)
law size_set:
  for -T: Data
  for a: Array<T>
  for +i: U32
  for +v: T
  {ASet.sz(T, Array.set(T, a, i, v)) == ASet.sz(T, a) : U32}

def size_set(T, a, i, v):
  ASet.top_size(T, a, i, v, ASet.copy(T, a))

# LAW: what Array.set writes at i, Array.get reads at i (any i: both wrap it the same way)
law get_after_set:
  for -T: Data
  for a: Array<T>
  for +i: U32
  for +v: T
  {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, a, i, v), i)) == v : T}

def get_after_set(T, a, i, v):
  ASet.top_get(T, a, i, v, ASet.copy(T, a))

# ASet.Apart(T, a, n, j, z, k, w): the write's descent (index j, first branch z) and the read's descent (k, w),
# both started from the size n, take different branches at some node of a before either reaches a leaf.
# It is Empty when both reach the same leaf, and Unit at the node where they split.
def ASet.Apart(-T: Data, a: Array<T>, +n: U32, +j: U32, +z: Bool, +k: U32, +w: Bool) -> Type:
  match a z w:
    case ALeaf{x} _ _:
      Empty
    case ANode{xs, ys} True{} True{}:
      ASet.Apart(T, xs, U32.shr(n), j, U32.is_lt(j, U32.shr(U32.shr(n))), k, U32.is_lt(k, U32.shr(U32.shr(n))))
    case ANode{xs, ys} False{} False{}:
      ASet.Apart(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n))))
    case ANode{xs, ys} True{} False{}:
      Unit
    case ANode{xs, ys} False{} True{}:
      Unit

# ASet.Side: what a pass over a subtree the write does not enter proves: Array.size(a) is (a, ASet.sz(a)), and the
# pair a read along (n, k, w) returns is (its first component, its second)
def ASet.Side(-T: Data, -a: Array<T>, -n: U32, -k: U32, -w: Bool) -> Type:
  {(a, ASet.sz(T, a)) == Array.size(T, a) : Array<T> & U32} & {(Pair.fst(Array<T>, T, Array.get.go(T, a, n, k, w)), Pair.snd(Array<T>, T, Array.get.go(T, a, n, k, w))) == Array.get.go(T, a, n, k, w) : Array<T> & T} & Unit

def ASet.side_node(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +k: U32, +w: Bool, sx: ASet.Side(T, xs, U32.shr(n), k, U32.is_lt(k, U32.shr(U32.shr(n)))), sy: ASet.Side(T, ys, U32.shr(n), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n))))) -> ASet.Side(T, ANode{xs, ys}, n, k, w):
  match w:
    case True{}:
      match sx:
        case Tuple{+s1, r1}:
          match r1:
            case Tuple{+g1, u1}:
              match sy:
                case Tuple{+s2, r2}:
                  match r2:
                    case Tuple{+g2, u2}:
                      ({(%s1 : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{xs, ys}, ASet.sz(T, ANode{xs, ys})) == Array.size(T, ANode{xs, ys}) : Array<T> & U32}}, {(%g1 : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{}))) == Array.get.go(T, ANode{xs, ys}, n, k, True{}) : Array<T> & T}}, Unit{})
    case False{}:
      match sx:
        case Tuple{+s1, r1}:
          match r1:
            case Tuple{+g1, u1}:
              match sy:
                case Tuple{+s2, r2}:
                  match r2:
                    case Tuple{+g2, u2}:
                      ({(%s1 : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{xs, ys}, ASet.sz(T, ANode{xs, ys})) == Array.size(T, ANode{xs, ys}) : Array<T> & U32}}, {(%g2 : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{}))) == Array.get.go(T, ANode{xs, ys}, n, k, False{}) : Array<T> & T}}, Unit{})

def ASet.side(-T: Data, a: Array<T>, +n: U32, +k: U32, +w: Bool) -> ASet.Side(T, a, n, k, w):
  match a:
    case ALeaf{x}:
      ({==}, {==}, Unit{})
    case ANode{xs, ys}:
      # bend-lint: disable-next bend-lint-rules/performance/missed_fork -- a proof by one pass over the tree not-expected-because: proofs are checked, never run
      ASet.side_node(T, xs, ys, n, k, w, ASet.side(T, xs, U32.shr(n), k, U32.is_lt(k, U32.shr(U32.shr(n)))), ASet.side(T, ys, U32.shr(n), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n)))))

# ASet.Reads: what one pass over the tree a proves about a write along (n, j, v, z) and a read along (n, k, w):
# 1. the pair the write's descent returns is (the written array, the old value);
# 2. the read of a returns a pair (first component, second);
# 3. so does the read of the written array;
# 4. if the descents split, the read finds the same value in the written array as in a.
def ASet.Reads(-T: Data, -a: Array<T>, -n: U32, -j: U32, -v: T, -z: Bool, -k: U32, -w: Bool) -> Type:
  {(ASet.NA(T, a, n, j, v, z), Pair.snd(Array<T>, T, ASet.G(T, a, n, j, v, z))) == ASet.G(T, a, n, j, v, z) : Array<T> & T} & {(Pair.fst(Array<T>, T, Array.get.go(T, a, n, k, w)), Pair.snd(Array<T>, T, Array.get.go(T, a, n, k, w))) == Array.get.go(T, a, n, k, w) : Array<T> & T} & {(Pair.fst(Array<T>, T, Array.get.go(T, ASet.NA(T, a, n, j, v, z), n, k, w)), Pair.snd(Array<T>, T, Array.get.go(T, ASet.NA(T, a, n, j, v, z), n, k, w))) == Array.get.go(T, ASet.NA(T, a, n, j, v, z), n, k, w) : Array<T> & T} & (ASet.Apart(T, a, n, j, z, k, w) -> {Pair.snd(Array<T>, T, Array.get.go(T, ASet.NA(T, a, n, j, v, z), n, k, w)) == Pair.snd(Array<T>, T, Array.get.go(T, a, n, k, w)) : T}) & Unit

# the written low half ASet.WL and the written high half ASet.WH of a node of size n
def ASet.WL(-T: Data, xs: Array<T>, +n: U32, +j: U32, +v: T) -> Array<T>:
  ASet.NA(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))

def ASet.WH(-T: Data, ys: Array<T>, +n: U32, +j: U32, +v: T) -> Array<T>:
  ASet.NA(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))

# the read's descent into the low half and into the high half of a node of size n
def ASet.RL(-T: Data, xs: Array<T>, +n: U32, +k: U32) -> Array<T> & T:
  Array.get.go(T, xs, U32.shr(n), k, U32.is_lt(k, U32.shr(U32.shr(n))))

def ASet.RH(-T: Data, ys: Array<T>, +n: U32, +k: U32) -> Array<T> & T:
  Array.get.go(T, ys, U32.shr(n), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n))))

def ASet.no_leaf(-T: Data, -x: T, -n: U32, -j: U32, -v: T, -z: Bool, -k: U32, -w: Bool, h: ASet.Apart(T, ALeaf{x}, n, j, z, k, w)) -> {Pair.snd(Array<T>, T, Array.get.go(T, ASet.NA(T, ALeaf{x}, n, j, v, z), n, k, w)) == Pair.snd(Array<T>, T, Array.get.go(T, ALeaf{x}, n, k, w)) : T}:
  match h:

# the write goes into the low half xs
def ASet.reads_t(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +k: U32, +w: Bool, +g: {(Pair.fst(Array<T>, T, ASet.RH(T, ys, n, k)), Pair.snd(Array<T>, T, ASet.RH(T, ys, n, k))) == ASet.RH(T, ys, n, k) : Array<T> & T}, r: ASet.Reads(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))), k, U32.is_lt(k, U32.shr(U32.shr(n))))) -> ASet.Reads(T, ANode{xs, ys}, n, j, v, True{}, k, w):
  match w:
    case True{}:
      match r:
        case Tuple{+d, r1}:
          match r1:
            case Tuple{+e1, r2}:
              match r2:
                case Tuple{+e2, r3}:
                  match r3:
                    case Tuple{e3, u}:
                      ({(%d : {(Array.set.fin(T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(ASet.NA(T, ANode{xs, ys}, n, j, v, True{}), Pair.snd(Array<T>, T, ASet.G(T, ANode{xs, ys}, n, j, v, True{}))) == ASet.G(T, ANode{xs, ys}, n, j, v, True{}) : Array<T> & T}},
                        {(%e1 : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{}))) == Array.get.go(T, ANode{xs, ys}, n, k, True{}) : Array<T> & T}},
                        {(%d : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, True{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, True{}) : Array<T> & T}; %e2 : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, True{}))) == Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, True{}) : Array<T> & T}},
                        (h => (%d : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, True{})) == Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{})) : T}; %e2 : {Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _)) == Pair.snd(Array<T>, T, Array.swap.lo(T, ys, ASet.RL(T, xs, n, k))) : T}; %e1 : {Pair.snd(Array<T>, T, ASet.RL(T, ASet.WL(T, xs, n, j, v), n, k)) == Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _)) : T}; e3(h))),
                        Unit{})
    case False{}:
      match r:
        case Tuple{+d, r1}:
          match r1:
            case Tuple{+e1, r2}:
              match r2:
                case Tuple{+e2, r3}:
                  match r3:
                    case Tuple{e3, u}:
                      ({(%d : {(Array.set.fin(T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(ASet.NA(T, ANode{xs, ys}, n, j, v, True{}), Pair.snd(Array<T>, T, ASet.G(T, ANode{xs, ys}, n, j, v, True{}))) == ASet.G(T, ANode{xs, ys}, n, j, v, True{}) : Array<T> & T}},
                        {(%g : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{}))) == Array.get.go(T, ANode{xs, ys}, n, k, False{}) : Array<T> & T}},
                        {(%d : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, False{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, False{}) : Array<T> & T}; %g : {(Pair.fst(Array<T>, T, Array.swap.hi(T, ASet.WL(T, xs, n, j, v), _)), Pair.snd(Array<T>, T, Array.swap.hi(T, ASet.WL(T, xs, n, j, v), _))) == Array.swap.hi(T, ASet.WL(T, xs, n, j, v), _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, False{}))) == Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, False{}) : Array<T> & T}},
                        (h => (%d : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, False{})) == Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{})) : T}; %g : {Pair.snd(Array<T>, T, Array.swap.hi(T, ASet.WL(T, xs, n, j, v), _)) == Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _)) : T}; {==})),
                        Unit{})

# the write goes into the high half ys
def ASet.reads_f(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +k: U32, +w: Bool, +g: {(Pair.fst(Array<T>, T, ASet.RL(T, xs, n, k)), Pair.snd(Array<T>, T, ASet.RL(T, xs, n, k))) == ASet.RL(T, xs, n, k) : Array<T> & T}, r: ASet.Reads(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n))))) -> ASet.Reads(T, ANode{xs, ys}, n, j, v, False{}, k, w):
  match w:
    case True{}:
      match r:
        case Tuple{+d, r1}:
          match r1:
            case Tuple{+e1, r2}:
              match r2:
                case Tuple{+e2, r3}:
                  match r3:
                    case Tuple{e3, u}:
                      ({(%d : {(Array.set.fin(T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(ASet.NA(T, ANode{xs, ys}, n, j, v, False{}), Pair.snd(Array<T>, T, ASet.G(T, ANode{xs, ys}, n, j, v, False{}))) == ASet.G(T, ANode{xs, ys}, n, j, v, False{}) : Array<T> & T}},
                        {(%g : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{}))) == Array.get.go(T, ANode{xs, ys}, n, k, True{}) : Array<T> & T}},
                        {(%d : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, True{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, True{}) : Array<T> & T}; %g : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ASet.WH(T, ys, n, j, v), _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ASet.WH(T, ys, n, j, v), _))) == Array.swap.lo(T, ASet.WH(T, ys, n, j, v), _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, True{}))) == Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, True{}) : Array<T> & T}},
                        (h => (%d : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, True{})) == Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{})) : T}; %g : {Pair.snd(Array<T>, T, Array.swap.lo(T, ASet.WH(T, ys, n, j, v), _)) == Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _)) : T}; {==})),
                        Unit{})
    case False{}:
      match r:
        case Tuple{+d, r1}:
          match r1:
            case Tuple{+e1, r2}:
              match r2:
                case Tuple{+e2, r3}:
                  match r3:
                    case Tuple{e3, u}:
                      ({(%d : {(Array.set.fin(T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(ASet.NA(T, ANode{xs, ys}, n, j, v, False{}), Pair.snd(Array<T>, T, ASet.G(T, ANode{xs, ys}, n, j, v, False{}))) == ASet.G(T, ANode{xs, ys}, n, j, v, False{}) : Array<T> & T}},
                        {(%e1 : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{}))) == Array.get.go(T, ANode{xs, ys}, n, k, False{}) : Array<T> & T}},
                        {(%d : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, False{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, False{}) : Array<T> & T}; %e2 : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, False{}))) == Array.get.go(T, ASet.NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, False{}) : Array<T> & T}},
                        (h => (%d : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, False{})) == Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{})) : T}; %e2 : {Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _)) == Pair.snd(Array<T>, T, Array.swap.hi(T, xs, ASet.RH(T, ys, n, k))) : T}; %e1 : {Pair.snd(Array<T>, T, ASet.RH(T, ASet.WH(T, ys, n, j, v), n, k)) == Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _)) : T}; e3(h))),
                        Unit{})

# ASet.Both: the six facts of the first two laws and the four of the third, from one pass
def ASet.Both(-T: Data, -a: Array<T>, -n: U32, -j: U32, -v: T, -z: Bool, -k: U32, -w: Bool) -> Type:
  ASet.Facts(T, a, n, j, v, z) & ASet.Reads(T, a, n, j, v, z, k, w)

def ASet.both_t(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +k: U32, +w: Bool, sy: ASet.Side(T, ys, U32.shr(n), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n)))), f: ASet.Both(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))), k, U32.is_lt(k, U32.shr(U32.shr(n))))) -> ASet.Both(T, ANode{xs, ys}, n, j, v, True{}, k, w):
  match sy:
    case Tuple{+s, r1}:
      match r1:
        case Tuple{+g, u}:
          match f:
            case Tuple{o, r}:
              (ASet.node_t(T, xs, ys, n, j, v, o), ASet.reads_t(T, xs, ys, n, j, v, k, w, g, r))

def ASet.both_f(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +k: U32, +w: Bool, sx: ASet.Side(T, xs, U32.shr(n), k, U32.is_lt(k, U32.shr(U32.shr(n)))), f: ASet.Both(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n))))) -> ASet.Both(T, ANode{xs, ys}, n, j, v, False{}, k, w):
  match sx:
    case Tuple{+s, r1}:
      match r1:
        case Tuple{+g, u}:
          match f:
            case Tuple{o, r}:
              (ASet.node_f(T, xs, ys, n, j, v, s, o), ASet.reads_f(T, xs, ys, n, j, v, k, w, g, r))

def ASet.both(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool, +k: U32, +w: Bool) -> ASet.Both(T, a, n, j, v, z, k, w):
  match a z:
    case ALeaf{x} _:
      (({==}, {==}, {==}, {==}, {==}, {==}, Unit{}), ({==}, {==}, {==}, (h => ASet.no_leaf(T, x, n, j, v, z, k, w, h)), Unit{}))
    case ANode{xs, ys} True{}:
      ASet.both_t(T, xs, ys, n, j, v, k, w, ASet.side(T, ys, U32.shr(n), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n)))), ASet.both(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))), k, U32.is_lt(k, U32.shr(U32.shr(n)))))
    case ANode{xs, ys} False{}:
      ASet.both_f(T, xs, ys, n, j, v, k, w, ASet.side(T, xs, U32.shr(n), k, U32.is_lt(k, U32.shr(U32.shr(n)))), ASet.both(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n)))))


def ASet.close_other(-T: Data, -b: Array<T>, +n: U32, +i: U32, +k: U32, +v: T, +q: {n == ASet.sz(T, b) : U32}, f: ASet.Both(T, b, n, ASet.J(i, n), v, ASet.Z(i, n), ASet.J(k, n), ASet.Z(k, n))) -> ASet.Apart(T, b, ASet.sz(T, b), ASet.J(i, ASet.sz(T, b)), ASet.Z(i, ASet.sz(T, b)), ASet.J(k, ASet.sz(T, b)), ASet.Z(k, ASet.sz(T, b))) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, b, i, v), k)) == Pair.snd(Array<T>, T, Array.get(T, b, k)) : T}:
  match f:
    case Tuple{o, r}:
      match o:
        case Tuple{+c1, o1}:
          match o1:
            case Tuple{+c2, o2}:
              match o2:
                case Tuple{+c3, o3}:
                  match o3:
                    case Tuple{+c4, o4}:
                      match r:
                        case Tuple{+d, r1}:
                          match r1:
                            case Tuple{+e1, r2}:
                              match r2:
                                case Tuple{+e2, r3}:
                                  match r3:
                                    case Tuple{e3, u}:
                                      %c1 : (ASet.Apart(T, b, ASet.sz(T, b), ASet.J(i, ASet.sz(T, b)), ASet.Z(i, ASet.sz(T, b)), ASet.J(k, ASet.sz(T, b)), ASet.Z(k, ASet.sz(T, b))) -> {Pair.snd(Array<T>, T, Array.get.at(T, k, Array.size(T, Array.set.fin(T, Array.swap.at(T, i, v, _))))) == Pair.snd(Array<T>, T, Array.get.at(T, k, _)) : T})
                                      %q : (ASet.Apart(T, b, _, ASet.J(i, _), ASet.Z(i, _), ASet.J(k, _), ASet.Z(k, _)) -> {Pair.snd(Array<T>, T, Array.get.at(T, k, Array.size(T, Array.set.fin(T, Array.swap.go(T, b, _, ASet.J(i, _), v, ASet.Z(i, _)))))) == Pair.snd(Array<T>, T, Array.get.go(T, b, _, ASet.J(k, _), ASet.Z(k, _))) : T})
                                      %c3 : (ASet.Apart(T, b, n, ASet.J(i, n), ASet.Z(i, n), ASet.J(k, n), ASet.Z(k, n)) -> {Pair.snd(Array<T>, T, Array.get.at(T, k, _)) == Pair.snd(Array<T>, T, Array.get.go(T, b, n, ASet.J(k, n), ASet.Z(k, n))) : T})
                                      %Equal.trans(U32, n, ASet.sz(T, b), ASet.sz(T, ASet.NA(T, b, n, ASet.J(i, n), v, ASet.Z(i, n))), q, Equal.sym(U32, ASet.sz(T, ASet.NA(T, b, n, ASet.J(i, n), v, ASet.Z(i, n))), ASet.sz(T, b), c4)) : (ASet.Apart(T, b, n, ASet.J(i, n), ASet.Z(i, n), ASet.J(k, n), ASet.Z(k, n)) -> {Pair.snd(Array<T>, T, Array.get.go(T, ASet.NA(T, b, n, ASet.J(i, n), v, ASet.Z(i, n)), _, ASet.J(k, _), ASet.Z(k, _))) == Pair.snd(Array<T>, T, Array.get.go(T, b, n, ASet.J(k, n), ASet.Z(k, n))) : T})
                                      e3


def ASet.top_other(-T: Data, -a: Array<T>, +i: U32, +k: U32, +v: T, r: ASet.Copy(T, a)) -> ASet.Apart(T, a, ASet.sz(T, a), ASet.J(i, ASet.sz(T, a)), ASet.Z(i, ASet.sz(T, a)), ASet.J(k, ASet.sz(T, a)), ASet.Z(k, ASet.sz(T, a))) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, a, i, v), k)) == Pair.snd(Array<T>, T, Array.get(T, a, k)) : T}:
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+n, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  %p : (ASet.Apart(T, _, ASet.sz(T, _), ASet.J(i, ASet.sz(T, _)), ASet.Z(i, ASet.sz(T, _)), ASet.J(k, ASet.sz(T, _)), ASet.Z(k, ASet.sz(T, _))) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, _, i, v), k)) == Pair.snd(Array<T>, T, Array.get(T, _, k)) : T})
                  ASet.close_other(T, b, n, i, k, v, q, ASet.both(T, b, n, ASet.J(i, n), v, ASet.Z(i, n), ASet.J(k, n), ASet.Z(k, n)))

# LAW: a write at i leaves alone the cell a read at k reaches, when the two descents part before a leaf
# (any i and k: both wrap them; h is what makes them different cells, see the top of the file)
law get_other_set:
  for -T: Data
  for a: Array<T>
  for +i: U32
  for +k: U32
  for +v: T
  for h: ASet.Apart(T, a, ASet.sz(T, a), ASet.J(i, ASet.sz(T, a)), ASet.Z(i, ASet.sz(T, a)), ASet.J(k, ASet.sz(T, a)), ASet.Z(k, ASet.sz(T, a)))
  {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, a, i, v), k)) == Pair.snd(Array<T>, T, Array.get(T, a, k)) : T}

def get_other_set(T, a, i, k, v, h):
  ASet.top_other(T, a, i, k, v, ASet.copy(T, a))(h)
