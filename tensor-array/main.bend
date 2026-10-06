# bend-ml-tensor-array: tensors over a flat Array<F32>, with the shape in the type.
#
#   import bend-ml-tensor-array@0.1.3.0/main.bend as TA
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

def dotg(k: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, acc: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> R:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      R{a, b, acc}
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      dotg(p, (ia + sa : U32), (ib + sb : U32), sa, sb, (acc + (x * y : F32) : F32), Array.get(F32, a, (ia + sa : U32)), Array.get(F32, b, (ib + sb : U32)))

# row i of C, columns j.. ; r is the product of the current column (already computed)

def gcols(mleft: Nat, +kn: Nat, +ia0: U32, +ib0: U32, +sa: U32, +sb: U32, +bcol: U32, +cidx: U32, c: Array<F32>, r: R) -> G:
  match mleft r:
    case 0n R{a, b, +x}:
      G{a, b, Array.set(F32, c, cidx, x)}
    case 1n+p R{a, b, +x}:
      gcols(p, kn, ia0, (ib0 + bcol : U32), sa, sb, bcol, (cidx + 1 : U32), Array.set(F32, c, cidx, x), dotg(kn, ia0, (ib0 + bcol : U32), sa, sb, 0.0, Array.get(F32, a, ia0), Array.get(F32, b, (ib0 + bcol : U32))))

def grows(nleft: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +mu: U32, g: G) -> G:
  match nleft g:
    case 0n G{a, b, c}:
      G{a, b, c}
    case 1n+p G{a, b, c}:
      grows(p, mm1, kn, (i + 1 : U32), arow, sa, sb, bcol, mu, gcols(mm1, kn, (i * arow : U32), 0, sa, sb, bcol, (i * mu : U32), c, dotg(kn, (i * arow : U32), 0, sa, sb, 0.0, Array.get(F32, a, (i * arow : U32)), Array.get(F32, b, 0))))

# C (n x m) = A · B with k steps; c is the output array (with capacity n*m)

def gemm(+n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  grows(n, Nat.sub(m, 1n), k, 0, arow, sa, sb, bcol, U32.from_nat(m), G{a, b, c})


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

# c[idx..] += list

def gp3(a: Array<F32>, b: Array<F32>, c: Array<F32>, xs: List<&2, F32>) -> G:
  G{a, b, write_l(xs, 0, c)}

def gp2(d: Nat, +n: Nat, +mm1: Nat, +kn: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, c: Array<F32>, pa: Array<F32> & Array<F32>, pb: Array<F32> & Array<F32>) -> G:
  match pa pb:
    case Tuple{a1, a2} Tuple{b1, b2}:
      gp3(a1, b1, c, lpar(d, n, mm1, kn, 0, arow, sa, sb, bcol, Array.clone(F32, a2), Array.clone(F32, b2)))

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

def cp2(d: Nat, +m: Nat, +kn: Nat, +sa: U32, +sb: U32, +bcol: U32, c: Array<F32>, pa: Array<F32> & Array<F32>, pb: Array<F32> & Array<F32>) -> G:
  match pa pb:
    case Tuple{a1, a2} Tuple{b1, b2}:
      gp3(a1, b1, c, cpar(d, m, kn, 0, 0, sa, sb, bcol, Array.clone(F32, a2), Array.clone(F32, b2)))

# C (1 x m) = A (1 x k) · B, with 2^d blocks of columns in parallel

def gemm_par_cols(d: Nat, +m: Nat, +k: Nat, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  cp2(d, m, k, sa, sb, bcol, c, Array.clone(F32, a), Array.clone(F32, b))

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

# list of m elements starting at idx (reads with Array.get, returning the array)

def read_l(mleft: Nat, +idx: U32, acc: List<&2, F32>, rc: Array<F32> & F32) -> AL:
  match mleft rc:
    case 0n Tuple{c, +v}:
      AL{c, List.reverse(&2, F32, v <> acc)}
    case 1n+p Tuple{c, +v}:
      read_l(p, (idx + 1 : U32), v <> acc, Array.get(F32, c, (idx + 1 : U32)))

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
      ce_rows(t, mm1, mu, nf, (i + 1 : U32), ce_row(read_l(mm1, (i * mu : U32), Nil{}, Array.get(F32, c, (i * mu : U32))), y, nf, (i * mu : U32), loss))

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
      count_rows(t, mm1, mu, (i + 1 : U32), count_rows3(y, n, read_l(mm1, (i * mu : U32), Nil{}, Array.get(F32, c, (i * mu : U32)))))


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

# the depth d such that an Array with 2^d slots holds n elements
def cap_depth(n: Nat) -> Nat:
  fst_nat(cap_pair(n))

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

def cap_final(n: Nat, pr: {snd_nat(cap_pair(n)) == pow2(fst_nat(cap_pair(n))) : Nat} & {True{} == leb(n, snd_nat(cap_pair(n))) : Bool}) -> {True{} == leb(n, pow2(cap_depth(n))) : Bool}:
  match pr:
    case Tuple{i1, i2}:
      %i1 : {True{} == leb(n, _) : Bool}
      i2

# LAW (cap_ok): an Array with 2^cap_depth(n) slots always has room for n elements.
law cap_ok:
  for +n: Nat
  {True{} == leb(n, pow2(cap_depth(n))) : Bool}

def cap_ok(n):
  cap_final(n, cap_inv(n))

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

# two independent copies

def Mat.to_list(+r: Nat, +c: Nat, m: Mat<r, c>) -> ML<r, c>:
  match m:
    case Mat{d}:
      Mat.to_list2(r, c, read_l(Nat.sub(Nat.mul(r, c), 1n), 0, Nil{}, Array.get(F32, d, 0)))

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

def gemm_any3(cols: Bool, d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  match cols:
    case True{}:
      gemm_par_cols(d, m, k, sa, sb, bcol, a, b, c)
    case False{}:
      gemm_par(d, n, m, k, arow, sa, sb, bcol, a, b, c)

def gemm_any2(d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  match d:
    case 0n:
      gemm(n, m, k, arow, sa, sb, bcol, a, b, c)
    case 1n+q:
      gemm_any3(Nat.is_eq(n, 1n), 1n+q, n, m, k, arow, sa, sb, bcol, a, b, c)

def gemm_any(d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  gemm_any2(d, n, m, k, arow, sa, sb, bcol, a, b, c)

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

def mm_a(+n: Nat, +k: Nat, +m: Nat, d: Nat, a: Array<F32>, b: Array<F32>) -> MMul<n, k, m>:
  mmul_fin(n, k, m, gemm_any(d, n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, a, b, Array.new(F32, cap_depth(Nat.mul(n, m)), 0.0)))

# C = A · B   (n x k) · (k x m); par = log2 of the number of parallel blocks (0 = sequential)

def Mat.matmul(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<n, k>, b: Mat<k, m>) -> MMul<n, k, m>:
  match a b:
    case Mat{x} Mat{y}:
      mm_a(n, k, m, par, x, y)

def mm_nt(+n: Nat, +k: Nat, +m: Nat, d: Nat, a: Array<F32>, b: Array<F32>) -> MMulNT<n, k, m>:
  mmul_nt_fin(n, k, m, gemm_any(d, n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), a, b, Array.new(F32, cap_depth(Nat.mul(n, m)), 0.0)))

# C = A · Bᵀ   (n x k) · (m x k)ᵀ: for weights stored with one row per output

def Mat.matmul_nt(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<n, k>, b: Mat<m, k>) -> MMulNT<n, k, m>:
  match a b:
    case Mat{x} Mat{y}:
      mm_nt(n, k, m, par, x, y)

def mm_tn(+n: Nat, +k: Nat, +m: Nat, d: Nat, a: Array<F32>, b: Array<F32>) -> MMulTN<n, k, m>:
  mmul_tn_fin(n, k, m, gemm_any(d, n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, a, b, Array.new(F32, cap_depth(Nat.mul(n, m)), 0.0)))

# C = Aᵀ · B   (k x n)ᵀ · (k x m): weight gradient, X^T · dY

def Mat.matmul_tn(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<k, n>, b: Mat<k, m>) -> MMulTN<n, k, m>:
  match a b:
    case Mat{x} Mat{y}:
      mm_tn(n, k, m, par, x, y)

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
      add_row2(n, m, n, m, cd, read_l(Nat.sub(m, 1n), 0, Nil{}, Array.get(F32, bd, 0)))

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
      ce_fin(n, c, F32.from_nat(n), ce_rows(labels, Nat.sub(c, 1n), U32.from_nat(c), F32.from_nat(n), 0, L2{0.0, d}))

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
      row_fin(r, c, read_l(Nat.sub(c, 1n), (i * U32.from_nat(c) : U32), Nil{}, Array.get(F32, d, (i * U32.from_nat(c) : U32))))

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

# internal: the bands and y as a list in row order
type BL<-r: Nat, -c: Nat> is Type:
  BL{b: Bands<r, c>, y: List<&2, F32>}

def bv_leaf(-r: Nat, -c: Nat, st: S3) -> BL<r, c>:
  match st:
    case S3{a, b, out}:
      BL{BLeaf{Mat{b}}, List.reverse(&2, F32, out)}

# the rows of one band, one dot product each: the same kernel (and the same order of
# additions) as Mat.matmul_nt, so the results are identical
def bv_rows(-r: Nat, -c: Nat, rows: Nat, +k: Nat, x: Array<F32>, w: Array<F32>) -> BL<r, c>:
  match rows:
    case 0n:
      BL{BLeaf{Mat{w}}, Nil{}}
    case 1n+p:
      bv_leaf(r, c, lcols(p, k, 0, 0, 1, 1, U32.from_nat(k), Nil{}, dotg(k, 0, 0, 1, 1, 0.0, Array.get(F32, x, 0), Array.get(F32, w, 0))))

def bv_mat(-r: Nat, +c: Nat, +rr: Nat, x: Array<F32>, m: Mat<r, c>) -> BL<r, c>:
  match m:
    case Mat{w}:
      bv_rows(r, c, rr, c, x, w)

def bv_join(-r: Nat, -c: Nat, p: BL<half(r), c>, q: BL<Nat.sub(r, half(r)), c>) -> BL<r, c>:
  match p q:
    case BL{bx, yx} BL{by, yy}:
      BL{BNode{bx, by}, List.append(&2, F32, yx, yy)}

# the two bands of a node run in parallel, each with its own copy of x (c numbers);
# xp is a pair of copies of x (a leaf uses one). xp comes after the erased e on purpose:
# in Bend 2.0.35 an erased parameter at the end of the list turns the parallel let
# into two sequential calls (NOTES.md, exp. 11).
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

# ---- reading rows and the whole matrix ----

type BRow<-r: Nat, -c: Nat> is Type:
  BRow{b: Bands<r, c>, row: List<&2, F32>}

def brow_fin(-r: Nat, -c: Nat, al: AL) -> BRow<r, c>:
  match al:
    case AL{d, l}:
      BRow{BLeaf{Mat{d}}, l}

# n numbers starting at flat index i of one band (a band may have no rows)
def brow_leaf2(-r: Nat, -c: Nat, n: Nat, +i: U32, d: Array<F32>) -> BRow<r, c>:
  match n:
    case 0n:
      BRow{BLeaf{Mat{d}}, Nil{}}
    case 1n+p:
      brow_fin(r, c, read_l(p, i, Nil{}, Array.get(F32, d, i)))

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
      bl_join(r, c, blist(c, half(r), x, half(rr), eq_half(rr, r, e)), blist(c, Nat.sub(r, half(r)), y, Nat.sub(rr, half(rr)), eq_rest(rr, r, e)))

# all r*c numbers in row order (in BRow.row)
def Bands.to_list(+r: Nat, +c: Nat, b: Bands<r, c>) -> BRow<r, c>:
  blist(c, r, b, r, {==})
