# C = A · Btᵀ  (A: n x k, Bt: m x k, both Array<F32> row by row) with blocks of rows in parallel.
# Data: A[r][c] = ((7r + 3c) mod 13) / 10,  Bt[j][c] = ((5j + 2c) mod 11) / 10.
import Base

type R is Type:
  R{a: Array<F32>, b: Array<F32>, x: F32}

type S is Type:
  S{a: Array<F32>, b: Array<F32>, out: List<&2, F32>}

# acc += a[ia+t] * b[ib+t], t = 0..n-1; ra, rb already carry the first (array, element) pair
def dotk(n: Nat, +ia: U32, +ib: U32, acc: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> R:
  match n ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      R{a, b, acc}
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      dotk(p, (ia + 1 : U32), (ib + 1 : U32), (acc + (x * y : F32) : F32), Array.get(F32, a, (ia + 1 : U32)), Array.get(F32, b, (ib + 1 : U32)))

# row i of C: one column at a time; r is the product of column j (already computed)
def cols(m: Nat, +kn: Nat, +ku: U32, +i: U32, +j: U32, out: List<&2, F32>, r: R) -> S:
  match m r:
    case 0n R{a, b, +x}:
      S{a, b, x <> out}
    case 1n+p R{a, b, +x}:
      cols(p, kn, ku, i, (j + 1 : U32), x <> out, dotk(kn, (i * ku : U32), ((j + 1 : U32) * ku : U32), 0.0, Array.get(F32, a, (i * ku : U32)), Array.get(F32, b, ((j + 1 : U32) * ku : U32))))

def first(+kn: Nat, +ku: U32, +i: U32, +j0: U32, a: Array<F32>, b: Array<F32>) -> R:
  dotk(kn, (i * ku : U32), (j0 * ku : U32), 0.0, Array.get(F32, a, (i * ku : U32)), Array.get(F32, b, (j0 * ku : U32)))

# cnt rows of C starting at row i
def rows(cnt: Nat, +mm1: Nat, +kn: Nat, +ku: U32, +i: U32, +j0: U32, st: S) -> S:
  match cnt st:
    case 0n S{a, b, out}:
      S{a, b, out}
    case 1n+p S{a, b, out}:
      rows(p, mm1, kn, ku, (i + 1 : U32), j0, cols(mm1, kn, ku, i, j0, out, first(kn, ku, i, j0, a, b)))

def drop_arrays(st: S) -> List<&2, F32>:
  match st:
    case S{a, b, out}:
      List.reverse(&2, F32, out)

# nb blocks of columns (each with mm1+1 columns), from j0 on; the whole Bt block fits in the cache
def tiles(nb: Nat, +cnt: Nat, +mm1: Nat, +kn: Nat, +ku: U32, +i: U32, +j0: U32, st: S) -> S:
  match nb:
    case 0n:
      st
    case 1n+p:
      tiles(p, cnt, mm1, kn, ku, i, (j0 + U32.from_nat(Nat.add(mm1, 1n)) : U32), rows(cnt, mm1, kn, ku, i, j0, st))

def block(cnt: Nat, +mm1: Nat, +kn: Nat, +ku: U32, +i: U32, a: Array<F32>, b: Array<F32>) -> List<&2, F32>:
  drop_arrays(tiles(TILES, cnt, mm1, kn, ku, i, 0, S{a, b, Nil{}}))

# 2^d blocks of rows in parallel; each block receives its own copy of A and Bt.
# ra and rb are (original, copy): one goes to each half.
def par(d: Nat, +cnt: Nat, +mm1: Nat, +kn: Nat, +ku: U32, +i: U32, ra: Array<F32> & Array<F32>, rb: Array<F32> & Array<F32>) -> List<&2, F32>:
  match d ra rb:
    case 0n Tuple{a, a2} Tuple{b, b2}:
      block(cnt, mm1, kn, ku, i, a, b)
    case 1n++q Tuple{a, a2} Tuple{b, b2}:
      x y = par(q, Nat.div(cnt, 2n), mm1, kn, ku, i, Array.clone(F32, a), Array.clone(F32, b)) par(q, Nat.sub(cnt, Nat.div(cnt, 2n)), mm1, kn, ku, (i + U32.from_nat(Nat.div(cnt, 2n)) : U32), Array.clone(F32, a2), Array.clone(F32, b2))
      List.append(&2, F32, x, y)

# ---- data and verification ----
def md(+x: U32, +m: U32) -> U32:
  (x - (U32.div(x, m) * m : U32) : U32)

# fills the n rows (each with k values) of an Array with f(r, c) = ((u*r + v*c) mod w) / 10
def fillrow(c: Nat, +cc: U32, +r: U32, +u: U32, +v: U32, +w: U32, +base: U32, a: Array<F32>) -> Array<F32>:
  match c:
    case 0n:
      a
    case 1n+p:
      fillrow(p, (cc + 1 : U32), r, u, v, w, base, Array.set(F32, a, (base + cc : U32), (U32.to_f32(md((u * r + v * cc : U32), w)) / 10.0 : F32)))

def fillmat(rn: Nat, +kn: Nat, +ku: U32, +r: U32, +u: U32, +v: U32, +w: U32, a: Array<F32>) -> Array<F32>:
  match rn:
    case 0n:
      a
    case 1n+p:
      fillmat(p, kn, ku, (r + 1 : U32), u, v, w, fillrow(kn, 0, r, u, v, w, (r * ku : U32), a))

def sum.l(xs: List<&2, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      sum.l(t, (acc + h : F32))

# n=100, k=784, m=128; 2^d blocks
def run(d: Nat, a: Array<F32>, b: Array<F32>) -> F32:
  sum.l(par(d, 100n, MM1, 784n, 784, 0, Array.clone(F32, a), Array.clone(F32, b)), 0.0)

# repeats the product k times (each time with fresh copies of A and Bt)
def many(k: Nat, +d: Nat, ra: Array<F32> & Array<F32>, rb: Array<F32> & Array<F32>, acc: F32) -> F32:
  match k ra rb:
    case 0n Tuple{a, a2} Tuple{b, b2}:
      acc
    case 1n+p Tuple{a, a2} Tuple{b, b2}:
      many(p, d, Array.clone(F32, a2), Array.clone(F32, b2), (acc + run(d, a, b) : F32))

def main() -> IO(Unit):
  IO.print(F32.show(many(REPS, D, Array.clone(F32, fillmat(100n, 784n, 784, 0, 7, 3, 13, [0.0 : F32^17n])), Array.clone(F32, fillmat(128n, 784n, 784, 0, 5, 2, 11, [0.0 : F32^17n])), 0.0)))
