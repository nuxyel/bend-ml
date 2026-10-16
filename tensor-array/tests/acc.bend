# The accumulating products (0.1.10.0) give the bits of computing the product and adding it: C0 + A·B from an _acc
# function equals Mat.add(C0, A·B) word for word, on every path of the dispatch (one row: the scalar loop and the
# parallel columns; two or more rows: cells of 8, sequential, parallel by rows and parallel by columns), every layout
# (nn, nt, tn), every k from 0 to 19, and through the four public functions; C0 has special values too (-0, inf, NaN, a
# subnormal). NaN counts as equal to NaN. Prints the number of differing words per case: all must be 0.
import Base
import ../main.bend as TA

# deterministic values, with specials mixed in by index (as tests/t8.bend)
def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  Bool.pick(F32, U32.is_eq(h, 7), (0.0 - 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 13), (1.0 / 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 17), (0.0 / 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 23), 0.00000000000000000000000000000000000001, (((U32.to_f32(h) / 250.0 : F32) - 2.0 : F32))))))

def fill(k: Nat, +i: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, (i + 1 : U32), seed, Array.set(F32, a, i, val(i, seed)))

def arr(+cnt: Nat, +seed: U32) -> Array<F32>:
  fill(cnt, 0, seed, Array.new(F32, TA.cap_depth(cnt), 0.0))

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def cmp(k: Nat, +i: U32, acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmp(p, (i + 1 : U32), (acc + Bool.pick(U32, Bool.or(U32.is_eq(bits(x), bits(y)), Bool.and(F32.is_ne(x, x), F32.is_ne(y, y))), 0, 1) : U32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def diff(+cnt: Nat, x: Array<F32>, y: Array<F32>) -> U32:
  cmp(cnt, 0, 0, Array.get(F32, x, 0), Array.get(F32, y, 0))

def g_c(g: TA.G) -> Array<F32>:
  match g:
    case TA.G{a, b, c}: c

def m_d(-r: Nat, -c: Nat, x: TA.Mat<r, c>) -> Array<F32>:
  match x:
    case TA.Mat{d}: d

def pair_a(-r: Nat, -c: Nat, p: TA.MPair<r, c>) -> Array<F32>:
  match p:
    case TA.MPair{a, b}: m_d(r, c, a)

# Mat.add(c0, p): the reference, computed apart from the product
def plus(+r: Nat, +c: Nat, c0: Array<F32>, p: Array<F32>) -> Array<F32>:
  pair_a(r, c, TA.Mat.add(r, c, TA.Mat{c0}, TA.Mat{p}))

# layout l: 0 = nn, 1 = nt, 2 = tn; the dispatch picks the path from n and d
def one(+d: Nat, +n: Nat, +m: Nat, +k: Nat, +l: Nat, +seed: U32) -> U32:
  +ku = U32.from_nat(k)
  +nu = U32.from_nat(n)
  +mu = U32.from_nat(m)
  +arow = Bool.pick(U32, Nat.is_eq(l, 2n), 1, ku)
  +sa = Bool.pick(U32, Nat.is_eq(l, 2n), nu, 1)
  +sb = Bool.pick(U32, Nat.is_eq(l, 1n), 1, mu)
  +bcol = Bool.pick(U32, Nat.is_eq(l, 1n), ku, 1)
  +nm = Nat.mul(n, m)
  diff(nm, g_c(TA.gemm_any(True{}, d, n, m, k, arow, sa, sb, bcol, arr(Nat.mul(n, k), seed), arr(Nat.mul(k, m), (seed + 1 : U32)), arr(nm, (seed + 2 : U32)))), plus(n, m, arr(nm, (seed + 2 : U32)), g_c(TA.gemm_any(False{}, d, n, m, k, arow, sa, sb, bcol, arr(Nat.mul(n, k), seed), arr(Nat.mul(k, m), (seed + 1 : U32)), Array.new(F32, TA.cap_depth(nm), 0.0)))))

def ks(k: Nat, +d: Nat, +n: Nat, +m: Nat, +l: Nat, acc: U32) -> U32:
  match k:
    case 0n: (acc + one(d, n, m, 0n, l, 3) : U32)
    case 1n++p: ks(p, d, n, m, l, (acc + one(d, n, m, 1n+p, l, U32.from_nat(p)) : U32))

# the five paths of one layout
def paths(+l: Nat) -> U32:
  (((ks(19n, 0n, 1n, 6n, l, 0) + ks(19n, 2n, 1n, 9n, l, 0) : U32) + (ks(19n, 0n, 3n, 5n, l, 0) + ks(19n, 2n, 7n, 6n, l, 0) : U32) : U32) + ks(19n, 3n, 2n, 9n, l, 0) : U32)

# the public functions, against Mat.add of the plain ones
def mm_c(-n: Nat, -k: Nat, -m: Nat, r: TA.MMul<n, k, m>) -> Array<F32>:
  match r:
    case TA.MMul{a, b, c}: m_d(n, m, c)

def mm_nt_c(-n: Nat, -k: Nat, -m: Nat, r: TA.MMulNT<n, k, m>) -> Array<F32>:
  match r:
    case TA.MMulNT{a, b, c}: m_d(n, m, c)

def mm_tn_c(-n: Nat, -k: Nat, -m: Nat, r: TA.MMulTN<n, k, m>) -> Array<F32>:
  match r:
    case TA.MMulTN{a, b, c}: m_d(n, m, c)

def cs_s(-n: Nat, -m: Nat, r: TA.MColSum<n, m>) -> Array<F32>:
  match r:
    case TA.MColSum{a, s}: m_d(1n, m, s)

def api(+n: Nat, +k: Nat, +m: Nat, +par: Nat, +seed: U32) -> U32:
  +nk = Nat.mul(n, k)
  +km = Nat.mul(k, m)
  +nm = Nat.mul(n, m)
  +s1 = (seed + 1 : U32)
  +s2 = (seed + 2 : U32)
  +nn = diff(nm, mm_c(n, k, m, TA.Mat.matmul_acc(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})), plus(n, m, arr(nm, s2), mm_c(n, k, m, TA.Mat.matmul(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}))))
  +nt = diff(nm, mm_nt_c(n, k, m, TA.Mat.matmul_nt_acc(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})), plus(n, m, arr(nm, s2), mm_nt_c(n, k, m, TA.Mat.matmul_nt(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}))))
  +tn = diff(nm, mm_tn_c(n, k, m, TA.Mat.matmul_tn_acc(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})), plus(n, m, arr(nm, s2), mm_tn_c(n, k, m, TA.Mat.matmul_tn(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}))))
  +cs = diff(m, cs_s(n, m, TA.Mat.col_sums_acc(n, m, TA.Mat{arr(nm, seed)}, TA.Mat{arr(m, s2)})), plus(1n, m, arr(m, s2), cs_s(n, m, TA.Mat.col_sums(n, m, TA.Mat{arr(nm, seed)}))))
  (((nn + nt : U32) + tn : U32) + cs : U32)

def main() -> String:
  U32.show(paths(0n)) ++ " " ++ U32.show(paths(1n)) ++ " " ++ U32.show(paths(2n)) ++ " " ++ U32.show((api(1n, 13n, 6n, 0n, 5) + api(1n, 13n, 6n, 2n, 6) : U32)) ++ " " ++ U32.show((api(5n, 13n, 6n, 0n, 7) + api(5n, 13n, 6n, 2n, 8) : U32)) ++ " " ++ U32.show(api(100n, 128n, 10n, 3n, 9))
