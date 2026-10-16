# gemm_t8 (cells of 8) gives the same bits as gemm (the scalar loop) on every layout (nn, nt, tn), every k from 0 to 19
# (all remainders mod 4 and mod 8), the sequential and the parallel path, and two larger products; NaN counts as equal
# to NaN (which NaN payload comes out may differ). Prints the number of differing words per case: all must be 0.
import Base
import ../main.bend as TA

# deterministic values, with specials mixed in by index
def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  Bool.pick(F32, U32.is_eq(h, 7), (0.0 - 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 13), (1.0 / 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 17), (0.0 / 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 23), 0.00000000000000000000000000000000000001, (((U32.to_f32(h) / 250.0 : F32) - 2.0 : F32))))))

def fill(k: Nat, +i: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, (i + 1 : U32), seed, Array.set(F32, a, i, val(i, seed)))

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def cmp(k: Nat, +i: U32, acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmp(p, (i + 1 : U32), (acc + Bool.pick(U32, Bool.or(U32.is_eq(bits(x), bits(y)), Bool.and(F32.is_ne(x, x), F32.is_ne(y, y))), 0, 1) : U32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def diff2(+cnt: Nat, g1: TA.G, g2: TA.G) -> U32:
  match g1 g2:
    case TA.G{a1, b1, c1} TA.G{a2, b2, c2}:
      cmp(cnt, 0, 0, Array.get(F32, c1, 0), Array.get(F32, c2, 0))

# layout l: 0 = nn, 1 = nt, 2 = tn; returns the number of output words that differ
def one(+d: Nat, +n: Nat, +m: Nat, +k: Nat, +l: Nat, +seed: U32) -> U32:
  +ku = U32.from_nat(k)
  +nu = U32.from_nat(n)
  +mu = U32.from_nat(m)
  +arow = Bool.pick(U32, Nat.is_eq(l, 2n), 1, ku)
  +sa = Bool.pick(U32, Nat.is_eq(l, 2n), nu, 1)
  +sb = Bool.pick(U32, Nat.is_eq(l, 1n), 1, mu)
  +bcol = Bool.pick(U32, Nat.is_eq(l, 1n), ku, 1)
  +na = TA.cap_depth(Nat.mul(n, k))
  +nb = TA.cap_depth(Nat.mul(k, m))
  +nc = TA.cap_depth(Nat.mul(n, m))
  diff2(Nat.mul(n, m), TA.gemm(n, m, k, arow, sa, sb, bcol, fill(Nat.mul(n, k), 0, seed, Array.new(F32, na, 0.0)), fill(Nat.mul(k, m), 0, (seed + 1 : U32), Array.new(F32, nb, 0.0)), Array.new(F32, nc, 0.0)), TA.gemm_t8(d, n, m, k, arow, sa, sb, bcol, fill(Nat.mul(n, k), 0, seed, Array.new(F32, na, 0.0)), fill(Nat.mul(k, m), 0, (seed + 1 : U32), Array.new(F32, nb, 0.0)), Array.new(F32, nc, 0.0)))

def ks(k: Nat, +d: Nat, +n: Nat, +m: Nat, +l: Nat, acc: U32) -> U32:
  match k:
    case 0n: (acc + one(d, n, m, 0n, l, 3) : U32)
    case 1n++p: ks(p, d, n, m, l, (acc + one(d, n, m, 1n+p, l, U32.from_nat(p)) : U32))

def main() -> String:
  U32.show(ks(19n, 0n, 3n, 5n, 0n, 0)) ++ " " ++ U32.show(ks(19n, 0n, 4n, 3n, 1n, 0)) ++ " " ++ U32.show(ks(19n, 0n, 5n, 2n, 2n, 0)) ++ " " ++ U32.show(ks(19n, 2n, 7n, 6n, 0n, 0)) ++ " " ++ U32.show(ks(19n, 3n, 9n, 4n, 1n, 0)) ++ " " ++ U32.show(ks(19n, 1n, 2n, 9n, 2n, 0)) ++ " " ++ U32.show(one(3n, 100n, 128n, 784n, 1n, 9)) ++ " " ++ U32.show(one(3n, 128n, 784n, 100n, 2n, 11))
