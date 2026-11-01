# dotg (the dot product unrolled by four, 0.1.19.0) gives the bits of dotr (one step at a time) for every k from 0
# to 40 (all remainders mod 4) with NaN, infinities and subnormals mixed in, unit and wider strides on either
# side, and three long dots without specials (768, 771, 3072); NaN counts as equal to NaN.
# Prints the number of differing results per case: all must be 0.
import Base
import ../main.bend as TA

# deterministic values, with specials mixed in by index (as tests/t8.bend) when sp; without them a long dot would
# almost always meet a NaN or an infinity and compare nothing
def val_sp(+h: U32) -> F32:
  Bool.pick(F32, U32.is_eq(h, 7), (0.0 - 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 13), (1.0 / 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 17), (0.0 / 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 23), 0.00000000000000000000000000000000000001, (((U32.to_f32(h) / 250.0 : F32) - 2.0 : F32))))))

def val(+sp: Bool, +i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  Bool.pick(F32, Bool.and(sp, U32.is_lt(h, 24)), val_sp(h), (((U32.to_f32(h) / 250.0 : F32) - 2.0 : F32)))

def fill(k: Nat, +sp: Bool, +i: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, sp, (i + 1 : U32), seed, Array.set(F32, a, i, val(sp, i, seed)))

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def same(+x: F32, +y: F32) -> U32:
  Bool.pick(U32, Bool.or(U32.is_eq(bits(x), bits(y)), Bool.and(F32.is_ne(x, x), F32.is_ne(y, y))), 0, 1)

def cmp(r1: TA.R, r2: TA.R) -> U32:
  match r1 r2:
    case TA.R{a1, b1, +x} TA.R{a2, b2, +y}: same(x, y)

def arr(+sp: Bool, +n: Nat, +seed: U32) -> Array<F32>:
  fill(n, sp, 0, seed, Array.new(F32, TA.cap_depth(n), 0.0))

# one dot of k products with strides sa, sb, starting at offsets ia, ib: dotg against dotr
def one(+sp: Bool, +k: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, +seed: U32) -> U32:
  +na = Nat.add(Nat.add(U32.to_nat(ia), Nat.mul(1n+k, U32.to_nat(sa))), 1n)
  +nb = Nat.add(Nat.add(U32.to_nat(ib), Nat.mul(1n+k, U32.to_nat(sb))), 1n)
  cmp(TA.dotg(k, ia, ib, sa, sb, 0.0, Array.get(F32, arr(sp, na, seed), ia), Array.get(F32, arr(sp, nb, (seed + 1 : U32)), ib)), TA.dotr(k, ia, ib, sa, sb, 0.0, 0.0, 0.0, 0.0, Array.get(F32, arr(sp, na, seed), ia), Array.get(F32, arr(sp, nb, (seed + 1 : U32)), ib)))

def ks(k: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, acc: U32) -> U32:
  match k:
    case 0n: (acc + one(True{}, 0n, ia, ib, sa, sb, 3) : U32)
    case 1n++p: ks(p, ia, ib, sa, sb, (acc + one(True{}, 1n+p, ia, ib, sa, sb, U32.from_nat(p)) : U32))

def main() -> String:
  U32.show(ks(40n, 0, 0, 1, 1, 0)) ++ " " ++ U32.show(ks(40n, 2, 5, 1, 7, 0)) ++ " " ++ U32.show(ks(40n, 3, 1, 5, 1, 0)) ++ " " ++ U32.show((one(False{}, 768n, 0, 0, 1, 1, 21) + (one(False{}, 771n, 4, 9, 1, 3, 22) + one(False{}, 3072n, 0, 0, 1, 1, 23) : U32) : U32))
