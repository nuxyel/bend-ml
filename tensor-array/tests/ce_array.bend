# Mat.softmax_ce's rows over the array (cea_rows, since 0.1.17.0) give the list version's bits (ce_rows): the loss and
# every cell of the gradient, word for word, on logits with special values (-0, inf, NaN, a subnormal), labels inside
# and past the row, and lists of labels shorter and longer than the matrix (both versions walk one row per label).
# Cases: 2x3, MNIST's 100x10, 7 rows of 49,152 (a language model's vocabulary), 1x1, 5x10 with 3 labels and with 7.
# Prints the number of differing words per case (the loss counts as one): all must be 0. NaN counts as equal to NaN.
import Base
import ../main.bend as TA

# deterministic values, with specials mixed in by index (as tests/acc.bend)
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

def same(+x: F32, +y: F32) -> U32:
  Bool.pick(U32, Bool.or(U32.is_eq(bits(x), bits(y)), Bool.and(F32.is_ne(x, x), F32.is_ne(y, y))), 0, 1)

def cmp(k: Nat, +i: U32, acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmp(p, (i + 1 : U32), (acc + same(x, y) : U32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def diff(+cnt: Nat, x: Array<F32>, y: Array<F32>) -> U32:
  cmp(cnt, 0, 0, Array.get(F32, x, 0), Array.get(F32, y, 0))

# k labels from seed, in 0 .. c + 1, so some fall past the row
def labels(k: Nat, +i: U32, +c: Nat, +seed: U32) -> List<&2, Nat>:
  match k:
    case 0n: Nil{}
    case 1n+p: Con{U32.to_nat(U32.mod(((i * 2246822519 : U32) + seed : U32), U32.from_nat(Nat.add(c, 2n)))), labels(p, (i + 1 : U32), c, seed)}

def d_l2(+cnt: Nat, a: TA.L2, b: TA.L2) -> U32:
  match a b:
    case TA.L2{+la, x} TA.L2{+lb, y}: (same(la, lb) + diff(cnt, x, y) : U32)

# rows of c from the same logits through both versions; cmpn words compared (k·c when k labels pass n rows)
def case_(+n: Nat, +c: Nat, +k: Nat, +cmpn: Nat, +seed: U32) -> U32:
  d_l2(cmpn, TA.ce_rows(labels(k, 0, c, seed), Nat.sub(c, 1n), U32.from_nat(c), F32.from_nat(n), 0, TA.L2{0.0, arr(Nat.mul(n, c), seed)}), TA.cea_rows(labels(k, 0, c, seed), Nat.sub(c, 1n), U32.from_nat(c), F32.from_nat(n), 0, TA.L2{0.0, arr(Nat.mul(n, c), seed)}))

def main() -> String:
  U32.show(case_(2n, 3n, 2n, 6n, 1)) ++ " " ++ U32.show(case_(100n, 10n, 100n, 1000n, 2)) ++ " " ++ U32.show(case_(7n, 49152n, 7n, 344064n, 3)) ++ " " ++ U32.show(case_(1n, 1n, 1n, 1n, 4)) ++ " " ++ U32.show(case_(5n, 10n, 3n, 50n, 5)) ++ " " ++ U32.show(case_(5n, 10n, 7n, 70n, 6))
