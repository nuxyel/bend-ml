# split_depth (0.1.11.0): the public products split only when the work pays, and the depth never changes a bit.
# First line: split_depth on the edges of the rule (64 multiply-adds per cell of C, 2^23 in all), which must print
# "0 3 0 0 3 0 4". Second line: on a product above the threshold (128 x 100 by 100 x 784, 10 million multiply-adds),
# Mat.matmul, matmul_nt and matmul_tn and their accumulating forms at par 4 against par 0, the number of differing
# words of each: all must be 0.
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

def mm_c(-n: Nat, -k: Nat, -m: Nat, r: TA.MMul<n, k, m>) -> Array<F32>:
  match r:
    case TA.MMul{a, b, TA.Mat{c}}: c

def mm_nt_c(-n: Nat, -k: Nat, -m: Nat, r: TA.MMulNT<n, k, m>) -> Array<F32>:
  match r:
    case TA.MMulNT{a, b, TA.Mat{c}}: c

def mm_tn_c(-n: Nat, -k: Nat, -m: Nat, r: TA.MMulTN<n, k, m>) -> Array<F32>:
  match r:
    case TA.MMulTN{a, b, TA.Mat{c}}: c

def edges() -> String:
  Nat.show(TA.split_depth(3n, 1n, 1n, 63n)) ++ " " ++ Nat.show(TA.split_depth(3n, 2048n, 64n, 64n)) ++ " " ++ Nat.show(TA.split_depth(3n, 2048n, 63n, 64n)) ++ " " ++ Nat.show(TA.split_depth(3n, 1536n, 576n, 32n)) ++ " " ++ Nat.show(TA.split_depth(3n, 128n, 784n, 100n)) ++ " " ++ Nat.show(TA.split_depth(4n, 8n, 1536n, 576n)) ++ " " ++ Nat.show(TA.split_depth(4n, 32n, 1536n, 576n))

def big(+n: Nat, +k: Nat, +m: Nat, +par: Nat, +seed: U32) -> String:
  +nk = Nat.mul(n, k)
  +km = Nat.mul(k, m)
  +nm = Nat.mul(n, m)
  +s1 = (seed + 1 : U32)
  +s2 = (seed + 2 : U32)
  +nn = diff(nm, mm_c(n, k, m, TA.Mat.matmul(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})), mm_c(n, k, m, TA.Mat.matmul(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})))
  +nt = diff(nm, mm_nt_c(n, k, m, TA.Mat.matmul_nt(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})), mm_nt_c(n, k, m, TA.Mat.matmul_nt(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})))
  +tn = diff(nm, mm_tn_c(n, k, m, TA.Mat.matmul_tn(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})), mm_tn_c(n, k, m, TA.Mat.matmul_tn(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})))
  +an = diff(nm, mm_c(n, k, m, TA.Mat.matmul_acc(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})), mm_c(n, k, m, TA.Mat.matmul_acc(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})))
  +at = diff(nm, mm_nt_c(n, k, m, TA.Mat.matmul_nt_acc(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})), mm_nt_c(n, k, m, TA.Mat.matmul_nt_acc(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})))
  +aw = diff(nm, mm_tn_c(n, k, m, TA.Mat.matmul_tn_acc(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})), mm_tn_c(n, k, m, TA.Mat.matmul_tn_acc(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})))
  U32.show(nn) ++ " " ++ U32.show(nt) ++ " " ++ U32.show(tn) ++ " " ++ U32.show(an) ++ " " ++ U32.show(at) ++ " " ++ U32.show(aw)

def main() -> String:
  edges() ++ "\n" ++ big(128n, 100n, 784n, 4n, 11)
