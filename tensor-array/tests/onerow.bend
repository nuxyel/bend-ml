# One row in parallel (0.1.12.0): a product with one row runs the scalar loop at every depth. Up to 0.1.11.0 it split by
# columns and cloned both operands at every node: this file's first product (one row of 768 against a 50,257 x 768
# weight matrix, GPT-2's output projection, at par 4) peaked at 9.5 GB on 0.1.11.0 (565 MB on 0.1.12.0, 16 threads), and
# in a benchmark that also held its inputs it ran out of the default heap. Prints the differing words of par 4
# against par 0 for Mat.matmul_nt and Mat.matmul_nt_acc at that shape and Mat.matmul at 1 x 768 x 3072: must be "0 0 0".
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

def one(+k: Nat, +m: Nat, +par: Nat, +seed: U32) -> String:
  +km = Nat.mul(k, m)
  +s1 = (seed + 1 : U32)
  +s2 = (seed + 2 : U32)
  +nt = diff(m, mm_nt_c(1n, k, m, TA.Mat.matmul_nt(1n, k, m, par, TA.Mat{arr(k, seed)}, TA.Mat{arr(km, s1)})), mm_nt_c(1n, k, m, TA.Mat.matmul_nt(1n, k, m, 0n, TA.Mat{arr(k, seed)}, TA.Mat{arr(km, s1)})))
  +at = diff(m, mm_nt_c(1n, k, m, TA.Mat.matmul_nt_acc(1n, k, m, par, TA.Mat{arr(k, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(m, s2)})), mm_nt_c(1n, k, m, TA.Mat.matmul_nt_acc(1n, k, m, 0n, TA.Mat{arr(k, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(m, s2)})))
  U32.show(nt) ++ " " ++ U32.show(at)

def one_nn(+k: Nat, +m: Nat, +par: Nat, +seed: U32) -> String:
  +km = Nat.mul(k, m)
  +s1 = (seed + 1 : U32)
  U32.show(diff(m, mm_c(1n, k, m, TA.Mat.matmul(1n, k, m, par, TA.Mat{arr(k, seed)}, TA.Mat{arr(km, s1)})), mm_c(1n, k, m, TA.Mat.matmul(1n, k, m, 0n, TA.Mat{arr(k, seed)}, TA.Mat{arr(km, s1)}))))

def test_main() -> IO(Unit):
  IO.print(one(768n, 50257n, 4n, 5) ++ " " ++ one_nn(768n, 3072n, 4n, 9))

# This program takes no arguments. Since Bend 2.0.38 a runtime option goes between +RTS and -RTS, and one written
# without them arrives here as an argument while the program runs on the default thread count, so a check of 1 and 16
# threads would pass without testing them: any argument is an error.
def alone(args: List<String>, go: IO(Unit)) -> IO(Unit):
  match args:
    case Nil{}:
      go
    case Con{p, Nil{}}:
      go
    case Con{p, Con{a, t}}:
      do IO<Unit>:
        IO.print_err("unexpected argument " ++ a ++ ": this program takes none (runtime options go between +RTS and -RTS)")
        IO.exit(2)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    alone(args, test_main())
