# TF.softmax_ce (the rows in 2^par blocks sharing the logits) against TA.Mat.softmax_ce, word for word: the loss and
# every cell of the gradient, at par 0..6, on logits with special values (-0, inf, NaN, a subnormal) and labels inside
# and past the row (tensor-array's tests/ce_array.bend's values and labels).
# Cases: 2x3, MNIST's 100x10, 7 rows of 49,152, 1x1, 5x10 with 3 labels and with 7 (rows past the matrix, run after
# the blocks), 33x17, 64 rows of 4,096, 5x0 (no columns), 0x10 with no labels and with 3.
# Prints the differing words of each case summed over the seven depths, then the number of runs: "0 ... 0 77".
# NaN counts as equal to NaN.
import Base
import bend-ml-tensor-array@0.1.21.0/main.bend as TA
import ../main.bend as TF

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

def d_mce(-n: Nat, -c: Nat, +cnt: Nat, a: TA.MCE<n, c>, b: TA.MCE<n, c>) -> U32:
  match a b:
    case TA.MCE{+la, TA.Mat{x}} TA.MCE{+lb, TA.Mat{y}}: (same(la, lb) + diff(cnt, x, y) : U32)

# one case at one depth; cmpn words compared (k·c when k labels pass n rows)
def one(+n: Nat, +c: Nat, +k: Nat, +cmpn: Nat, +seed: U32, +par: Nat) -> U32:
  d_mce(n, c, cmpn, TF.softmax_ce(n, c, par, TA.Mat{arr(Nat.mul(n, c), seed)}, labels(k, 0, c, seed)), TA.Mat.softmax_ce(n, c, TA.Mat{arr(Nat.mul(n, c), seed)}, labels(k, 0, c, seed)))

# par d .. d+p-1 summed
def deps(p: Nat, +d: Nat, +n: Nat, +c: Nat, +k: Nat, +cmpn: Nat, +seed: U32, acc: U32) -> U32:
  match p:
    case 0n: acc
    case 1n+q: deps(q, 1n+d, n, c, k, cmpn, seed, (acc + one(n, c, k, cmpn, seed, d) : U32))

def case_(+n: Nat, +c: Nat, +k: Nat, +cmpn: Nat, +seed: U32) -> String:
  U32.show(deps(7n, 0n, n, c, k, cmpn, seed, 0)) ++ " "

def results() -> String:
  case_(2n, 3n, 2n, 6n, 1) ++ case_(100n, 10n, 100n, 1000n, 2) ++ case_(7n, 49152n, 7n, 344064n, 3) ++ case_(1n, 1n, 1n, 1n, 4) ++ case_(5n, 10n, 3n, 50n, 5) ++ case_(5n, 10n, 7n, 70n, 6) ++ case_(33n, 17n, 33n, 561n, 7) ++ case_(64n, 4096n, 64n, 262144n, 8) ++ case_(5n, 0n, 5n, 1n, 9) ++ case_(0n, 10n, 0n, 1n, 10) ++ case_(0n, 10n, 3n, 30n, 11) ++ "77"

# This program takes no arguments (as tests/bits.bend): a runtime option written outside +RTS ... -RTS arrives here as
# an argument while the program runs on the default thread count, so it is an error.
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
    alone(args, IO.print(results()))
