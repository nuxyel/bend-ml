# tensor-gpu's matmul_precise against tensor-array's, bit for bit, on the GPU and the CPU threads: one row and several rows,
# k from inside one block to many (up to 49,152), three times each. Prints the differing words over every
# product, then the number of products: "0 21".
import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import ../main.bend as TG

def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  Bool.pick(F32, U32.is_eq(h, 7), (0.0 - 0.0 : F32), (((U32.to_f32(h) / 250.0 : F32) - 2.0 : F32)))

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

def mmc(-n: Nat, -k: Nat, -m: Nat, r: TA.MMul<n, k, m>) -> Array<F32>:
  match r:
    case TA.MMul{a, b, TA.Mat{c}}: c

@unsafe
def one(+p: Nat, +n: Nat, +m: Nat, +k: Nat) -> U32:
  diff(Nat.mul(n, m), mmc(n, k, m, TG.matmul_precise(n, k, m, p, TA.Mat{arr(Nat.mul(n, k), 3)}, TA.Mat{arr(Nat.mul(k, m), 4)})), mmc(n, k, m, TA.Mat.matmul_precise(n, k, m, 0n, TA.Mat{arr(Nat.mul(n, k), 3)}, TA.Mat{arr(Nat.mul(k, m), 4)})))

type S is Data:
  S{n: Nat, m: Nat, k: Nat}

@unsafe
def gs(ss: List<&2, S>, +p: Nat, acc: U32) -> U32:
  match ss:
    case Nil{}: acc
    case Con{S{+n, +m, +k}, r}: gs(r, p, (acc + one(p, n, m, k) : U32))

def shapes() -> List<&2, S>:
  [S{1n, 5n, 3001n}, S{3n, 4n, 1000n}, S{9n, 6n, 1025n}, S{5n, 7n, 2049n}, S{64n, 33n, 4100n}, S{70n, 9n, 2048n}, S{2n, 3n, 49152n}]

@unsafe
def test_main() -> IO(Unit):
  IO.print(U32.show(((gs(shapes(), 0n, 0) + gs(shapes(), 4n, 0) : U32) + gs(shapes(), 6n, 0) : U32)) ++ " " ++ U32.show((7 * 3 : U32)))

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
