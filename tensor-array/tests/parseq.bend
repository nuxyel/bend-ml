# Parallel = sequential (the laws track's first target, step L1): for every shape on a grid, every layout (nn, nt, tn)
# and every split depth 1..4, gemm_any gives the same bits as at depth 0, plain and accumulating. Odd counts at a node of
# the fork tree are where the packing (pk_tree) and the blocks (t8_rpar, t8_cpar, cpar) must cut alike. Prints the
# number of differing words over the whole grid, then the number of products compared: the first must be 0.
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

# one shape, one layout: the differing words of depth d against depth 0, plain and accumulating
def one(+d: Nat, +n: Nat, +m: Nat, +k: Nat, +l: Nat, +seed: U32) -> U32:
  +ku = U32.from_nat(k)
  +nu = U32.from_nat(n)
  +mu = U32.from_nat(m)
  +arow = Bool.pick(U32, Nat.is_eq(l, 2n), 1, ku)
  +sa = Bool.pick(U32, Nat.is_eq(l, 2n), nu, 1)
  +sb = Bool.pick(U32, Nat.is_eq(l, 1n), 1, mu)
  +bcol = Bool.pick(U32, Nat.is_eq(l, 1n), ku, 1)
  +nm = Nat.mul(n, m)
  +plain = diff(nm, g_c(TA.gemm_any(False{}, d, n, m, k, arow, sa, sb, bcol, arr(Nat.mul(n, k), seed), arr(Nat.mul(k, m), (seed + 1 : U32)), Array.new(F32, TA.cap_depth(nm), 0.0))), g_c(TA.gemm_any(False{}, 0n, n, m, k, arow, sa, sb, bcol, arr(Nat.mul(n, k), seed), arr(Nat.mul(k, m), (seed + 1 : U32)), Array.new(F32, TA.cap_depth(nm), 0.0))))
  +acc = diff(nm, g_c(TA.gemm_any(True{}, d, n, m, k, arow, sa, sb, bcol, arr(Nat.mul(n, k), seed), arr(Nat.mul(k, m), (seed + 1 : U32)), arr(nm, (seed + 2 : U32)))), g_c(TA.gemm_any(True{}, 0n, n, m, k, arow, sa, sb, bcol, arr(Nat.mul(n, k), seed), arr(Nat.mul(k, m), (seed + 1 : U32)), arr(nm, (seed + 2 : U32)))))
  (plain + acc : U32)

# the grid, one loop per axis (no mutual recursion: each loop calls the next one down)
def gl(l: Nat, +d: Nat, +n: Nat, +m: Nat, +k: Nat, acc: U32) -> U32:
  match l:
    case 0n:
      acc
    case 1n++p:
      gl(p, d, n, m, k, (acc + one(d, n, m, k, p, U32.from_nat(Nat.add(Nat.mul(n, 31n), Nat.add(Nat.mul(m, 7n), k)))) : U32))

def gk(ks: List<&2, Nat>, +d: Nat, +n: Nat, +m: Nat, acc: U32) -> U32:
  match ks:
    case Nil{}:
      acc
    case Con{+k, r}:
      gk(r, d, n, m, gl(3n, d, n, m, k, acc))

def gm(mi: Nat, +d: Nat, +n: Nat, acc: U32) -> U32:
  match mi:
    case 0n:
      acc
    case 1n++p:
      gm(p, d, n, gk([0n, 1n, 3n, 8n, 9n, 17n], d, n, 1n+p, acc))

def gn(ni: Nat, +d: Nat, acc: U32) -> U32:
  match ni:
    case 0n:
      acc
    case 1n++p:
      gn(p, d, gm(9n, d, 1n+p, acc))

def gd(di: Nat, acc: U32) -> U32:
  match di:
    case 0n:
      acc
    case 1n++p:
      gd(p, gn(9n, 1n+p, acc))

def test_main() -> IO(Unit):
  IO.print(U32.show(gd(4n, 0)) ++ " " ++ Nat.show(Nat.mul(Nat.mul(4n, 81n), Nat.mul(6n, 6n))))

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
