# bend-ml-tensor-gpu against bend-ml-tensor-array, bit for bit.
#
# First line: TG.gemm (one bang) against TA.gemm_any at depth 0, in every layout (nn, nt, tn) and both modes (C = A · B
# and C + A · B), on n, m = 1..9 with k = 0, 1, 3, 8, 9, 17 (tensor-array's tests/parseq.bend grid) and on (n, m, k) =
# (33, 65, 70), (64, 64, 64), (100, 31, 17), (1, 200, 64), (7, 300, 9), (300, 7, 100), (130, 130, 131): the differing
# words in C, then in A and B (the shared operands must come back as they went in), then the number of products:
# "0 0 2958". The counts sit in 16-bit fields of one U32; only "not 0" matters.
# Second line: the six public products against tensor-array's on 128 x 100 by 100 x 784, differing words of each:
# "0 0 0 0 0 0".
# Third line: matmul, matmul_nt and matmul_tn_acc on 512 x 576 by 576 x 1536 against tensor-array's: "0 0 0".
# The values mix in -0, +inf, NaN and a subnormal (as tensor-array's tests); NaN equals NaN whatever its payload.
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ../main.bend as TG

# deterministic values with specials mixed in by index (as tensor-array's tests/parseq.bend)
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

def g_c(g: TA.G) -> Array<F32>:
  match g:
    case TA.G{a, b, c}: c

# differing words: (in C) << 16 | (in A and B)
def g_cmp(+nm: Nat, +na: Nat, +nb: Nat, +seed: U32, want: Array<F32>, g: TA.G) -> U32:
  match g:
    case TA.G{a, b, c}:
      +dc = diff(nm, c, want)
      +dab = (diff(na, a, arr(na, seed)) + diff(nb, b, arr(nb, (seed + 1 : U32))) : U32)
      ((dc * 65536 : U32) + dab : U32)

def c0(ac: Bool, +nm: Nat, +seed: U32) -> Array<F32>:
  match ac:
    case True{}: arr(nm, (seed + 2 : U32))
    case False{}: Array.new(F32, TA.cap_depth(nm), 0.0)

# one shape, one layout (l: 0 nn, 1 nt, 2 tn), one mode
@unsafe
def one(+ac: Bool, +d: Nat, +n: Nat, +m: Nat, +k: Nat, +l: Nat) -> U32:
  +ku = U32.from_nat(k)
  +nu = U32.from_nat(n)
  +mu = U32.from_nat(m)
  +arow = Bool.pick(U32, Nat.is_eq(l, 2n), 1, ku)
  +sa = Bool.pick(U32, Nat.is_eq(l, 2n), nu, 1)
  +sb = Bool.pick(U32, Nat.is_eq(l, 1n), 1, mu)
  +bcol = Bool.pick(U32, Nat.is_eq(l, 1n), ku, 1)
  +nm = Nat.mul(n, m)
  +na = Nat.mul(n, k)
  +nb = Nat.mul(k, m)
  +seed = U32.from_nat(Nat.add(Nat.mul(n, 31n), Nat.add(Nat.mul(m, 7n), Nat.add(k, Nat.mul(l, 1000n)))))
  want = g_c(TA.gemm_any(ac, 0n, n, m, k, arow, sa, sb, bcol, arr(na, seed), arr(nb, (seed + 1 : U32)), c0(ac, nm, seed)))
  g_cmp(nm, na, nb, seed, want, abc_g(TG.gemm(ac, n, m, k, arow, sa, sb, bcol, arr(na, seed), arr(nb, (seed + 1 : U32)), c0(ac, nm, seed))))

# layouts and modes of one shape at one depth
@unsafe
def lm(i: Nat, +d: Nat, +n: Nat, +m: Nat, +k: Nat, acc: U32) -> U32:
  match i:
    case 0n:
      acc
    case 1n++p:
      lm(p, d, n, m, k, (acc + one(Nat.is_le(3n, p), d, n, m, k, Nat.mod(p, 3n)) : U32))

@unsafe
def gk(ks: List<&2, Nat>, +d: Nat, +n: Nat, +m: Nat, acc: U32) -> U32:
  match ks:
    case Nil{}:
      acc
    case Con{+k, r}:
      gk(r, d, n, m, lm(6n, d, n, m, k, acc))

@unsafe
def gm(mi: Nat, +d: Nat, +n: Nat, acc: U32) -> U32:
  match mi:
    case 0n:
      acc
    case 1n++p:
      gm(p, d, n, gk([0n, 1n, 3n, 8n, 9n, 17n], d, n, 1n+p, acc))

@unsafe
def gn(ni: Nat, +d: Nat, acc: U32) -> U32:
  match ni:
    case 0n:
      acc
    case 1n++p:
      gn(p, d, gm(9n, d, 1n+p, acc))


type S is Data:
  S{n: Nat, m: Nat, k: Nat}

@unsafe
def gs(ss: List<&2, S>, +d: Nat, acc: U32) -> U32:
  match ss:
    case Nil{}:
      acc
    case Con{S{+n, +m, +k}, r}:
      gs(r, d, lm(6n, d, n, m, k, acc))


def abc_g(r: TG.ABC) -> TA.G:
  match r:
    case TG.ABC{a, b, c}: TA.G{a, b, c}

# the number of products the grid runs: 9 x 9 shapes x 6 k x 6 (layout, mode), and 7 shapes x 6
def gcount() -> U32:
  ((((9 * 9 : U32) * 36 : U32) + (7 * 6 : U32)) : U32)

@unsafe
def grid() -> U32:
  gs([S{33n, 65n, 70n}, S{64n, 64n, 64n}, S{100n, 31n, 17n}, S{1n, 200n, 64n}, S{7n, 300n, 9n}, S{300n, 7n, 100n}, S{130n, 130n, 131n}], 0n, gn(9n, 0n, 0))

# the six public products on one shape against tensor-array's, differing words of C (A and B checked by the grid)
@unsafe
def pub(+n: Nat, +k: Nat, +m: Nat, which: Nat) -> U32:
  match which:
    case 0n: diff(Nat.mul(n, m), mm_c(n, k, m, TG.matmul(n, k, m, 0n, TA.Mat{arr(Nat.mul(n, k), 5)}, TA.Mat{arr(Nat.mul(k, m), 6)})), mm_c(n, k, m, TA.Mat.matmul(n, k, m, 0n, TA.Mat{arr(Nat.mul(n, k), 5)}, TA.Mat{arr(Nat.mul(k, m), 6)})))
    case 1n: diff(Nat.mul(n, m), mm_c(n, k, m, TG.matmul_acc(n, k, m, 0n, TA.Mat{arr(Nat.mul(n, k), 5)}, TA.Mat{arr(Nat.mul(k, m), 6)}, TA.Mat{arr(Nat.mul(n, m), 7)})), mm_c(n, k, m, TA.Mat.matmul_acc(n, k, m, 0n, TA.Mat{arr(Nat.mul(n, k), 5)}, TA.Mat{arr(Nat.mul(k, m), 6)}, TA.Mat{arr(Nat.mul(n, m), 7)})))
    case 2n: diff(Nat.mul(n, m), mm_nt_c(n, k, m, TG.matmul_nt(n, k, m, 0n, TA.Mat{arr(Nat.mul(n, k), 5)}, TA.Mat{arr(Nat.mul(m, k), 6)})), mm_nt_c(n, k, m, TA.Mat.matmul_nt(n, k, m, 0n, TA.Mat{arr(Nat.mul(n, k), 5)}, TA.Mat{arr(Nat.mul(m, k), 6)})))
    case 3n: diff(Nat.mul(n, m), mm_nt_c(n, k, m, TG.matmul_nt_acc(n, k, m, 0n, TA.Mat{arr(Nat.mul(n, k), 5)}, TA.Mat{arr(Nat.mul(m, k), 6)}, TA.Mat{arr(Nat.mul(n, m), 7)})), mm_nt_c(n, k, m, TA.Mat.matmul_nt_acc(n, k, m, 0n, TA.Mat{arr(Nat.mul(n, k), 5)}, TA.Mat{arr(Nat.mul(m, k), 6)}, TA.Mat{arr(Nat.mul(n, m), 7)})))
    case 4n: diff(Nat.mul(n, m), mm_tn_c(n, k, m, TG.matmul_tn(n, k, m, 0n, TA.Mat{arr(Nat.mul(k, n), 5)}, TA.Mat{arr(Nat.mul(k, m), 6)})), mm_tn_c(n, k, m, TA.Mat.matmul_tn(n, k, m, 0n, TA.Mat{arr(Nat.mul(k, n), 5)}, TA.Mat{arr(Nat.mul(k, m), 6)})))
    case _: diff(Nat.mul(n, m), mm_tn_c(n, k, m, TG.matmul_tn_acc(n, k, m, 0n, TA.Mat{arr(Nat.mul(k, n), 5)}, TA.Mat{arr(Nat.mul(k, m), 6)}, TA.Mat{arr(Nat.mul(n, m), 7)})), mm_tn_c(n, k, m, TA.Mat.matmul_tn_acc(n, k, m, 0n, TA.Mat{arr(Nat.mul(k, n), 5)}, TA.Mat{arr(Nat.mul(k, m), 6)}, TA.Mat{arr(Nat.mul(n, m), 7)})))

def line1(+g: U32) -> IO(Unit):
  IO.print(U32.show(U32.shrn(g, 16n)) ++ " " ++ U32.show(U32.and(g, 65535)) ++ " " ++ U32.show(gcount()))

@unsafe
def test_main() -> IO(Unit):
  do IO<Unit>:
    line1(grid())
    IO.print(U32.show(pub(128n, 100n, 784n, 0n)) ++ " " ++ U32.show(pub(128n, 100n, 784n, 1n)) ++ " " ++ U32.show(pub(128n, 100n, 784n, 2n)) ++ " " ++ U32.show(pub(128n, 100n, 784n, 3n)) ++ " " ++ U32.show(pub(128n, 100n, 784n, 4n)) ++ " " ++ U32.show(pub(128n, 100n, 784n, 5n)))
    IO.print(U32.show(pub(512n, 576n, 1536n, 0n)) ++ " " ++ U32.show(pub(512n, 576n, 1536n, 2n)) ++ " " ++ U32.show(pub(512n, 576n, 1536n, 5n)))

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
