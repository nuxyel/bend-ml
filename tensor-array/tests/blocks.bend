# Mat.matmul_precise (0.1.14.0): every cell of a product with k > 1024 equals its dot summed as blocks of 1024 (the last
# one 1..1024), each block computed by dotr (the four partial sums, unchanged since 0.1.8.0) and the block totals added
# one after another from the first. The reference below builds each cell that way from dotr itself, independently of
# matmul_precise's slices; they are compared on one row (the scalar path) and on several (the cells of 8), at par 0 and
# 4, for k around one, two and many blocks (up to 49,152, SmolLM2's vocabulary); for k <= 1024 matmul_precise must be
# Mat.matmul's bits. Prints the differing words over every product, then the number of products: "0 22".
import Base
import ../main.bend as TA

def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  Bool.pick(F32, U32.is_eq(h, 7), (0.0 - 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 13), 0.00000000000000000000000000000000000001, (((U32.to_f32(h) / 250.0 : F32) - 2.0 : F32))))

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

# ---- the reference: a cell as its blocks, each from dotr ----

def r_then(r: TA.R, k: @+x: F32 -> Array<F32> -> Array<F32> -> TA.R) -> TA.R:
  match r:
    case TA.R{a, b, +x}: k(x, a, b)

def comb(first: Bool, +acc: F32, +x: F32) -> F32:
  match first:
    case True{}: x
    case False{}: (acc + x : F32)

# the length of block bi of a dot of k products
def blen(+bi: U32, +k: U32) -> Nat:
  U32.to_nat(Bool.pick(U32, U32.is_lt((k - (bi * 1024 : U32) : U32), 1024), (k - (bi * 1024 : U32) : U32), 1024))

def rblocks(nb: Nat, +bi: U32, +k: U32, +ia0: U32, +ib0: U32, +sa: U32, +sb: U32, +first: Bool, +acc: F32, a: Array<F32>, b: Array<F32>) -> TA.R:
  match nb:
    case 0n: TA.R{a, b, acc}
    case 1n++q:
      +ia = (ia0 + ((bi * 1024 : U32) * sa : U32) : U32)
      +ib = (ib0 + ((bi * 1024 : U32) * sb : U32) : U32)
      r_then(TA.dotr(blen(bi, k), ia, ib, sa, sb, 0.0, 0.0, 0.0, 0.0, Array.get(F32, a, ia), Array.get(F32, b, ib)), +x => a2 => b2 => rblocks(q, (bi + 1 : U32), k, ia0, ib0, sa, sb, False{}, comb(first, acc, x), a2, b2))

type RC is Type:
  RC{a: Array<F32>, b: Array<F32>, c: Array<F32>}

def r_put(+ac: Bool, +idx: U32, c: Array<F32>, r: TA.R) -> RC:
  match r:
    case TA.R{a, b, +x}: RC{a, b, TA.cwr(ac, idx, x, c)}

type Dims is Data:
  Dims{nb: Nat, k: U32, arow: U32, sa: U32, sb: U32, bcol: U32, m: U32}

def mk_dims(+k: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +m: U32) -> Dims:
  Dims{U32.to_nat(U32.div((k + 1023 : U32), 1024)), k, arow, sa, sb, bcol, m}

def rcols(mleft: Nat, +ac: Bool, +i: U32, +j: U32, +d: Dims, r: RC) -> RC:
  match mleft d r:
    case 0n _ r: r
    case 1n+p Dims{+nb, +k, +arow, +sa, +sb, +bcol, +m} RC{a, b, c}:
      rcols(p, ac, i, (j + 1 : U32), d, r_put(ac, ((i * m : U32) + j : U32), c, rblocks(nb, 0, k, (i * arow : U32), (j * bcol : U32), sa, sb, True{}, 0.0, a, b)))

def rrows(nleft: Nat, +mn: Nat, +ac: Bool, +i: U32, +d: Dims, r: RC) -> RC:
  match nleft:
    case 0n: r
    case 1n+p: rrows(p, mn, ac, (i + 1 : U32), d, rcols(mn, ac, i, 0, d, r))

def rc_c(r: RC) -> Array<F32>:
  match r:
    case RC{a, b, c}: c

# ---- one shape against matmul_precise at par p ----

def mmc(-n: Nat, -k: Nat, -m: Nat, r: TA.MMul<n, k, m>) -> Array<F32>:
  match r:
    case TA.MMul{a, b, TA.Mat{c}}: c

def one(+p: Nat, +n: Nat, +m: Nat, +k: Nat, +seed: U32) -> U32:
  +ku = U32.from_nat(k)
  +nm = Nat.mul(n, m)
  +dims = mk_dims(ku, ku, 1, U32.from_nat(m), 1, U32.from_nat(m))
  diff(nm, mmc(n, k, m, TA.Mat.matmul_precise(n, k, m, p, TA.Mat{arr(Nat.mul(n, k), seed)}, TA.Mat{arr(Nat.mul(k, m), (seed + 1 : U32))})), rc_c(rrows(n, m, False{}, 0, dims, RC{arr(Nat.mul(n, k), seed), arr(Nat.mul(k, m), (seed + 1 : U32)), Array.new(F32, TA.cap_depth(nm), 0.0)})))

# k <= 1024: matmul_precise is Mat.matmul
def short(+p: Nat, +n: Nat, +m: Nat, +k: Nat, +seed: U32) -> U32:
  diff(Nat.mul(n, m), mmc(n, k, m, TA.Mat.matmul_precise(n, k, m, p, TA.Mat{arr(Nat.mul(n, k), seed)}, TA.Mat{arr(Nat.mul(k, m), (seed + 1 : U32))})), mmc(n, k, m, TA.Mat.matmul(n, k, m, p, TA.Mat{arr(Nat.mul(n, k), seed)}, TA.Mat{arr(Nat.mul(k, m), (seed + 1 : U32))})))

type S is Data:
  S{n: Nat, m: Nat, k: Nat}

def gs(ss: List<&2, S>, +p: Nat, acc: U32) -> U32:
  match ss:
    case Nil{}: acc
    case Con{S{+n, +m, +k}, r}: gs(r, p, (acc + one(p, n, m, k, 7) : U32))

def shapes() -> List<&2, S>:
  [S{1n, 5n, 3001n}, S{1n, 3n, 1025n}, S{3n, 4n, 1023n}, S{3n, 4n, 1024n}, S{3n, 4n, 1025n}, S{5n, 7n, 2048n}, S{5n, 7n, 2049n}, S{9n, 2n, 8200n}, S{2n, 3n, 49152n}]

def test_main() -> IO(Unit):
  IO.print(U32.show((((gs(shapes(), 0n, 0) + gs(shapes(), 4n, 0) : U32) + ((short(0n, 3n, 4n, 1023n, 9) + short(4n, 5n, 7n, 1024n, 9) : U32) + (short(0n, 1n, 9n, 700n, 9) + short(4n, 64n, 32n, 1000n, 9) : U32) : U32)) : U32)) ++ " " ++ U32.show(((9 * 2 : U32) + 4 : U32)))

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
