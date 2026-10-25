# bend-ml-tensor-fork against bend-ml-tensor-array, bit for bit.
#
# First line: TF.gemm_any against TA.gemm_any at depth 0, in every layout (nn, nt, tn) and both modes (C = A · B and
# C + A · B), on two grids: n, m = 1..9 with k = 0, 1, 3, 8, 9, 17 at depths 0..4 (tensor-array's tests/parseq.bend
# grid), and (n, m, k) = (33, 65, 70), (64, 64, 64), (100, 31, 17), (1, 200, 64), (7, 300, 9), (300, 7, 100) at depths
# 0..6. It prints the differing words in C, then in A and B (the shared operands must come back as they went in), then
# the number of products: "0 0 14832". The counts sit in 16-bit fields of one U32; only "not 0" matters.
# Second line: the six public products at par 4 against tensor-array's at par 0 on 128 x 100 by 100 x 784 (above
# split_depth's threshold), differing words of each: "0 0 0 0 0 0".
# Third line: matmul, matmul_nt and matmul_tn_acc at par 6 on 512 x 576 by 576 x 1536 (big enough for the packing to
# run in blocks too), then par1..par4 with a kernel over blocks of 0..cnt-1 at depths 0..6 and cnt = 0, 1, 7, 100,
# 1000 against the same kernel in one loop, differing words in every array (the ones only read included):
# "0 0 0 0 0 0 0".
import Base
import bend-ml-tensor-array@0.1.16.0/main.bend as TA
import ../main.bend as TF

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
  g_cmp(nm, na, nb, seed, want, TF.gemm_any(ac, d, n, m, k, arow, sa, sb, bcol, arr(na, seed), arr(nb, (seed + 1 : U32)), c0(ac, nm, seed)))

# layouts and modes of one shape at one depth
def lm(i: Nat, +d: Nat, +n: Nat, +m: Nat, +k: Nat, acc: U32) -> U32:
  match i:
    case 0n:
      acc
    case 1n++p:
      lm(p, d, n, m, k, (acc + one(Nat.is_le(3n, p), d, n, m, k, Nat.mod(p, 3n)) : U32))

def gk(ks: List<&2, Nat>, +d: Nat, +n: Nat, +m: Nat, acc: U32) -> U32:
  match ks:
    case Nil{}:
      acc
    case Con{+k, r}:
      gk(r, d, n, m, lm(6n, d, n, m, k, acc))

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
      gd(p, gn(9n, p, acc))

type S is Data:
  S{n: Nat, m: Nat, k: Nat}

def gs(ss: List<&2, S>, +d: Nat, acc: U32) -> U32:
  match ss:
    case Nil{}:
      acc
    case Con{S{+n, +m, +k}, r}:
      gs(r, d, lm(6n, d, n, m, k, acc))

def gl(di: Nat, acc: U32) -> U32:
  match di:
    case 0n:
      acc
    case 1n++p:
      gl(p, gs([S{33n, 65n, 70n}, S{64n, 64n, 64n}, S{100n, 31n, 17n}, S{1n, 200n, 64n}, S{7n, 300n, 9n}, S{300n, 7n, 100n}], p, acc))

def report(+n: U32) -> String:
  U32.show(U32.shrn(n, 16n)) ++ " " ++ U32.show(U32.and(n, 65535)) ++ " " ++ Nat.show(Nat.add(Nat.mul(5n, Nat.mul(81n, 36n)), Nat.mul(7n, 36n)))
def api(+n: Nat, +k: Nat, +m: Nat, +par: Nat, +seed: U32) -> String:
  +nk = Nat.mul(n, k)
  +km = Nat.mul(k, m)
  +nm = Nat.mul(n, m)
  +s1 = (seed + 1 : U32)
  +s2 = (seed + 2 : U32)
  +nn = diff(nm, mm_c(n, k, m, TF.matmul(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})), mm_c(n, k, m, TA.Mat.matmul(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})))
  +nt = diff(nm, mm_nt_c(n, k, m, TF.matmul_nt(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})), mm_nt_c(n, k, m, TA.Mat.matmul_nt(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})))
  +tn = diff(nm, mm_tn_c(n, k, m, TF.matmul_tn(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})), mm_tn_c(n, k, m, TA.Mat.matmul_tn(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)})))
  +an = diff(nm, mm_c(n, k, m, TF.matmul_acc(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})), mm_c(n, k, m, TA.Mat.matmul_acc(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})))
  +at = diff(nm, mm_nt_c(n, k, m, TF.matmul_nt_acc(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})), mm_nt_c(n, k, m, TA.Mat.matmul_nt_acc(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})))
  +aw = diff(nm, mm_tn_c(n, k, m, TF.matmul_tn_acc(n, k, m, par, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})), mm_tn_c(n, k, m, TA.Mat.matmul_tn_acc(n, k, m, 0n, TA.Mat{arr(nk, seed)}, TA.Mat{arr(km, s1)}, TA.Mat{arr(nm, s2)})))
  U32.show(nn) ++ " " ++ U32.show(nt) ++ " " ++ U32.show(tn) ++ " " ++ U32.show(an) ++ " " ++ U32.show(at) ++ " " ++ U32.show(aw)


def big() -> String:
  +nn = diff(Nat.mul(512n, 1536n), mm_c(512n, 576n, 1536n, TF.matmul(512n, 576n, 1536n, 6n, TA.Mat{arr(Nat.mul(512n, 576n), 21)}, TA.Mat{arr(Nat.mul(576n, 1536n), 22)})), mm_c(512n, 576n, 1536n, TA.Mat.matmul(512n, 576n, 1536n, 0n, TA.Mat{arr(Nat.mul(512n, 576n), 21)}, TA.Mat{arr(Nat.mul(576n, 1536n), 22)})))
  +nt = diff(Nat.mul(512n, 1536n), mm_nt_c(512n, 576n, 1536n, TF.matmul_nt(512n, 576n, 1536n, 6n, TA.Mat{arr(Nat.mul(512n, 576n), 23)}, TA.Mat{arr(Nat.mul(576n, 1536n), 24)})), mm_nt_c(512n, 576n, 1536n, TA.Mat.matmul_nt(512n, 576n, 1536n, 0n, TA.Mat{arr(Nat.mul(512n, 576n), 23)}, TA.Mat{arr(Nat.mul(576n, 1536n), 24)})))
  +tn = diff(Nat.mul(576n, 1536n), mm_tn_c(576n, 512n, 1536n, TF.matmul_tn_acc(576n, 512n, 1536n, 6n, TA.Mat{arr(Nat.mul(512n, 576n), 25)}, TA.Mat{arr(Nat.mul(512n, 1536n), 26)}, TA.Mat{arr(Nat.mul(576n, 1536n), 27)})), mm_tn_c(576n, 512n, 1536n, TA.Mat.matmul_tn_acc(576n, 512n, 1536n, 0n, TA.Mat{arr(Nat.mul(512n, 576n), 25)}, TA.Mat{arr(Nat.mul(512n, 1536n), 26)}, TA.Mat{arr(Nat.mul(576n, 1536n), 27)})))
  U32.show(nn) ++ " " ++ U32.show(nt) ++ " " ++ U32.show(tn)

# ---- par1..par4: x[i] = f(i, ...) on the last array, the others read ----

def k1_set(+i: U32, r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{a, +x}:
      Array.set(F32, a, i, ((x * 2.0 : F32) + U32.to_f32(i) : F32))

def k1(k: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n:
      a
    case 1n+q:
      k1(q, (i + 1 : U32), k1_set(i, Array.get(F32, a, i)))

def l1(+e: Unit, +i0: U32, +cnt: Nat, a: Array<F32>) -> Array<F32>:
  k1(cnt, i0, a)

def k2_add(+i: U32, +v: F32, +s: F32, x: Array<F32>, r: Array<F32> & F32) -> TF.B2<F32, F32>:
  match r:
    case Tuple{y, +w}:
      TF.B2{x, Array.set(F32, y, i, ((v * s : F32) + w : F32))}

# y[i] = x[i] * s + y[i], for the arrays (x, y)
def k2_set(+i: U32, +s: F32, y: Array<F32>, r: Array<F32> & F32) -> TF.B2<F32, F32>:
  match r:
    case Tuple{x, +v}:
      k2_add(i, v, s, x, Array.get(F32, y, i))

def k2(k: Nat, +i: U32, +s: F32, r: TF.B2<F32, F32>) -> TF.B2<F32, F32>:
  match k r:
    case 0n r:
      r
    case 1n+q TF.B2{x, y}:
      k2(q, (i + 1 : U32), s, k2_set(i, s, y, Array.get(F32, x, i)))

def l2(+e: F32, +i0: U32, +cnt: Nat, r: TF.B2<F32, F32>) -> TF.B2<F32, F32>:
  k2(cnt, i0, e, r)

def b3_of(z: Array<F32>, r: TF.B2<F32, F32>) -> TF.B3<F32, F32, F32>:
  match r:
    case TF.B2{x, y}:
      TF.B3{z, x, y}

# (z, x, y): y[i] = x[i] * z[i] + y[i]
def k3_s(+i: U32, x: Array<F32>, y: Array<F32>, r: Array<F32> & F32) -> TF.B3<F32, F32, F32>:
  match r:
    case Tuple{z, +s}:
      b3_of(z, k2_set(i, s, y, Array.get(F32, x, i)))

def k3(k: Nat, +i: U32, r: TF.B3<F32, F32, F32>) -> TF.B3<F32, F32, F32>:
  match k r:
    case 0n r:
      r
    case 1n+q TF.B3{z, x, y}:
      k3(q, (i + 1 : U32), k3_s(i, x, y, Array.get(F32, z, i)))

def l3(+e: Unit, +i0: U32, +cnt: Nat, r: TF.B3<F32, F32, F32>) -> TF.B3<F32, F32, F32>:
  k3(cnt, i0, r)

def b4_of(u: Array<F32>, r: TF.B3<F32, F32, F32>) -> TF.B4<F32, F32, F32, F32>:
  match r:
    case TF.B3{z, x, y}:
      TF.B4{u, z, x, y}

# (u, z, x, y): y[i] = x[i] * z[i] + y[i], then u[i] read and added: y[i] += u[i] (as one more k3 over (u, ...))
def k4_s(+i: U32, z: Array<F32>, x: Array<F32>, y: Array<F32>, r: Array<F32> & F32) -> TF.B4<F32, F32, F32, F32>:
  match r:
    case Tuple{u, +s}:
      b4_of(u, k3_s(i, x, y, Array.get(F32, z, i)))

def k4(k: Nat, +i: U32, r: TF.B4<F32, F32, F32, F32>) -> TF.B4<F32, F32, F32, F32>:
  match k r:
    case 0n r:
      r
    case 1n+q TF.B4{u, z, x, y}:
      k4(q, (i + 1 : U32), k4_s(i, z, x, y, Array.get(F32, u, i)))

def l4(+e: Unit, +i0: U32, +cnt: Nat, r: TF.B4<F32, F32, F32, F32>) -> TF.B4<F32, F32, F32, F32>:
  k4(cnt, i0, r)

def c2(+cnt: Nat, +s: U32, a: TF.B2<F32, F32>, b: TF.B2<F32, F32>) -> U32:
  match a b:
    case TF.B2{x1, y1} TF.B2{x2, y2}:
      (diff(cnt, x1, x2) + diff(cnt, y1, y2) : U32)

def c3(+cnt: Nat, a: TF.B3<F32, F32, F32>, b: TF.B3<F32, F32, F32>) -> U32:
  match a b:
    case TF.B3{z1, x1, y1} TF.B3{z2, x2, y2}:
      ((diff(cnt, z1, z2) + diff(cnt, x1, x2) : U32) + diff(cnt, y1, y2) : U32)

def c4(+cnt: Nat, a: TF.B4<F32, F32, F32, F32>, b: TF.B4<F32, F32, F32, F32>) -> U32:
  match a b:
    case TF.B4{u1, z1, x1, y1} TF.B4{u2, z2, x2, y2}:
      (((diff(cnt, u1, u2) + diff(cnt, z1, z2) : U32) + diff(cnt, x1, x2) : U32) + diff(cnt, y1, y2) : U32)

# the four checks at one depth and one count
def blk(+d: Nat, +cnt: Nat, +s: U32) -> U32:
  +p1 = diff(cnt, TF.par1(~Unit, ~F32, ~l1, Unit{}, d, cnt, arr(cnt, s)), k1(cnt, 0, arr(cnt, s)))
  +p2 = c2(cnt, s, TF.par2(~F32, ~F32, ~F32, ~l2, 0.5, d, cnt, arr(cnt, s), arr(cnt, (s + 1 : U32))), k2(cnt, 0, 0.5, TF.B2{arr(cnt, s), arr(cnt, (s + 1 : U32))}))
  +p3 = c3(cnt, TF.par3(~Unit, ~F32, ~F32, ~F32, ~l3, Unit{}, d, cnt, arr(cnt, s), arr(cnt, (s + 1 : U32)), arr(cnt, (s + 2 : U32))), k3(cnt, 0, TF.B3{arr(cnt, s), arr(cnt, (s + 1 : U32)), arr(cnt, (s + 2 : U32))}))
  +p4 = c4(cnt, TF.par4(~Unit, ~F32, ~F32, ~F32, ~F32, ~l4, Unit{}, d, cnt, arr(cnt, s), arr(cnt, (s + 1 : U32)), arr(cnt, (s + 2 : U32)), arr(cnt, (s + 3 : U32))), k4(cnt, 0, TF.B4{arr(cnt, s), arr(cnt, (s + 1 : U32)), arr(cnt, (s + 2 : U32)), arr(cnt, (s + 3 : U32))}))
  (((p1 * 16777216 : U32) + (p2 * 65536 : U32) : U32) + ((p3 * 256 : U32) + p4 : U32) : U32)

def bcnt(cs: List<&2, Nat>, +d: Nat, acc: U32) -> U32:
  match cs:
    case Nil{}:
      acc
    case Con{+c, r}:
      bcnt(r, d, (acc + blk(d, c, (U32.from_nat(c) + 5 : U32)) : U32))

def bdep(di: Nat, acc: U32) -> U32:
  match di:
    case 0n:
      acc
    case 1n++p:
      bdep(p, bcnt([0n, 1n, 7n, 100n, 1000n], p, acc))

def blocks() -> String:
  +r = bdep(7n, 0)
  U32.show(U32.shrn(r, 24n)) ++ " " ++ U32.show(U32.and(U32.shrn(r, 16n), 255)) ++ " " ++ U32.show(U32.and(U32.shrn(r, 8n), 255)) ++ " " ++ U32.show(U32.and(r, 255))

def test_main() -> IO(Unit):
  do IO<Unit>:
    IO.print(report(gl(7n, gd(5n, 0))))
    IO.print(api(128n, 100n, 784n, 4n, 11))
    IO.print(big() ++ " " ++ blocks())

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
