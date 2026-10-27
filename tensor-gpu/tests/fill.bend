# fill_uniform, fill_gauss and gauss_mat against their cell functions run once on the host (rng_uniform(s, i),
# rng_gauss(s, i)), bit for bit, on n = 1, 2, 7, 1000, 16389 and 70001 cells (more than the 2^14 leaves, odd counts)
# and on a 256 x 3072 Mat; the offset fills (fill_uniform_at, fill_gauss_at) against rng_uniform(s, i0 + k) and
# rng_gauss(s, i0 + k), for odd and even i0 and n (n = 1 with i0 odd, an i0 near 2^32 whose indexes wrap); slices of
# fill_uniform and fill_gauss against the same cells of a full fill, and a 16-row gauss_mat_at against those rows of
# the 256 x 3072 gauss_mat (with a uniform_mat_at at an odd i0 against its cell function); then, on the host only, how far rng_ln, rng_sin and rng_cos are from libm's F32.log, sin and
# cos, and Box-Muller with them from Box-Muller with libm's; then the moments of 2^20 Gaussians and uniforms.
# Prints: "0 0" six times and "0" (differing words: uniform, Gaussian; the Mat), "0 0" twelve times (the same for the
# offset fills, the slices and the Mat rows), the words of mlbend-generative's
# uniform formula that differ (0), the largest ulp distances of ln, sin and cos, the largest |difference| from libm's
# Box-Muller, the moments, and "ok" when every moment is inside its bound. The output is the same on every lane. The
# test sees a lane that differs: with F32.sin and F32.cos in rng_sc_x instead of rng_sin and rng_cos, the GPU run gives
# 1 to 6,410 differing Gaussians a line and 69,913 in the Mat (2026-10-27), the CPU threads 0.
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ../main.bend as TG

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def cmp(k: Nat, +i: U32, acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmp(p, (i + 1 : U32), (acc + Bool.pick(U32, U32.is_eq(bits(x), bits(y)), 0, 1) : U32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def diff(+cnt: Nat, x: Array<F32>, y: Array<F32>) -> U32:
  cmp(cnt, 0, 0, Array.get(F32, x, 0), Array.get(F32, y, 0))

# cells 0 .. cnt - 1 of x against cells j0 .. j0 + cnt - 1 of y
def cmpo(k: Nat, +i: U32, +j: U32, acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmpo(p, (i + 1 : U32), (j + 1 : U32), (acc + Bool.pick(U32, U32.is_eq(bits(x), bits(y)), 0, 1) : U32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (j + 1 : U32)))

def diffo(+cnt: Nat, x: Array<F32>, +j0: U32, y: Array<F32>) -> U32:
  cmpo(cnt, 0, j0, 0, Array.get(F32, x, 0), Array.get(F32, y, j0))

# the host references: the cell functions once over every cell, in order (cell k from index i0 + k, mod 2^32)
def hu(k: Nat, +i: U32, +i0: U32, +s: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: hu(p, (i + 1 : U32), i0, s, Array.set(F32, a, i, TG.rng_uniform(s, (i0 + i : U32))))

def hg(k: Nat, +i: U32, +i0: U32, +s: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: hg(p, (i + 1 : U32), i0, s, Array.set(F32, a, i, TG.rng_gauss(s, (i0 + i : U32))))

def ref_u_at(+s: U32, +i0: U32, +n: Nat) -> Array<F32>:
  hu(n, 0, i0, s, Array.new(F32, TA.cap_depth(n), 0.0))

def ref_g_at(+s: U32, +i0: U32, +n: Nat) -> Array<F32>:
  hg(n, 0, i0, s, Array.new(F32, TA.cap_depth(n), 0.0))

def ref_u(+s: U32, +n: Nat) -> Array<F32>:
  ref_u_at(s, 0, n)

def ref_g(+s: U32, +n: Nat) -> Array<F32>:
  ref_g_at(s, 0, n)

@unsafe
def line(+s: U32, +n: Nat) -> String:
  U32.show(diff(n, TG.fill_uniform(s, n, TA.cap_depth(n)), ref_u(s, n))) ++ " " ++ U32.show(diff(n, TG.fill_gauss(s, n, TA.cap_depth(n)), ref_g(s, n)))

def mat_d(m: TA.Mat<256n, 3072n>) -> Array<F32>:
  match m:
    case TA.Mat{d}: d

@unsafe
def mat_line(+s: U32) -> String:
  U32.show(diff(786432n, mat_d(TG.gauss_mat(s, 256n, 3072n)), ref_g(s, 786432n)))

# the offset fills against their cell functions at i0 + k
@unsafe
def line_at(+s: U32, +i0: U32, +n: Nat) -> String:
  U32.show(diff(n, TG.fill_uniform_at(s, i0, n, TA.cap_depth(n)), ref_u_at(s, i0, n))) ++ " " ++ U32.show(diff(n, TG.fill_gauss_at(s, i0, n, TA.cap_depth(n)), ref_g_at(s, i0, n)))

# a slice of n cells from i0 against cells i0 .. i0 + n - 1 of a full fill of nf cells
@unsafe
def slice_line(+s: U32, +nf: Nat, +i0: U32, +n: Nat) -> String:
  U32.show(diffo(n, TG.fill_uniform_at(s, i0, n, TA.cap_depth(n)), i0, TG.fill_uniform(s, nf, TA.cap_depth(nf)))) ++ " " ++ U32.show(diffo(n, TG.fill_gauss_at(s, i0, n, TA.cap_depth(n)), i0, TG.fill_gauss(s, nf, TA.cap_depth(nf))))

def mat16_d(m: TA.Mat<16n, 3072n>) -> Array<F32>:
  match m:
    case TA.Mat{d}: d

def mat7_d(m: TA.Mat<7n, 3n>) -> Array<F32>:
  match m:
    case TA.Mat{d}: d

# a 7 x 3 uniform_mat_at from the odd index 12345 against its cell functions; rows 37 .. 52 of the 256 x 3072
# gauss_mat (from index 37 * 3072 = 113664) against a 16-row gauss_mat_at
@unsafe
def mat_at_line(+s: U32) -> String:
  U32.show(diff(21n, mat7_d(TG.uniform_mat_at(s, 12345, 7n, 3n)), ref_u_at(s, 12345, 21n))) ++ " " ++ U32.show(diffo(49152n, mat16_d(TG.gauss_mat_at(s, 113664, 16n, 3072n)), 113664, mat_d(TG.gauss_mat(s, 256n, 3072n))))

# mlbend-generative 0.1.0.0's uniform, as it writes it (the 24 bits plus one through Nat)
def gen_uniform(+s: U32, +i: U32) -> F32:
  +m = (U32.and(TG.rng_hash(s, i), 16777215) + 1 : U32)
  (F32.from_nat(U32.to_nat(m)) / 16777216.0 : F32)

def gen_u(k: Nat, +i: U32, +s: U32, acc: U32) -> U32:
  match k:
    case 0n: acc
    case 1n+p: gen_u(p, (i + 1 : U32), s, (acc + Bool.pick(U32, U32.is_eq(bits(TG.rng_uniform(s, i)), bits(gen_uniform(s, i))), 0, 1) : U32))

# the distance in ulp of two F32 of one sign
def ulp(+a: F32, +b: F32) -> U32:
  +x = bits(a)
  +y = bits(b)
  Bool.pick(U32, U32.is_lt(x, y), (y - x : U32), (x - y : U32))

# ln of k / 2^24 for k = 1, 1 + st, 1 + 2 st, .. (cnt of them): the largest distance to F32.log
def ln_go(cnt: Nat, +k: U32, +st: U32, acc: U32) -> U32:
  match cnt:
    case 0n: acc
    case 1n+p:
      +u = (U32.to_f32(k) / 16777216.0 : F32)
      ln_go(p, (k + st : U32), st, U32.max(acc, ulp(TG.rng_ln(u), F32.log(u))))

# sin and cos of w * (pi/2) / 2^22 for w = 0 .. 2^21, the arguments the polynomials see: the largest distances
type SCD is Data:
  SCD{s: U32, c: U32}

def sc_go(cnt: Nat, +w: U32, +ds: U32, +dc: U32) -> SCD:
  match cnt:
    case 0n: SCD{ds, dc}
    case 1n+p:
      +x = (U32.to_f32(w) * 0.00000037450702 : F32)
      sc_go(p, (w + 1 : U32), U32.max(ds, ulp(TG.rng_sin(x), F32.sin(x))), U32.max(dc, ulp(TG.rng_cos(x), F32.cos(x))))

def show_scd(d: SCD) -> String:
  match d:
    case SCD{+s, +c}: U32.show(s) ++ " " ++ U32.show(c)

# Box-Muller with libm's functions, as mlbend-generative's gauss_at writes it
def lib_gauss(+s: U32, +i: U32) -> F32:
  +pr = U32.and(i, 4294967294)
  +r = F32.sqrt((0.0 - (2.0 * F32.log(TG.rng_uniform(s, pr)) : F32) : F32))
  +ang = (6.2831855 * TG.rng_uniform(s, (pr + 1 : U32)) : F32)
  Bool.pick(F32, U32.is_eq(U32.and(i, 1), 0), (r * F32.cos(ang) : F32), (r * F32.sin(ang) : F32))

def bm_go(cnt: Nat, +i: U32, +s: U32, +acc: F32) -> F32:
  match cnt:
    case 0n: acc
    case 1n+p: bm_go(p, (i + 1 : U32), s, F32.max(acc, F32.abs((TG.rng_gauss(s, i) - lib_gauss(s, i) : F32))))

# the moments of n numbers in chunks of 1024 (each chunk summed alone, then the chunks): sum, sum of squares, and how
# many lie inside 1, 2 and 3 of 0
type Mo is Data:
  Mo{s1: F32, s2: F32, c1: U32, c2: U32, c3: U32}

def in_b(+x: F32, +b: F32) -> U32:
  Bool.pick(U32, F32.is_lt(F32.abs(x), b), 1, 0)

def mo_chunk(k: Nat, +s1: F32, +s2: F32, +c1: U32, +c2: U32, +c3: U32, r: Array<F32> & F32, +i: U32) -> Array<F32> & Mo:
  match k r:
    case 0n Tuple{a, x}: Tuple{a, Mo{s1, s2, c1, c2, c3}}
    case 1n+p Tuple{a, +x}:
      mo_chunk(p, (s1 + x : F32), (s2 + (x * x : F32) : F32), (c1 + in_b(x, 1.0) : U32), (c2 + in_b(x, 2.0) : U32), (c3 + in_b(x, 3.0) : U32), Array.get(F32, a, (i + 1 : U32)), (i + 1 : U32))

def mo_add(t: Mo, r: Array<F32> & Mo) -> Array<F32> & Mo:
  match t r:
    case Mo{+a1, +a2, +b1, +b2, +b3} Tuple{a, Mo{+x1, +x2, +y1, +y2, +y3}}:
      Tuple{a, Mo{(a1 + x1 : F32), (a2 + x2 : F32), (b1 + y1 : U32), (b2 + y2 : U32), (b3 + y3 : U32)}}

def mo_go(chunks: Nat, +i: U32, r: Array<F32> & Mo) -> Mo:
  match chunks r:
    case 0n Tuple{a, t}: t
    case 1n+p Tuple{a, t}: mo_go(p, (i + 1024 : U32), mo_add(t, mo_chunk(1024n, 0.0, 0.0, 0, 0, 0, Array.get(F32, a, i), i)))

# 2^20 numbers: the mean, the variance and the three fractions, then whether each is inside its bound (5 standard
# errors: the mean 0 or 1/2 within 0.005, the variance 1 or 1/12 within 0.007 or 0.0007, the fractions 0.6827, 0.9545,
# 0.9973 within 0.0023, 0.0010 and 0.00026; for the uniforms the fractions are all 1)
def near(+x: F32, +want: F32, +tol: F32) -> Bool:
  F32.is_lt(F32.abs((x - want : F32)), tol)

def mo_show(+gauss: Bool, m: Mo) -> String:
  match m:
    case Mo{+s1, +s2, +c1, +c2, +c3}:
      +mean = (s1 / 1048576.0 : F32)
      +var = ((s2 / 1048576.0 : F32) - (mean * mean : F32) : F32)
      +f1 = (U32.to_f32(c1) / 1048576.0 : F32)
      +f2 = (U32.to_f32(c2) / 1048576.0 : F32)
      +f3 = (U32.to_f32(c3) / 1048576.0 : F32)
      +ok = Bool.pick(Bool, gauss, Bool.and(Bool.and(near(mean, 0.0, 0.005), near(var, 1.0, 0.007)), Bool.and(Bool.and(near(f1, 0.6827, 0.0023), near(f2, 0.9545, 0.0010)), near(f3, 0.9973, 0.00026))), Bool.and(Bool.and(near(mean, 0.5, 0.005), near(var, 0.083333336, 0.0007)), U32.is_eq(c1, 1048576)))
      F32.show(mean) ++ " " ++ F32.show(var) ++ " " ++ F32.show(f1) ++ " " ++ F32.show(f2) ++ " " ++ F32.show(f3) ++ " " ++ Bool.pick(String, ok, "ok", "OUT")

@unsafe
def stats(+gauss: Bool, +s: U32) -> String:
  mo_show(gauss, mo_go(1024n, 0, Tuple{Bool.pick(Array<F32>, gauss, TG.fill_gauss(s, 1048576n, 20n), TG.fill_uniform(s, 1048576n, 20n)), Mo{0.0, 0.0, 0, 0, 0}}))

@unsafe
def test_main() -> IO(Unit):
  do IO<Unit>:
    a : Unit <- IO.print(line(1, 1n))
    b : Unit <- IO.print(line(2, 2n))
    c : Unit <- IO.print(line(3, 7n))
    d : Unit <- IO.print(line(4, 1000n))
    e : Unit <- IO.print(line(5, 16389n))
    f : Unit <- IO.print(line(4294967295, 70001n))
    g : Unit <- IO.print(mat_line(7))
    o1 : Unit <- IO.print(line_at(1, 1, 1n))
    o2 : Unit <- IO.print(line_at(2, 6, 1n))
    o3 : Unit <- IO.print(line_at(3, 1, 2n))
    o4 : Unit <- IO.print(line_at(4, 5, 7n))
    o5 : Unit <- IO.print(line_at(5, 8, 1000n))
    o6 : Unit <- IO.print(line_at(8, 2, 999n))
    o7 : Unit <- IO.print(line_at(6, 1001, 16389n))
    o8 : Unit <- IO.print(line_at(4294967295, 4294967291, 70001n))
    s1 : Unit <- IO.print(slice_line(9, 16389n, 12345, 777n))
    s2 : Unit <- IO.print(slice_line(9, 16389n, 4001, 1n))
    s3 : Unit <- IO.print(slice_line(9, 16389n, 1, 16388n))
    m1 : Unit <- IO.print(mat_at_line(7))
    h : Unit <- IO.print(U32.show(gen_u(1048576n, 0, 9, 0)))
    i : Unit <- IO.print(U32.show(ln_go(1048576n, 1, 16, 0)))
    j : Unit <- IO.print(show_scd(sc_go(2097153n, 0, 0, 0)))
    k : Unit <- IO.print(F32.show(bm_go(262144n, 0, 11, 0.0)))
    l : Unit <- IO.print(stats(True{}, 1))
    IO.print(stats(False{}, 1))

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
