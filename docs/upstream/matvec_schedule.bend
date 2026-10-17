# One matrix · vector product y = x · Wᵀ (W stored r x c), repeated, with bend-ml-tensor-array@0.1.11.0 from BendHub.
#   op 0: TA.Mat.matmul_nt(1, c, r, par)  the general product with one row (par 0: the scalar loop)
#   op 1: TA.Bands.matvec, W in 2^par bands
# Usage: matvec_schedule <op> <r> <c> <reps> <par>; prints the milliseconds of the loop and the sum of the last y.
# W and x are filled before the clock starts; every repetition allocates its own y.
import Base
import bend-ml-tensor-array@0.1.11.0/main.bend as TA

def val(+j: U32, +seed: U32) -> F32:
  ((U32.to_f32(U32.mod(((j * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)) / 500.0 : F32) - 1.0 : F32)

def fill(k: Nat, +j: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, (j + 1 : U32), seed, Array.set(F32, a, j, val(j, seed)))

def arr(+len: Nat, +seed: U32) -> Array<F32>:
  fill(len, 0, seed, Array.new(F32, TA.cap_depth(len), 0.0))

def sum_l(k: Nat, +j: U32, acc: F32, ra: Array<F32> & F32) -> F32:
  match k ra:
    case 0n Tuple{a, x}: acc
    case 1n+p Tuple{a, +x}: sum_l(p, (j + 1 : U32), (acc + x : F32), Array.get(F32, a, (j + 1 : U32)))

def csum(+len: Nat, c: Array<F32>) -> F32:
  sum_l(len, 0, 0.0, Array.get(F32, c, 0))

# ---- op 0: the published product ----

type S0 is Type:
  S0{x: Array<F32>, w: Array<F32>, s: F32}

def f0(+r: Nat, +c: Nat, m: TA.MMulNT<1n, c, r>) -> S0:
  match m:
    case TA.MMulNT{TA.Mat{x}, TA.Mat{w}, TA.Mat{y}}: S0{x, w, csum(r, y)}

def l0(k: Nat, +r: Nat, +c: Nat, +par: Nat, s: S0) -> S0:
  match k s:
    case 0n s: s
    case 1n+p S0{x, w, z}: l0(p, r, c, par, f0(r, c, TA.Mat.matmul_nt(1n, c, r, par, TA.Mat{x}, TA.Mat{w})))

def s0(s: S0) -> F32:
  match s:
    case S0{x, w, z}: z

# ---- op 1: Bands ----

type S1<-r: Nat, -c: Nat> is Type:
  S1{x: Array<F32>, b: TA.Bands<r, c>, s: F32}

def f1(+r: Nat, +c: Nat, x: Array<F32>, v: TA.BV<r, c>) -> S1<r, c>:
  match v:
    case TA.BV{b, TA.Mat{y}}: S1{x, b, csum(r, y)}

def b1(+r: Nat, +c: Nat, p: Array<F32> & Array<F32>, b: TA.Bands<r, c>) -> S1<r, c>:
  match p:
    case Tuple{xa, xb}: f1(r, c, xa, TA.Bands.matvec(r, c, b, TA.Mat{xb}))

def l1(k: Nat, +r: Nat, +c: Nat, s: S1<r, c>) -> S1<r, c>:
  match k s:
    case 0n s: s
    case 1n+p S1{x, b, z}: l1(p, r, c, b1(r, c, Array.clone(F32, x), b))

def s1(-r: Nat, -c: Nat, s: S1<r, c>) -> F32:
  match s:
    case S1{x, b, z}: z

# row i of W as a list (the values arr(r * c, seed) holds at i * c ..)
def rowl(k: Nat, +base: U32, +seed: U32, acc: List<&2, F32>) -> List<&2, F32>:
  match k:
    case 0n: acc
    case 1n++p: rowl(p, base, seed, val((base + U32.from_nat(p) : U32), seed) <> acc)

def bfill(k: Nat, +i: Nat, +r: Nat, +c: Nat, +seed: U32, b: TA.Bands<r, c>) -> TA.Bands<r, c>:
  match k:
    case 0n: b
    case 1n+p: bfill(p, Nat.add(i, 1n), r, c, seed, TA.Bands.fill_at(r, c, Nat.mul(i, c), c, rowl(c, U32.from_nat(Nat.mul(i, c)), seed, Nil{}), b))

# ---- the driver ----

def nat2(m: Maybe<&2, Nat>) -> Nat:
  match m:
    case Some{n}: n
    case None{}: 0n

def nat(s: String) -> Nat:
  nat2(Nat.read(s))

def out(+t0: Nat, +cs: F32) -> IO(Unit):
  do IO<Unit>:
    t1 : Nat <- IO.now()
    IO.print("ms=" ++ Nat.show(Nat.sub(t1, t0)) ++ " checksum=" ++ F32.show(cs))

def run0(+r: Nat, +c: Nat, +k: Nat, +par: Nat) -> IO(Unit):
  do IO<Unit>:
    st : S0 <- IO.pure(S0, S0{arr(c, 1), arr(Nat.mul(r, c), 2), 0.0})
    t0 : Nat <- IO.now()
    out(t0, s0(l0(k, r, c, par, st)))

def run1(+r: Nat, +c: Nat, +k: Nat, +par: Nat) -> IO(Unit):
  do IO<Unit>:
    st : S1<r, c> <- IO.pure(S1<r, c>, S1{arr(c, 1), bfill(r, 0n, r, c, 2, TA.Bands.zeros(par, r, c)), 0.0})
    t0 : Nat <- IO.now()
    out(t0, s1(r, c, l1(k, r, c, st)))

def run(+op: Nat, +r: Nat, +c: Nat, +k: Nat, +par: Nat) -> IO(Unit):
  match op:
    case 0n: run0(r, c, k, par)
    case _: run1(r, c, k, par)

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+o, Con{+r, Con{+c, Con{+k, Con{+p, Nil{}}}}}}}:
      run(nat(o), nat(r), nat(c), nat(k), nat(p))
    case _:
      IO.print("usage: matvec_schedule <op 0|1> <r> <c> <reps> <par>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
