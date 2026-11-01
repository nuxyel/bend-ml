import TAPATH as TA

# The per-element cost of Array operations on one thread (NOTES.md, 2026-11-01), against bench/array_cost.c.
# bench/array_cost.py substitutes OPNAME, DEPTH (the array holds 2^DEPTH numbers) and REPS, and times REPS = 0
# against REPS = N: the difference divided by N * 2^DEPTH is the cost of one element.
#
#   get     sum of every cell of an Array<U32> through Array.get (a U32 add, so the read is what is timed)
#   fold    the same over an Array<F32> with one F32 accumulator (tensor-array's loops: read_l, cea_max, ...)
#   set     Array.set of every cell of an Array<U32>
#   new     Array.new(F32, DEPTH, v), then one read
#   newrt   new with a depth computed at run time, as tensor-array's Mat.zeros computes it
#   tolist  tensor-array's Mat.to_list (read_l), then the list summed with TA.suml
#   round   Mat.to_list, then Mat.fill_at (fill_list, what Mat.from_list runs) back into the same array
#   btolist Base's Array.to_list (a match per node) of a fresh Array.new, then the list summed

# get: n + 1 reads from index i, summed into acc
def gsum(n: Nat, +i: U32, acc: U32, r: Array<U32> & U32) -> Array<U32> & U32:
  match n r:
    case 0n Tuple{a, +v}:
      (a, (acc + v : U32))
    case 1n+p Tuple{a, +v}:
      gsum(p, (i + 1 : U32), (acc + v : U32), Array.get(U32, a, (i + 1 : U32)))

def greps(k: Nat, +nm1: Nat, r: Array<U32> & U32) -> U32:
  match k r:
    case 0n Tuple{a, +s}:
      s
    case 1n+p Tuple{a, +s}:
      greps(p, nm1, gsum(nm1, 0, s, Array.get(U32, a, 0)))

# set: cell i becomes i + x for n cells from i
def sloop(n: Nat, +i: U32, +x: U32, a: Array<U32>) -> Array<U32>:
  match n:
    case 0n:
      a
    case 1n+p:
      sloop(p, (i + 1 : U32), x, Array.set(U32, a, i, (i + x : U32)))

def sreps(k: Nat, +n: Nat, +x: U32, a: Array<U32>) -> Array<U32>:
  match k:
    case 0n:
      a
    case 1n+p:
      sreps(p, n, (x + 1 : U32), sloop(n, 0, x, a))

def ufin2(r: Array<U32> & U32) -> U32:
  match r:
    case Tuple{a, +v}:
      v

def ufin(+nm1: U32, r: Array<U32> & U32) -> U32:
  match r:
    case Tuple{a, +v}:
      U32.add(v, ufin2(Array.get(U32, a, nm1)))

# fold: n + 1 reads of an Array<F32> into one F32 sum
def fsum(n: Nat, +i: U32, acc: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match n r:
    case 0n Tuple{a, +v}:
      (a, (acc + v : F32))
    case 1n+p Tuple{a, +v}:
      fsum(p, (i + 1 : U32), (acc + v : F32), Array.get(F32, a, (i + 1 : U32)))

def freps(k: Nat, +nm1: Nat, r: Array<F32> & F32) -> F32:
  match k r:
    case 0n Tuple{a, +s}:
      s
    case 1n+p Tuple{a, +s}:
      freps(p, nm1, fsum(nm1, 0, (s * 0.5 : F32), Array.get(F32, a, 0)))

def ffill(n: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      a
    case 1n+p:
      ffill(p, (i + 1 : U32), Array.set(F32, a, i, U32.to_f32(U32.and(i, 1023))))

# new: a fresh array per rep, filled with the running value
def ffin(r: Array<F32> & F32) -> F32:
  match r:
    case Tuple{a, +v}:
      v

def nreps(k: Nat, +d: Nat, +acc: F32) -> F32:
  match k:
    case 0n:
      acc
    case 1n+p:
      nreps(p, d, (ffin(Array.get(F32, Array.new(F32, d, acc), 7)) + 1.0 : F32))

# tolist: Mat.to_list's list summed, then read again (the first list is read before the reps, as in round)
def treps(k: Nat, +n: Nat, +s: F32, ml: TA.ML<1n, n>) -> F32:
  match k ml:
    case 0n TA.ML{m, l}:
      s
    case 1n+p TA.ML{m, l}:
      treps(p, n, TA.suml(l, (s * 0.5 : F32)), TA.Mat.to_list(1n, n, m))

# round: Mat.to_list's list written back with Mat.fill_at (fill_list, what Mat.from_list runs)
def rreps(k: Nat, +n: Nat, ml: TA.ML<1n, n>) -> TA.ML<1n, n>:
  match k ml:
    case 0n TA.ML{m, l}:
      TA.ML{m, l}
    case 1n+p TA.ML{m, l}:
      rreps(p, n, TA.Mat.to_list(1n, n, TA.Mat.fill_at(1n, n, 0, l, m)))

def rfin(-n: Nat, ml: TA.ML<1n, n>) -> F32:
  match ml:
    case TA.ML{m, l}:
      TA.suml(l, 0.0)

# btolist: Base's Array.to_list of a fresh array, summed
def lsum1(xs: List<F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      lsum1(t, (acc + h : F32))

def breps(k: Nat, +d: Nat, +acc: F32) -> F32:
  match k:
    case 0n:
      acc
    case 1n+p:
      breps(p, d, (lsum1(Array.to_list(~F32, Array.new(F32, d, acc)), 0.0) * 0.000001 : F32))

def op_get(+k: Nat, +d: Nat) -> U32:
  +n = TA.pow2(d)
  greps(k, Nat.sub(n, 1n), (sloop(n, 0, 3, Array.new(U32, d, 0)), 0))

def op_set(+k: Nat, +d: Nat) -> U32:
  +n = TA.pow2(d)
  ufin(4294967295, Array.get(U32, sreps(k, n, 5, sloop(n, 0, 3, Array.new(U32, d, 0))), 0))

def op_fold(+k: Nat, +d: Nat) -> F32:
  +n = TA.pow2(d)
  freps(k, Nat.sub(n, 1n), (ffill(n, 0, Array.new(F32, d, 0.0)), 0.0))

def op_new(+k: Nat, +d: Nat) -> F32:
  nreps(k, d, 1.0)

# newrt: new with the depth computed at run time (TA.cap_depth, as Mat.zeros and Mat.from_list do), not a literal
def op_newrt(+k: Nat, +d: Nat) -> F32:
  nreps(k, TA.cap_depth(TA.pow2(d)), 1.0)

def op_tolist(+k: Nat, +d: Nat) -> F32:
  +n = TA.pow2(d)
  treps(k, n, 0.0, TA.Mat.to_list(1n, n, TA.Mat{ffill(n, 0, Array.new(F32, d, 0.0))}))

def op_round(+k: Nat, +d: Nat) -> F32:
  +n = TA.pow2(d)
  rfin(n, rreps(k, n, TA.Mat.to_list(1n, n, TA.Mat{ffill(n, 0, Array.new(F32, d, 0.0))})))

def op_btolist(+k: Nat, +d: Nat) -> F32:
  breps(k, d, 1.0)

def main() -> IO(Unit):
  IO.print(SHOW(OPNAME(REPS, DEPTH)))
