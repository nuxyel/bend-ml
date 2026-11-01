# The lemma the round trip needs, checked where it can be checked today: on a
# perfect tree of 2^d leaves, a write at j and a read at k reach different leaves
# whenever j and k differ, for every pair of indices.
#
#   bend tensor-array/tests/apart.bend
#
# The law `to_list (from_list xs) == xs` needs that lemma symbolically, and the
# lemma needs four U32 facts Base does not state. Until they are proved, this is
# the oracle: the semantic property, computed on every ordered pair of distinct
# indices up to depth 7, and the number of pairs it actually checked is printed
# beside the failures so that a zero cannot mean "nothing ran".
#
# n = 2^d and every index used is below n, so nothing wraps; the checks are
#   set at j, read at j:  v[j] is 1.0
#   set at j, read at k:  v[k] is still 0.0 when j != k
import Base
import ../main.bend as TA

type CS is Type:
  CS{bad: Nat, chk: Nat}

def cs_bad(s: CS) -> CS:
  match s:
    case CS{b, c}:
      CS{Nat.add(b, 1n), c}

def cs_chk(s: CS) -> CS:
  match s:
    case CS{b, c}:
      CS{b, Nat.add(c, 1n)}

def bump(ok: Bool, s: CS) -> CS:
  match ok:
    case True{}:
      s
    case False{}:
      cs_bad(s)

def second_read(r: Array<F32> & F32, +ok_j: Bool, s: CS) -> CS:
  match r:
    case Tuple{_, +vk}:
      bump(Bool.and(ok_j, F32.is_eq(vk, 0.0)), s)

def first_read(r: Array<F32> & F32, +k: U32, s: CS) -> CS:
  match r:
    case Tuple{b, +vj}:
      second_read(Array.get(F32, b, k), F32.is_eq(vj, 1.0), s)

def pair_bad(+d: Nat, +j: U32, +k: U32, s: CS) -> CS:
  a = Array.new(F32, d, 0.0)
  b = Array.set(F32, a, j, 1.0)
  first_read(Array.get(F32, b, j), k, s)

def pair_of(eq: Bool, +d: Nat, +j: U32, +k: U32, s: CS) -> CS:
  match eq:
    case True{}:
      s
    case False{}:
      pair_bad(d, j, k, cs_chk(s))

# one row: k from 0 to n-1, every k != j
def row(+fuel: Nat, +d: Nat, +j: Nat, +k: Nat, +n: Nat, s: CS) -> CS:
  match fuel:
    case 0n:
      s
    case 1n+rest:
      row(rest, d, j, Nat.add(k, 1n), n,
        pair_of(Nat.is_eq(j, k), d, U32.from_nat(j), U32.from_nat(k), s))

# j from 0 to n-1
def rows(+fuel: Nat, +d: Nat, +j: Nat, +n: Nat, s: CS) -> CS:
  match fuel:
    case 0n:
      s
    case 1n+rest:
      rows(rest, d, Nat.add(j, 1n), n, row(n, d, j, 0n, n, s))

def depth(+d: Nat, +n: Nat, s: CS) -> CS:
  rows(n, d, 0n, n, s)

def upto(+left: Nat, +d: Nat, +n: Nat, s: CS) -> CS:
  match left:
    case 0n:
      s
    case 1n+rest:
      upto(rest, Nat.add(d, 1n), Nat.mul(n, 2n), depth(d, n, s))

def report(s: CS) -> IO(Unit):
  match s:
    case CS{b, c}:
      do IO<Unit>:
        IO.print(String.append("apart_pairs: ", Nat.show(c)))
        IO.print(String.append("apart_failures: ", Nat.show(b)))
        return Unit{}

def main() -> IO(Unit):
  report(upto(8n, 0n, 1n, CS{0n, 0n}))
