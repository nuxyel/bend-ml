# cap_depth (by doubling, 0.1.10.0) equals cap_pair's depth (the n-step recursion the law was first proved on) for every
# n from 0 to 2000, and is the smallest d with n <= 2^d at a few large sizes. Prints the number of mismatches: 0.
import Base
import ../main.bend as TA

def same(+n: Nat) -> U32:
  Bool.pick(U32, Nat.is_eq(TA.cap_depth(n), TA.fst_nat(TA.cap_pair(n))), 0, 1)

def upto(k: Nat, acc: U32) -> U32:
  match k:
    case 0n: (acc + same(0n) : U32)
    case 1n++p: upto(p, (acc + same(1n+p) : U32))

# the smallest d with n <= 2^d, for n >= 2: 2^(d-1) < n <= 2^d
def tight(+n: Nat) -> U32:
  +d = TA.cap_depth(n)
  # bend-lint: disable-next bend-lint-rules/performance/eager_branch -- a one-shot test not-expected-because: 4 calls, pow2(21) at most
  Bool.pick(U32, Bool.and(Nat.is_le(n, TA.pow2(d)), Nat.is_lt(TA.pow2(Nat.sub(d, 1n)), n)), 0, 1)

def main() -> U32:
  (((upto(2000n, 0) + tight(78400n) : U32) + (tight(100352n) + tight(1048576n) : U32) : U32) + tight(1048577n) : U32)
