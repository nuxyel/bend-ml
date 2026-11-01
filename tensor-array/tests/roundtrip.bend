# Mat.of_list -> Mat.to_list returns the same numbers, for empty, one-element, row, column and
# rectangular matrices, with +0, -0, +inf, -inf, NaN and a subnormal among the values. Prints the
# number of cases that differ from their list, which must be 0.
#
# This is the value half of the conversion guarantee. The count is the law `to_list_len` in
# main.bend; the values are tested here because a content law needs an Array.set/get lemma, which
# tensor-array now carries: `size_set`, `get_after_set` and `get_other_set` (with `ASet.Apart` as
# the side condition, issue #26). The content law is statable over them from here on.
#
# The law is shape-blind by construction, and this file says so: `of_list`/`to_list` are row-major,
# so the same 15 numbers round-trip as 3x5 and as 5x3 alike. What a shape test can catch is the
# *count* (`to_list_len`) and `Mat`'s indexed accessors, not the list.
import Base
import ../main.bend as TA

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def same_list(+xs: List<&2, F32>, +ys: List<&2, F32>) -> Bool:
  match xs ys:
    case Nil{} Nil{}:
      True{}
    case Con{+h, +t} Con{+h2, +t2}:
      # bend-lint: disable-next bend-lint-rules/performance/eager_branch -- a one-shot test not-expected-because: equal lists are walked to the end anyway
      Bool.and(U32.is_eq(bits(h), bits(h2)), same_list(t, t2))
    case _ _:
      False{}

def check_ml(-r: Nat, -c: Nat, +xs: List<&2, F32>, ml: TA.ML<r, c>) -> Nat:
  match ml:
    case TA.ML{m, l}:
      Bool.pick(Nat, same_list(xs, l), 0n, 1n)

def check_m(+r: Nat, +c: Nat, +xs: List<&2, F32>, m: Maybe<&1, TA.Mat<r, c>>) -> Nat:
  match m:
    case Some{mat}:
      check_ml(r, c, xs, TA.Mat.to_list(r, c, mat))
    case None{}:
      1n

def check(+r: Nat, +c: Nat, +xs: List<&2, F32>) -> Nat:
  check_m(r, c, xs, TA.Mat.of_list(r, c, xs))

# 0, 1, 2, ... (k numbers) — enough for a 64x64 mat without a literal of 4096 entries
def ramp(+k: Nat, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match k:
    case 0n:
      acc
    case 1n+q:
      ramp(q, (1n + i : Nat), Con{F32.from_nat(i), acc})

def sum(+xs: List<&2, Nat>, +acc: Nat) -> Nat:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      sum(t, Nat.add(acc, h))

def cases() -> Nat:
  sum([
    check(0n, 0n, Nil{}),
    check(0n, 5n, Nil{}),
    check(5n, 0n, Nil{}),
    check(1n, 1n, [1.5]),
    check(1n, 2n, [(0.0 - 2.0 : F32), 3.5]),
    check(2n, 1n, [1.0, (0.0 - 1.0 : F32)]),
    check(2n, 2n, [1.0, 2.0, 3.0, 4.0]),
    check(2n, 3n, [1.0, (0.0 - 0.0 : F32), 3.5, 4.25, (0.0 - 5.0 : F32), 6.0]),
    check(3n, 4n, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    check(1n, 8n, [0.0, ((0.0 - 1.0 : F32) * 0.0 : F32), (1.0 / 0.0 : F32), ((0.0 - 1.0 : F32) / 0.0 : F32), (0.0 / 0.0 : F32), 1.0, 0.00000000000000000000000000000000000001, 3.0]),
    check(4n, 1n, [0.5, (0.0 - 0.5 : F32), 1.5, (0.0 - 1.5 : F32)]),
    check(3n, 5n, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
    check(64n, 64n, ramp(4096n, 0n, Nil{}))
  ], 0n)

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(Nat.show(cases()))
