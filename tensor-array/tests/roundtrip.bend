# Mat.of_list -> Mat.to_list returns the same numbers, for empty, one-element, row, column and
# rectangular matrices, with +0, -0, +inf, -inf, NaN and a subnormal among the values. Prints the
# number of cases that differ from their list, which must be 0.
#
# This is the value half of the conversion guarantee. The count is the law `to_list_len` in
# main.bend; the values are tested here because Bend's checker does not follow Array.set/Array.get
# on a neutral array, so a content law needs an Array.set/get lemma that Base does not have.
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

def cases() -> Nat:
  Nat.add(check(0n, 0n, Nil{}),
  Nat.add(check(0n, 5n, Nil{}),
  Nat.add(check(5n, 0n, Nil{}),
  Nat.add(check(1n, 1n, [1.5]),
  Nat.add(check(1n, 2n, [(0.0 - 2.0 : F32), 3.5]),
  Nat.add(check(2n, 1n, [1.0, (0.0 - 1.0 : F32)]),
  Nat.add(check(2n, 2n, [1.0, 2.0, 3.0, 4.0]),
  Nat.add(check(2n, 3n, [1.0, (0.0 - 0.0 : F32), 3.5, 4.25, (0.0 - 5.0 : F32), 6.0]),
  Nat.add(check(3n, 4n, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
  Nat.add(check(1n, 8n, [0.0, ((0.0 - 1.0 : F32) * 0.0 : F32), (1.0 / 0.0 : F32), ((0.0 - 1.0 : F32) / 0.0 : F32), (0.0 / 0.0 : F32), 1.0, 0.00000000000000000000000000000000000001, 3.0]),
  check(4n, 1n, [0.5, (0.0 - 0.5 : F32), 1.5, (0.0 - 1.5 : F32)])))))))))))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(Nat.show(cases()))
