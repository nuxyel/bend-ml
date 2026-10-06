# A reshape whose size is only known at run time, checked for EVERY size at compile time.
#
# n comes from the command line. x : Mat<n, 6> is regrouped as Mat<n*2, 3>. A reshape only
# compiles with a proof that the number of elements does not change, and here that proof
# is about a variable: n*(2*3) == (n*2)*3 holds for every n (associativity of
# multiplication, `mul_assoc` from bend-ml-nat-lemmas). So the checker accepts `regroup`
# once, for all n, and nothing about the size is checked at run time.
#
#   bend examples/symbolic_reshape.bend -- 2
#
# A library that checks shapes with constant sizes has to fall back to a run-time assertion
# once a size is a run-time number. examples/symbolic_reshape_bad.bend asks for n x 5
# instead of n*2 x 3: n*6 == n*5 is false for every n > 0, no proof exists, and it does
# not compile.

import Base
import bend-ml-nat-lemmas@0.1.1.0/main.bend as NL
import bend-ml-tensor@0.1.2.0/main.bend as T

# n x 6 -> (n*2) x 3, for every n
def regroup(+n: Nat, x: T.Mat<n, 6n>) -> T.Mat<Nat.mul(n, 2n), 3n>:
  T.Mat.reshape(n, 6n, Nat.mul(n, 2n), 3n, NL.mul_assoc(n, 2n, 3n), x)

# ---- input and output ----

# 0, 1, 2, ... k-1
def count(k: Nat, +i: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match k:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      count(p, 1n+i, U32.to_f32(U32.from_nat(i)) <> acc)

def row_str(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      " " ++ F32.show(h) ++ row_str(t)

def print_rows(rs: List<&2, List<&2, F32>>) -> IO(Unit):
  match rs:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{h, t}:
      do IO<Unit>:
        IO.print(" " ++ row_str(h))
        print_rows(t)

def run2(+n: Nat, mx: Maybe<&2, T.Mat<n, 6n>>) -> IO(Unit):
  match mx:
    case None{}:
      IO.print("error: could not build the matrix")
    case Some{x}:
      do IO<Unit>:
        IO.print("n read from the command line: " ++ Nat.show(n))
        IO.print("x : Mat<" ++ Nat.show(n) ++ ", 6>  ->  regroup(x) : Mat<" ++ Nat.show(Nat.mul(n, 2n)) ++ ", 3>")
        print_rows(T.Mat.to_rows(Nat.mul(n, 2n), 3n, regroup(n, x)))

def run(+n: Nat) -> IO(Unit):
  run2(n, T.Mat.of(n, 6n, T.chunk.l(n, 6n, count(Nat.mul(n, 6n), 0n, Nil{}))))

# the first command-line argument, as a number
def arg1(args: List<String>) -> Maybe<&2, Nat>:
  match args:
    case Con{prog, Con{a, rest}}:
      Nat.read(a)
    case _:
      None{}

def nat_of(m: Maybe<&2, Nat>) -> Nat:
  match m:
    case Some{n}:
      n
    case None{}:
      2n

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    run(nat_of(arg1(args)))
