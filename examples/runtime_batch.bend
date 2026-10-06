# End-to-end example: the batch size is NOT known when the program is compiled.
#
# The input file starts with the batch size n, followed by n rows of 4 numbers:
#
#   3
#   0.5  1.0 -0.5  2.0
#   ...
#
# The program reads n from the file, builds X : Mat<n, 4> and runs one layer,
# H = relu(X·W + b) : Mat<n, 3>. The shapes are still checked at compile time,
# because `layer` is written for every n at once: the checker never needs to
# know n, only that X's 4 columns meet W's 4 rows. The one thing that can go
# wrong at run time, the file holding the wrong amount of numbers, is caught
# once at the boundary: Mat.of_list returns None instead of a matrix.
#
#   bend examples/runtime_batch.bend -- examples/data/batch3.txt
#   bend examples/runtime_batch.bend -- examples/data/wrong_count.txt
#
# examples/runtime_batch_bad.bend is the same program with W of the wrong
# shape: it does not compile, even though n is unknown.

import Base
import bend-ml-tensor-array@0.1.3.0/main.bend as TA

# ---- the model: fixed weights, any batch size ----

# Bend has no negative number literals
def neg(+x: F32) -> F32:
  (0.0 - x : F32)

def weights() -> TA.Mat<4n, 3n>:
  TA.Mat.from_list(4n, 3n, [0.5, neg(1.0), 0.25, 1.0, 0.5, neg(0.5), neg(0.25), 0.75, 1.0, 0.1, 0.2, 0.3])

def bias() -> TA.Mat<1n, 3n>:
  TA.Mat.from_list(1n, 3n, [0.1, neg(0.2), 0.05])

def take_c(-n: Nat, x: TA.MMul<n, 4n, 3n>) -> TA.Mat<n, 3n>:
  match x:
    case TA.MMul{a, w, c}:
      c

def take_b(-n: Nat, x: TA.MBias<n, 3n>) -> TA.Mat<n, 3n>:
  match x:
    case TA.MBias{h, b}:
      h

# one dense layer for a batch of ANY size n: Mat<n, 4> -> Mat<n, 3>
def layer(+n: Nat, x: TA.Mat<n, 4n>) -> TA.Mat<n, 3n>:
  TA.Mat.relu(n, 3n, take_b(n, TA.Mat.add_row(n, 3n, take_c(n, TA.Mat.matmul(n, 4n, 3n, 0n, x, weights())), bias())))

# ---- reading the file ----

def rb_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  (f, r) = m
  do IO<List<&2, U32>>:
    bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    File.close(f)
    return bs

def read_bytes(path: String) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_bytes(f, 100000000)
    rb_done(m)

def is_ws(xs: List<&2, U32>) -> Bool:
  match xs:
    case Nil{}:
      True{}
    case Con{+c, t}:
      U32.is_le(c, 32)

def push(m: Maybe<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match m:
    case None{}:
      acc
    case Some{v}:
      v <> acc

def flush(cur: List<&2, Char>, acc: List<&2, F32>) -> List<&2, F32>:
  match cur:
    case Nil{}:
      acc
    case Con{h, t}:
      push(F32.read(String.from_list(List.reverse(&2, Char, Con{h, t}))), acc)

# every whitespace-separated number in the text, in order
def floats.go(xs: List<&2, U32>, ws: Bool, cur: List<&2, Char>, acc: List<&2, F32>) -> List<&2, F32>:
  match xs ws:
    case Nil{} _:
      List.reverse(&2, F32, flush(cur, acc))
    case Con{c, +t} True{}:
      floats.go(t, is_ws(t), Nil{}, flush(cur, acc))
    case Con{+c, +t} False{}:
      floats.go(t, is_ws(t), Chr{c} <> cur, acc)

def floats(+xs: List<&2, U32>) -> List<&2, F32>:
  floats.go(xs, is_ws(xs), Nil{}, Nil{})

# ---- printing ----

# the first c numbers of xs, as one line
def row_str(c: Nat, xs: List<&2, F32>) -> String:
  match c xs:
    case 1n+p Con{h, t}:
      F32.show(h) ++ " " ++ row_str(p, t)
    case _ _:
      ""

def print_rows(k: Nat, +xs: List<&2, F32>) -> IO(Unit):
  match k:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        IO.print("  " ++ row_str(3n, xs))
        print_rows(p, List.drop(&2, F32, xs, 3n))

def show_out(+n: Nat, r: TA.ML<n, 3n>) -> IO(Unit):
  match r:
    case TA.ML{m, l}:
      do IO<Unit>:
        IO.print("H = relu(X·W + b) : Mat<" ++ Nat.show(n) ++ ", 3>")
        print_rows(n, l)

# ---- the boundary: one run-time check, then everything is typed ----

def run2(+n: Nat, +count: Nat, mx: Maybe<&1, TA.Mat<n, 4n>>) -> IO(Unit):
  match mx:
    case None{}:
      IO.print("error: the file says the batch has " ++ Nat.show(n) ++ " rows of 4 numbers (" ++ Nat.show(Nat.mul(n, 4n)) ++ " in total), but it holds " ++ Nat.show(count))
    case Some{x}:
      do IO<Unit>:
        IO.print("batch size read from the file: " ++ Nat.show(n))
        IO.print("X : Mat<" ++ Nat.show(n) ++ ", 4>")
        show_out(n, TA.Mat.to_list(n, 3n, layer(n, x)))

def run(+n: Nat, +rest: List<&2, F32>) -> IO(Unit):
  run2(n, List.length(&2, F32, rest), TA.Mat.of_list(n, 4n, rest))

def start(fs: List<&2, F32>) -> IO(Unit):
  match fs:
    case Nil{}:
      IO.print("error: empty file, expected the batch size first")
    case Con{h, rest}:
      run(F32.to_nat(h), rest)

def go(path: String) -> IO(Unit):
  do IO<Unit>:
    bs : List<&2, U32> <- read_bytes(path)
    start(floats(bs))

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{path, Nil{}}}:
      go(path)
    case _:
      IO.print("usage: runtime_batch <file>   (first number: batch size n, then n rows of 4 numbers)")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
