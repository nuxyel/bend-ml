import Base

# r x c matrix: the dimensions are in the TYPE (erased at run time);
# the data lives in an Array<F32>, row by row (index = i*c + j).
type Mat<-r: Nat, -c: Nat> is Type:
  Mat{data: Array<F32>}

def Mat.new(-r: Nat, -c: Nat, depth: Nat, +v: F32) -> Mat<r, c>:
  Mat{Array.new(F32, depth, v)}

# "a match does not inspect a computed value": the pair returned by Array.get
# needs its own function to be opened.
def Mat.get.fin(-r: Nat, -c: Nat, t: Array<F32> & F32) -> Mat<r, c> & F32:
  (d2, x) = t
  (Mat{d2}, x)

def Mat.get(-r: Nat, +c: Nat, m: Mat<r, c>, +i: U32, +j: U32) -> Mat<r, c> & F32:
  match m:
    case Mat{d}:
      Mat.get.fin(r, c, Array.get(F32, d, (i * U32.from_nat(c) + j : U32)))

def Mat.set(-r: Nat, +c: Nat, m: Mat<r, c>, +i: U32, +j: U32, v: F32) -> Mat<r, c>:
  match m:
    case Mat{d}:
      Mat{Array.set(F32, d, (i * U32.from_nat(c) + j : U32), v)}

# acc += A[i][p] * B[p][j] for p = 0..k-1. The (matrix, element) pairs
# arrive as parameters, which is what match accepts to open.
def Mat.dot(-n: Nat, +k: Nat, +m: Nat, steps: Nat, +i: U32, +j: U32, +p: U32, acc: F32, ax: Mat<n, k> & F32, by: Mat<k, m> & F32) -> Mat<n, k> & Mat<k, m> & F32:
  match steps ax by:
    case 0n Tuple{a, x} Tuple{b, y}:
      (a, b, acc)
    case 1n+s Tuple{a, +x} Tuple{b, +y}:
      Mat.dot(n, k, m, s, i, j, (p + 1 : U32), (acc + (x * y : F32) : F32), Mat.get(n, k, a, i, (p + 1 : U32)), Mat.get(k, m, b, (p + 1 : U32), j))

def show3(t: Mat<2n, 2n> & Mat<2n, 2n> & F32) -> IO(Unit):
  match t:
    case Tuple{a, bf}:
      match bf:
        case Tuple{b, r}:
          IO.print(F32.show(r))

def dot22(a: Mat<2n, 2n>, b: Mat<2n, 2n>) -> IO(Unit):
  show3(Mat.dot(2n, 2n, 2n, 2n, 0, 0, 0, 0.0, Mat.get(2n, 2n, a, 0, 0), Mat.get(2n, 2n, b, 0, 0)))

def main() -> IO(Unit):
  dot22(Mat.new(2n, 2n, 2n, 1.5), Mat.new(2n, 2n, 2n, 2.0))
