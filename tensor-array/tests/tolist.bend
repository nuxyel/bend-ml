# Mat.to_list and Mat.read_row (read_l, which reads from the last index down since 2026-11-01, so its list needs no
# reverse) give the numbers in row order: against a list built by a plain forward recursion, bit for bit, on shapes
# of one number, of one row, of non-powers of two and of sizes just past a power of two (a capacity with spare cells,
# which a wrong index would read as 0.0). Also the count of each list. Prints the number of differences per shape:
# all must be 0.
import Base
import ../main.bend as TA

def val(+i: U32) -> F32:
  (U32.to_f32(U32.mod((i * 2654435761 : U32), 1000)) - 499.5 : F32)

# the reference: val(i), ..., val(i + n - 1) by a forward recursion (no Array)
def ref(n: Nat, +i: U32) -> List<&2, F32>:
  match n:
    case 0n: Nil{}
    case 1n+p: val(i) <> ref(p, (i + 1 : U32))

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def diff(xs: List<&2, F32>, ys: List<&2, F32>, +acc: U32) -> U32:
  match xs ys:
    case Nil{} Nil{}: acc
    case Con{+x, xt} Con{+y, yt}: diff(xt, yt, (acc + Bool.pick(U32, U32.is_eq(bits(x), bits(y)), 0, 1) : U32))
    case _ _: (acc + 1000000 : U32)

def mat(+r: Nat, +c: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, ref(Nat.mul(r, c), 0))

def whole2(+r: Nat, +c: Nat, ml: TA.ML<r, c>) -> U32:
  match ml:
    case TA.ML{m, l}: diff(l, ref(Nat.mul(r, c), 0), 0)

def whole(+r: Nat, +c: Nat) -> U32:
  whole2(r, c, TA.Mat.to_list(r, c, mat(r, c)))

def row_d(+r: Nat, +c: Nat, +i: U32, rr: TA.MRow<r, c>) -> TA.Mat<r, c> & U32:
  match rr:
    case TA.MRow{m, l}: (m, diff(l, ref(c, (i * U32.from_nat(c) : U32)), 0))

# every row of the matrix, from row i on, read with Mat.read_row
def rows(k: Nat, +r: Nat, +c: Nat, +i: U32, +acc: U32, mu: TA.Mat<r, c> & U32) -> U32:
  match k mu:
    case 0n Tuple{m, +d}: (acc + d : U32)
    case 1n+p Tuple{m, +d}: rows(p, r, c, (i + 1 : U32), (acc + d : U32), row_d(r, c, i, TA.Mat.read_row(r, c, m, i)))

def shape(+r: Nat, +c: Nat) -> U32:
  (whole(r, c) + rows(r, r, c, 0, 0, (mat(r, c), 0)) : U32)

def main() -> String:
  U32.show(shape(1n, 1n)) ++ " " ++ U32.show(shape(1n, 5n)) ++ " " ++ U32.show(shape(3n, 4n)) ++ " " ++ U32.show(shape(7n, 9n)) ++ " " ++ U32.show(shape(1n, 1024n)) ++ " " ++ U32.show(shape(1n, 1025n)) ++ " " ++ U32.show(shape(33n, 31n)) ++ " " ++ U32.show(shape(5n, 1025n))
