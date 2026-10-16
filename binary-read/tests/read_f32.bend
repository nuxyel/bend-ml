# read_f32 on a 30-byte file holding 1.0, -2.5, 3.25, 0.1, -0.0, +inf and the smallest subnormal
# (little-endian F32), then 2 bytes of a number cut short: a whole read, a read at an offset, a
# read clipped by the array's capacity, the special values, and a read cut short by the end of the
# file, whose last 2 bytes go over the low bytes of a cell holding 9.0 (0x41100000 becomes
# 0x41100001). Each line prints the bytes read, then the cells. Both lanes print the same lines:
#   bend binary-read/tests/read_f32.bend                        (JS)
#   bend binary-read/tests/read_f32.bend -o /tmp/rf && /tmp/rf  (C)
import Base
import ../main.bend as B

def path() -> String:
  "/tmp/bend_ml_read_f32.bin"

# 1.0 = 0x3F800000, -2.5 = 0xC0200000, 3.25 = 0x40500000, 0.1 = 0x3DCCCCCD, -0.0 = 0x80000000,
# +inf = 0x7F800000, the smallest subnormal = 0x00000001, low byte first; then 0x01, 0x00
def bytes() -> List<&2, U32>:
  [0, 0, 128, 63, 0, 0, 32, 192, 0, 0, 80, 64, 205, 204, 204, 61, 0, 0, 0, 128, 0, 0, 128, 127, 1, 0, 0, 0, 1, 0]

def wrote(p: File & Result<&1, &1, U32 & String, Unit>) -> IO(Unit):
  match p:
    case Tuple{f, r}:
      do IO<Unit>:
        IO.pass(Unit, r)
        File.close(f)

# shows k + 1 cells: the one already read (r) and the k after it
def cells(k: Nat, r: Array<F32> & F32, +i: U32, acc: String) -> String:
  match k r:
    case 0n Tuple{a, x}:
      acc ++ " " ++ F32.show(x)
    case 1n+p Tuple{a, x}:
      cells(p, Array.get(F32, a, (i + 1 : U32)), (i + 1 : U32), acc ++ " " ++ F32.show(x))

def show(p: File & (Array<F32> & Result<&1, &1, U32 & String, U32>), +k: Nat, name: String) -> IO(Unit):
  match p:
    case Tuple{f, Tuple{a, r}}:
      do IO<Unit>:
        +n : U32 <- IO.pass(U32, r)
        File.close(f)
        IO.print(name ++ ": " ++ U32.show(n) ++ " bytes," ++ cells(k, Array.get(F32, a, 0), 0, ""))

def probe(+off: U32, +n: U32, +d: Nat, +k: Nat, name: String) -> IO(Unit):
  do IO<Unit>:
    f : File <- IO.try(File, File.open(path(), "r"))
    p : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- B.read_f32(f, off, n, Array.new(F32, d, 9.0))
    show(p, k, name)

def main() -> IO(Unit):
  do IO<Unit>:
    f : File <- IO.try(File, File.open(path(), "w"))
    w : File & Result<&1, &1, U32 & String, Unit> <- File.write_bytes(f, bytes())
    wrote(w)
    probe(0, 4, 2n, 3n, "whole")
    probe(4, 2, 2n, 3n, "offset")
    probe(0, 4, 1n, 1n, "capacity")
    probe(16, 3, 2n, 3n, "special values")
    probe(24, 4, 2n, 3n, "end of file")

#| whole: 16 bytes, 1 -2.5 3.25 0.1
#| offset: 8 bytes, -2.5 3.25 9 9
#| capacity: 8 bytes, 1 -2.5
#| special values: 12 bytes, -0 inf 1e-45 9
#| end of file: 6 bytes, 1e-45 9.000001 9 9
