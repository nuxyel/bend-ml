# The 64-bit and BF16 reads (size64, read_f32_at, read_bf16_at, read_f32_into, read_bf16_into) and read_f32, driven by
# binary-read/tests/test_read_at.py, which writes the files and knows every expected bit. An offset is given as two
# U32s, HI and LO, meaning HI * 2^32 + LO, as the package takes it.
#
#   f32 FILE HI LO N         read_f32_at into 2^d cells holding 9.0 (2^d >= N): "count: bits of cells 0 .. N-1"
#   bf16 FILE HI LO N        the same with read_bf16_at
#   f32into FILE HI LO N AT  read_f32_into into 2^d cells holding 9.0 (2^d >= AT + N + 1), from cell AT:
#                            "count: bits of cells 0 .. AT + N" (so the cell after the read shows it was left alone)
#   bf16into FILE HI LO N AT the same with read_bf16_into
#   f32cap FILE HI LO N D    read_f32_at of N numbers into 2^D cells holding 9.0 (the read stops at the capacity):
#                            "count: bits of all 2^D cells"
#   bf16cap FILE HI LO N D   the same with read_bf16_at
#   old FILE OFF N           read_f32 (a U32 offset): "bytes: bits of cells 0 .. N-1"
#   size FILE                size64: "HI LO"
#
#   bend binary-read/tests/read_at.bend f32 FILE 1 8 4               (JavaScript)
#   bend binary-read/tests/read_at.bend -o /tmp/ra && /tmp/ra ...    (C)
import Base
import ../main.bend as B

# U32 and F32 hold the same 32-bit word: the cell's bits
def bits(x: F32) -> U32:
  match x:
    case F32{w}:
      U32{w}

# the smallest d with 2^d >= n (0 for n <= 1); p is 2^d, ok says whether it holds n
def depth_go(f: Nat, ok: Bool, +n: U32, +p: U32, d: Nat) -> Nat:
  match f ok:
    case _ True{}:
      d
    case 0n False{}:
      d
    case 1n+g False{}:
      depth_go(g, U32.is_le(n, (p + p : U32)), n, (p + p : U32), 1n+d)

def depth(+n: U32) -> Nat:
  depth_go(32n, U32.is_le(n, 1), n, 1, 0n)

def nines(+n: U32) -> Array<F32>:
  Array.new(F32, depth(n), 9.0)

# " b_0 b_1 .. b_i" (the cells' bits in decimal), built from the end so each step only prepends
def dump(k: Nat, r: Array<F32> & F32, +i: U32, acc: String) -> String:
  match k r:
    case 0n Tuple{a, x}:
      " " ++ U32.show(bits(x)) ++ acc
    case 1n+p Tuple{a, x}:
      dump(p, Array.get(F32, a, U32.sub(i, 1)), U32.sub(i, 1), " " ++ U32.show(bits(x)) ++ acc)

def cells(z: Bool, a: Array<F32>, +n: U32) -> String:
  match z:
    case True{}: ""
    case False{}: dump(U32.to_nat(U32.sub(n, 1)), Array.get(F32, a, U32.sub(n, 1)), U32.sub(n, 1), "")

# the count (numbers or bytes) and the first k cells; an error stops the program with its message
def show(r: File & (Array<F32> & Result<&1, &1, U32 & String, U32>), +k: U32) -> IO(Unit):
  match r:
    case Tuple{f, Tuple{a, res}}:
      do IO<Unit>:
        +got : U32 <- IO.pass(U32, res)
        File.close(f)
        IO.print(U32.show(got) ++ ":" ++ cells(U32.is_zero(k), a, k))

def at_go(bf16: Bool, f: File, hi: U32, lo: U32, +n: U32) -> IO(Unit):
  match bf16:
    case True{}:
      do IO<Unit>:
        r : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- B.read_bf16_at(f, hi, lo, n, nines(n))
        show(r, n)
    case False{}:
      do IO<Unit>:
        r : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- B.read_f32_at(f, hi, lo, n, nines(n))
        show(r, n)

def into_go(bf16: Bool, f: File, hi: U32, lo: U32, +n: U32, +at: U32) -> IO(Unit):
  match bf16:
    case True{}:
      do IO<Unit>:
        r : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- B.read_bf16_into(f, hi, lo, n, nines((at + n + 1 : U32)), at)
        show(r, (at + n + 1 : U32))
    case False{}:
      do IO<Unit>:
        r : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- B.read_f32_into(f, hi, lo, n, nines((at + n + 1 : U32)), at)
        show(r, (at + n + 1 : U32))

def cap_go(bf16: Bool, f: File, hi: U32, lo: U32, +n: U32, +d: U32) -> IO(Unit):
  match bf16:
    case True{}:
      do IO<Unit>:
        r : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- B.read_bf16_at(f, hi, lo, n, Array.new(F32, U32.to_nat(d), 9.0))
        show(r, U32.shln(1, U32.to_nat(d)))
    case False{}:
      do IO<Unit>:
        r : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- B.read_f32_at(f, hi, lo, n, Array.new(F32, U32.to_nat(d), 9.0))
        show(r, U32.shln(1, U32.to_nat(d)))

def old_go(f: File, off: U32, +n: U32) -> IO(Unit):
  do IO<Unit>:
    r : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- B.read_f32(f, off, n, nines(n))
    show(r, n)

def print_pair(s: U32 & U32) -> IO(Unit):
  match s:
    case Tuple{hi, lo}: IO.print(U32.show(hi) ++ " " ++ U32.show(lo))

def sized(r: File & Result<&1, &1, U32 & String, U32 & U32>) -> IO(Unit):
  match r:
    case Tuple{f, res}:
      do IO<Unit>:
        s : U32 & U32 <- IO.pass(U32 & U32, res)
        File.close(f)
        print_pair(s)

def opened(+path: String) -> IO(File):
  IO.try(File, File.open(path, "r"))

def usage() -> IO(Unit):
  IO.die(Unit, 1, "usage: read_at f32|bf16 FILE HI LO N | f32into|bf16into FILE HI LO N AT | f32cap|bf16cap FILE HI LO N D | old FILE OFF N | size FILE")

def at_m(bf16: Bool, +path: String, mh: Maybe<&2, U32>, ml: Maybe<&2, U32>, mn: Maybe<&2, U32>) -> IO(Unit):
  match mh ml mn:
    case Some{hi} Some{lo} Some{+n}:
      do IO<Unit>:
        f : File <- opened(path)
        at_go(bf16, f, hi, lo, n)
    case _ _ _: usage()

def cap_m(bf16: Bool, +path: String, mh: Maybe<&2, U32>, ml: Maybe<&2, U32>, mn: Maybe<&2, U32>, md: Maybe<&2, U32>) -> IO(Unit):
  match mh ml mn md:
    case Some{hi} Some{lo} Some{+n} Some{+d}:
      do IO<Unit>:
        f : File <- opened(path)
        cap_go(bf16, f, hi, lo, n, d)
    case _ _ _ _: usage()

def into_m(bf16: Bool, +path: String, mh: Maybe<&2, U32>, ml: Maybe<&2, U32>, mn: Maybe<&2, U32>, ma: Maybe<&2, U32>) -> IO(Unit):
  match mh ml mn ma:
    case Some{hi} Some{lo} Some{+n} Some{+at}:
      do IO<Unit>:
        f : File <- opened(path)
        into_go(bf16, f, hi, lo, n, at)
    case _ _ _ _: usage()

def old_m(+path: String, mo: Maybe<&2, U32>, mn: Maybe<&2, U32>) -> IO(Unit):
  match mo mn:
    case Some{off} Some{+n}:
      do IO<Unit>:
        f : File <- opened(path)
        old_go(f, off, n)
    case _ _: usage()

def size_m(+path: String) -> IO(Unit):
  do IO<Unit>:
    f : File <- opened(path)
    r : File & Result<&1, &1, U32 & String, U32 & U32> <- B.size64(f)
    sized(r)

def five(f32: Bool, bf16: Bool, +path: String, +hi: String, +lo: String, +n: String) -> IO(Unit):
  match f32 bf16:
    case True{} _: at_m(False{}, path, U32.read(hi), U32.read(lo), U32.read(n))
    case _ True{}: at_m(True{}, path, U32.read(hi), U32.read(lo), U32.read(n))
    case _ _: usage()

def six(f32: Bool, bf16: Bool, f32c: Bool, bf16c: Bool, +path: String, +hi: String, +lo: String, +n: String, +at: String) -> IO(Unit):
  match f32 bf16 f32c bf16c:
    case True{} _ _ _: into_m(False{}, path, U32.read(hi), U32.read(lo), U32.read(n), U32.read(at))
    case _ True{} _ _: into_m(True{}, path, U32.read(hi), U32.read(lo), U32.read(n), U32.read(at))
    case _ _ True{} _: cap_m(False{}, path, U32.read(hi), U32.read(lo), U32.read(n), U32.read(at))
    case _ _ _ True{}: cap_m(True{}, path, U32.read(hi), U32.read(lo), U32.read(n), U32.read(at))
    case _ _ _ _: usage()

def four(is_old: Bool, +path: String, +off: String, +n: String) -> IO(Unit):
  match is_old:
    case True{}: old_m(path, U32.read(off), U32.read(n))
    case False{}: usage()

def two(is_size: Bool, +path: String) -> IO(Unit):
  match is_size:
    case True{}: size_m(path)
    case False{}: usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, Con{path, Con{hi, Con{lo, Con{n, Con{at, Nil{}}}}}}}}:
      six(String.eq(mode, "f32into"), String.eq(mode, "bf16into"), String.eq(mode, "f32cap"), String.eq(mode, "bf16cap"), path, hi, lo, n, at)
    case Con{prog, Con{+mode, Con{path, Con{hi, Con{lo, Con{n, Nil{}}}}}}}:
      five(String.eq(mode, "f32"), String.eq(mode, "bf16"), path, hi, lo, n)
    case Con{prog, Con{mode, Con{path, Con{off, Con{n, Nil{}}}}}}:
      four(String.eq(mode, "old"), path, off, n)
    case Con{prog, Con{mode, Con{path, Nil{}}}}:
      two(String.eq(mode, "size"), path)
    case _: usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
