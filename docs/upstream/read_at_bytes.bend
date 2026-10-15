import Base

# Reads w.bin (384 MiB; make it with `head -c 402653184 /dev/urandom > w.bin`) in 1 MiB blocks with
# File.read_at and adds up its bytes. Each block arrives as a List<U32> of 2^20 cells, one per byte.
# The same loop in C is read_at_bytes.c; both print the same sum.

def walk(bs: List<&2, U32>, acc: U32) -> U32:
  match bs:
    case Nil{}:
      acc
    case Con{b, t}:
      walk(t, (acc + b : U32))

def loop(k: Nat, +off: U32, m: File & Result<&1, &1, U32 & String, List<&2, U32>>, acc: U32) -> IO(U32):
  match k m:
    case 0n Tuple{f, r}:
      do IO<U32>:
        File.close(f)
        return acc
    case 1n+p Tuple{f, r}:
      do IO<U32>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
        m2 : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, (off + 1048576 : U32), 1048576)
        loop(p, (off + 1048576 : U32), m2, walk(bs, acc))

def main() -> IO(Unit):
  do IO<Unit>:
    t0 : Nat <- IO.now()
    f : File <- IO.try(File, File.open("w.bin", "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, 0, 1048576)
    s : U32 <- loop(384n, 0, m, 0)
    t1 : Nat <- IO.now()
    IO.print("sum " ++ U32.show(s) ++ "  ms " ++ Nat.show(Nat.sub(t1, t0)))
