import Base
import bend-ml-tensor@0.1.2.0/main.bend as T

# 4 little-endian bytes -> F32 (U32 and F32 hold the same Word(32n): it is a bitcast)
def bitcast(u: U32) -> F32:
  match u:
    case U32{w}:
      F32{w}

def f32_le(+b0: U32, +b1: U32, +b2: U32, +b3: U32) -> F32:
  bitcast((b0 + (b1 * 256 : U32) + (b2 * 65536 : U32) + (b3 * 16777216 : U32) : U32))

def f32s(bs: List<&2, U32>) -> List<&2, F32>:
  match bs:
    case Con{b0, Con{b1, Con{b2, Con{b3, t}}}}:
      f32_le(b0, b1, b2, b3) <> f32s(t)
    case _:
      Nil{}

# reads the file in blocks of `blk` bytes; k blocks remaining; m is the block already read
def rf_loop(k: Nat, +off: U32, +blk: U32, m: File & Result<&1, &1, U32 & String, List<&2, U32>>, acc: List<&2, List<&2, F32>>) -> IO(List<&2, F32>):
  match k m:
    case 0n Tuple{f, r}:
      do IO<List<&2, F32>>:
        File.close(f)
        return List.concat(&2, F32, List.reverse(&2, List<&2, F32>, acc))
    case 1n+p Tuple{f, r}:
      do IO<List<&2, F32>>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
        m2 : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, (off + blk : U32), blk)
        rf_loop(p, (off + blk : U32), blk, m2, f32s(bs) <> acc)

def blocks_of(n: Nat, +per: Nat) -> Nat:
  Nat.div(Nat.add(n, Nat.sub(per, 1n)), per)

# n F32 numbers (little-endian) from a file
def read_floats(path: String, +n: Nat) -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, 0, 1048576)
    fl : List<&2, F32> <- rf_loop(blocks_of(Nat.mul(n, 4n), 1048576n), 0, 1048576, m, Nil{})
    return List.take(&2, F32, fl, n)

def take_rows(r: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, List<&2, F32>>:
  match r:
    case 0n:
      Nil{}
    case 1n+p:
      List.take(&2, F32, xs, c) <> take_rows(p, c, List.drop(&2, F32, xs, c))

def unwrap_mat(-r: Nat, -c: Nat, m: Maybe<&2, T.Mat<r, c>>) -> IO(T.Mat<r, c>):
  match m:
    case None{}:
      IO.die(T.Mat<r, c>, 1, "invalid matrix size")
    case Some{x}:
      IO.pure(T.Mat<r, c>, x)

def unwrap_vec(-n: Nat, m: Maybe<&2, T.Vec<n>>) -> IO(T.Vec<n>):
  match m:
    case None{}:
      IO.die(T.Vec<n>, 1, "invalid vector size")
    case Some{x}:
      IO.pure(T.Vec<n>, x)

def get_mat(name: String, +r: Nat, +c: Nat) -> IO(T.Mat<r, c>):
  do IO<T.Mat<r, c>>:
    fl : List<&2, F32> <- read_floats("demos/gpt2/data/w/" ++ name ++ ".bin", Nat.mul(r, c))
    unwrap_mat(r, c, T.Mat.of(r, c, take_rows(r, c, fl)))

def get_vec(name: String, +n: Nat) -> IO(T.Vec<n>):
  do IO<T.Vec<n>>:
    fl : List<&2, F32> <- read_floats("demos/gpt2/data/w/" ++ name ++ ".bin", n)
    unwrap_vec(n, T.Vec.of(n, fl))

def show_f(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      F32.show(h) ++ " " ++ show_f(t)

def first5(m: Maybe<&2, List<&2, F32>>) -> String:
  match m:
    case None{}:
      "nenhuma"
    case Some{r}:
      show_f(List.take(&2, F32, r, 5n))

def main() -> IO(Unit):
  do IO<Unit>:
    g : T.Vec<768n> <- get_vec("lnf_g", 768n)
    IO.print(show_f(List.take(&2, F32, T.Vec.to_list(768n, g), 5n)))
    w : T.Mat<2304n, 768n> <- get_mat("h0.aw", 2304n, 768n)
    IO.print(first5(List.get(&2, List<&2, F32>, T.Mat.to_rows(2304n, 768n, w), 2303n)))
