import Base
import ./main.bend as TA

# ---- reading numbers from text ----

def rb_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  (f, r) = m
  do IO<List<&2, U32>>:
    bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    File.close(f)
    return bs

def read_bytes(path: String) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_bytes(f, 400000000)
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

def unwrap(-r: Nat, -c: Nat, m: Maybe<&1, TA.Mat<r, c>>) -> IO(TA.Mat<r, c>):
  match m:
    case None{}:
      IO.die(TA.Mat<r, c>, 1, "invalid size")
    case Some{x}:
      IO.pure(TA.Mat<r, c>, x)

# reads an r x c matrix from a text file

def load(+r: Nat, +c: Nat, path: String) -> IO(TA.Mat<r, c>):
  do IO<TA.Mat<r, c>>:
    bs : List<&2, U32> <- read_bytes(path)
    unwrap(r, c, TA.Mat.of_list(r, c, floats(bs)))

def fshow(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      F32.show(h) ++ " " ++ fshow(t)

def show2(-r: Nat, -c: Nat, x: TA.ML<r, c>) -> IO(Unit):
  match x:
    case TA.ML{m, l}:
      IO.print(fshow(l))

def show(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> IO(Unit):
  show2(r, c, TA.Mat.to_list(r, c, m))

def nat2(m: Maybe<&2, Nat>) -> Nat:
  match m:
    case Some{n}:
      n
    case None{}:
      0n

# ---- modes ----

def nat(s: String) -> Nat:
  nat2(Nat.read(s))

def r_nn2(+n: Nat, -k: Nat, +m: Nat, x: TA.MMul<n, k, m>) -> IO(Unit):
  match x:
    case TA.MMul{a, b, c}:
      show(n, m, c)

def r_nn(+n: Nat, +k: Nat, +m: Nat, +p: Nat, fa: String, fb: String) -> IO(Unit):
  do IO<Unit>:
    a : TA.Mat<n, k> <- load(n, k, fa)
    b : TA.Mat<k, m> <- load(k, m, fb)
    r_nn2(n, k, m, TA.Mat.matmul(n, k, m, p, a, b))

def r_nt2(+n: Nat, -k: Nat, +m: Nat, x: TA.MMulNT<n, k, m>) -> IO(Unit):
  match x:
    case TA.MMulNT{a, b, c}:
      show(n, m, c)

def r_nt(+n: Nat, +k: Nat, +m: Nat, +p: Nat, fa: String, fb: String) -> IO(Unit):
  do IO<Unit>:
    a : TA.Mat<n, k> <- load(n, k, fa)
    b : TA.Mat<m, k> <- load(m, k, fb)
    r_nt2(n, k, m, TA.Mat.matmul_nt(n, k, m, p, a, b))

def r_tn2(+n: Nat, -k: Nat, +m: Nat, x: TA.MMulTN<n, k, m>) -> IO(Unit):
  match x:
    case TA.MMulTN{a, b, c}:
      show(n, m, c)

def r_tn(+n: Nat, +k: Nat, +m: Nat, +p: Nat, fa: String, fb: String) -> IO(Unit):
  do IO<Unit>:
    a : TA.Mat<k, n> <- load(k, n, fa)
    b : TA.Mat<k, m> <- load(k, m, fb)
    r_tn2(n, k, m, TA.Mat.matmul_tn(n, k, m, p, a, b))

def r_bias2(+n: Nat, +m: Nat, x: TA.MBias<n, m>) -> IO(Unit):
  match x:
    case TA.MBias{c, b}:
      show(n, m, c)

def r_bias(+n: Nat, +m: Nat, fc: String, fb: String) -> IO(Unit):
  do IO<Unit>:
    c : TA.Mat<n, m> <- load(n, m, fc)
    b : TA.Mat<1n, m> <- load(1n, m, fb)
    r_bias2(n, m, TA.Mat.add_row(n, m, c, b))

def r_relu(+r: Nat, +c: Nat, fa: String) -> IO(Unit):
  do IO<Unit>:
    a : TA.Mat<r, c> <- load(r, c, fa)
    show(r, c, TA.Mat.relu(r, c, a))

def r_pair(+r: Nat, +c: Nat, x: TA.MPair<r, c>) -> IO(Unit):
  match x:
    case TA.MPair{a, b}:
      show(r, c, a)

def r_relu_bwd(+r: Nat, +c: Nat, fd: String, fh: String) -> IO(Unit):
  do IO<Unit>:
    d : TA.Mat<r, c> <- load(r, c, fd)
    h : TA.Mat<r, c> <- load(r, c, fh)
    r_pair(r, c, TA.Mat.relu_bwd(r, c, d, h))

def r_sgd(+r: Nat, +c: Nat, +lr: F32, fw: String, fd: String) -> IO(Unit):
  do IO<Unit>:
    w : TA.Mat<r, c> <- load(r, c, fw)
    d : TA.Mat<r, c> <- load(r, c, fd)
    r_pair(r, c, TA.Mat.sgd(r, c, lr, w, d))

def r_add(+r: Nat, +c: Nat, fa: String, fb: String) -> IO(Unit):
  do IO<Unit>:
    a : TA.Mat<r, c> <- load(r, c, fa)
    b : TA.Mat<r, c> <- load(r, c, fb)
    r_pair(r, c, TA.Mat.add(r, c, a, b))

def r_colsum2(-n: Nat, +m: Nat, x: TA.MColSum<n, m>) -> IO(Unit):
  match x:
    case TA.MColSum{a, s}:
      show(1n, m, s)

def r_colsum(+n: Nat, +m: Nat, fa: String) -> IO(Unit):
  do IO<Unit>:
    a : TA.Mat<n, m> <- load(n, m, fa)
    r_colsum2(n, m, TA.Mat.col_sums(n, m, a))

def labs(+bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> labs(t)

def nats_of(xs: List<&2, F32>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      F32.to_nat(h) <> nats_of(t)

def r_ce2(+n: Nat, +c: Nat, x: TA.MCE<n, c>) -> IO(Unit):
  match x:
    case TA.MCE{loss, g}:
      do IO<Unit>:
        IO.print(F32.show(loss))
        show(n, c, g)

def r_ce(+n: Nat, +c: Nat, fz: String, fy: String) -> IO(Unit):
  do IO<Unit>:
    z : TA.Mat<n, c> <- load(n, c, fz)
    ys : List<&2, U32> <- read_bytes(fy)
    r_ce2(n, c, TA.Mat.softmax_ce(n, c, z, nats_of(floats(ys))))

def r_hits2(+n: Nat, +c: Nat, x: TA.MHits<n, c>) -> IO(Unit):
  match x:
    case TA.MHits{z, k}:
      IO.print(Nat.show(k))

def r_hits(+n: Nat, +c: Nat, fz: String, fy: String) -> IO(Unit):
  do IO<Unit>:
    z : TA.Mat<n, c> <- load(n, c, fz)
    ys : List<&2, U32> <- read_bytes(fy)
    r_hits2(n, c, TA.Mat.count_correct(n, c, z, nats_of(floats(ys))))

def f32b(m: Maybe<&2, F32>) -> F32:
  match m:
    case Some{v}:
      v
    case None{}:
      0.0

def f32(s: String) -> F32:
  f32b(F32.read(s))

# ---- Bands: fill in blocks of blk numbers, then B · x, row i and the whole matrix ----

def b_fill(k: Nat, +r: Nat, +c: Nat, +blk: Nat, +i: Nat, +xs: List<&2, F32>, b: TA.Bands<r, c>) -> TA.Bands<r, c>:
  match k:
    case 0n:
      b
    case 1n+p:
      b_fill(p, r, c, blk, Nat.add(i, blk), List.drop(&2, F32, xs, blk), TA.Bands.fill_at(r, c, i, Nat.min(blk, List.length(&2, F32, xs)), List.take(&2, F32, xs, blk), b))

def b_out3(-r: Nat, -c: Nat, y: List<&2, F32>, row: List<&2, F32>, al: TA.BRow<r, c>) -> IO(Unit):
  match al:
    case TA.BRow{b, all}:
      IO.print(fshow(y) ++ fshow(row) ++ fshow(all))

def b_out2(+r: Nat, +c: Nat, y: List<&2, F32>, br: TA.BRow<r, c>) -> IO(Unit):
  match br:
    case TA.BRow{b, row}:
      b_out3(r, c, y, row, TA.Bands.to_list(r, c, b))

def b_out(+r: Nat, +c: Nat, +i: Nat, bv: TA.BV<r, c>) -> IO(Unit):
  match bv:
    case TA.BV{b, y}:
      b_out2(r, c, y, TA.Bands.read_row(r, c, b, i))

def r_bands(+d: Nat, +r: Nat, +c: Nat, +blk: Nat, +i: Nat, fa: String, fx: String) -> IO(Unit):
  do IO<Unit>:
    ba : List<&2, U32> <- read_bytes(fa)
    bx : List<&2, U32> <- read_bytes(fx)
    b_out(r, c, i, TA.Bands.matvec(r, c, b_fill(Nat.add(Nat.div(Nat.mul(r, c), blk), 1n), r, c, blk, 0n, floats(ba), TA.Bands.zeros(d, r, c)), floats(bx)))

def dispatch2(a1: Bool, a2: Bool, a3: Bool, a4: Bool, a5: Bool, a6: Bool, a7: Bool, a8: Bool, a9: Bool, a10: Bool, a11: Bool, args: List<String>) -> IO(Unit):
  match a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 args:
    case True{} _ _ _ _ _ _ _ _ _ _ Con{p, Con{n, Con{k, Con{m, Con{fa, Con{fb, Nil{}}}}}}}:
      r_nn(nat(n), nat(k), nat(m), nat(p), fa, fb)
    case False{} True{} _ _ _ _ _ _ _ _ _ Con{p, Con{n, Con{k, Con{m, Con{fa, Con{fb, Nil{}}}}}}}:
      r_nt(nat(n), nat(k), nat(m), nat(p), fa, fb)
    case False{} False{} True{} _ _ _ _ _ _ _ _ Con{p, Con{n, Con{k, Con{m, Con{fa, Con{fb, Nil{}}}}}}}:
      r_tn(nat(n), nat(k), nat(m), nat(p), fa, fb)
    case False{} False{} False{} True{} _ _ _ _ _ _ _ Con{n, Con{m, Con{fc, Con{fb, Nil{}}}}}:
      r_bias(nat(n), nat(m), fc, fb)
    case False{} False{} False{} False{} True{} _ _ _ _ _ _ Con{r, Con{c, Con{fa, Nil{}}}}:
      r_relu(nat(r), nat(c), fa)
    case False{} False{} False{} False{} False{} True{} _ _ _ _ _ Con{r, Con{c, Con{fd, Con{fh, Nil{}}}}}:
      r_relu_bwd(nat(r), nat(c), fd, fh)
    case False{} False{} False{} False{} False{} False{} True{} _ _ _ _ Con{r, Con{c, Con{lr, Con{fw, Con{fd, Nil{}}}}}}:
      r_sgd(nat(r), nat(c), f32(lr), fw, fd)
    case False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ Con{r, Con{c, Con{fa, Con{fb, Nil{}}}}}:
      r_add(nat(r), nat(c), fa, fb)
    case False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ Con{n, Con{m, Con{fa, Nil{}}}}:
      r_colsum(nat(n), nat(m), fa)
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ Con{n, Con{c, Con{fz, Con{fy, Nil{}}}}}:
      r_ce(nat(n), nat(c), fz, fy)
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} Con{n, Con{c, Con{fz, Con{fy, Nil{}}}}}:
      r_hits(nat(n), nat(c), fz, fy)
    case _ _ _ _ _ _ _ _ _ _ _ _:
      IO.print("invalid usage")

def dispatch1(isb: Bool, +mode: String, rest: List<String>) -> IO(Unit):
  match isb rest:
    case True{} Con{d, Con{r, Con{c, Con{blk, Con{i, Con{fa, Con{fx, Nil{}}}}}}}}:
      r_bands(nat(d), nat(r), nat(c), nat(blk), nat(i), fa, fx)
    case True{} _:
      IO.print("invalid usage")
    case False{} xs:
      dispatch2(String.eq(mode, "nn"), String.eq(mode, "nt"), String.eq(mode, "tn"), String.eq(mode, "bias"), String.eq(mode, "relu"), String.eq(mode, "relu_bwd"), String.eq(mode, "sgd"), String.eq(mode, "add"), String.eq(mode, "colsum"), String.eq(mode, "ce"), String.eq(mode, "hits"), xs)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, rest}}:
      dispatch1(String.eq(mode, "bands"), mode, rest)
    case _:
      IO.print("invalid usage")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
