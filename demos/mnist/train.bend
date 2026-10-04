# MLP 784-128-10 on MNIST, trained in Bend with layers whose shapes are checked by the type.
#
#   bend demos/mnist/train.bend -- <epochs> <max batches (0 = all)> <lr>
#
# PyTorch twin: reference/mnist_torch.py (same initial weights, same batch order,
# plain SGD). Run from the repository root.

import Base
import bend-ml-tensor@0.1.2.0/main.bend as T
import bend-ml-autograd@0.1.1.0/main.bend as AG

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

def take_rows(r: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, List<&2, F32>>:
  match r:
    case 0n:
      Nil{}
    case 1n+p:
      List.take(&2, F32, xs, c) <> take_rows(p, c, List.drop(&2, F32, xs, c))

def raw_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  (f, r) = m
  do IO<List<&2, U32>>:
    bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    File.close(f)
    return bs

# reads `n` bytes starting at `off`
def read_at(path: String, off: U32, n: U32) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, off, n)
    raw_done(m)

# ---------------------------------------------------------------------
# model and data (the shapes enter the TYPES)
# ---------------------------------------------------------------------

type P is Data:
  P{w1: T.Mat<784n, 128n>, b1: T.Vec<128n>, w2: T.Mat<128n, 10n>, b2: T.Vec<10n>}

type Batch is Data:
  Batch{x: T.Mat<100n, 784n>, y: List<&2, Nat>}

# byte (0..255) -> F32 in [0, 1]
def px(+b: U32) -> F32:
  (U32.to_f32(b) / 255.0 : F32)

def px.l(bs: List<&2, U32>) -> List<&2, F32>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      px(b) <> px.l(t)

def labels.l(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> labels.l(t)

def mk_batch(mx: Maybe<&2, T.Mat<100n, 784n>>, y: List<&2, Nat>) -> Maybe<&2, Batch>:
  match mx:
    case None{}:
      None{}
    case Some{x}:
      Some{Batch{x, y}}

def make(imgs: List<&2, U32>, labs: List<&2, U32>) -> Maybe<&2, Batch>:
  mk_batch(T.Mat.of(100n, 784n, take_rows(100n, 784n, px.l(List.take(&2, U32, imgs, 78400n)))), labels.l(List.take(&2, U32, labs, 100n)))

# n batches of 100 images from `imgs` (784 bytes each) and `labs`; mb is the current batch
def batches(n: Nat, +imgs: List<&2, U32>, +labs: List<&2, U32>, mb: Maybe<&2, Batch>) -> List<&2, Batch>:
  match n mb:
    case 0n _:
      Nil{}
    case 1n+p None{}:
      Nil{}
    case 1n+p Some{b}:
      b <> batches(p, List.drop(&2, U32, imgs, 78400n), List.drop(&2, U32, labs, 100n), make(List.drop(&2, U32, imgs, 78400n), List.drop(&2, U32, labs, 100n)))

def load_batches(n: Nat, +imgs: List<&2, U32>, +labs: List<&2, U32>) -> List<&2, Batch>:
  batches(n, imgs, labs, make(imgs, labs))

# ---------------------------------------------------------------------
# one training step: forward, loss, backward and SGD
# ---------------------------------------------------------------------

def s5(+w1: T.Mat<784n, 128n>, +b1: T.Vec<128n>, +w2: T.Mat<128n, 10n>, +b2: T.Vec<10n>, +lr: F32, +x: T.Mat<100n, 784n>, loss: F32, dw2: T.Mat<128n, 10n>, db2: T.Vec<10n>, +dz1: T.Mat<100n, 128n>) -> F32 & P:
  (loss, P{AG.sgd(784n, 128n, lr, w1, T.Mat.matmul(784n, 100n, 128n, T.Mat.transpose(100n, 784n, x), dz1)), AG.sgd_vec(128n, lr, b1, T.Mat.col_sums(100n, 128n, dz1)), AG.sgd(128n, 10n, lr, w2, dw2), AG.sgd_vec(10n, lr, b2, db2)})

def s4(+w1: T.Mat<784n, 128n>, +b1: T.Vec<128n>, +w2: T.Mat<128n, 10n>, +b2: T.Vec<10n>, +lr: F32, +x: T.Mat<100n, 784n>, z1: T.Mat<100n, 128n>, loss: F32, t: T.Mat<100n, 128n> & (T.Mat<128n, 10n> & T.Vec<10n>)) -> F32 & P:
  match t:
    case Tuple{dh, r}:
      match r:
        case Tuple{dw2, db2}:
          s5(w1, b1, w2, b2, lr, x, loss, dw2, db2, AG.relu_bwd(100n, 128n, z1, dh))

def s3(+w1: T.Mat<784n, 128n>, +b1: T.Vec<128n>, +w2: T.Mat<128n, 10n>, +b2: T.Vec<10n>, +lr: F32, +x: T.Mat<100n, 784n>, +y: List<&2, Nat>, z1: T.Mat<100n, 128n>, +h: T.Mat<100n, 128n>, +z2: T.Mat<100n, 10n>) -> F32 & P:
  s4(w1, b1, w2, b2, lr, x, z1, AG.ce_loss(100n, 10n, z2, y), AG.linear_bwd(100n, 128n, 10n, h, w2, AG.ce_grad(100n, 10n, z2, y)))

def s2(+w1: T.Mat<784n, 128n>, +b1: T.Vec<128n>, +w2: T.Mat<128n, 10n>, +b2: T.Vec<10n>, +lr: F32, +x: T.Mat<100n, 784n>, y: List<&2, Nat>, +z1: T.Mat<100n, 128n>) -> F32 & P:
  s3(w1, b1, w2, b2, lr, x, y, z1, T.Mat.relu(100n, 128n, z1), AG.linear(100n, 128n, 10n, T.Mat.relu(100n, 128n, z1), w2, b2))

def train_step(p: P, b: Batch, +lr: F32) -> F32 & P:
  match p b:
    case P{+w1, +b1, +w2, +b2} Batch{+x, +y}:
      s2(w1, b1, w2, b2, lr, x, y, AG.linear(100n, 784n, 128n, x, w1, b1))

# walks the batches; r is the result of the previous step (loss, parameters)
def epoch(bs: List<&2, Batch>, +lr: F32, +acc: F32, r: F32 & P) -> F32 & P:
  match bs r:
    case Nil{} Tuple{l, p}:
      ((acc + l : F32), p)
    case Con{b, t} Tuple{+l, p}:
      epoch(t, lr, (acc + l : F32), train_step(p, b, lr))

# ---------------------------------------------------------------------
# evaluation: accuracy
# ---------------------------------------------------------------------

def gt_head(xs: List<&2, F32>, +v: F32) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Con{+h, t}:
      F32.is_gt(h, v)

def argmax.go(xs: List<&2, F32>, g: Bool, +i: Nat, +bi: Nat, +bv: F32) -> Nat:
  match xs g:
    case Nil{} _:
      bi
    case Con{+h, +t} True{}:
      argmax.go(t, gt_head(t, h), 1n+i, i, h)
    case Con{h, +t} False{}:
      argmax.go(t, gt_head(t, bv), 1n+i, bi, bv)

def argmax(xs: List<&2, F32>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{+h, +t}:
      argmax.go(t, gt_head(t, h), 1n, 0n, h)

def hit(ok: Bool) -> Nat:
  match ok:
    case True{}:
      1n
    case False{}:
      0n

def correct(rows: List<&2, List<&2, F32>>, ys: List<&2, Nat>) -> Nat:
  match rows ys:
    case Con{r, t} Con{y, u}:
      Nat.add(hit(Nat.is_eq(argmax(r), y)), correct(t, u))
    case _ _:
      0n

def predict(+p: P, +x: T.Mat<100n, 784n>) -> T.Mat<100n, 10n>:
  match p:
    case P{w1, b1, w2, b2}:
      AG.linear(100n, 128n, 10n, T.Mat.relu(100n, 128n, AG.linear(100n, 784n, 128n, x, w1, b1)), w2, b2)

def batch_ok(+p: P, b: Batch) -> Nat:
  match b:
    case Batch{x, y}:
      correct(T.Mat.to_rows(100n, 10n, predict(p, x)), y)

def eval(bs: List<&2, Batch>, +p: P, acc: Nat) -> Nat:
  match bs:
    case Nil{}:
      acc
    case Con{b, t}:
      eval(t, p, Nat.add(acc, batch_ok(p, b)))

# ---------------------------------------------------------------------
# loading the initial weights (written by reference/mnist_torch.py)
# ---------------------------------------------------------------------

def mk_p(m1: Maybe<&2, T.Mat<784n, 128n>>, m2: Maybe<&2, T.Vec<128n>>, m3: Maybe<&2, T.Mat<128n, 10n>>, m4: Maybe<&2, T.Vec<10n>>) -> Maybe<&2, P>:
  match m1 m2 m3 m4:
    case Some{a} Some{b} Some{c} Some{d}:
      Some{P{a, b, c, d}}
    case _ _ _ _:
      None{}

def load_p(w1: List<&2, U32>, b1: List<&2, U32>, w2: List<&2, U32>, b2: List<&2, U32>) -> Maybe<&2, P>:
  mk_p(T.Mat.of(784n, 128n, take_rows(784n, 128n, floats(w1))), T.Vec.of(128n, floats(b1)), T.Mat.of(128n, 10n, take_rows(128n, 10n, floats(w2))), T.Vec.of(10n, floats(b2)))

# ---------------------------------------------------------------------
# epoch loop with timing
# ---------------------------------------------------------------------

def ms_show(+ms: Nat) -> String:
  Nat.show(Nat.div(ms, 1000n)) ++ "." ++ Nat.show(Nat.div(Nat.mod(ms, 1000n), 100n))

# forces the epoch (when opening r), measures the time, prints and returns the parameters
def report(+k: Nat, +nb: Nat, t0: Nat, r: F32 & P, test: List<&2, Batch>) -> IO(P):
  match r:
    case Tuple{l, +p}:
      do IO<P>:
        t1 : Nat <- IO.now()
        IO.print("epoch " ++ Nat.show(k) ++ " train_loss=" ++ F32.show((l / F32.from_nat(nb) : F32)) ++ " test_correct=" ++ Nat.show(eval(test, p, 0n)) ++ "/10000 seconds=" ++ ms_show(Nat.sub(t1, t0)))
        return p

def run(e: Nat, +k: Nat, +nb: Nat, +lr: F32, +train: List<&2, Batch>, +test: List<&2, Batch>, p: P) -> IO(Unit):
  match e:
    case 0n:
      IO.print("done")
    case 1n+q:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        p2 : P <- report(k, nb, t0, epoch(train, lr, 0.0, (0.0, p)), test)
        run(q, 1n+k, nb, lr, train, test, p2)

def go(e: Nat, +nb: Nat, +lr: F32, mp: Maybe<&2, P>, imgs: List<&2, U32>, labs: List<&2, U32>, timgs: List<&2, U32>, tlabs: List<&2, U32>) -> IO(Unit):
  match mp:
    case None{}:
      IO.print("invalid initial weights (run reference/mnist_torch.py --make-init)")
    case Some{p}:
      run(e, 1n, nb, lr, load_batches(nb, imgs, labs), load_batches(100n, timgs, tlabs), p)

def nb2(all: Bool, m: Nat) -> Nat:
  match all:
    case True{}:
      600n
    case False{}:
      m

def nb_of(+m: Nat) -> Nat:
  nb2(Nat.is_eq(m, 0n), m)

def go3(+e: Nat, +nb: Nat, +lr: F32) -> IO(Unit):
  do IO<Unit>:
    w1 : List<&2, U32> <- read_bytes("demos/mnist/data/init/w1.txt")
    b1 : List<&2, U32> <- read_bytes("demos/mnist/data/init/b1.txt")
    w2 : List<&2, U32> <- read_bytes("demos/mnist/data/init/w2.txt")
    b2 : List<&2, U32> <- read_bytes("demos/mnist/data/init/b2.txt")
    imgs : List<&2, U32> <- read_at("demos/mnist/data/train-images-idx3-ubyte", 16, 47040000)
    labs : List<&2, U32> <- read_at("demos/mnist/data/train-labels-idx1-ubyte", 8, 60000)
    timgs : List<&2, U32> <- read_at("demos/mnist/data/t10k-images-idx3-ubyte", 16, 7840000)
    tlabs : List<&2, U32> <- read_at("demos/mnist/data/t10k-labels-idx1-ubyte", 8, 10000)
    go(e, nb, lr, load_p(w1, b1, w2, b2), imgs, labs, timgs, tlabs)

def go2(me: Maybe<&2, Nat>, mm: Maybe<&2, Nat>, ml: Maybe<&2, F32>) -> IO(Unit):
  match me mm ml:
    case Some{e} Some{m} Some{lr}:
      go3(e, nb_of(m), lr)
    case _ _ _:
      IO.print("usage: train.bend -- epochs max_batches lr")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{e, Con{m, Con{lr, Nil{}}}}}:
      go2(Nat.read(e), Nat.read(m), F32.read(lr))
    case _:
      IO.print("usage: train.bend -- epochs max_batches lr")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)

