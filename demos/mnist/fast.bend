# MLP 784-128-10 no MNIST com o pacote bend-ml-tensor-array: cada produto, cada
# gradiente e cada atualização tem o formato conferido pelo tipo.
#
#   bend demos/mnist/fast.bend -o /tmp/mnist_fast && /tmp/mnist_fast <épocas> <máx. lotes (0 = todos)> <lr>
#
# Mesmos dados, pesos iniciais, ordem de lotes e lr do train.bend e do reference/mnist_torch.py.

import Base
import bend-ml-tensor-array@0.1.1.0/main.bend as TA

# ---- leitura de números em texto ----

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

def raw_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  (f, r) = m
  do IO<List<&2, U32>>:
    bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    File.close(f)
    return bs

# lê `n` bytes a partir de `off`

def read_at(path: String, off: U32, n: U32) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, off, n)
    raw_done(m)



# ---------------------------------------------------------------------
# parâmetros e passo de treino (lote de 100)
# ---------------------------------------------------------------------

type Pm is Type:
  Pm{w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>}

type St2 is Type:
  St2{loss: F32, pm: Pm}

# 4. SGD nos quatro parâmetros

def u4(loss: F32, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, r: TA.MPair<1n, 10n>) -> St2:
  match r:
    case TA.MPair{b2, db2}:
      St2{loss, Pm{w1, b1, w2, b2}}

def u3(+lr: F32, loss: F32, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, b2: TA.Mat<1n, 10n>, db2: TA.Mat<1n, 10n>, r: TA.MPair<128n, 10n>) -> St2:
  match r:
    case TA.MPair{w2, dw2}:
      u4(loss, w1, b1, w2, TA.Mat.sgd(1n, 10n, lr, b2, db2))

def u2(+lr: F32, loss: F32, w1: TA.Mat<784n, 128n>, w2: TA.Mat<128n, 10n>, dw2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, db2: TA.Mat<1n, 10n>, r: TA.MPair<1n, 128n>) -> St2:
  match r:
    case TA.MPair{b1, db1}:
      u3(lr, loss, w1, b1, b2, db2, TA.Mat.sgd(128n, 10n, lr, w2, dw2))

def u1(+lr: F32, loss: F32, b1: TA.Mat<1n, 128n>, db1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, dw2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, db2: TA.Mat<1n, 10n>, r: TA.MPair<784n, 128n>) -> St2:
  match r:
    case TA.MPair{w1, dw1}:
      u2(lr, loss, w1, w2, dw2, b2, db2, TA.Mat.sgd(1n, 128n, lr, b1, db1))

# 3. gradiente da primeira camada: dW1 = Xᵀ · dZ1 (784 x 128), db1 = soma das linhas de dZ1

def s13(+lr: F32, loss: F32, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, dw2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, db2: TA.Mat<1n, 10n>, dw1: TA.Mat<784n, 128n>, r: TA.MColSum<100n, 128n>) -> St2:
  match r:
    case TA.MColSum{dz1, db1}:
      u1(lr, loss, b1, db1, w2, dw2, b2, db2, TA.Mat.sgd(784n, 128n, lr, w1, dw1))

def s12(+lr: F32, loss: F32, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, dw2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, db2: TA.Mat<1n, 10n>, r: TA.MMulTN<784n, 100n, 128n>) -> St2:
  match r:
    case TA.MMulTN{x, dz1, dw1}:
      s13(lr, loss, w1, b1, w2, dw2, b2, db2, dw1, TA.Mat.col_sums(100n, 128n, dz1))

def s11(+lr: F32, loss: F32, x: TA.Mat<100n, 784n>, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, dw2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, db2: TA.Mat<1n, 10n>, r: TA.MPair<100n, 128n>) -> St2:
  match r:
    case TA.MPair{dz1, h}:
      s12(lr, loss, w1, b1, w2, dw2, b2, db2, TA.Mat.matmul_tn(784n, 100n, 128n, 3n, x, dz1))

# 2. gradientes da segunda camada e da relu

def s10(+lr: F32, loss: F32, x: TA.Mat<100n, 784n>, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, dw2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, h: TA.Mat<100n, 128n>, dh: TA.Mat<100n, 128n>, r: TA.MColSum<100n, 10n>) -> St2:
  match r:
    case TA.MColSum{g2, db2}:
      s11(lr, loss, x, w1, b1, w2, dw2, b2, db2, TA.Mat.relu_bwd(100n, 128n, dh, h))

def s9(+lr: F32, loss: F32, x: TA.Mat<100n, 784n>, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, dw2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, h: TA.Mat<100n, 128n>, r: TA.MMulNT<100n, 10n, 128n>) -> St2:
  match r:
    case TA.MMulNT{g2, w2, dh}:
      s10(lr, loss, x, w1, b1, w2, dw2, b2, h, dh, TA.Mat.col_sums(100n, 10n, g2))

def s8(+lr: F32, loss: F32, x: TA.Mat<100n, 784n>, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, r: TA.MMulTN<128n, 100n, 10n>) -> St2:
  match r:
    case TA.MMulTN{h, g2, dw2}:
      s9(lr, loss, x, w1, b1, dw2, b2, h, TA.Mat.matmul_nt(100n, 10n, 128n, 0n, g2, w2))

# 1. forward da segunda camada e perda

def s7(+lr: F32, x: TA.Mat<100n, 784n>, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, h: TA.Mat<100n, 128n>, w2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, r: TA.MCE<100n, 10n>) -> St2:
  match r:
    case TA.MCE{loss, g2}:
      s8(lr, loss, x, w1, b1, w2, b2, TA.Mat.matmul_tn(128n, 100n, 10n, 0n, h, g2))

def s6(+lr: F32, x: TA.Mat<100n, 784n>, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, h: TA.Mat<100n, 128n>, w2: TA.Mat<128n, 10n>, labels: List<&2, Nat>, r: TA.MBias<100n, 10n>) -> St2:
  match r:
    case TA.MBias{z2, b2}:
      s7(lr, x, w1, b1, h, w2, b2, TA.Mat.softmax_ce(100n, 10n, z2, labels))

def s5(+lr: F32, x: TA.Mat<100n, 784n>, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, b2: TA.Mat<1n, 10n>, labels: List<&2, Nat>, r: TA.MMul<100n, 128n, 10n>) -> St2:
  match r:
    case TA.MMul{h, w2, z2}:
      s6(lr, x, w1, b1, h, w2, labels, TA.Mat.add_row(100n, 10n, z2, b2))

def s4(+lr: F32, x: TA.Mat<100n, 784n>, w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, labels: List<&2, Nat>, h: TA.Mat<100n, 128n>) -> St2:
  s5(lr, x, w1, b1, b2, labels, TA.Mat.matmul(100n, 128n, 10n, 0n, h, w2))

# 0. forward da primeira camada

def s3(+lr: F32, x: TA.Mat<100n, 784n>, w1: TA.Mat<784n, 128n>, w2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, labels: List<&2, Nat>, r: TA.MBias<100n, 128n>) -> St2:
  match r:
    case TA.MBias{z1, b1}:
      s4(lr, x, w1, b1, w2, b2, labels, TA.Mat.relu(100n, 128n, z1))

def s2(+lr: F32, w2: TA.Mat<128n, 10n>, b1: TA.Mat<1n, 128n>, b2: TA.Mat<1n, 10n>, labels: List<&2, Nat>, r: TA.MMul<100n, 784n, 128n>) -> St2:
  match r:
    case TA.MMul{x, w1, z1}:
      s3(lr, x, w1, w2, b2, labels, TA.Mat.add_row(100n, 128n, z1, b1))

def step(pm: Pm, x: TA.Mat<100n, 784n>, labels: List<&2, Nat>, +lr: F32) -> St2:
  match pm:
    case Pm{w1, b1, w2, b2}:
      s2(lr, w2, b1, b2, labels, TA.Mat.matmul(100n, 784n, 128n, 3n, x, w1))

# ---------------------------------------------------------------------
# avaliação: acertos em um lote (só forward)
# ---------------------------------------------------------------------

type Cn is Type:
  Cn{n: Nat, pm: Pm}

def e5(w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, r: TA.MHits<100n, 10n>) -> Cn:
  match r:
    case TA.MHits{z, k}:
      Cn{k, Pm{w1, b1, w2, b2}}

def e4(w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, w2: TA.Mat<128n, 10n>, labels: List<&2, Nat>, r: TA.MBias<100n, 10n>) -> Cn:
  match r:
    case TA.MBias{z2, b2}:
      e5(w1, b1, w2, b2, TA.Mat.count_correct(100n, 10n, z2, labels))

def e3(w1: TA.Mat<784n, 128n>, b1: TA.Mat<1n, 128n>, b2: TA.Mat<1n, 10n>, labels: List<&2, Nat>, r: TA.MMul<100n, 128n, 10n>) -> Cn:
  match r:
    case TA.MMul{h, w2, z2}:
      e4(w1, b1, w2, labels, TA.Mat.add_row(100n, 10n, z2, b2))

def e2(w1: TA.Mat<784n, 128n>, w2: TA.Mat<128n, 10n>, b2: TA.Mat<1n, 10n>, labels: List<&2, Nat>, r: TA.MBias<100n, 128n>) -> Cn:
  match r:
    case TA.MBias{z1, b1}:
      e3(w1, b1, b2, labels, TA.Mat.matmul(100n, 128n, 10n, 0n, TA.Mat.relu(100n, 128n, z1), w2))

def e1(w2: TA.Mat<128n, 10n>, b1: TA.Mat<1n, 128n>, b2: TA.Mat<1n, 10n>, labels: List<&2, Nat>, r: TA.MMul<100n, 784n, 128n>) -> Cn:
  match r:
    case TA.MMul{x, w1, z1}:
      e2(w1, w2, b2, labels, TA.Mat.add_row(100n, 128n, z1, b1))

def evb(pm: Pm, x: TA.Mat<100n, 784n>, labels: List<&2, Nat>) -> Cn:
  match pm:
    case Pm{w1, b1, w2, b2}:
      e1(w2, b1, b2, labels, TA.Mat.matmul(100n, 784n, 128n, 3n, x, w1))

# ---------------------------------------------------------------------
# dados e laço de épocas
# ---------------------------------------------------------------------

def scale255(bs: List<&2, U32>) -> List<&2, F32>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      (U32.to_f32(b) / 255.0 : F32) <> scale255(t)

def labs_nat(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> labs_nat(t)

def unwrap_x(m: Maybe<&1, TA.Mat<100n, 784n>>) -> IO(TA.Mat<100n, 784n>):
  match m:
    case None{}:
      IO.die(TA.Mat<100n, 784n>, 1, "lote invalido")
    case Some{x}:
      IO.pure(TA.Mat<100n, 784n>, x)

def load_batch(img: String, +off: U32) -> IO(TA.Mat<100n, 784n>):
  do IO<TA.Mat<100n, 784n>>:
    xs : List<&2, U32> <- read_at(img, off, 78400)
    unwrap_x(TA.Mat.of_list(100n, 784n, scale255(xs)))

def tloop(k: Nat, +b: U32, +lr: F32, acc: F32, r: St2) -> IO(St2):
  match k r:
    case 0n St2{l, pm}:
      IO.pure(St2, St2{(acc + l : F32), pm})
    case 1n+p St2{l, pm}:
      do IO<St2>:
        x : TA.Mat<100n, 784n> <- load_batch("demos/mnist/data/train-images-idx3-ubyte", (16 + (b * 784 * 100 : U32) : U32))
        ys : List<&2, U32> <- read_at("demos/mnist/data/train-labels-idx1-ubyte", (8 + (b * 100 : U32) : U32), 100)
        tloop(p, (b + 1 : U32), lr, (acc + l : F32), step(pm, x, labs_nat(ys), lr))

def eloop(k: Nat, +b: U32, acc: Nat, r: Cn) -> IO(Cn):
  match k r:
    case 0n Cn{n, pm}:
      IO.pure(Cn, Cn{Nat.add(acc, n), pm})
    case 1n+p Cn{n, pm}:
      do IO<Cn>:
        x : TA.Mat<100n, 784n> <- load_batch("demos/mnist/data/t10k-images-idx3-ubyte", (16 + (b * 784 * 100 : U32) : U32))
        ys : List<&2, U32> <- read_at("demos/mnist/data/t10k-labels-idx1-ubyte", (8 + (b * 100 : U32) : U32), 100)
        eloop(p, (b + 1 : U32), Nat.add(acc, n), evb(pm, x, labs_nat(ys)))

def ms_show(+ms: Nat) -> String:
  Nat.show(Nat.div(ms, 1000n)) ++ "." ++ Nat.show(Nat.div(Nat.mod(ms, 1000n), 100n))

def report(+k: Nat, +nb: Nat, t0: Nat, t1: Nat, loss: F32, c: Cn) -> IO(Pm):
  match c:
    case Cn{n, pm}:
      do IO<Pm>:
        IO.print("epoca " ++ Nat.show(k) ++ " loss_treino=" ++ F32.show((loss / F32.from_nat(nb) : F32)) ++ " acertos_teste=" ++ Nat.show(n) ++ "/10000 segundos_treino=" ++ ms_show(Nat.sub(t1, t0)))
        return pm

def epoch_end(+k: Nat, +nb: Nat, t0: Nat, t1: Nat, r: St2) -> IO(Pm):
  match r:
    case St2{loss, pm}:
      do IO<Pm>:
        c : Cn <- eloop(100n, 0, 0n, Cn{0n, pm})
        report(k, nb, t0, t1, loss, c)

def epochs(e: Nat, +k: Nat, +nb: Nat, +lr: F32, pm: Pm) -> IO(Unit):
  match e:
    case 0n:
      IO.print("fim")
    case 1n++q:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        r : St2 <- tloop(nb, 0, lr, 0.0, St2{0.0, pm})
        t1 : Nat <- IO.now()
        pm2 : Pm <- epoch_end(k, nb, t0, t1, r)
        epochs(q, 1n+k, nb, lr, pm2)

def unwrap(-r: Nat, -c: Nat, m: Maybe<&1, TA.Mat<r, c>>) -> IO(TA.Mat<r, c>):
  match m:
    case None{}:
      IO.die(TA.Mat<r, c>, 1, "pesos iniciais com tamanho invalido")
    case Some{x}:
      IO.pure(TA.Mat<r, c>, x)

def load_mat(+r: Nat, +c: Nat, path: String) -> IO(TA.Mat<r, c>):
  do IO<TA.Mat<r, c>>:
    bs : List<&2, U32> <- read_bytes(path)
    unwrap(r, c, TA.Mat.of_list(r, c, floats(bs)))

def go3(+e: Nat, +nb: Nat, +lr: F32) -> IO(Unit):
  do IO<Unit>:
    w1 : TA.Mat<784n, 128n> <- load_mat(784n, 128n, "demos/mnist/data/init/w1.txt")
    b1 : TA.Mat<1n, 128n> <- load_mat(1n, 128n, "demos/mnist/data/init/b1.txt")
    w2 : TA.Mat<128n, 10n> <- load_mat(128n, 10n, "demos/mnist/data/init/w2.txt")
    b2 : TA.Mat<1n, 10n> <- load_mat(1n, 10n, "demos/mnist/data/init/b2.txt")
    epochs(e, 1n, nb, lr, Pm{w1, b1, w2, b2})

def nb2(all: Bool, m: Nat) -> Nat:
  match all:
    case True{}:
      600n
    case False{}:
      m

def nb_of(+m: Nat) -> Nat:
  nb2(Nat.is_eq(m, 0n), m)

def go2(me: Maybe<&2, Nat>, mm: Maybe<&2, Nat>, ml: Maybe<&2, F32>) -> IO(Unit):
  match me mm ml:
    case Some{e} Some{m} Some{lr}:
      go3(e, nb_of(m), lr)
    case _ _ _:
      IO.print("uso: fast.bend -- epocas max_lotes lr")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{e, Con{m, Con{lr, Nil{}}}}}:
      go2(Nat.read(e), Nat.read(m), F32.read(lr))
    case _:
      IO.print("uso: fast.bend -- epocas max_lotes lr")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
