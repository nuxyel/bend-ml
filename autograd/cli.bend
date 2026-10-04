import Base
import ./main.bend as AG
import bend-ml-tensor@0.1.0.0/main.bend as T

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

def take_rows(r: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, List<&2, F32>>:
  match r:
    case 0n:
      Nil{}
    case 1n+p:
      List.take(&2, F32, xs, c) <> take_rows(p, c, List.drop(&2, F32, xs, c))

# ---- saída ----
def fshow(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      F32.show(h) ++ " " ++ fshow(t)

def rshow(rs: List<&2, List<&2, F32>>) -> String:
  match rs:
    case Nil{}:
      ""
    case Con{r, t}:
      fshow(r) ++ rshow(t)

def nats(xs: List<&2, F32>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      F32.to_nat(h) <> nats(t)

# ---- escalares: expressões fixas, em 3 variáveis ----
def x0() -> AG.G:
  AG.GVar{0n}
def x1() -> AG.G:
  AG.GVar{1n}
def x2() -> AG.G:
  AG.GVar{2n}

def expr(k: Nat) -> AG.G:
  match k:
    case 1n:
      AG.GAdd{AG.GMul{x0(), x1()}, x0()}
    case 2n:
      AG.GRelu{AG.GAdd{AG.GMul{x0(), x1()}, AG.GMul{AG.GCst{(0.0 - 1.0 : F32)}, x2()}}}
    case 3n:
      AG.GTanh{AG.GAdd{AG.GMul{x0(), x1()}, x2()}}
    case 4n:
      AG.GAdd{AG.GMul{AG.GExp{x0()}, AG.GTanh{x1()}}, AG.GMul{x2(), x2()}}
    case _:
      AG.GTanh{AG.GAdd{AG.GRelu{AG.GMul{x0(), x1()}}, AG.GExp{AG.GMul{x2(), x0()}}}}

def run_scalar(+e: AG.G, +env: List<&2, F32>) -> IO(Unit):
  IO.print(F32.show(AG.gval(e, env)) ++ " " ++ fshow(AG.ggrad(e, env)))

def scalar4(mk: Maybe<&2, Nat>, a: Maybe<&2, F32>, b: Maybe<&2, F32>, c: Maybe<&2, F32>) -> IO(Unit):
  match mk a b c:
    case Some{k} Some{u} Some{v} Some{w}:
      run_scalar(expr(k), [u, v, w])
    case _ _ _ _:
      IO.print("argumentos invalidos")

# ---- camadas: as dimensões vêm da linha de comando e entram nos tipos ----
def lin_show(n: Nat, i: Nat, o: Nat, t: T.Mat<n, i> & (T.Mat<i, o> & T.Vec<o>)) -> IO(Unit):
  match t:
    case Tuple{dx, rest}:
      match rest:
        case Tuple{dw, db}:
          IO.print(rshow(T.Mat.to_rows(n, i, dx)) ++ rshow(T.Mat.to_rows(i, o, dw)) ++ fshow(T.Vec.to_list(o, db)))

def lin_run(+n: Nat, +i: Nat, +o: Nat, mx: Maybe<&2, T.Mat<n, i>>, mw: Maybe<&2, T.Mat<i, o>>, md: Maybe<&2, T.Mat<n, o>>) -> IO(Unit):
  match mx mw md:
    case Some{x} Some{w} Some{dy}:
      lin_show(n, i, o, AG.linear_bwd(n, i, o, x, w, dy))
    case _ _ _:
      IO.print("tamanho invalido")

def do_linear(+n: Nat, +i: Nat, +o: Nat, fx: List<&2, U32>, fw: List<&2, U32>, fd: List<&2, U32>) -> IO(Unit):
  lin_run(n, i, o, T.Mat.of(n, i, take_rows(n, i, floats(fx))), T.Mat.of(i, o, take_rows(i, o, floats(fw))), T.Mat.of(n, o, take_rows(n, o, floats(fd))))

def relu_run(+r: Nat, +c: Nat, mx: Maybe<&2, T.Mat<r, c>>, md: Maybe<&2, T.Mat<r, c>>) -> IO(Unit):
  match mx md:
    case Some{x} Some{d}:
      IO.print(rshow(T.Mat.to_rows(r, c, AG.relu_bwd(r, c, x, d))))
    case _ _:
      IO.print("tamanho invalido")

def do_relu(+r: Nat, +c: Nat, fx: List<&2, U32>, fd: List<&2, U32>) -> IO(Unit):
  relu_run(r, c, T.Mat.of(r, c, take_rows(r, c, floats(fx))), T.Mat.of(r, c, take_rows(r, c, floats(fd))))

def ce_run(+n: Nat, +c: Nat, ml: Maybe<&2, T.Mat<n, c>>, +labels: List<&2, Nat>) -> IO(Unit):
  match ml:
    case None{}:
      IO.print("tamanho invalido")
    case Some{+l}:
      IO.print(F32.show(AG.ce_loss(n, c, l, labels)) ++ " " ++ rshow(T.Mat.to_rows(n, c, AG.ce_grad(n, c, l, labels))))

def do_ce(+n: Nat, +c: Nat, fl: List<&2, U32>, fy: List<&2, U32>) -> IO(Unit):
  ce_run(n, c, T.Mat.of(n, c, take_rows(n, c, floats(fl))), nats(floats(fy)))

# ---- argumentos ----
def go_scalar(k: String, a: String, b: String, c: String) -> IO(Unit):
  scalar4(Nat.read(k), F32.read(a), F32.read(b), F32.read(c))

def go_linear(mn: Maybe<&2, Nat>, mi: Maybe<&2, Nat>, mo: Maybe<&2, Nat>, fx: String, fw: String, fd: String) -> IO(Unit):
  match mn mi mo:
    case Some{n} Some{i} Some{o}:
      do IO<Unit>:
        x : List<&2, U32> <- read_bytes(fx)
        w : List<&2, U32> <- read_bytes(fw)
        d : List<&2, U32> <- read_bytes(fd)
        do_linear(n, i, o, x, w, d)
    case _ _ _:
      IO.print("dimensoes invalidas")

def go_relu(mr: Maybe<&2, Nat>, mc: Maybe<&2, Nat>, fx: String, fd: String) -> IO(Unit):
  match mr mc:
    case Some{r} Some{c}:
      do IO<Unit>:
        x : List<&2, U32> <- read_bytes(fx)
        d : List<&2, U32> <- read_bytes(fd)
        do_relu(r, c, x, d)
    case _ _:
      IO.print("dimensoes invalidas")

def go_ce(mn: Maybe<&2, Nat>, mc: Maybe<&2, Nat>, fl: String, fy: String) -> IO(Unit):
  match mn mc:
    case Some{n} Some{c}:
      do IO<Unit>:
        l : List<&2, U32> <- read_bytes(fl)
        y : List<&2, U32> <- read_bytes(fy)
        do_ce(n, c, l, y)
    case _ _:
      IO.print("dimensoes invalidas")

def dispatch2(m1: Bool, m2: Bool, m3: Bool, m4: Bool, args: List<String>) -> IO(Unit):
  match m1 m2 m3 m4 args:
    case True{} _ _ _ Con{k, Con{a, Con{b, Con{c, Nil{}}}}}:
      go_scalar(k, a, b, c)
    case False{} True{} _ _ Con{n, Con{i, Con{o, Con{fx, Con{fw, Con{fd, Nil{}}}}}}}:
      go_linear(Nat.read(n), Nat.read(i), Nat.read(o), fx, fw, fd)
    case False{} False{} True{} _ Con{r, Con{c, Con{fx, Con{fd, Nil{}}}}}:
      go_relu(Nat.read(r), Nat.read(c), fx, fd)
    case False{} False{} False{} True{} Con{n, Con{c, Con{fl, Con{fy, Nil{}}}}}:
      go_ce(Nat.read(n), Nat.read(c), fl, fy)
    case _ _ _ _ _:
      IO.print("uso invalido")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, rest}}:
      dispatch2(String.eq(mode, "scalar"), String.eq(mode, "linear_bwd"), String.eq(mode, "relu_bwd"), String.eq(mode, "ce"), rest)
    case _:
      IO.print("uso invalido")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
