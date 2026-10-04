import Base
import ./main.bend as T

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

def show_mat(-r: Nat, -c: Nat, m: Maybe<&2, T.Mat<r, c>>) -> IO(Unit):
  match m:
    case None{}:
      IO.print("erro: tamanho")
    case Some{x}:
      IO.print(rshow(T.Mat.to_rows(r, c, x)))

# ---- operações (as dimensões vêm da linha de comando e entram nos TIPOS) ----
def mm(n: Nat, +k: Nat, +m: Nat, ma: Maybe<&2, T.Mat<n, k>>, mb: Maybe<&2, T.Mat<k, m>>) -> Maybe<&2, T.Mat<n, m>>:
  match ma mb:
    case Some{a} Some{b}:
      Some{T.Mat.matmul(n, k, m, a, b)}
    case _ _:
      None{}

def op1(r: Nat, c: Nat, which: Nat, mx: Maybe<&2, T.Mat<r, c>>) -> IO(Unit):
  match which mx:
    case _ None{}:
      IO.print("erro: tamanho")
    case 0n Some{x}:
      show_mat(r, c, Some{T.Mat.relu(r, c, x)})
    case 1n Some{x}:
      show_mat(r, c, Some{T.Mat.softmax(r, c, x)})
    case 2n Some{x}:
      show_mat(r, c, Some{T.Mat.gelu(r, c, x)})
    case 1n+p Some{x}:
      show_mat(c, r, Some{T.Mat.transpose(r, c, x)})

def do_unary(+r: Nat, +c: Nat, which: Nat, fa: List<&2, U32>) -> IO(Unit):
  op1(r, c, which, T.Mat.of(r, c, take_rows(r, c, floats(fa))))

def do_matmul(+n: Nat, +k: Nat, +m: Nat, fa: List<&2, U32>, fb: List<&2, U32>) -> IO(Unit):
  show_mat(n, m, mm(n, k, m, T.Mat.of(n, k, take_rows(n, k, floats(fa))), T.Mat.of(k, m, take_rows(k, m, floats(fb)))))

# matmul_t: B chega como (m x k) já transposta (Bt = Bᵀ)
def mm_t(n: Nat, k: Nat, m: Nat, ma: Maybe<&2, T.Mat<n, k>>, mb: Maybe<&2, T.Mat<m, k>>) -> Maybe<&2, T.Mat<n, m>>:
  match ma mb:
    case Some{a} Some{b}:
      Some{T.Mat.matmul_t(n, k, m, a, b)}
    case _ _:
      None{}

def do_matmul_t(+n: Nat, +k: Nat, +m: Nat, fa: List<&2, U32>, fb: List<&2, U32>) -> IO(Unit):
  show_mat(n, m, mm_t(n, k, m, T.Mat.of(n, k, take_rows(n, k, floats(fa))), T.Mat.of(m, k, take_rows(m, k, floats(fb)))))

def ln_run(n: Nat, d: Nat, eps: F32, mx: Maybe<&2, T.Mat<n, d>>, mg: Maybe<&2, T.Vec<d>>, mb: Maybe<&2, T.Vec<d>>) -> IO(Unit):
  match mx mg mb:
    case Some{x} Some{g} Some{b}:
      show_mat(n, d, Some{T.Mat.layernorm(n, d, eps, g, b, x)})
    case _ _ _:
      IO.print("erro: tamanho")

def do_layernorm(+n: Nat, +d: Nat, eps: F32, fx: List<&2, U32>, fg: List<&2, U32>, fb: List<&2, U32>) -> IO(Unit):
  ln_run(n, d, eps, T.Mat.of(n, d, take_rows(n, d, floats(fx))), T.Vec.of(d, floats(fg)), T.Vec.of(d, floats(fb)))

# ---- argumentos ----
def which2(is_relu: Bool, is_soft: Bool, is_gelu: Bool) -> Nat:
  match is_relu is_soft is_gelu:
    case True{} _ _:
      0n
    case False{} True{} _:
      1n
    case False{} False{} True{}:
      2n
    case False{} False{} False{}:
      3n

def which_of(+a: String) -> Nat:
  which2(String.eq(a, "relu"), String.eq(a, "softmax"), String.eq(a, "gelu"))

def run_unary(mr: Maybe<&2, Nat>, mc: Maybe<&2, Nat>, which: Nat, f: String) -> IO(Unit):
  match mr mc:
    case Some{r} Some{c}:
      do IO<Unit>:
        fa : List<&2, U32> <- read_bytes(f)
        do_unary(r, c, which, fa)
    case _ _:
      IO.print("dimensoes invalidas")

def run_matmul_t(mn: Maybe<&2, Nat>, mk: Maybe<&2, Nat>, mm_: Maybe<&2, Nat>, fa: String, fb: String) -> IO(Unit):
  match mn mk mm_:
    case Some{n} Some{k} Some{m}:
      do IO<Unit>:
        a : List<&2, U32> <- read_bytes(fa)
        b : List<&2, U32> <- read_bytes(fb)
        do_matmul_t(n, k, m, a, b)
    case _ _ _:
      IO.print("dimensoes invalidas")

def run_matmul(mn: Maybe<&2, Nat>, mk: Maybe<&2, Nat>, mm_: Maybe<&2, Nat>, fa: String, fb: String) -> IO(Unit):
  match mn mk mm_:
    case Some{n} Some{k} Some{m}:
      do IO<Unit>:
        a : List<&2, U32> <- read_bytes(fa)
        b : List<&2, U32> <- read_bytes(fb)
        do_matmul(n, k, m, a, b)
    case _ _ _:
      IO.print("dimensoes invalidas")

def run_ln2(mn: Maybe<&2, Nat>, md: Maybe<&2, Nat>, me: Maybe<&2, F32>, fx: String, fg: String, fb: String) -> IO(Unit):
  match mn md me:
    case Some{n} Some{d} Some{eps}:
      do IO<Unit>:
        x : List<&2, U32> <- read_bytes(fx)
        g : List<&2, U32> <- read_bytes(fg)
        b : List<&2, U32> <- read_bytes(fb)
        do_layernorm(n, d, eps, x, g, b)
    case _ _ _:
      IO.print("dimensoes invalidas")

def dispatch2(is_mm: Bool, is_mt: Bool, is_ln: Bool, args: List<String>) -> IO(Unit):
  match is_mm is_mt is_ln args:
    case True{} _ _ Con{n, Con{k, Con{m, Con{fa, Con{fb, Nil{}}}}}}:
      run_matmul(Nat.read(n), Nat.read(k), Nat.read(m), fa, fb)
    case False{} True{} _ Con{n, Con{k, Con{m, Con{fa, Con{fb, Nil{}}}}}}:
      run_matmul_t(Nat.read(n), Nat.read(k), Nat.read(m), fa, fb)
    case False{} False{} True{} Con{n, Con{d, Con{eps, Con{fx, Con{fg, Con{fb, Nil{}}}}}}}:
      run_ln2(Nat.read(n), Nat.read(d), F32.read(eps), fx, fg, fb)
    case _ _ _ _:
      IO.print("uso invalido")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, Con{r, Con{c, Con{f, Nil{}}}}}}:
      run_unary(Nat.read(r), Nat.read(c), which_of(mode), f)
    case Con{prog, Con{+mode, rest}}:
      dispatch2(String.eq(mode, "matmul"), String.eq(mode, "matmul_t"), String.eq(mode, "layernorm"), rest)
    case _:
      IO.print("uso invalido")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
