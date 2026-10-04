# bend-ml-tensor: tensores com a shape no TIPO. Erro de shape é erro de tipo.
#
#   import bend-ml-tensor@0.1.0.0/main.bend as T
#
# Vec<n> é um vetor de n números F32; Mat<r, c> é uma matriz r x c (lista de r
# linhas com c números). As dimensões são parâmetros de tipo apagados: não custam
# nada em tempo de execução, mas o checker as confere em toda operação.

import Base
import bend-ml-nat-lemmas@0.1.0.0/main.bend as NL

type Vec<-n: Nat> is Data:
  Vec{xs: List<&2, F32>}

type Mat<-r: Nat, -c: Nat> is Data:
  Mat{rows: List<&2, List<&2, F32>>}

# ---------------------------------------------------------------------
# listas de F32 (internas; as operações públicas fixam as dimensões)
# ---------------------------------------------------------------------

def zip_with.add(xs: List<&2, F32>, ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Con{+h, t} Con{+g, u}:
      (h + g : F32) <> zip_with.add(t, u)
    case _ _:
      Nil{}

def zip_with.sub(xs: List<&2, F32>, ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Con{+h, t} Con{+g, u}:
      (h - g : F32) <> zip_with.sub(t, u)
    case _ _:
      Nil{}

def zip_with.mul(xs: List<&2, F32>, ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Con{+h, t} Con{+g, u}:
      (h * g : F32) <> zip_with.mul(t, u)
    case _ _:
      Nil{}

def scale.l(+k: F32, xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      (k * h : F32) <> scale.l(k, t)

def dot.go(xs: List<&2, F32>, ys: List<&2, F32>, acc: F32) -> F32:
  match xs ys:
    case Con{+h, t} Con{+g, u}:
      dot.go(t, u, (acc + (h * g : F32) : F32))
    case _ _:
      acc

def sum.go(xs: List<&2, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      sum.go(t, (acc + h : F32))

def relu.l(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      F32.max(h, 0.0) <> relu.l(t)

def fill.l(n: Nat, +v: F32) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      v <> fill.l(p, v)

# ---------------------------------------------------------------------
# Vec
# ---------------------------------------------------------------------

def Vec.zeros(n: Nat) -> Vec<n>:
  Vec{fill.l(n, 0.0)}

def Vec.fill(n: Nat, +v: F32) -> Vec<n>:
  Vec{fill.l(n, v)}

def Vec.add(-n: Nat, a: Vec<n>, b: Vec<n>) -> Vec<n>:
  match a b:
    case Vec{xs} Vec{ys}:
      Vec{zip_with.add(xs, ys)}

def Vec.sub(-n: Nat, a: Vec<n>, b: Vec<n>) -> Vec<n>:
  match a b:
    case Vec{xs} Vec{ys}:
      Vec{zip_with.sub(xs, ys)}

def Vec.mul(-n: Nat, a: Vec<n>, b: Vec<n>) -> Vec<n>:
  match a b:
    case Vec{xs} Vec{ys}:
      Vec{zip_with.mul(xs, ys)}

def Vec.scale(-n: Nat, +k: F32, a: Vec<n>) -> Vec<n>:
  match a:
    case Vec{xs}:
      Vec{scale.l(k, xs)}

def Vec.dot(-n: Nat, a: Vec<n>, b: Vec<n>) -> F32:
  match a b:
    case Vec{xs} Vec{ys}:
      dot.go(xs, ys, 0.0)

def Vec.sum(-n: Nat, a: Vec<n>) -> F32:
  match a:
    case Vec{xs}:
      sum.go(xs, 0.0)

def Vec.relu(-n: Nat, a: Vec<n>) -> Vec<n>:
  match a:
    case Vec{xs}:
      Vec{relu.l(xs)}

# ---------------------------------------------------------------------
# listas de linhas (internas)
# ---------------------------------------------------------------------

def heads(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Con{Con{+h, t}, rest}:
      h <> heads(rest)
    case _:
      Nil{}

def tails(rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Con{Con{h, t}, rest}:
      t <> tails(rest)
    case _:
      Nil{}

# transposta de uma matriz com `c` colunas (c é quem encolhe, vem primeiro)
def transpose.go(c: Nat, +rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match c:
    case 0n:
      Nil{}
    case 1n+p:
      heads(rows) <> transpose.go(p, tails(rows))

def rows.zip_add(xs: List<&2, List<&2, F32>>, ys: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs ys:
    case Con{a, t} Con{b, u}:
      zip_with.add(a, b) <> rows.zip_add(t, u)
    case _ _:
      Nil{}

def rows.zip_sub(xs: List<&2, List<&2, F32>>, ys: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs ys:
    case Con{a, t} Con{b, u}:
      zip_with.sub(a, b) <> rows.zip_sub(t, u)
    case _ _:
      Nil{}

def rows.zip_mul(xs: List<&2, List<&2, F32>>, ys: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs ys:
    case Con{a, t} Con{b, u}:
      zip_with.mul(a, b) <> rows.zip_mul(t, u)
    case _ _:
      Nil{}

def rows.scale(+k: F32, xs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      scale.l(k, r) <> rows.scale(k, t)

def rows.relu(xs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      relu.l(r) <> rows.relu(t)

# soma por colunas
def rows.col_sums(c: Nat, +rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match c:
    case 0n:
      Nil{}
    case 1n+p:
      sum.go(heads(rows), 0.0) <> rows.col_sums(p, tails(rows))

def rows.add_row(xs: List<&2, List<&2, F32>>, +b: List<&2, F32>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      zip_with.add(r, b) <> rows.add_row(t, b)

def matmul.row(+ra: List<&2, F32>, bt: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match bt:
    case Nil{}:
      Nil{}
    case Con{col, t}:
      dot.go(ra, col, 0.0) <> matmul.row(ra, t)

def matmul.rows(a: List<&2, List<&2, F32>>, +bt: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match a:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      matmul.row(r, bt) <> matmul.rows(t, bt)

# junta todas as linhas numa lista só
def flatten.l(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      List.append(&2, F32, r, flatten.l(t))

# quebra uma lista em r linhas de c números
def chunk.l(r: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, List<&2, F32>>:
  match r:
    case 0n:
      Nil{}
    case 1n+p:
      List.take(&2, F32, xs, c) <> chunk.l(p, c, List.drop(&2, F32, xs, c))

# ---------------------------------------------------------------------
# Mat
# ---------------------------------------------------------------------

def Mat.zeros(+r: Nat, +c: Nat) -> Mat<r, c>:
  Mat{chunk.l(r, c, fill.l(Nat.mul(r, c), 0.0))}

def Mat.fill(+r: Nat, +c: Nat, +v: F32) -> Mat<r, c>:
  Mat{chunk.l(r, c, fill.l(Nat.mul(r, c), v))}

def Mat.add(-r: Nat, -c: Nat, a: Mat<r, c>, b: Mat<r, c>) -> Mat<r, c>:
  match a b:
    case Mat{x} Mat{y}:
      Mat{rows.zip_add(x, y)}

def Mat.sub(-r: Nat, -c: Nat, a: Mat<r, c>, b: Mat<r, c>) -> Mat<r, c>:
  match a b:
    case Mat{x} Mat{y}:
      Mat{rows.zip_sub(x, y)}

# produto elemento a elemento (Hadamard)
def Mat.mul(-r: Nat, -c: Nat, a: Mat<r, c>, b: Mat<r, c>) -> Mat<r, c>:
  match a b:
    case Mat{x} Mat{y}:
      Mat{rows.zip_mul(x, y)}

def Mat.scale(-r: Nat, -c: Nat, +k: F32, a: Mat<r, c>) -> Mat<r, c>:
  match a:
    case Mat{x}:
      Mat{rows.scale(k, x)}

def Mat.relu(-r: Nat, -c: Nat, a: Mat<r, c>) -> Mat<r, c>:
  match a:
    case Mat{x}:
      Mat{rows.relu(x)}

def Mat.transpose(-r: Nat, +c: Nat, a: Mat<r, c>) -> Mat<c, r>:
  match a:
    case Mat{x}:
      Mat{transpose.go(c, x)}

# (n x k) · (k x m) = (n x m): as dimensões internas têm que ser a mesma k
def Mat.matmul(-n: Nat, +k: Nat, +m: Nat, a: Mat<n, k>, b: Mat<k, m>) -> Mat<n, m>:
  match a b:
    case Mat{x} Mat{y}:
      Mat{matmul.rows(x, transpose.go(m, y))}

# (n x k) · (m x k)ᵀ = (n x m): o segundo operando já vem TRANSPOSTO (guardado como
# m linhas de k números). Evita transpor pesos grandes a cada chamada; o tipo
# continua exigindo a mesma dimensão interna k.
def Mat.matmul_t(-n: Nat, -k: Nat, -m: Nat, a: Mat<n, k>, bt: Mat<m, k>) -> Mat<n, m>:
  match a bt:
    case Mat{x} Mat{y}:
      Mat{matmul.rows(x, y)}

# soma bias (um vetor de m números) a cada linha de uma matriz n x m
def Mat.add_row(-n: Nat, -m: Nat, a: Mat<n, m>, b: Vec<m>) -> Mat<n, m>:
  match a b:
    case Mat{x} Vec{v}:
      Mat{rows.add_row(x, v)}

# soma de cada coluna: Mat<n, m> -> Vec<m>
def Mat.col_sums(-n: Nat, +m: Nat, a: Mat<n, m>) -> Vec<m>:
  match a:
    case Mat{x}:
      Vec{rows.col_sums(m, x)}

# matriz · vetor: (n x k) · k = n
def Mat.matvec(-n: Nat, +k: Nat, a: Mat<n, k>, v: Vec<k>) -> Vec<n>:
  match a v:
    case Mat{x} Vec{ys}:
      Vec{heads(matmul.rows(x, ys <> Nil{}))}

# Reshape: só compila com a PROVA de que o número de elementos não muda.
def Mat.reshape(-r1: Nat, -c1: Nat, r2: Nat, +c2: Nat, -p: {Nat.mul(r1, c1) == Nat.mul(r2, c2) : Nat}, a: Mat<r1, c1>) -> Mat<r2, c2>:
  match a:
    case Mat{x}:
      Mat{chunk.l(r2, c2, flatten.l(x))}

# ---------------------------------------------------------------------
# funções por linha: softmax, GELU, layernorm
# ---------------------------------------------------------------------

def max.go(xs: List<&2, F32>, cur: F32) -> F32:
  match xs:
    case Nil{}:
      cur
    case Con{+h, t}:
      max.go(t, F32.max(cur, h))

def max.l(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{+h, t}:
      max.go(t, h)

def exp_shift(xs: List<&2, F32>, +m: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      F32.exp((h - m : F32)) <> exp_shift(t, m)

def div.l(xs: List<&2, F32>, +s: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      (h / s : F32) <> div.l(t, s)

# softmax estável: subtrai o máximo antes de exponenciar
def softmax.fin(+es: List<&2, F32>) -> List<&2, F32>:
  div.l(es, sum.go(es, 0.0))

def softmax.l(+xs: List<&2, F32>) -> List<&2, F32>:
  softmax.fin(exp_shift(xs, max.l(xs)))

def rows.softmax(xs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      softmax.l(r) <> rows.softmax(t)

# GELU com a aproximação por tanh (a mesma do GPT-2)
def gelu.f(+x: F32) -> F32:
  (0.5 * (x * (1.0 + F32.tanh((0.7978846 * (x + (0.044715 * (x * (x * x : F32) : F32) : F32) : F32) : F32)) : F32) : F32) : F32)

def gelu.l(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      gelu.f(h) <> gelu.l(t)

def rows.gelu(xs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      gelu.l(r) <> rows.gelu(t)

def sub_const(xs: List<&2, F32>, +k: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      (h - k : F32) <> sub_const(t, k)

def sq_sum(xs: List<&2, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      sq_sum(t, (acc + (h * h : F32) : F32))

# (x - média) / sqrt(variância + eps), depois * gamma + beta
def layernorm.fin(+cs: List<&2, F32>, +d: F32, +eps: F32, +gamma: List<&2, F32>, +beta: List<&2, F32>) -> List<&2, F32>:
  zip_with.add(zip_with.mul(scale.l((1.0 / F32.sqrt(((sq_sum(cs, 0.0) / d : F32) + eps : F32)) : F32), cs), gamma), beta)

def layernorm.l(+xs: List<&2, F32>, +d: F32, +eps: F32, +gamma: List<&2, F32>, +beta: List<&2, F32>) -> List<&2, F32>:
  layernorm.fin(sub_const(xs, (sum.go(xs, 0.0) / d : F32)), d, eps, gamma, beta)

def rows.layernorm(xs: List<&2, List<&2, F32>>, +d: F32, +eps: F32, +gamma: List<&2, F32>, +beta: List<&2, F32>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      layernorm.l(r, d, eps, gamma, beta) <> rows.layernorm(t, d, eps, gamma, beta)

def Mat.softmax(-r: Nat, -c: Nat, a: Mat<r, c>) -> Mat<r, c>:
  match a:
    case Mat{x}:
      Mat{rows.softmax(x)}

def Mat.gelu(-r: Nat, -c: Nat, a: Mat<r, c>) -> Mat<r, c>:
  match a:
    case Mat{x}:
      Mat{rows.gelu(x)}

# normaliza cada linha (d colunas) e aplica gamma e beta, vetores de d números
def Mat.layernorm(-n: Nat, d: Nat, eps: F32, g: Vec<d>, bt: Vec<d>, a: Mat<n, d>) -> Mat<n, d>:
  match g bt a:
    case Vec{gamma} Vec{beta} Mat{x}:
      Mat{rows.layernorm(x, F32.from_nat(d), eps, gamma, beta)}

def Vec.softmax(-n: Nat, a: Vec<n>) -> Vec<n>:
  match a:
    case Vec{xs}:
      Vec{softmax.l(xs)}

# ---------------------------------------------------------------------
# construção e leitura com verificação de tamanho
# ---------------------------------------------------------------------

def len.eq(xs: List<&2, F32>, n: Nat) -> Bool:
  match xs n:
    case Nil{} 0n:
      True{}
    case Con{h, t} 1n+p:
      len.eq(t, p)
    case _ _:
      False{}

def rows.ok(rs: List<&2, List<&2, F32>>, +c: Nat, r: Nat) -> Bool:
  match rs r:
    case Nil{} 0n:
      True{}
    case Con{row, t} 1n+p:
      Bool.and(len.eq(row, c), rows.ok(t, c, p))
    case _ _:
      False{}

def Vec.of2(-n: Nat, ok: Bool, xs: List<&2, F32>) -> Maybe<&2, Vec<n>>:
  match ok:
    case True{}:
      Some{Vec{xs}}
    case False{}:
      None{}

# Vec<n> a partir de uma lista, só se ela tem exatamente n números
def Vec.of(n: Nat, +xs: List<&2, F32>) -> Maybe<&2, Vec<n>>:
  Vec.of2(n, len.eq(xs, n), xs)

def Mat.of2(-r: Nat, -c: Nat, ok: Bool, rs: List<&2, List<&2, F32>>) -> Maybe<&2, Mat<r, c>>:
  match ok:
    case True{}:
      Some{Mat{rs}}
    case False{}:
      None{}

# Mat<r, c> a partir de linhas, só se são exatamente r linhas de c números
def Mat.of(+r: Nat, +c: Nat, +rs: List<&2, List<&2, F32>>) -> Maybe<&2, Mat<r, c>>:
  Mat.of2(r, c, rows.ok(rs, c, r), rs)

def Vec.to_list(-n: Nat, a: Vec<n>) -> List<&2, F32>:
  match a:
    case Vec{xs}:
      xs

def Mat.to_rows(-r: Nat, -c: Nat, a: Mat<r, c>) -> List<&2, List<&2, F32>>:
  match a:
    case Mat{x}:
      x

# ---------------------------------------------------------------------
# LAWS: quando um reshape é possível, e uma prova reaproveitada do nat-lemmas
# ---------------------------------------------------------------------

# LEI (reshape_swap): r x c e c x r têm o mesmo número de elementos
# (a multiplicação é comutativa), então esse reshape sempre existe.
law reshape_swap:
  for +r: Nat
  for +c: Nat
  {Nat.mul(r, c) == Nat.mul(c, r) : Nat}

def reshape_swap(r, c):
  NL.mul_comm(r, c)

# LEI (reshape_flat): r x c tem o mesmo número de elementos que 1 x (r*c)
# (multiplicar por 1 não muda nada), então achatar uma matriz sempre existe.
law reshape_flat:
  for +r: Nat
  for +c: Nat
  {Nat.mul(r, c) == Nat.mul(1n, Nat.mul(r, c)) : Nat}

def reshape_flat(r, c):
  Equal.sym(Nat, Nat.mul(1n, Nat.mul(r, c)), Nat.mul(r, c), NL.mul_one_l(Nat.mul(r, c)))

# Mat<r, c> -> Mat<1, r*c>: a prova vem da lei acima
def Mat.flatten(+r: Nat, +c: Nat, a: Mat<r, c>) -> Mat<1n, Nat.mul(r, c)>:
  Mat.reshape(r, c, 1n, Nat.mul(r, c), reshape_flat(r, c), a)

# Mat<r, c> -> Mat<c, r> pelos mesmos dados (não é a transposta!): só reinterpreta
def Mat.reshape_swap(+r: Nat, +c: Nat, a: Mat<r, c>) -> Mat<c, r>:
  Mat.reshape(r, c, c, r, reshape_swap(r, c), a)
