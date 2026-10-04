# bend-ml-autograd: automatic differentiation with a proved law.
#
#   import bend-ml-autograd@0.1.1.0/main.bend as AG
#
# 1) PROVED MODEL (over Nat): the reverse mode of autodiff gives the same result
#    as the forward mode (dual numbers). Proved by the kernel, using the sum and
#    product lemmas of bend-ml-nat-lemmas.
# 2) SCALAR AUTOGRAD in F32 (micrograd style), with the same structure.
# 3) LAYERS WITH TENSORS (forward and backward) whose shapes are checked by the type.
#
# F32 is not a real number, so the NUMERICAL correctness of F32 is validated by
# tests against PyTorch (reference/test_autograd.py), not by proof.

import Base
import bend-ml-nat-lemmas@0.1.0.0/main.bend as NL
import bend-ml-tensor@0.1.0.0/main.bend as T

# =====================================================================
# 1) Proved model
# =====================================================================

# Expressions with one variable NX, over Nat (a commutative semiring).
type NE is Data:
  NCst{n: Nat}
  NX{}
  NAdd{a: NE, b: NE}
  NMul{a: NE, b: NE}

# value of the expression at x
def nval(e: NE, +x: Nat) -> Nat:
  match e:
    case NCst{n}:
      n
    case NX{}:
      x
    case NAdd{a, b}:
      Nat.add(nval(a, x), nval(b, x))
    case NMul{+a, +b}:
      Nat.mul(nval(a, x), nval(b, x))

# FORWARD-mode derivative (dual numbers): derivative of the sum and the product rule
def nfwd(e: NE, +x: Nat) -> Nat:
  match e:
    case NCst{n}:
      0n
    case NX{}:
      1n
    case NAdd{a, b}:
      Nat.add(nfwd(a, x), nfwd(b, x))
    case NMul{+a, +b}:
      Nat.add(Nat.mul(nfwd(a, x), nval(b, x)), Nat.mul(nval(a, x), nfwd(b, x)))

# REVERSE-mode derivative: g is the gradient arriving from the output; each node passes on
# g (sum) or g * value-of-the-other-factor (product) to its children and adds up what
# arrives at NX.
def nbwd(e: NE, +x: Nat, +g: Nat) -> Nat:
  match e:
    case NCst{n}:
      0n
    case NX{}:
      g
    case NAdd{a, b}:
      Nat.add(nbwd(a, x, g), nbwd(b, x, g))
    case NMul{+a, +b}:
      Nat.add(nbwd(a, x, Nat.mul(g, nval(b, x))), nbwd(b, x, Nat.mul(g, nval(a, x))))

# a * (b + c) == a*b + a*c   (left distributivity; nat-lemmas only has the right one)
def ndist_l(+a: Nat, +b: Nat, +c: Nat) -> {Nat.mul(a, Nat.add(b, c)) == Nat.add(Nat.mul(a, b), Nat.mul(a, c)) : Nat}:
  Equal.trans(Nat, Nat.mul(a, Nat.add(b, c)), Nat.mul(Nat.add(b, c), a), Nat.add(Nat.mul(a, b), Nat.mul(a, c)),
    NL.mul_comm(a, Nat.add(b, c)),
    Equal.trans(Nat, Nat.mul(Nat.add(b, c), a), Nat.add(Nat.mul(b, a), Nat.mul(c, a)), Nat.add(Nat.mul(a, b), Nat.mul(a, c)),
      Equal.sym(Nat, Nat.add(Nat.mul(b, a), Nat.mul(c, a)), Nat.mul(Nat.add(b, c), a), NL.mul_dist(b, c, a)),
      Equal.trans(Nat, Nat.add(Nat.mul(b, a), Nat.mul(c, a)), Nat.add(Nat.mul(a, b), Nat.mul(c, a)), Nat.add(Nat.mul(a, b), Nat.mul(a, c)),
        Equal.cong(Nat, Nat, k => Nat.add(k, Nat.mul(c, a)), Nat.mul(b, a), Nat.mul(a, b), NL.mul_comm(b, a)),
        Equal.cong(Nat, Nat, k => Nat.add(Nat.mul(a, b), k), Nat.mul(c, a), Nat.mul(a, c), NL.mul_comm(c, a)))))

# The product case, with everything abstracted into numbers:
# (g*vb)*fa + (g*va)*fb == g * (fa*vb + va*fb)
def nmul_case(+g: Nat, +fa: Nat, +fb: Nat, +va: Nat, +vb: Nat, +ba: Nat, +bb: Nat, iha: {ba == Nat.mul(Nat.mul(g, vb), fa) : Nat}, ihb: {bb == Nat.mul(Nat.mul(g, va), fb) : Nat}) -> {Nat.add(ba, bb) == Nat.mul(g, Nat.add(Nat.mul(fa, vb), Nat.mul(va, fb))) : Nat}:
  Equal.trans(Nat, Nat.add(ba, bb), Nat.add(Nat.mul(Nat.mul(g, vb), fa), bb), Nat.mul(g, Nat.add(Nat.mul(fa, vb), Nat.mul(va, fb))),
    Equal.cong(Nat, Nat, k => Nat.add(k, bb), ba, Nat.mul(Nat.mul(g, vb), fa), iha),
    Equal.trans(Nat, Nat.add(Nat.mul(Nat.mul(g, vb), fa), bb), Nat.add(Nat.mul(Nat.mul(g, vb), fa), Nat.mul(Nat.mul(g, va), fb)), Nat.mul(g, Nat.add(Nat.mul(fa, vb), Nat.mul(va, fb))),
      Equal.cong(Nat, Nat, k => Nat.add(Nat.mul(Nat.mul(g, vb), fa), k), bb, Nat.mul(Nat.mul(g, va), fb), ihb),
      Equal.trans(Nat, Nat.add(Nat.mul(Nat.mul(g, vb), fa), Nat.mul(Nat.mul(g, va), fb)), Nat.add(Nat.mul(g, Nat.mul(fa, vb)), Nat.mul(Nat.mul(g, va), fb)), Nat.mul(g, Nat.add(Nat.mul(fa, vb), Nat.mul(va, fb))),
        Equal.cong(Nat, Nat, k => Nat.add(k, Nat.mul(Nat.mul(g, va), fb)), Nat.mul(Nat.mul(g, vb), fa), Nat.mul(g, Nat.mul(fa, vb)),
          Equal.trans(Nat, Nat.mul(Nat.mul(g, vb), fa), Nat.mul(g, Nat.mul(vb, fa)), Nat.mul(g, Nat.mul(fa, vb)),
            Equal.sym(Nat, Nat.mul(g, Nat.mul(vb, fa)), Nat.mul(Nat.mul(g, vb), fa), NL.mul_assoc(g, vb, fa)),
            Equal.cong(Nat, Nat, k => Nat.mul(g, k), Nat.mul(vb, fa), Nat.mul(fa, vb), NL.mul_comm(vb, fa)))),
        Equal.trans(Nat, Nat.add(Nat.mul(g, Nat.mul(fa, vb)), Nat.mul(Nat.mul(g, va), fb)), Nat.add(Nat.mul(g, Nat.mul(fa, vb)), Nat.mul(g, Nat.mul(va, fb))), Nat.mul(g, Nat.add(Nat.mul(fa, vb), Nat.mul(va, fb))),
          Equal.cong(Nat, Nat, k => Nat.add(Nat.mul(g, Nat.mul(fa, vb)), k), Nat.mul(Nat.mul(g, va), fb), Nat.mul(g, Nat.mul(va, fb)),
            Equal.sym(Nat, Nat.mul(g, Nat.mul(va, fb)), Nat.mul(Nat.mul(g, va), fb), NL.mul_assoc(g, va, fb))),
          Equal.sym(Nat, Nat.mul(g, Nat.add(Nat.mul(fa, vb), Nat.mul(va, fb))), Nat.add(Nat.mul(g, Nat.mul(fa, vb)), Nat.mul(g, Nat.mul(va, fb))), ndist_l(g, Nat.mul(fa, vb), Nat.mul(va, fb)))))))

# The reverse mode returns g times the forward-mode derivative.
def nbwd_ok(e: NE, +x: Nat, +g: Nat) -> {nbwd(e, x, g) == Nat.mul(g, nfwd(e, x)) : Nat}:
  match e:
    case NCst{n}:
      Equal.sym(Nat, Nat.mul(g, 0n), 0n, NL.mul_zero(g))
    case NX{}:
      Equal.sym(Nat, Nat.mul(g, 1n), g, NL.mul_one_r(g))
    case NAdd{+a, +b}:
      Equal.trans(Nat, Nat.add(nbwd(a, x, g), nbwd(b, x, g)), Nat.add(Nat.mul(g, nfwd(a, x)), nbwd(b, x, g)), Nat.mul(g, Nat.add(nfwd(a, x), nfwd(b, x))),
        Equal.cong(Nat, Nat, k => Nat.add(k, nbwd(b, x, g)), nbwd(a, x, g), Nat.mul(g, nfwd(a, x)), nbwd_ok(a, x, g)),
        Equal.trans(Nat, Nat.add(Nat.mul(g, nfwd(a, x)), nbwd(b, x, g)), Nat.add(Nat.mul(g, nfwd(a, x)), Nat.mul(g, nfwd(b, x))), Nat.mul(g, Nat.add(nfwd(a, x), nfwd(b, x))),
          Equal.cong(Nat, Nat, k => Nat.add(Nat.mul(g, nfwd(a, x)), k), nbwd(b, x, g), Nat.mul(g, nfwd(b, x)), nbwd_ok(b, x, g)),
          Equal.sym(Nat, Nat.mul(g, Nat.add(nfwd(a, x), nfwd(b, x))), Nat.add(Nat.mul(g, nfwd(a, x)), Nat.mul(g, nfwd(b, x))), ndist_l(g, nfwd(a, x), nfwd(b, x)))))
    case NMul{+a, +b}:
      nmul_case(g, nfwd(a, x), nfwd(b, x), nval(a, x), nval(b, x), nbwd(a, x, Nat.mul(g, nval(b, x))), nbwd(b, x, Nat.mul(g, nval(a, x))), nbwd_ok(a, x, Nat.mul(g, nval(b, x))), nbwd_ok(b, x, Nat.mul(g, nval(a, x))))

# LAW (reverse_eq_forward): for any expression made of constants, X, sums
# and products, the reverse-mode gradient (starting with gradient 1 at the output)
# equals the forward-mode derivative. It holds for every x.
law reverse_eq_forward:
  for +e: NE
  for +x: Nat
  {nbwd(e, x, 1n) == nfwd(e, x) : Nat}

def reverse_eq_forward(e, x):
  Equal.trans(Nat, nbwd(e, x, 1n), Nat.mul(1n, nfwd(e, x)), nfwd(e, x), nbwd_ok(e, x, 1n), NL.mul_one_l(nfwd(e, x)))


# =====================================================================
# 2) Scalar autograd in F32 (reverse mode, micrograd style)
# =====================================================================

type G is Data:
  GCst{c: F32}
  GVar{i: Nat}
  GAdd{a: G, b: G}
  GMul{a: G, b: G}
  GRelu{a: G}
  GTanh{a: G}
  GExp{a: G}

def lookup(env: List<&2, F32>, +i: Nat) -> F32:
  match env i:
    case Nil{} _:
      0.0
    case Con{h, t} 0n:
      h
    case Con{h, t} 1n+p:
      lookup(t, p)

def gval(e: G, +env: List<&2, F32>) -> F32:
  match e:
    case GCst{c}:
      c
    case GVar{i}:
      lookup(env, i)
    case GAdd{a, b}:
      (gval(a, env) + gval(b, env) : F32)
    case GMul{a, b}:
      (gval(a, env) * gval(b, env) : F32)
    case GRelu{a}:
      F32.max(gval(a, env), 0.0)
    case GTanh{a}:
      F32.tanh(gval(a, env))
    case GExp{a}:
      F32.exp(gval(a, env))

# adds v to the gradient of variable i
def acc_add(acc: List<&2, F32>, +i: Nat, +v: F32) -> List<&2, F32>:
  match acc i:
    case Nil{} _:
      Nil{}
    case Con{+h, t} 0n:
      (h + v : F32) <> t
    case Con{h, t} 1n+p:
      h <> acc_add(t, p, v)

def mask2(pos: Bool) -> F32:
  match pos:
    case True{}:
      1.0
    case False{}:
      0.0

# local derivative of relu: 1 if the input > 0, else 0
def relu_d(+x: F32) -> F32:
  mask2(F32.is_gt(x, 0.0))

def tanh_d(+x: F32) -> F32:
  (1.0 - (F32.tanh(x) * F32.tanh(x) : F32) : F32)

# g is the gradient arriving at the node; acc accumulates the gradient of each variable
def gbwd(e: G, +env: List<&2, F32>, +g: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match e:
    case GCst{c}:
      acc
    case GVar{+i}:
      acc_add(acc, i, g)
    case GAdd{a, b}:
      gbwd(b, env, g, gbwd(a, env, g, acc))
    case GMul{+a, +b}:
      gbwd(b, env, (g * gval(a, env) : F32), gbwd(a, env, (g * gval(b, env) : F32), acc))
    case GRelu{+a}:
      gbwd(a, env, (g * relu_d(gval(a, env)) : F32), acc)
    case GTanh{+a}:
      gbwd(a, env, (g * tanh_d(gval(a, env)) : F32), acc)
    case GExp{+a}:
      gbwd(a, env, (g * F32.exp(gval(a, env)) : F32), acc)

# gradient of the expression with respect to each variable of env
def ggrad(e: G, +env: List<&2, F32>) -> List<&2, F32>:
  gbwd(e, env, 1.0, T.fill.l(List.length(&2, F32, env), 0.0))

# =====================================================================
# 3) Layers with tensors: the shape of each gradient is imposed by the TYPE
# =====================================================================

# y = x·W + b           x: n x i,  W: i x o,  b: o   ->   y: n x o
def linear(-n: Nat, +i: Nat, +o: Nat, x: T.Mat<n, i>, w: T.Mat<i, o>, b: T.Vec<o>) -> T.Mat<n, o>:
  T.Mat.add_row(n, o, T.Mat.matmul(n, i, o, x, w), b)

# Given the gradient dy arriving at y, returns (dx, (dW, db)):
#   dx = dy · Wᵀ   (n x o)·(o x i) = n x i
#   dW = xᵀ · dy   (i x n)·(n x o) = i x o
#   db = sum of the rows of dy
# If any of these products had the wrong dimensions, the program would not compile.
def linear_bwd(+n: Nat, +i: Nat, +o: Nat, x: T.Mat<n, i>, w: T.Mat<i, o>, +dy: T.Mat<n, o>) -> T.Mat<n, i> & (T.Mat<i, o> & T.Vec<o>):
  (T.Mat.matmul(n, o, i, dy, T.Mat.transpose(i, o, w)), (T.Mat.matmul(i, n, o, T.Mat.transpose(n, i, x), dy), T.Mat.col_sums(n, o, dy)))

def mask.l(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      relu_d(h) <> mask.l(t)

def mask.rows(xs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      mask.l(r) <> mask.rows(t)

# y = relu(x); the gradient passes where x > 0
def relu_bwd(-r: Nat, -c: Nat, x: T.Mat<r, c>, dy: T.Mat<r, c>) -> T.Mat<r, c>:
  match x:
    case T.Mat{rows}:
      T.Mat.mul(r, c, dy, T.Mat{mask.rows(rows)})

# one-hot: n rows of c numbers, with 1 at the label's position
def onehot.cell(hit: Bool) -> F32:
  mask2(hit)

def onehot.row(c: Nat, +k: Nat, +j: Nat) -> List<&2, F32>:
  match c:
    case 0n:
      Nil{}
    case 1n+p:
      onehot.cell(Nat.is_eq(k, j)) <> onehot.row(p, k, 1n+j)

def onehot.rows(labels: List<&2, Nat>, +c: Nat) -> List<&2, List<&2, F32>>:
  match labels:
    case Nil{}:
      Nil{}
    case Con{k, t}:
      onehot.row(c, k, 0n) <> onehot.rows(t, c)

def log.l(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      F32.log(F32.max(h, 0.000000000001)) <> log.l(t)

def log.rows(xs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      log.l(r) <> log.rows(t)

def rows.total(xs: List<&2, List<&2, F32>>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{r, t}:
      rows.total(t, (acc + T.sum.go(r, 0.0) : F32))

def ce_loss2(-n: Nat, -c: Nat, +cc: Nat, +nf: F32, p: T.Mat<n, c>, labels: List<&2, Nat>) -> F32:
  match p:
    case T.Mat{rows}:
      (0.0 - (rows.total(T.rows.zip_mul(log.rows(rows), onehot.rows(labels, cc)), 0.0) / nf : F32) : F32)

# mean cross-entropy loss: -(1/n) Σ log p[i][label_i]
def ce_loss(+n: Nat, +c: Nat, logits: T.Mat<n, c>, labels: List<&2, Nat>) -> F32:
  ce_loss2(n, c, c, F32.from_nat(n), T.Mat.softmax(n, c, logits), labels)

# gradient of the loss with respect to the logits: (softmax - one-hot) / n
def ce_grad(+n: Nat, +c: Nat, logits: T.Mat<n, c>, labels: List<&2, Nat>) -> T.Mat<n, c>:
  T.Mat.scale(n, c, (1.0 / F32.from_nat(n) : F32), T.Mat.sub(n, c, T.Mat.softmax(n, c, logits), T.Mat{onehot.rows(labels, c)}))

def sgd(-r: Nat, -c: Nat, +lr: F32, w: T.Mat<r, c>, dw: T.Mat<r, c>) -> T.Mat<r, c>:
  T.Mat.sub(r, c, w, T.Mat.scale(r, c, lr, dw))

def sgd_vec(-n: Nat, +lr: F32, b: T.Vec<n>, db: T.Vec<n>) -> T.Vec<n>:
  T.Vec.sub(n, b, T.Vec.scale(n, lr, db))
