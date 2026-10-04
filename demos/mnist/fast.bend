# MLP 784-128-10 no MNIST sobre Array<F32> plano (v2): gemm com strides, sem listas.
#
#   bend demos/mnist/fast.bend -o /tmp/mnist_fast && /tmp/mnist_fast <épocas> <máx. lotes (0 = todos)> <lr>
#
# Mesmos dados, pesos iniciais, ordem de lotes e lr do train.bend e do reference/mnist_torch.py.

import Base

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
# tipos de apoio
# ---------------------------------------------------------------------

type R is Type:
  R{a: Array<F32>, b: Array<F32>, x: F32}

type G is Type:
  G{a: Array<F32>, b: Array<F32>, c: Array<F32>}

# menor d com 2^d >= n (n >= 2)

def dotg(k: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, acc: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> R:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      R{a, b, acc}
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      dotg(p, (ia + sa : U32), (ib + sb : U32), sa, sb, (acc + (x * y : F32) : F32), Array.get(F32, a, (ia + sa : U32)), Array.get(F32, b, (ib + sb : U32)))

# a linha i de C, colunas j.. ; r é o produto da coluna atual (já calculado)

def gcols(mleft: Nat, +kn: Nat, +ia0: U32, +ib0: U32, +sa: U32, +sb: U32, +bcol: U32, +cidx: U32, c: Array<F32>, r: R) -> G:
  match mleft r:
    case 0n R{a, b, +x}:
      G{a, b, Array.set(F32, c, cidx, x)}
    case 1n+p R{a, b, +x}:
      gcols(p, kn, ia0, (ib0 + bcol : U32), sa, sb, bcol, (cidx + 1 : U32), Array.set(F32, c, cidx, x), dotg(kn, ia0, (ib0 + bcol : U32), sa, sb, 0.0, Array.get(F32, a, ia0), Array.get(F32, b, (ib0 + bcol : U32))))

def grows(nleft: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +mu: U32, g: G) -> G:
  match nleft g:
    case 0n G{a, b, c}:
      G{a, b, c}
    case 1n+p G{a, b, c}:
      grows(p, mm1, kn, (i + 1 : U32), arow, sa, sb, bcol, mu, gcols(mm1, kn, (i * arow : U32), 0, sa, sb, bcol, (i * mu : U32), c, dotg(kn, (i * arow : U32), 0, sa, sb, 0.0, Array.get(F32, a, (i * arow : U32)), Array.get(F32, b, 0))))

# C (n x m) = A · B con k passos; c é o array de saída (com capacidade n*m)

def gemm(+n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  grows(n, Nat.sub(m, 1n), k, 0, arow, sa, sb, bcol, U32.from_nat(m), G{a, b, c})


# ---------------------------------------------------------------------
# gemm paralelo: 2^d blocos de linhas de C, cada um com a sua cópia de A e B,
# devolvendo listas que são escritas em C no fim
# ---------------------------------------------------------------------

type S3 is Type:
  S3{a: Array<F32>, b: Array<F32>, out: List<&2, F32>}

def lcols(mleft: Nat, +kn: Nat, +ia0: U32, +ib0: U32, +sa: U32, +sb: U32, +bcol: U32, out: List<&2, F32>, r: R) -> S3:
  match mleft r:
    case 0n R{a, b, +x}:
      S3{a, b, x <> out}
    case 1n+p R{a, b, +x}:
      lcols(p, kn, ia0, (ib0 + bcol : U32), sa, sb, bcol, x <> out, dotg(kn, ia0, (ib0 + bcol : U32), sa, sb, 0.0, Array.get(F32, a, ia0), Array.get(F32, b, (ib0 + bcol : U32))))

def lrows(nleft: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, st: S3) -> S3:
  match nleft st:
    case 0n S3{a, b, out}:
      S3{a, b, out}
    case 1n+p S3{a, b, out}:
      lrows(p, mm1, kn, (i + 1 : U32), arow, sa, sb, bcol, lcols(mm1, kn, (i * arow : U32), 0, sa, sb, bcol, out, dotg(kn, (i * arow : U32), 0, sa, sb, 0.0, Array.get(F32, a, (i * arow : U32)), Array.get(F32, b, 0))))

def lblock_fin(st: S3) -> List<&2, F32>:
  match st:
    case S3{a, b, out}:
      List.reverse(&2, F32, out)

def lblock(cnt: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>) -> List<&2, F32>:
  lblock_fin(lrows(cnt, mm1, kn, i, arow, sa, sb, bcol, S3{a, b, Nil{}}))

def lpar(d: Nat, +cnt: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, ra: Array<F32> & Array<F32>, rb: Array<F32> & Array<F32>) -> List<&2, F32>:
  match d ra rb:
    case 0n Tuple{a, a2} Tuple{b, b2}:
      lblock(cnt, mm1, kn, i, arow, sa, sb, bcol, a, b)
    case 1n++q Tuple{a, a2} Tuple{b, b2}:
      x y = lpar(q, Nat.div(cnt, 2n), mm1, kn, i, arow, sa, sb, bcol, Array.clone(F32, a), Array.clone(F32, b)) lpar(q, Nat.sub(cnt, Nat.div(cnt, 2n)), mm1, kn, (i + U32.from_nat(Nat.div(cnt, 2n)) : U32), arow, sa, sb, bcol, Array.clone(F32, a2), Array.clone(F32, b2))
      List.append(&2, F32, x, y)

def write_l(xs: List<&2, F32>, +idx: U32, c: Array<F32>) -> Array<F32>:
  match xs:
    case Nil{}:
      c
    case Con{h, t}:
      write_l(t, (idx + 1 : U32), Array.set(F32, c, idx, h))

# c[idx..] += lista

def gp3(a: Array<F32>, b: Array<F32>, c: Array<F32>, xs: List<&2, F32>) -> G:
  G{a, b, write_l(xs, 0, c)}

def gp2(d: Nat, +n: Nat, +mm1: Nat, +kn: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, c: Array<F32>, pa: Array<F32> & Array<F32>, pb: Array<F32> & Array<F32>) -> G:
  match pa pb:
    case Tuple{a1, a2} Tuple{b1, b2}:
      gp3(a1, b1, c, lpar(d, n, mm1, kn, 0, arow, sa, sb, bcol, Array.clone(F32, a2), Array.clone(F32, b2)))

# C (n x m) = A · B, com 2^d blocos de linhas em paralelo; devolve A, B (intactos) e C

def gemm_par(d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  gp2(d, n, Nat.sub(m, 1n), k, arow, sa, sb, bcol, c, Array.clone(F32, a), Array.clone(F32, b))

# ---------------------------------------------------------------------
# operações elementwise (em lugar, por índice)
# ---------------------------------------------------------------------

def depth(n: Nat) -> Nat:
  Nat.add(U32.log2(U32.from_nat(Nat.sub(n, 1n))), 1n)

# ---------------------------------------------------------------------
# gemm: C[i][j] = soma_p A[i*arow + p*sa] * B[p*sb + j*bcol]
# (nn: arow=K, sa=1, sb=M, bcol=1 | nt: B guardada m x k: sb=1, bcol=K | tn: A guardada k x n: arow=1, sa=N)
# ---------------------------------------------------------------------

# acc += A[ia + t*sa] * B[ib + t*sb] por k passos; ra, rb já trazem o primeiro par

type P2 is Type:
  P2{a: Array<F32>, b: Array<F32>}

type AL is Type:
  AL{c: Array<F32>, l: List<&2, F32>}

# lista de m elementos a partir de idx (lê com Array.get, devolvendo o array)

def read_l(mleft: Nat, +idx: U32, acc: List<&2, F32>, rc: Array<F32> & F32) -> AL:
  match mleft rc:
    case 0n Tuple{c, +v}:
      AL{c, List.reverse(&2, F32, v <> acc)}
    case 1n+p Tuple{c, +v}:
      read_l(p, (idx + 1 : U32), v <> acc, Array.get(F32, c, (idx + 1 : U32)))

# escreve a lista a partir de idx

def addl(xs: List<&2, F32>, +idx: U32, rc: Array<F32> & F32) -> Array<F32>:
  match xs rc:
    case Nil{} Tuple{c, v}:
      c
    case Con{+h, t} Tuple{c, +v}:
      addl(t, (idx + 1 : U32), Array.get(F32, Array.set(F32, c, idx, (v + h : F32)), (idx + 1 : U32)))

# soma bias (lista de m) a cada uma das n linhas de c (n x m)

def bias_rows(nleft: Nat, +i: U32, +mu: U32, +bl: List<&2, F32>, c: Array<F32>) -> Array<F32>:
  match nleft:
    case 0n:
      c
    case 1n+p:
      bias_rows(p, (i + 1 : U32), mu, bl, addl(bl, (i * mu : U32), Array.get(F32, c, (i * mu : U32))))

# relu em lugar nos primeiros n elementos

def relu_l(nleft: Nat, +t: U32, rc: Array<F32> & F32) -> Array<F32>:
  match nleft rc:
    case 0n Tuple{c, v}:
      c
    case 1n+p Tuple{c, +v}:
      relu_l(p, (t + 1 : U32), Array.get(F32, Array.set(F32, c, t, F32.max(v, 0.0)), (t + 1 : U32)))

def pos2(b: Bool) -> F32:
  match b:
    case True{}:
      1.0
    case False{}:
      0.0

def pos(+y: F32) -> F32:
  pos2(F32.is_gt(y, 0.0))

# d[t] *= (h[t] > 0): gradiente da relu; devolve os dois arrays

def mask_l(nleft: Nat, +t: U32, rd: Array<F32> & F32, rh: Array<F32> & F32) -> P2:
  match nleft rd rh:
    case 0n Tuple{d, x} Tuple{h, y}:
      P2{d, h}
    case 1n+p Tuple{d, +x} Tuple{h, +y}:
      mask_l(p, (t + 1 : U32), Array.get(F32, Array.set(F32, d, t, (x * pos(y) : F32)), (t + 1 : U32)), Array.get(F32, h, (t + 1 : U32)))

# w[t] -= lr * dw[t]

def maxl2(xs: List<&2, F32>, cur: F32) -> F32:
  match xs:
    case Nil{}:
      cur
    case Con{+h, t}:
      maxl2(t, F32.max(cur, h))

def maxl(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{+h, t}:
      maxl2(t, h)

def expl(xs: List<&2, F32>, +m: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      F32.exp((h - m : F32)) <> expl(t, m)

def suml(xs: List<&2, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      suml(t, (acc + h : F32))

def divl(xs: List<&2, F32>, +s: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      (h / s : F32) <> divl(t, s)

def softmax2(+es: List<&2, F32>) -> List<&2, F32>:
  divl(es, suml(es, 0.0))

def softmax(+xs: List<&2, F32>) -> List<&2, F32>:
  softmax2(expl(xs, maxl(xs)))

# p[label]

def sgd_l(nleft: Nat, +t: U32, +lr: F32, rw: Array<F32> & F32, rd: Array<F32> & F32) -> P2:
  match nleft rw rd:
    case 0n Tuple{w, x} Tuple{d, y}:
      P2{w, d}
    case 1n+p Tuple{w, +x} Tuple{d, +y}:
      sgd_l(p, (t + 1 : U32), lr, Array.get(F32, Array.set(F32, w, t, (x - (lr * y : F32) : F32)), (t + 1 : U32)), Array.get(F32, d, (t + 1 : U32)))

# ---------------------------------------------------------------------
# softmax + entropia cruzada por linha (listas de 10 elementos: custo desprezível)
# ---------------------------------------------------------------------

def pick(ps: List<&2, F32>, n: Nat) -> F32:
  match ps n:
    case Con{+h, t} 0n:
      h
    case Con{h, t} 1n+q:
      pick(t, q)
    case Nil{} _:
      0.0

# (p - onehot(label)) / n

def hit(b: Bool) -> F32:
  match b:
    case True{}:
      1.0
    case False{}:
      0.0

def grad_l(ps: List<&2, F32>, +label: Nat, +j: Nat, +nf: F32) -> List<&2, F32>:
  match ps:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      (((h - hit(Nat.is_eq(label, j)) : F32) / nf) : F32) <> grad_l(t, label, 1n+j, nf)

type L2 is Type:
  L2{loss: F32, c: Array<F32>}

def ce_row2(+ps: List<&2, F32>, +label: Nat, +nf: F32, +idx: U32, loss: F32, c: Array<F32>) -> L2:
  L2{(loss - F32.log(F32.max(pick(ps, label), 0.000000000001)) : F32), write_l(grad_l(ps, label, 0n, nf), idx, c)}

def ce_row(al: AL, +label: Nat, +nf: F32, +idx: U32, loss: F32) -> L2:
  match al:
    case AL{c, l}:
      ce_row2(softmax(l), label, nf, idx, loss, c)

# percorre as linhas de z2 (n x m): r traz a perda acumulada e o array; ao fim, o array virou o gradiente

def ce_rows(labels: List<&2, Nat>, +mm1: Nat, +mu: U32, +nf: F32, +i: U32, r: L2) -> L2:
  match labels r:
    case Nil{} L2{loss, c}:
      L2{loss, c}
    case Con{+y, t} L2{loss, c}:
      ce_rows(t, mm1, mu, nf, (i + 1 : U32), ce_row(read_l(mm1, (i * mu : U32), Nil{}, Array.get(F32, c, (i * mu : U32))), y, nf, (i * mu : U32), loss))

# ---------------------------------------------------------------------
# um passo de treino: forward, perda, backward e SGD (lote de 100)
# ---------------------------------------------------------------------

type Pm is Type:
  Pm{w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, b2: Array<F32>}

type St2 is Type:
  St2{loss: F32, pm: Pm}

# 4. atualização dos parâmetros

def u4(+lr: F32, w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, loss: F32, r: P2) -> St2:
  match r:
    case P2{b2, db2}:
      St2{loss, Pm{w1, b1, w2, b2}}

def u3(+lr: F32, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, db2: Array<F32>, loss: F32, r: P2) -> St2:
  match r:
    case P2{w2, dw2}:
      u4(lr, w1, b1, w2, loss, sgd_l(10n, 0, lr, Array.get(F32, b2, 0), Array.get(F32, db2, 0)))

def u3b(+lr: F32, w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, dw2: Array<F32>, b2: Array<F32>, db2: Array<F32>, loss: F32) -> St2:
  u3(lr, w1, b1, b2, db2, loss, sgd_l(1280n, 0, lr, Array.get(F32, w2, 0), Array.get(F32, dw2, 0)))

def u2(+lr: F32, w1: Array<F32>, w2: Array<F32>, dw2: Array<F32>, b2: Array<F32>, db2: Array<F32>, loss: F32, r: P2) -> St2:
  match r:
    case P2{b1, db1}:
      u3b(lr, w1, b1, w2, dw2, b2, db2, loss)

def u1(+lr: F32, b1: Array<F32>, w2: Array<F32>, dw2: Array<F32>, b2: Array<F32>, db2: Array<F32>, db1: Array<F32>, loss: F32, r: P2) -> St2:
  match r:
    case P2{w1, dw1}:
      u2(lr, w1, w2, dw2, b2, db2, loss, sgd_l(128n, 0, lr, Array.get(F32, b1, 0), Array.get(F32, db1, 0)))

# 3. gradientes da primeira camada

def f12(+lr: F32, w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, dw2: Array<F32>, b2: Array<F32>, db2: Array<F32>, dw1: Array<F32>, loss: F32, g: G) -> St2:
  match g:
    case G{ones, dz1, db1}:
      u1(lr, b1, w2, dw2, b2, db2, db1, loss, sgd_l(100352n, 0, lr, Array.get(F32, w1, 0), Array.get(F32, dw1, 0)))

def f11(+lr: F32, w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, dw2: Array<F32>, b2: Array<F32>, db2: Array<F32>, loss: F32, g: G) -> St2:
  match g:
    case G{x, dz1, dw1}:
      f12(lr, w1, b1, w2, dw2, b2, db2, dw1, loss, gemm(1n, 128n, 100n, 0, 1, 128, 1, Array.new(F32, 7n, 1.0), dz1, Array.new(F32, 7n, 0.0)))

def f10(+lr: F32, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, dw2: Array<F32>, b2: Array<F32>, db2: Array<F32>, loss: F32, r: P2) -> St2:
  match r:
    case P2{dz1, h}:
      f11(lr, w1, b1, w2, dw2, b2, db2, loss, gemm_par(3n, 784n, 128n, 100n, 1, 784, 128, 1, x, dz1, Array.new(F32, 17n, 0.0)))

# 2. gradientes da segunda camada e da relu

def f9(+lr: F32, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, dw2: Array<F32>, b2: Array<F32>, db2: Array<F32>, h: Array<F32>, loss: F32, g: G) -> St2:
  match g:
    case G{g2, w2b, dh}:
      f10(lr, x, w1, b1, w2b, dw2, b2, db2, loss, mask_l(12800n, 0, Array.get(F32, dh, 0), Array.get(F32, h, 0)))

def f8(+lr: F32, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, dw2: Array<F32>, b2: Array<F32>, h: Array<F32>, loss: F32, g: G) -> St2:
  match g:
    case G{ones, g2, db2}:
      f9(lr, x, w1, b1, dw2, b2, db2, h, loss, gemm(100n, 128n, 10n, 10, 1, 1, 10, g2, w2, Array.new(F32, 14n, 0.0)))

def f7(+lr: F32, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, b2: Array<F32>, loss: F32, g: G) -> St2:
  match g:
    case G{h, g2, dw2}:
      f8(lr, x, w1, b1, w2, dw2, b2, h, loss, gemm(1n, 10n, 100n, 0, 1, 10, 1, Array.new(F32, 7n, 1.0), g2, Array.new(F32, 4n, 0.0)))

def f6(+lr: F32, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, h: Array<F32>, w2: Array<F32>, b2: Array<F32>, r: L2) -> St2:
  match r:
    case L2{loss, g2}:
      f7(lr, x, w1, b1, w2, b2, (loss / 100.0 : F32), gemm(128n, 10n, 100n, 1, 128, 10, 1, h, g2, Array.new(F32, 11n, 0.0)))

# 1. forward da segunda camada e perda

def f5(+lr: F32, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, h: Array<F32>, w2: Array<F32>, z2: Array<F32>, labels: List<&2, Nat>, al: AL) -> St2:
  match al:
    case AL{b2, bl2}:
      f6(lr, x, w1, b1, h, w2, b2, ce_rows(labels, 9n, 10, 100.0, 0, L2{0.0, bias_rows(100n, 0, 10, bl2, z2)}))

def f4(+lr: F32, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, labels: List<&2, Nat>, g: G) -> St2:
  match g:
    case G{h, w2, z2}:
      f5(lr, x, w1, b1, h, w2, z2, labels, read_l(9n, 0, Nil{}, Array.get(F32, b2, 0)))

# 0. forward da primeira camada

def f4b(+lr: F32, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, b2: Array<F32>, labels: List<&2, Nat>, h: Array<F32>) -> St2:
  f4(lr, x, w1, b1, b2, labels, gemm(100n, 10n, 128n, 128, 1, 10, 1, h, w2, Array.new(F32, 10n, 0.0)))

def f3(+lr: F32, x: Array<F32>, w1: Array<F32>, w2: Array<F32>, b2: Array<F32>, labels: List<&2, Nat>, z1: Array<F32>, al: AL) -> St2:
  match al:
    case AL{b1, bl}:
      f4b(lr, x, w1, b1, w2, b2, labels, relu_l(12800n, 0, Array.get(F32, bias_rows(100n, 0, 128, bl, z1), 0)))

def f2(+lr: F32, w2: Array<F32>, b1: Array<F32>, b2: Array<F32>, labels: List<&2, Nat>, g: G) -> St2:
  match g:
    case G{x, w1, z1}:
      f3(lr, x, w1, w2, b2, labels, z1, read_l(127n, 0, Nil{}, Array.get(F32, b1, 0)))

def step(pm: Pm, x: Array<F32>, labels: List<&2, Nat>, +lr: F32) -> St2:
  match pm:
    case Pm{w1, b1, w2, b2}:
      f2(lr, w2, b1, b2, labels, gemm_par(3n, 100n, 128n, 784n, 784, 1, 128, 1, x, w1, Array.new(F32, 14n, 0.0)))

# ---------------------------------------------------------------------
# avaliação: acertos em um lote de 100 (só forward)
# ---------------------------------------------------------------------

type Cn is Type:
  Cn{n: Nat, pm: Pm}

type CR is Type:
  CR{c: Array<F32>, n: Nat}

def gt_head(xs: List<&2, F32>, +v: F32) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Con{+h, t}:
      F32.is_gt(h, v)

def argmax2(xs: List<&2, F32>, g: Bool, +i: Nat, +bi: Nat, +bv: F32) -> Nat:
  match xs g:
    case Nil{} _:
      bi
    case Con{+h, +t} True{}:
      argmax2(t, gt_head(t, h), 1n+i, i, h)
    case Con{h, +t} False{}:
      argmax2(t, gt_head(t, bv), 1n+i, bi, bv)

def argmax(xs: List<&2, F32>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{+h, +t}:
      argmax2(t, gt_head(t, h), 1n, 0n, h)

def one(b: Bool) -> Nat:
  match b:
    case True{}:
      1n
    case False{}:
      0n

def count_rows3(+y: Nat, n: Nat, al: AL) -> CR:
  match al:
    case AL{c, l}:
      CR{c, Nat.add(n, one(Nat.is_eq(argmax(l), y)))}

def count_rows(labels: List<&2, Nat>, +mm1: Nat, +mu: U32, +i: U32, r: CR) -> CR:
  match labels r:
    case Nil{} CR{c, n}:
      CR{c, n}
    case Con{+y, t} CR{c, n}:
      count_rows(t, mm1, mu, (i + 1 : U32), count_rows3(y, n, read_l(mm1, (i * mu : U32), Nil{}, Array.get(F32, c, (i * mu : U32)))))

def e7(w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, b2: Array<F32>, r: CR) -> Cn:
  match r:
    case CR{c, n}:
      Cn{n, Pm{w1, b1, w2, b2}}

def e6(w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, labels: List<&2, Nat>, z2: Array<F32>, al: AL) -> Cn:
  match al:
    case AL{b2, bl2}:
      e7(w1, b1, w2, b2, count_rows(labels, 9n, 10, 0, CR{bias_rows(100n, 0, 10, bl2, z2), 0n}))

def e5(w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, labels: List<&2, Nat>, g: G) -> Cn:
  match g:
    case G{h, w2, z2}:
      e6(w1, b1, w2, labels, z2, read_l(9n, 0, Nil{}, Array.get(F32, b2, 0)))

def e4(w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, b2: Array<F32>, labels: List<&2, Nat>, h: Array<F32>) -> Cn:
  e5(w1, b1, b2, labels, gemm(100n, 10n, 128n, 128, 1, 10, 1, h, w2, Array.new(F32, 10n, 0.0)))

def e3(w1: Array<F32>, w2: Array<F32>, b2: Array<F32>, labels: List<&2, Nat>, z1: Array<F32>, al: AL) -> Cn:
  match al:
    case AL{b1, bl}:
      e4(w1, b1, w2, b2, labels, relu_l(12800n, 0, Array.get(F32, bias_rows(100n, 0, 128, bl, z1), 0)))

def e2(b1: Array<F32>, w2: Array<F32>, b2: Array<F32>, labels: List<&2, Nat>, g: G) -> Cn:
  match g:
    case G{x, w1, z1}:
      e3(w1, w2, b2, labels, z1, read_l(127n, 0, Nil{}, Array.get(F32, b1, 0)))

def evb(pm: Pm, x: Array<F32>, labels: List<&2, Nat>) -> Cn:
  match pm:
    case Pm{w1, b1, w2, b2}:
      e2(b1, w2, b2, labels, gemm(100n, 128n, 784n, 784, 1, 128, 1, x, w1, Array.new(F32, 14n, 0.0)))

# ---------------------------------------------------------------------
# dados e laço de épocas
# ---------------------------------------------------------------------

def fillx(bs: List<&2, U32>, +i: U32, a: Array<F32>) -> Array<F32>:
  match bs:
    case Nil{}:
      a
    case Con{+b, t}:
      fillx(t, (i + 1 : U32), Array.set(F32, a, i, (U32.to_f32(b) / 255.0 : F32)))

def fill_l(xs: List<&2, F32>, +i: U32, a: Array<F32>) -> Array<F32>:
  match xs:
    case Nil{}:
      a
    case Con{h, t}:
      fill_l(t, (i + 1 : U32), Array.set(F32, a, i, h))

def labs_nat(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> labs_nat(t)

def mkx(bs: List<&2, U32>) -> Array<F32>:
  fillx(bs, 0, Array.new(F32, 17n, 0.0))

# treina nb lotes; r traz a perda acumulada e os parâmetros do passo anterior

def tloop(k: Nat, +b: U32, +lr: F32, acc: F32, r: St2) -> IO(St2):
  match k r:
    case 0n St2{l, pm}:
      IO.pure(St2, St2{(acc + l : F32), pm})
    case 1n+p St2{l, pm}:
      do IO<St2>:
        xs : List<&2, U32> <- read_at("demos/mnist/data/train-images-idx3-ubyte", (16 + (b * 784 * 100 : U32) : U32), 78400)
        ys : List<&2, U32> <- read_at("demos/mnist/data/train-labels-idx1-ubyte", (8 + (b * 100 : U32) : U32), 100)
        tloop(p, (b + 1 : U32), lr, (acc + l : F32), step(pm, mkx(xs), labs_nat(ys), lr))

def eloop(k: Nat, +b: U32, acc: Nat, r: Cn) -> IO(Cn):
  match k r:
    case 0n Cn{n, pm}:
      IO.pure(Cn, Cn{Nat.add(acc, n), pm})
    case 1n+p Cn{n, pm}:
      do IO<Cn>:
        xs : List<&2, U32> <- read_at("demos/mnist/data/t10k-images-idx3-ubyte", (16 + (b * 784 * 100 : U32) : U32), 78400)
        ys : List<&2, U32> <- read_at("demos/mnist/data/t10k-labels-idx1-ubyte", (8 + (b * 100 : U32) : U32), 100)
        eloop(p, (b + 1 : U32), Nat.add(acc, n), evb(pm, mkx(xs), labs_nat(ys)))

def ms_show(+ms: Nat) -> String:
  Nat.show(Nat.div(ms, 1000n)) ++ "." ++ Nat.show(Nat.div(Nat.mod(ms, 1000n), 100n))

def report(+k: Nat, +nb: Nat, t0: Nat, t1: Nat, loss: F32, c: Cn) -> IO(Pm):
  match c:
    case Cn{n, pm}:
      do IO<Pm>:
        IO.print("epoca " ++ Nat.show(k) ++ " loss_treino=" ++ F32.show((loss / F32.from_nat(nb) : F32)) ++ " acertos_teste=" ++ Nat.show(n) ++ "/10000 segundos_treino=" ++ ms_show(Nat.sub(t1, t0)))
        return pm

def after(+k: Nat, +nb: Nat, t0: Nat, t1: Nat, loss: F32, r: Cn) -> IO(Pm):
  report(k, nb, t0, t1, loss, r)

def epoch_end(+k: Nat, +nb: Nat, t0: Nat, t1: Nat, r: St2) -> IO(Pm):
  match r:
    case St2{loss, pm}:
      do IO<Pm>:
        c : Cn <- eloop(100n, 0, 0n, Cn{0n, pm})
        after(k, nb, t0, t1, loss, c)

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

def load_arr(path: String, +n: Nat, +d: Nat) -> IO(Array<F32>):
  do IO<Array<F32>>:
    bs : List<&2, U32> <- read_bytes(path)
    return fill_l(floats(bs), 0, Array.new(F32, d, 0.0))

def go3(+e: Nat, +nb: Nat, +lr: F32) -> IO(Unit):
  do IO<Unit>:
    w1 : Array<F32> <- load_arr("demos/mnist/data/init/w1.txt", 100352n, 17n)
    b1 : Array<F32> <- load_arr("demos/mnist/data/init/b1.txt", 128n, 7n)
    w2 : Array<F32> <- load_arr("demos/mnist/data/init/w2.txt", 1280n, 11n)
    b2 : Array<F32> <- load_arr("demos/mnist/data/init/b2.txt", 10n, 4n)
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
